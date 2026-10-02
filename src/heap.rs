//! A general-purpose heap: boundary tags, segregated free lists, splitting and coalescing.
//!
//! The heap owns *chunks* of address space obtained from [`crate::os`] and carves them into blocks.
//!
//! ```text
//! chunk:  | Chunk header (40 bytes) | block | block | ... | block | sentinel (8 bytes) |
//!
//! allocated block:  [ header ][ payload ............................................ ]
//! free block:       [ header ][ next ][ prev ] ..................... [ footer = size ]
//! header:           size (a multiple of 16) | PREV_INUSE (2) | INUSE (1)
//! ```
//!
//! * **Alignment.** Blocks start at addresses that are 8 mod 16, so a payload (header + 8) is 16-byte
//!   aligned. Sizes are multiples of 16 and at least 32 (header, two list links, footer).
//! * **Boundary tags.** A free block repeats its size in its last word. When a block is freed, the
//!   word before it says whether the block before is free (the `PREV_INUSE` bit of its own header)
//!   and, if so, how far back that block starts. The word after it is the next block's header. So
//!   both neighbours are found in constant time and merged, and two free blocks are never adjacent.
//!   An allocated block carries no footer: its payload runs to the end of the block.
//! * **Free lists.** Free blocks are kept in 160 bins by size: exact bins every 16 bytes up to 512,
//!   then four bins per doubling. A bitmap says which bins are non-empty, so the next bin that can
//!   satisfy a request is found with a few bit operations. Within the request's own bin the best of
//!   the first 64 candidates is taken.
//! * **Splitting.** The front of a block is used and the remainder, if it is at least 32 bytes,
//!   becomes a new free block.
//! * **Alignment above 16.** A larger-aligned request takes a block with room to spare, gives the
//!   gap in front back as a free block, and uses the aligned part.
//! * **Giving memory back.** When a block turns out to be an entire chunk, the chunk is returned to
//!   the system, except that one standard-size spare chunk is kept to avoid mapping and unmapping in
//!   a loop.
//!
//! The heap is not thread-safe; [`crate::Tmalloc`] puts a lock around it.

use crate::os;

const HDR: usize = 8;
const MIN_BLOCK: usize = 32;
const INUSE: usize = 1;
const PREV_INUSE: usize = 2;
const FLAGS: usize = 15;
/// Offset of the first block from the start of its chunk (so that blocks sit at 8 mod 16).
const FIRST: usize = 40;
/// The end-of-chunk sentinel: a header that is always "in use" and has size zero.
const SENTINEL: usize = 8;
/// Space in a chunk that is not usable by blocks.
const OVERHEAD: usize = FIRST + SENTINEL;

const EXACT_BINS: usize = 31; // sizes 32, 48, ..., 512
const NBINS: usize = 160;
/// How far into the request's own bin to look for a better fit.
const SCAN_LIMIT: usize = 64;
/// How many fully free standard chunks are kept instead of unmapped.
const SPARE_CHUNKS: usize = 1;

#[repr(C)]
struct Chunk {
    next: *mut Chunk,
    /// Total bytes mapped for this chunk.
    size: usize,
}

#[repr(C)]
struct Free {
    hdr: usize,
    next: *mut Free,
    prev: *mut Free,
}

/// Counters, for tests and for the benchmark.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HeapStats {
    /// Bytes of address space currently mapped for chunks.
    pub mapped: usize,
    pub peak_mapped: usize,
    /// Bytes in allocated blocks, headers included.
    pub in_use: usize,
    pub peak_in_use: usize,
    pub chunks: usize,
    pub allocs: u64,
    pub frees: u64,
}

pub struct Heap {
    bins: [*mut Free; NBINS],
    bitmap: [u64; 3],
    chunks: *mut Chunk,
    chunk_size: usize,
    free_chunks: usize,
    limit: usize,
    stats: HeapStats,
}

// The heap owns the memory its raw pointers point into.
unsafe impl Send for Heap {}

// ---- block field access --------------------------------------------------------------------

#[inline(always)]
unsafe fn hdr(b: *mut u8) -> usize {
    *(b as *const usize)
}

#[inline(always)]
unsafe fn set_hdr(b: *mut u8, v: usize) {
    *(b as *mut usize) = v
}

#[inline(always)]
unsafe fn size(b: *mut u8) -> usize {
    hdr(b) & !FLAGS
}

#[inline(always)]
unsafe fn next_block(b: *mut u8) -> *mut u8 {
    b.add(size(b))
}

#[inline(always)]
unsafe fn set_footer(b: *mut u8, sz: usize) {
    *(b.add(sz - HDR) as *mut usize) = sz
}

/// Sets or clears the PREV_INUSE bit of the block that follows a block whose state just changed.
#[inline(always)]
unsafe fn set_prev_flag(b: *mut u8, in_use: bool) {
    let h = hdr(b);
    set_hdr(b, if in_use { h | PREV_INUSE } else { h & !PREV_INUSE })
}

/// The block size needed for a request of `n` payload bytes, or None if it cannot be represented.
#[inline(always)]
fn block_size(n: usize) -> Option<usize> {
    let total = n.checked_add(HDR + 15)? & !15;
    Some(total.max(MIN_BLOCK))
}

fn bin_index(sz: usize) -> usize {
    debug_assert!(sz >= MIN_BLOCK && sz.is_multiple_of(16));
    if sz <= 512 {
        (sz - MIN_BLOCK) / 16
    } else {
        let lg = 63 - sz.leading_zeros() as usize;
        (EXACT_BINS + (lg - 9) * 4 + ((sz >> (lg - 2)) & 3)).min(NBINS - 1)
    }
}

impl Heap {
    /// A heap that maps `chunk_size` bytes (a multiple of the page size, at least 64 KiB) at a time.
    pub const fn new(chunk_size: usize) -> Heap {
        Heap {
            bins: [std::ptr::null_mut(); NBINS],
            bitmap: [0; 3],
            chunks: std::ptr::null_mut(),
            chunk_size,
            free_chunks: 0,
            limit: usize::MAX,
            stats: HeapStats { mapped: 0, peak_mapped: 0, in_use: 0, peak_in_use: 0, chunks: 0, allocs: 0, frees: 0 },
        }
    }

    /// Refuses to map more than `bytes` in total, so `alloc` returns null instead. For testing
    /// out-of-memory behaviour.
    pub fn set_limit(&mut self, bytes: usize) {
        self.limit = bytes;
    }

    pub fn stats(&self) -> HeapStats {
        self.stats
    }

    /// The largest payload that fits in a standard chunk.
    pub fn max_standard_payload(&self) -> usize {
        self.chunk_size - OVERHEAD - HDR
    }

    // ---- free lists -------------------------------------------------------------------------

    unsafe fn bin_insert(&mut self, b: *mut u8) {
        let sz = size(b);
        let i = bin_index(sz);
        let f = b as *mut Free;
        (*f).prev = std::ptr::null_mut();
        (*f).next = self.bins[i];
        if !self.bins[i].is_null() {
            (*self.bins[i]).prev = f;
        }
        self.bins[i] = f;
        self.bitmap[i / 64] |= 1 << (i % 64);
        if sz == self.chunk_size - OVERHEAD && self.whole_chunk(b, sz).is_some() {
            self.free_chunks += 1;
        }
    }

    unsafe fn bin_remove(&mut self, b: *mut u8) {
        let sz = size(b);
        let i = bin_index(sz);
        let f = b as *mut Free;
        if !(*f).prev.is_null() {
            (*(*f).prev).next = (*f).next;
        } else {
            self.bins[i] = (*f).next;
        }
        if !(*f).next.is_null() {
            (*(*f).next).prev = (*f).prev;
        }
        if self.bins[i].is_null() {
            self.bitmap[i / 64] &= !(1 << (i % 64));
        }
        if sz == self.chunk_size - OVERHEAD && self.whole_chunk(b, sz).is_some() {
            self.free_chunks -= 1;
        }
    }

    /// The first non-empty bin at or after `from`.
    fn next_bin(&self, from: usize) -> Option<usize> {
        if from >= NBINS {
            return None;
        }
        let mut w = from / 64;
        let mut bits = self.bitmap[w] & (!0u64 << (from % 64));
        loop {
            if bits != 0 {
                return Some(w * 64 + bits.trailing_zeros() as usize);
            }
            w += 1;
            if w == self.bitmap.len() {
                return None;
            }
            bits = self.bitmap[w];
        }
    }

    /// Finds a free block of at least `need` bytes and removes it from its bin.
    unsafe fn find(&mut self, need: usize) -> Option<*mut u8> {
        let bin = bin_index(need);
        if bin < EXACT_BINS {
            if !self.bins[bin].is_null() {
                let b = self.bins[bin] as *mut u8;
                self.bin_remove(b);
                return Some(b);
            }
        } else {
            // A range bin may hold blocks smaller than `need`: take the smallest that fits.
            let mut best: *mut Free = std::ptr::null_mut();
            let mut cur = self.bins[bin];
            let mut seen = 0;
            while !cur.is_null() && seen < SCAN_LIMIT {
                let s = size(cur as *mut u8);
                if s >= need && (best.is_null() || s < size(best as *mut u8)) {
                    best = cur;
                    if s == need {
                        break;
                    }
                }
                cur = (*cur).next;
                seen += 1;
            }
            if !best.is_null() {
                self.bin_remove(best as *mut u8);
                return Some(best as *mut u8);
            }
        }
        // Every block in a later bin is larger than anything in this one.
        let i = self.next_bin(bin + 1)?;
        let b = self.bins[i] as *mut u8;
        self.bin_remove(b);
        Some(b)
    }

    // ---- chunks -----------------------------------------------------------------------------

    /// Maps a new chunk that can hold a block of `need` bytes and puts its free block in a bin.
    unsafe fn grow(&mut self, need: usize) -> bool {
        let want = need.checked_add(OVERHEAD).map(os::round_up_page);
        let size_ = match want {
            Some(w) => w.max(self.chunk_size),
            None => return false,
        };
        if self.stats.mapped.checked_add(size_).is_none_or(|m| m > self.limit) {
            return false;
        }
        let base = os::map(size_);
        if base.is_null() {
            return false;
        }
        let c = base as *mut Chunk;
        (*c).next = self.chunks;
        (*c).size = size_;
        self.chunks = c;
        self.stats.mapped += size_;
        self.stats.peak_mapped = self.stats.peak_mapped.max(self.stats.mapped);
        self.stats.chunks += 1;
        let b = base.add(FIRST);
        let usable = size_ - OVERHEAD;
        set_hdr(b, usable | PREV_INUSE);
        set_footer(b, usable);
        // The sentinel: always in use; the block before it is free.
        set_hdr(b.add(usable), INUSE);
        self.bin_insert(b);
        true
    }

    /// The chunk whose entire usable area is the block `b` of size `sz`, if there is one.
    unsafe fn whole_chunk(&self, b: *mut u8, sz: usize) -> Option<*mut Chunk> {
        let mut c = self.chunks;
        while !c.is_null() {
            if (c as *mut u8).add(FIRST) == b {
                return ((*c).size - OVERHEAD == sz).then_some(c);
            }
            c = (*c).next;
        }
        None
    }

    unsafe fn release_chunk(&mut self, c: *mut Chunk) {
        let mut link: *mut *mut Chunk = &mut self.chunks;
        while *link != c {
            link = &mut (**link).next;
        }
        *link = (*c).next;
        let sz = (*c).size;
        self.stats.mapped -= sz;
        self.stats.chunks -= 1;
        os::unmap(c as *mut u8, sz);
    }

    // ---- allocation -------------------------------------------------------------------------

    /// Turns the unlinked free block `b` into an allocated block of `need` bytes, returning the
    /// rest, if it is big enough to be a block, to the free lists.
    unsafe fn use_block(&mut self, b: *mut u8, need: usize) {
        let total = size(b);
        let prev = hdr(b) & PREV_INUSE;
        let rest = total - need;
        if rest >= MIN_BLOCK {
            set_hdr(b, need | INUSE | prev);
            let r = b.add(need);
            set_hdr(r, rest | PREV_INUSE);
            set_footer(r, rest);
            // The block after `r` already records that its predecessor is free.
            self.bin_insert(r);
            self.stats.in_use += need;
        } else {
            set_hdr(b, total | INUSE | prev);
            set_prev_flag(b.add(total), true);
            self.stats.in_use += total;
        }
        self.stats.peak_in_use = self.stats.peak_in_use.max(self.stats.in_use);
        self.stats.allocs += 1;
    }

    /// Allocates `size` bytes aligned to `align` (a power of two). Returns null on failure.
    ///
    /// # Safety
    /// `align` must be a power of two.
    pub unsafe fn alloc(&mut self, size: usize, align: usize) -> *mut u8 {
        debug_assert!(align.is_power_of_two());
        let Some(need) = block_size(size) else { return std::ptr::null_mut() };
        if align <= 16 {
            let b = match self.find(need) {
                Some(b) => b,
                None => {
                    if !self.grow(need) {
                        return std::ptr::null_mut();
                    }
                    match self.find(need) {
                        Some(b) => b,
                        None => return std::ptr::null_mut(),
                    }
                }
            };
            self.use_block(b, need);
            return b.add(HDR);
        }
        self.alloc_aligned(need, align)
    }

    unsafe fn alloc_aligned(&mut self, need: usize, align: usize) -> *mut u8 {
        // Room for the request, for the worst gap before the aligned address, and for the gap to be
        // a block of its own.
        let Some(worst) = need.checked_add(align).and_then(|n| n.checked_add(MIN_BLOCK)) else { return std::ptr::null_mut() };
        let b = match self.find(worst) {
            Some(b) => b,
            None => {
                if !self.grow(worst) {
                    return std::ptr::null_mut();
                }
                match self.find(worst) {
                    Some(b) => b,
                    None => return std::ptr::null_mut(),
                }
            }
        };
        let total = size(b);
        let prev = hdr(b) & PREV_INUSE;
        let payload = b as usize + HDR;
        let mut aligned = (payload + align - 1) & !(align - 1);
        if aligned != payload && aligned - payload < MIN_BLOCK {
            // The gap would be too small to be a block; take the next aligned address.
            aligned += align;
        }
        let gap = aligned - payload;
        let a = if gap == 0 {
            b
        } else {
            // Give the front back as a free block and continue with the rest.
            set_hdr(b, gap | prev);
            set_footer(b, gap);
            self.bin_insert(b);
            let a = b.add(gap);
            // The block in front of `a` is free.
            set_hdr(a, (total - gap) | INUSE);
            a
        };
        self.use_block(a, need);
        a.add(HDR)
    }

    /// Frees a block returned by `alloc`.
    ///
    /// # Safety
    /// `p` came from this heap's `alloc` (or a successful `realloc_in_place`) and is not used again.
    pub unsafe fn free(&mut self, p: *mut u8) {
        let b = p.sub(HDR);
        let h = hdr(b);
        debug_assert!(h & INUSE != 0, "double free or a pointer that is not a block");
        let mut start = b;
        let mut total = h & !FLAGS;
        self.stats.in_use -= total;
        self.stats.frees += 1;
        let following = b.add(total);
        // Merge with the block before, if it is free: its size is the word just before ours.
        if h & PREV_INUSE == 0 {
            let before = *(b.sub(HDR) as *const usize);
            start = b.sub(before);
            self.bin_remove(start);
            total += before;
        }
        // Merge with the block after, if it is free.
        if hdr(following) & INUSE == 0 {
            self.bin_remove(following);
            total += size(following);
        }
        // By the invariant that no two free blocks touch, the block before `start` is in use.
        set_hdr(start, total | PREV_INUSE);
        set_footer(start, total);
        set_prev_flag(start.add(total), false);
        // A block that is a whole chunk goes back to the system unless it is the one spare.
        if total >= self.chunk_size - OVERHEAD {
            if let Some(c) = self.whole_chunk(start, total) {
                if (*c).size != self.chunk_size || self.free_chunks >= SPARE_CHUNKS {
                    self.release_chunk(c);
                    return;
                }
            }
        }
        self.bin_insert(start);
    }

    /// Payload bytes available in the block `p` points to.
    ///
    /// # Safety
    /// `p` is a live allocation from this heap.
    pub unsafe fn usable_size(&self, p: *mut u8) -> usize {
        size(p.sub(HDR)) - HDR
    }

    /// Tries to resize the allocation `p` to `new_size` bytes without moving it, by splitting off
    /// the tail or by absorbing a free neighbour. Returns false, changing nothing, if that is not
    /// possible.
    ///
    /// # Safety
    /// `p` is a live allocation from this heap.
    pub unsafe fn realloc_in_place(&mut self, p: *mut u8, new_size: usize) -> bool {
        let Some(need) = block_size(new_size) else { return false };
        let b = p.sub(HDR);
        let mut total = size(b);
        let prev = hdr(b) & PREV_INUSE;
        let following = b.add(total);
        if need > total {
            // Grow into a free block after this one, if it is big enough.
            if hdr(following) & INUSE != 0 || total + size(following) < need {
                return false;
            }
            self.bin_remove(following);
            let merged = total + size(following);
            self.stats.in_use += merged - total;
            self.stats.peak_in_use = self.stats.peak_in_use.max(self.stats.in_use);
            total = merged;
            set_hdr(b, total | INUSE | prev);
            // The block after the absorbed one used to follow a free block and now follows ours.
            set_prev_flag(b.add(total), true);
        }
        let rest = total - need;
        if rest >= MIN_BLOCK {
            // Split the tail off, merging it with a free block after it.
            set_hdr(b, need | INUSE | prev);
            self.stats.in_use -= rest;
            let mut r_size = rest;
            let r = b.add(need);
            let after = r.add(rest);
            if hdr(after) & INUSE == 0 {
                self.bin_remove(after);
                r_size += size(after);
            }
            set_hdr(r, r_size | PREV_INUSE);
            set_footer(r, r_size);
            set_prev_flag(r.add(r_size), false);
            self.bin_insert(r);
        }
        true
    }

    // ---- inspection -------------------------------------------------------------------------

    /// Walks every chunk and every bin and checks every invariant the heap relies on. Slow; for
    /// tests. Returns a description of the first violation.
    pub fn check(&self) -> Result<(), String> {
        unsafe { self.check_inner() }
    }

    unsafe fn check_inner(&self) -> Result<(), String> {
        let mut free_blocks = 0usize;
        let mut used_bytes = 0usize;
        let mut mapped = 0usize;
        let mut nchunks = 0usize;
        let mut whole_std = 0usize;
        let mut c = self.chunks;
        while !c.is_null() {
            nchunks += 1;
            let csize = (*c).size;
            mapped += csize;
            if !(c as usize).is_multiple_of(os::GRANULE) || !csize.is_multiple_of(os::PAGE) {
                return Err(format!("chunk {c:p} is misaligned or has a bad size {csize}"));
            }
            let end = (c as *mut u8).add(csize - SENTINEL);
            let mut b = (c as *mut u8).add(FIRST);
            let mut prev_used = true;
            let mut first = true;
            while b != end {
                if b > end {
                    return Err(format!("block {b:p} runs past the end of its chunk"));
                }
                if (b as usize) % 16 != 8 {
                    return Err(format!("block {b:p} is not at 8 mod 16"));
                }
                let h = hdr(b);
                let sz = h & !FLAGS;
                if sz < MIN_BLOCK || !sz.is_multiple_of(16) {
                    return Err(format!("block {b:p} has size {sz}"));
                }
                if (h & PREV_INUSE != 0) != prev_used {
                    return Err(format!(
                        "block {b:p}: PREV_INUSE is {} but the block before is {}",
                        h & PREV_INUSE != 0,
                        if prev_used { "in use" } else { "free" }
                    ));
                }
                if h & INUSE != 0 {
                    used_bytes += sz;
                    prev_used = true;
                } else {
                    if !prev_used {
                        return Err(format!("two free blocks touch at {b:p}"));
                    }
                    let footer = *(b.add(sz - HDR) as *const usize);
                    if footer != sz {
                        return Err(format!("free block {b:p} has footer {footer}, size {sz}"));
                    }
                    free_blocks += 1;
                    prev_used = false;
                    if first && b.add(sz) == end && csize == self.chunk_size {
                        whole_std += 1;
                    }
                }
                first = false;
                b = b.add(sz);
            }
            let sh = hdr(end);
            if sh & INUSE == 0 || sh & !FLAGS != 0 {
                return Err(format!("sentinel {end:p} is damaged: {sh:#x}"));
            }
            if (sh & PREV_INUSE != 0) != prev_used {
                return Err(format!("sentinel {end:p} disagrees about the last block"));
            }
            c = (*c).next;
        }
        // Every free block is in exactly the right bin, and the lists are well formed.
        let mut listed = 0usize;
        for (i, &head) in self.bins.iter().enumerate() {
            let bit = self.bitmap[i / 64] & (1 << (i % 64)) != 0;
            if bit != !head.is_null() {
                return Err(format!("bin {i}: bitmap says {bit}, head is {head:p}"));
            }
            let mut prev: *mut Free = std::ptr::null_mut();
            let mut cur = head;
            while !cur.is_null() {
                let b = cur as *mut u8;
                let h = hdr(b);
                if h & INUSE != 0 {
                    return Err(format!("bin {i} holds the allocated block {b:p}"));
                }
                if bin_index(h & !FLAGS) != i {
                    return Err(format!("block {b:p} of size {} is in bin {i}", h & !FLAGS));
                }
                if (*cur).prev != prev {
                    return Err(format!("block {b:p} has a wrong prev link"));
                }
                listed += 1;
                prev = cur;
                cur = (*cur).next;
                if listed > free_blocks {
                    return Err("a free list is longer than the number of free blocks (cycle?)".into());
                }
            }
        }
        if listed != free_blocks {
            return Err(format!("{free_blocks} free blocks in the chunks, {listed} in the bins"));
        }
        if mapped != self.stats.mapped || nchunks != self.stats.chunks {
            return Err(format!(
                "mapped {mapped} / {nchunks} chunks, counters say {} / {}",
                self.stats.mapped, self.stats.chunks
            ));
        }
        if used_bytes != self.stats.in_use {
            return Err(format!("{used_bytes} bytes in allocated blocks, counter says {}", self.stats.in_use));
        }
        if whole_std != self.free_chunks {
            return Err(format!("{whole_std} fully free standard chunks, counter says {}", self.free_chunks));
        }
        Ok(())
    }

    /// How many free blocks there are, and the size of the largest.
    pub fn free_blocks(&self) -> (usize, usize) {
        let (mut n, mut largest) = (0, 0);
        unsafe {
            for &head in &self.bins {
                let mut cur = head;
                while !cur.is_null() {
                    n += 1;
                    largest = largest.max(size(cur as *mut u8));
                    cur = (*cur).next;
                }
            }
        }
        (n, largest)
    }

    /// Calls `f` with the start and payload size of every allocated block.
    pub fn for_each_allocated(&self, mut f: impl FnMut(*mut u8, usize)) {
        unsafe {
            let mut c = self.chunks;
            while !c.is_null() {
                let end = (c as *mut u8).add((*c).size - SENTINEL);
                let mut b = (c as *mut u8).add(FIRST);
                while b != end {
                    if hdr(b) & INUSE != 0 {
                        f(b.add(HDR), size(b) - HDR);
                    }
                    b = next_block(b);
                }
                c = (*c).next;
            }
        }
    }
}

impl Drop for Heap {
    fn drop(&mut self) {
        unsafe {
            while !self.chunks.is_null() {
                let c = self.chunks;
                self.chunks = (*c).next;
                os::unmap(c as *mut u8, (*c).size);
            }
        }
    }
}

#[cfg(test)]
mod tests;
