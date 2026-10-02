use super::*;
use std::collections::BTreeMap;

const SMALL_CHUNK: usize = 64 * 1024;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn ok(h: &Heap) {
    if let Err(e) = h.check() {
        panic!("heap invariant broken: {e}");
    }
}

#[test]
fn an_empty_heap_is_consistent() {
    let h = Heap::new(SMALL_CHUNK);
    ok(&h);
    assert_eq!(h.stats(), HeapStats::default());
}

#[test]
fn alloc_write_free() {
    let mut h = Heap::new(SMALL_CHUNK);
    unsafe {
        let p = h.alloc(100, 8);
        assert!(!p.is_null());
        assert_eq!(p as usize % 16, 0);
        assert!(h.usable_size(p) >= 100);
        std::ptr::write_bytes(p, 0xAB, 100);
        ok(&h);
        assert_eq!(h.stats().in_use, 112);
        h.free(p);
        ok(&h);
        assert_eq!(h.stats().in_use, 0);
        assert_eq!((h.stats().allocs, h.stats().frees), (1, 1));
    }
}

#[test]
fn block_sizes_follow_the_header_and_rounding_rules() {
    assert_eq!(block_size(0), Some(32));
    assert_eq!(block_size(1), Some(32));
    assert_eq!(block_size(24), Some(32));
    assert_eq!(block_size(25), Some(48));
    assert_eq!(block_size(40), Some(48));
    assert_eq!(block_size(41), Some(64));
    assert_eq!(block_size(usize::MAX), None);
    assert_eq!(block_size(usize::MAX - 20), None);
}

#[test]
fn bins_are_monotonic_and_cover_every_size() {
    let mut last = 0;
    let mut sz = MIN_BLOCK;
    while sz < (1usize << 40) {
        let b = bin_index(sz);
        assert!(b >= last && b < NBINS, "size {sz} bin {b}");
        assert!(b <= last + 1 || sz > 512, "exact bins are consecutive");
        last = b;
        sz += if sz < 4096 { 16 } else { sz / 7 / 16 * 16 + 16 };
    }
    assert_eq!(bin_index(32), 0);
    assert_eq!(bin_index(512), 30);
    assert_eq!(bin_index(528), 31);
    assert_eq!(bin_index(1 << 62), NBINS - 1);
}

/// Allocates `n` adjacent blocks of one size from a fresh heap.
unsafe fn adjacent(h: &mut Heap, n: usize, size: usize) -> Vec<*mut u8> {
    let v: Vec<_> = (0..n).map(|_| h.alloc(size, 16)).collect();
    for w in v.windows(2) {
        assert_eq!(w[1] as usize - w[0] as usize, block_size(size).unwrap(), "blocks are carved in order");
    }
    v
}

#[test]
fn freeing_in_any_order_coalesces_back_to_one_block() {
    let orders = [[0, 1, 2, 3], [3, 2, 1, 0], [1, 3, 0, 2], [2, 0, 3, 1], [0, 2, 1, 3], [1, 0, 3, 2], [3, 0, 2, 1]];
    for order in orders {
        let mut h = Heap::new(SMALL_CHUNK);
        unsafe {
            let v = adjacent(&mut h, 4, 200);
            // Keep a block after them so the run does not merge into the end of the chunk.
            let fence = h.alloc(64, 16);
            for &i in &order {
                h.free(v[i]);
                ok(&h);
            }
            let (n, _) = h.free_blocks();
            assert_eq!(n, 2, "the freed run and the rest of the chunk after the fence, order {order:?}");
            h.free(fence);
            ok(&h);
            assert_eq!(h.free_blocks(), (1, SMALL_CHUNK - OVERHEAD));
            assert_eq!(h.stats().in_use, 0);
        }
    }
}

#[test]
fn a_freed_block_is_reused_exactly() {
    let mut h = Heap::new(SMALL_CHUNK);
    unsafe {
        let a = h.alloc(100, 16);
        let _keep = h.alloc(100, 16);
        h.free(a);
        let b = h.alloc(100, 16);
        assert_eq!(a, b, "an exact-size bin hit returns the same block");
        ok(&h);
    }
}

#[test]
fn a_remainder_too_small_to_be_a_block_stays_with_the_allocation() {
    let mut h = Heap::new(SMALL_CHUNK);
    unsafe {
        let a = h.alloc(100, 16); // block 112
        let _keep = h.alloc(8, 16);
        h.free(a);
        // 112 - 96 = 16 left over: not a block, so the allocation keeps all 112.
        let b = h.alloc(81, 16); // needs 96
        assert_eq!(a, b);
        assert_eq!(h.usable_size(b), 104);
        ok(&h);
        // 112 - 64 = 48 left over: a block.
        h.free(b);
        let c = h.alloc(49, 16); // needs 64
        assert_eq!(a, c);
        assert_eq!(h.usable_size(c), 56);
        let (n, _) = h.free_blocks();
        assert_eq!(n, 2);
        ok(&h);
    }
}

#[test]
fn a_remainder_of_exactly_the_minimum_block_is_split_off() {
    let mut h = Heap::new(SMALL_CHUNK);
    unsafe {
        let a = h.alloc(100, 16); // block 112
        let _keep = h.alloc(8, 16);
        h.free(a);
        let before = h.free_blocks().0;
        let b = h.alloc(72, 16); // needs 80, leaves exactly 32
        assert_eq!(a, b);
        assert_eq!(h.usable_size(b), 72);
        assert_eq!(h.free_blocks().0, before, "one block was taken and the 32-byte remainder became a new one");
        ok(&h);
    }
}

#[test]
fn a_range_bin_hands_out_the_smallest_block_that_fits() {
    let mut h = Heap::new(SMALL_CHUNK);
    unsafe {
        // Two free blocks in the same bin (528..=639): 560 freed first, 624 freed second, so 624 is at
        // the head of the list. A request for 544 must still get the 560 block.
        let x1 = h.alloc(552, 16); // block 560
        let f1 = h.alloc(8, 16);
        let x2 = h.alloc(616, 16); // block 624
        let _f2 = h.alloc(8, 16);
        assert_eq!(bin_index(560), bin_index(624));
        h.free(x1);
        h.free(x2);
        let p = h.alloc(536, 16); // block 544
        assert_eq!(p, x1, "best fit, not first fit");
        let _ = f1;
        ok(&h);
    }
}

#[test]
fn alignment_is_honoured_up_to_the_granule_and_gaps_are_recycled() {
    let mut h = Heap::new(1 << 20);
    unsafe {
        let mut live = Vec::new();
        for align in [32usize, 64, 128, 256, 1024, 4096, 16384, 65536] {
            for size in [1usize, 17, 100, 5000] {
                let p = h.alloc(size, align);
                assert!(!p.is_null(), "align {align} size {size}");
                assert_eq!(p as usize % align, 0, "align {align} size {size}");
                assert!(h.usable_size(p) >= size);
                std::ptr::write_bytes(p, 0x5A, size);
                live.push((p, size));
                ok(&h);
            }
        }
        // Mixed with ordinary blocks, and freed in a scrambled order.
        for i in 0..20 {
            live.push((h.alloc(40 + i, 16), 40 + i));
        }
        let mut rng = Rng(7);
        while !live.is_empty() {
            let i = rng.below(live.len() as u64) as usize;
            let (p, _) = live.swap_remove(i);
            h.free(p);
            ok(&h);
        }
        assert_eq!(h.stats().in_use, 0);
        assert_eq!(h.free_blocks().0, 1, "all gaps merged back");
    }
}

#[test]
fn aligned_requests_that_need_a_fresh_chunk() {
    let mut h = Heap::new(SMALL_CHUNK);
    unsafe {
        // Larger than a standard chunk, and aligned.
        let p = h.alloc(100_000, 4096);
        assert!(!p.is_null());
        assert_eq!(p as usize % 4096, 0);
        ok(&h);
        h.free(p);
        ok(&h);
        assert_eq!(h.stats().chunks, 0, "an oversize chunk is returned as soon as it is free");
    }
}

#[test]
fn realloc_in_place_shrinks_and_grows_when_it_can() {
    let mut h = Heap::new(SMALL_CHUNK);
    unsafe {
        let a = h.alloc(1000, 16);
        let b = h.alloc(1000, 16);
        std::ptr::write_bytes(a, 0x11, 1000);
        // Cannot grow: the next block is in use.
        assert!(!h.realloc_in_place(a, 2000));
        ok(&h);
        // Shrinks, leaving a free tail.
        assert!(h.realloc_in_place(a, 300));
        assert!(h.usable_size(a) >= 300 && h.usable_size(a) < 400);
        ok(&h);
        assert!(std::slice::from_raw_parts(a, 300).iter().all(|&x| x == 0x11));
        // Now it can grow back into its own tail.
        assert!(h.realloc_in_place(a, 1000));
        assert!(h.usable_size(a) >= 1000);
        ok(&h);
        // Free the neighbour: growth can absorb it, then the surplus is split off again.
        h.free(b);
        assert!(h.realloc_in_place(a, 1800));
        assert!(h.usable_size(a) >= 1800);
        ok(&h);
        assert!(h.realloc_in_place(a, 100));
        ok(&h);
        h.free(a);
        ok(&h);
        assert_eq!(h.stats().in_use, 0);
        assert_eq!(h.free_blocks().0, 1);
    }
}

#[test]
fn growing_to_the_end_of_a_chunk_and_back() {
    let mut h = Heap::new(SMALL_CHUNK);
    unsafe {
        let a = h.alloc(64, 16);
        let max = h.max_standard_payload();
        assert!(h.realloc_in_place(a, max));
        ok(&h);
        assert_eq!(h.free_blocks().0, 0);
        assert!(!h.realloc_in_place(a, max + 1));
        assert!(h.realloc_in_place(a, 64));
        ok(&h);
        h.free(a);
        ok(&h);
    }
}

#[test]
fn spare_chunks_are_kept_once_and_the_rest_unmapped() {
    let mut h = Heap::new(SMALL_CHUNK);
    unsafe {
        // Each 40 KB block needs its own chunk (two do not fit in 64 KiB).
        let blocks: Vec<_> = (0..6).map(|_| h.alloc(40_000, 16)).collect();
        assert_eq!(h.stats().chunks, 6);
        ok(&h);
        for &b in &blocks {
            h.free(b);
            ok(&h);
        }
        assert_eq!(h.stats().chunks, 1, "one spare chunk is kept");
        assert_eq!(h.stats().mapped, SMALL_CHUNK);
        // The spare is reused without mapping again.
        let peak = h.stats().peak_mapped;
        let p = h.alloc(40_000, 16);
        assert_eq!((h.stats().chunks, h.stats().peak_mapped), (1, peak));
        ok(&h);
        h.free(p);
        ok(&h);
    }
}

#[test]
fn the_limit_makes_allocation_fail_and_leaves_the_heap_intact() {
    let mut h = Heap::new(SMALL_CHUNK);
    h.set_limit(3 * SMALL_CHUNK);
    unsafe {
        let mut got = Vec::new();
        loop {
            let p = h.alloc(30_000, 16);
            if p.is_null() {
                break;
            }
            got.push(p);
            assert!(got.len() < 100);
        }
        assert_eq!(got.len(), 6, "two 30,000-byte blocks fit in each of three chunks");
        ok(&h);
        assert!(h.alloc(usize::MAX, 16).is_null());
        assert!(h.alloc(usize::MAX / 2, 4096).is_null());
        ok(&h);
        for p in got {
            h.free(p);
        }
        ok(&h);
        // Memory freed is available again.
        assert!(!h.alloc(30_000, 16).is_null());
    }
}

/// A random mix of every operation, checked against an independent record of what is live: no two
/// allocations may overlap, and every byte written must still be there when its block is freed.
fn model(seed: u64, chunk: usize, ops: usize, max_size: u64) {
    let mut h = Heap::new(chunk);
    let mut rng = Rng(seed);
    // start address -> (payload size, fill byte, alignment)
    let mut live: BTreeMap<usize, (usize, u8, usize)> = BTreeMap::new();
    let mut keys: Vec<usize> = Vec::new();
    unsafe {
        for step in 0..ops {
            let r = rng.below(100);
            if r < 45 || live.is_empty() {
                let size = match rng.below(10) {
                    0..=5 => 1 + rng.below(max_size.min(200)),
                    6..=8 => 1 + rng.below(max_size.min(4000)),
                    _ => 1 + rng.below(max_size),
                } as usize;
                let align = if rng.below(12) == 0 { 16usize << rng.below(9) } else { 16 };
                let p = h.alloc(size, align);
                assert!(!p.is_null(), "step {step}: allocation of {size} (align {align}) failed");
                assert_eq!(p as usize % align, 0, "step {step}: misaligned");
                assert!(h.usable_size(p) >= size);
                let addr = p as usize;
                if let Some((&before, &(bsize, _, _))) = live.range(..addr).next_back() {
                    assert!(before + bsize <= addr, "step {step}: {before:#x}+{bsize} overlaps new block at {addr:#x}");
                }
                if let Some((&after, _)) = live.range(addr..).next() {
                    assert!(addr + size <= after, "step {step}: new block at {addr:#x}+{size} overlaps {after:#x}");
                }
                let fill = (rng.next() & 0xFF) as u8;
                std::ptr::write_bytes(p, fill, size);
                live.insert(addr, (size, fill, align));
                keys.push(addr);
            } else if r < 85 {
                let i = rng.below(keys.len() as u64) as usize;
                let addr = keys.swap_remove(i);
                let (size, fill, _) = live.remove(&addr).unwrap();
                let s = std::slice::from_raw_parts(addr as *const u8, size);
                assert!(s.iter().all(|&b| b == fill), "step {step}: block {addr:#x} was overwritten");
                h.free(addr as *mut u8);
            } else {
                let i = rng.below(keys.len() as u64) as usize;
                let addr = keys[i];
                let (size, fill, align) = live[&addr];
                let new_size = 1 + rng.below(max_size.min(3000)) as usize;
                if h.realloc_in_place(addr as *mut u8, new_size) {
                    let keep = size.min(new_size);
                    let s = std::slice::from_raw_parts(addr as *const u8, keep);
                    assert!(s.iter().all(|&b| b == fill), "step {step}: realloc damaged the kept bytes");
                    assert!(h.usable_size(addr as *mut u8) >= new_size);
                    if new_size > size {
                        // The grown part must not run into a neighbour.
                        if let Some((&after, _)) = live.range(addr + 1..).next() {
                            assert!(addr + new_size <= after, "step {step}: realloc grew into a live block");
                        }
                    }
                    std::ptr::write_bytes(addr as *mut u8, fill, new_size);
                    live.insert(addr, (new_size, fill, align));
                }
            }
            if step % 97 == 0 {
                ok(&h);
            }
        }
        ok(&h);
        let mut counted = 0;
        h.for_each_allocated(|p, _| {
            assert!(live.contains_key(&(p as usize)), "the heap lists a block the model does not know");
            counted += 1;
        });
        assert_eq!(counted, live.len());
        for (addr, (size, fill, _)) in std::mem::take(&mut live) {
            let s = std::slice::from_raw_parts(addr as *const u8, size);
            assert!(s.iter().all(|&b| b == fill));
            h.free(addr as *mut u8);
        }
        ok(&h);
        assert_eq!(h.stats().in_use, 0);
        assert!(h.stats().chunks <= 1, "everything freed leaves at most the spare chunk");
        assert!(h.free_blocks().0 <= 1);
    }
}

#[test]
fn random_operations_small_chunks() {
    for seed in 1..=8 {
        model(seed * 0x9E37, SMALL_CHUNK, 20_000, 20_000);
    }
}

#[test]
fn random_operations_big_chunks_and_big_blocks() {
    for seed in 1..=4 {
        model(seed * 0x1234_5677, 1 << 22, 15_000, 300_000);
    }
}

#[test]
fn random_operations_tiny_blocks_only() {
    for seed in 1..=4 {
        model(seed * 77, 1 << 20, 40_000, 64);
    }
}
