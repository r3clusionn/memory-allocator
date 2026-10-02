//! The allocator: three paths chosen by request size, behind [`GlobalAlloc`].
//!
//! | Size | Path | Notes |
//! |---|---|---|
//! | up to 8 KiB | size classes | per-thread caches in front of a shared pool per class |
//! | 8 KiB to 256 KiB | [`Heap`] | boundary tags, coalescing, one lock |
//! | 256 KiB and up | direct mapping | straight from the system, returned on free |
//!
//! The path is a pure function of the request's size and alignment ([`route`]). `dealloc` is given
//! the same layout as `alloc` was, so it recomputes the path and needs no header on small blocks.
//!
//! # Small blocks
//!
//! Each class has a shared pool and every thread has a private cache.
//!
//! * The **cache** is a free list per class. Allocating pops from it and freeing pushes to it, with
//!   no lock and no atomic operation: a thread owns its cache slot exclusively. When the list is
//!   empty the thread takes a *batch* of objects from the pool; when it holds twice a batch it gives
//!   one back.
//! * The **pool** keeps freed objects as ready-made batches (linked lists of exactly one batch each),
//!   so taking or giving a batch is a constant-time pointer swap under a short lock, however many
//!   objects it holds. It also owns a *slab* (64 KiB or more from the system) from which fresh
//!   objects are carved when no batch is waiting.
//! * A thread gets its cache slot the first time it allocates and gives it up when it exits: the
//!   system calls back at thread exit (see [`crate::threads`]) and the cache is flushed to the pool.
//!   Threads beyond the number of slots, and threads that are already exiting, use the pool
//!   directly, one object at a time.
//!
//! Small memory is never given back to the system: a class keeps the slabs it has mapped. Memory
//! freed in one class is not reused by another.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::{Cell, UnsafeCell};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::classes::{batch, class_of, slab_bytes, NUM_CLASSES, SIZES, SMALL_MAX};
use crate::heap::{Heap, HeapStats};
use crate::os;
use crate::spin::SpinLock;
use crate::threads;

/// Requests at least this big are mapped directly.
pub const HUGE_MIN: usize = 256 * 1024;
/// Bytes the heap maps at a time.
const HEAP_CHUNK: usize = 4 << 20;
/// Number of thread cache slots; threads beyond this many use the shared pools directly.
const SLOTS: usize = 256;

/// Which part of the allocator serves a request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    /// Size class index.
    Small(usize),
    Heap,
    Huge,
    /// Alignment above 64 KiB: delegated to the system allocator.
    System,
}

/// The path for a request of `size` bytes aligned to `align`.
///
/// For alignments above 16 a small request is rounded up to a power-of-two class: objects of a
/// power-of-two class are carved at multiples of their size from a slab that starts on a 64 KiB
/// boundary, so they are naturally aligned.
pub fn route(size: usize, align: usize) -> Route {
    if align > os::GRANULE {
        return Route::System;
    }
    let size = size.max(1);
    if align <= 16 {
        if size <= SMALL_MAX {
            return Route::Small(class_of(size));
        }
    } else if let Some(s) = size.max(align).checked_next_power_of_two() {
        if s <= SMALL_MAX {
            return Route::Small(class_of(s));
        }
    }
    if size >= HUGE_MIN {
        Route::Huge
    } else {
        Route::Heap
    }
}

// ---- the shared pool of one size class -------------------------------------------------------

/// Objects are linked through their first word. A *batch* is a null-terminated list of exactly
/// `batch(class size)` objects whose head also holds, in its second word, the next batch.
struct Central {
    batches: *mut u8,
    /// Single objects, for callers that move one at a time.
    loose: *mut u8,
    /// Unused part of the current slab.
    bump: *mut u8,
    end: *mut u8,
}

unsafe impl Send for Central {}

#[repr(align(128))]
struct PaddedCentral(SpinLock<Central>);

impl PaddedCentral {
    const fn new() -> Self {
        PaddedCentral(SpinLock::new(Central {
            batches: std::ptr::null_mut(),
            loose: std::ptr::null_mut(),
            bump: std::ptr::null_mut(),
            end: std::ptr::null_mut(),
        }))
    }
}

// ---- thread caches -----------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct CacheClass {
    head: *mut u8,
    count: usize,
}

/// One thread's cache. Only the thread that owns the slot touches `classes`.
#[repr(align(128))]
struct Slot {
    owned: AtomicBool,
    /// The allocator this slot belongs to, and how to give the slot back to it.
    instance: UnsafeCell<*const ()>,
    release: UnsafeCell<Option<unsafe fn(*mut Slot)>>,
    classes: UnsafeCell<[CacheClass; NUM_CLASSES]>,
}

unsafe impl Sync for Slot {}

impl Slot {
    const fn new() -> Self {
        Slot {
            owned: AtomicBool::new(false),
            instance: UnsafeCell::new(std::ptr::null()),
            release: UnsafeCell::new(None),
            classes: UnsafeCell::new([CacheClass { head: std::ptr::null_mut(), count: 0 }; NUM_CLASSES]),
        }
    }
}

#[derive(Clone, Copy)]
struct Tls {
    /// Which allocator instance `slot` belongs to (0 when none; `DEAD` once the thread has exited).
    id: usize,
    slot: *mut Slot,
}

const DEAD: usize = usize::MAX;
const NO_SLOT: *mut Slot = usize::MAX as *mut Slot;

thread_local! {
    // A plain `Cell` with a constant initialiser: no lazy setup and no destructor, so touching it
    // from inside the allocator never allocates.
    static TLS: Cell<Tls> = const { Cell::new(Tls { id: 0, slot: std::ptr::null_mut() }) };
}

static NEXT_INSTANCE: AtomicUsize = AtomicUsize::new(1);
static NEXT_START: AtomicUsize = AtomicUsize::new(0);

#[inline(always)]
unsafe fn link(p: *mut u8) -> *mut *mut u8 {
    p as *mut *mut u8
}

/// Called by the system when a thread that owns a cache slot exits.
///
/// # Safety
/// `p` is the slot the thread registered, and the allocator it belongs to is still alive.
pub(crate) unsafe fn on_thread_exit(p: *mut u8) {
    let slot = p as *mut Slot;
    if let Some(release) = *(*slot).release.get() {
        release(slot);
    }
    // Anything the thread frees from here on (other thread-exit destructors run around this one)
    // goes to the shared pools: the slot may already belong to another thread.
    TLS.with(|t| t.set(Tls { id: DEAD, slot: NO_SLOT }));
}

/// Memory held, by where it came from.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    /// Bytes mapped for size-class slabs. Never returned.
    pub small_mapped: usize,
    /// Bytes currently mapped for direct (huge) allocations.
    pub huge_mapped: usize,
    pub heap: HeapStats,
}

impl Stats {
    /// Everything this allocator currently holds from the system.
    pub fn total_mapped(&self) -> usize {
        self.small_mapped + self.huge_mapped + self.heap.mapped
    }
}

/// The allocator. `CACHE = false` removes the per-thread caches (every small request goes to the
/// shared pool, one object at a time), which exists so the benchmark can show what the caches buy.
///
/// An instance must not be moved once it has served a request (threads keep pointers into it), and
/// it must outlive every thread that used it. A `static` satisfies both.
///
/// ```
/// use tmalloc::Tmalloc;
///
/// #[global_allocator]
/// static ALLOC: Tmalloc = Tmalloc::new();
///
/// let v: Vec<u64> = (0..1000).collect();
/// assert_eq!(v.iter().sum::<u64>(), 499_500);
/// ```
pub struct Tmalloc<const CACHE: bool = true> {
    id: AtomicUsize,
    central: [PaddedCentral; NUM_CLASSES],
    slots: [Slot; SLOTS],
    heap: SpinLock<Heap>,
    small_mapped: AtomicUsize,
    huge_mapped: AtomicUsize,
}

impl<const CACHE: bool> Default for Tmalloc<CACHE> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const CACHE: bool> Drop for Tmalloc<CACHE> {
    fn drop(&mut self) {
        // The dropping thread's own cache must not outlive the allocator it points into. (Other
        // threads must have exited, which releases theirs.)
        if CACHE {
            let id = self.id.load(Ordering::Relaxed);
            let e = TLS.with(|t| t.get());
            if id != 0 && e.id == id && e.slot != NO_SLOT && !e.slot.is_null() {
                unsafe {
                    release_slot::<CACHE>(e.slot);
                    threads::set_exit_value(std::ptr::null_mut());
                }
                TLS.with(|t| t.set(Tls { id: 0, slot: std::ptr::null_mut() }));
            }
        }
    }
}

/// Flushes a slot's cache to its allocator's pools and makes the slot available again.
unsafe fn release_slot<const CACHE: bool>(slot: *mut Slot) {
    let a = &*((*(*slot).instance.get()) as *const Tmalloc<CACHE>);
    let classes = &mut *(*slot).classes.get();
    for (class, c) in classes.iter_mut().enumerate() {
        a.flush_class(class, c);
    }
    (*slot).owned.store(false, Ordering::Release);
}

impl<const CACHE: bool> Tmalloc<CACHE> {
    pub const fn new() -> Self {
        Tmalloc {
            id: AtomicUsize::new(0),
            central: [const { PaddedCentral::new() }; NUM_CLASSES],
            slots: [const { Slot::new() }; SLOTS],
            heap: SpinLock::new(Heap::new(HEAP_CHUNK)),
            small_mapped: AtomicUsize::new(0),
            huge_mapped: AtomicUsize::new(0),
        }
    }

    pub fn stats(&self) -> Stats {
        Stats {
            small_mapped: self.small_mapped.load(Ordering::Relaxed),
            huge_mapped: self.huge_mapped.load(Ordering::Relaxed),
            heap: self.heap.lock().stats(),
        }
    }

    /// How many threads currently own a cache slot. A thread's slot is released when the operating
    /// system reports that the thread has exited, which can be a moment after `thread::scope` or a
    /// join returns, so a test that inspects the pools waits for this to drop back first.
    pub fn live_caches(&self) -> usize {
        self.slots.iter().filter(|s| s.owned.load(Ordering::Acquire)).count()
    }

    /// The number of free objects of a class waiting in the shared pool: whole batches plus loose
    /// objects. Objects in thread caches are not counted. For tests and diagnostics.
    pub fn pooled_objects(&self, class: usize) -> usize {
        let g = self.central[class].0.lock();
        let b = batch(SIZES[class]);
        let mut n = 0;
        let mut head = g.batches;
        while !head.is_null() {
            n += b;
            head = unsafe { *link(head.add(8)) };
        }
        let mut p = g.loose;
        while !p.is_null() {
            n += 1;
            p = unsafe { *link(p) };
        }
        n
    }

    /// Checks the heap's invariants (slow).
    pub fn check_heap(&self) -> Result<(), String> {
        self.heap.lock().check()
    }

    /// Checks every thread cache and every shared pool: counts equal list lengths, caches are under
    /// their flush threshold, every pooled batch holds exactly one batch of objects. Slow; for tests,
    /// when no other thread is using the allocator.
    pub fn check_pools(&self) -> Result<(), String> {
        let walk = |mut p: *mut u8, limit: usize| -> Result<usize, String> {
            let mut n = 0;
            while !p.is_null() {
                n += 1;
                if n > limit {
                    return Err("a list is longer than it can be (cycle?)".into());
                }
                p = unsafe { *link(p) };
            }
            Ok(n)
        };
        for (si, slot) in self.slots.iter().enumerate() {
            if !slot.owned.load(Ordering::Acquire) {
                continue;
            }
            let classes = unsafe { &*slot.classes.get() };
            for (class, c) in classes.iter().enumerate() {
                let b = batch(SIZES[class]);
                let n = walk(c.head, 2 * b).map_err(|e| format!("slot {si} class {class}: {e}"))?;
                if n != c.count {
                    return Err(format!("slot {si} class {class}: count says {}, the list has {n}", c.count));
                }
                if c.count >= 2 * b {
                    return Err(format!("slot {si} class {class}: {} cached, flush threshold is {}", c.count, 2 * b));
                }
            }
        }
        for (class, central) in self.central.iter().enumerate() {
            let g = central.0.lock();
            let b = batch(SIZES[class]);
            let mut head = g.batches;
            let mut batches = 0;
            while !head.is_null() {
                batches += 1;
                if batches > 10_000_000 {
                    return Err(format!("class {class}: batch list too long (cycle?)"));
                }
                let n = walk(head, b).map_err(|e| format!("class {class} batch {batches}: {e}"))?;
                if n != b {
                    return Err(format!("class {class}: a pooled batch holds {n} objects, not {b}"));
                }
                head = unsafe { *link(head.add(8)) };
            }
            walk(g.loose, usize::MAX).map_err(|e| format!("class {class} loose list: {e}"))?;
        }
        Ok(())
    }

    /// The number of bytes `alloc` really gives for this layout.
    pub fn usable_size(&self, layout: Layout) -> usize {
        match route(layout.size(), layout.align()) {
            Route::Small(c) => SIZES[c],
            _ => layout.size(),
        }
    }

    // ---- the shared pool --------------------------------------------------------------------

    /// Reserves up to `want` fresh objects from the slab (mapping a new one if it is used up) and
    /// returns where they start and how many there are. The caller links them.
    unsafe fn carve(&self, c: &mut Central, size: usize, want: usize) -> (*mut u8, usize) {
        if (c.end as usize) - (c.bump as usize) < size {
            let bytes = slab_bytes(size);
            let slab = os::map(bytes);
            if slab.is_null() {
                return (std::ptr::null_mut(), 0);
            }
            self.small_mapped.fetch_add(bytes, Ordering::Relaxed);
            c.bump = slab;
            c.end = slab.add(bytes);
        }
        let n = want.min((c.end as usize - c.bump as usize) / size);
        let start = c.bump;
        c.bump = start.add(n * size);
        (start, n)
    }

    /// Takes a batch (up to `batch(size)` objects) as a null-terminated list.
    #[cold]
    unsafe fn take_batch(&self, class: usize) -> (*mut u8, usize) {
        let size = SIZES[class];
        let b = batch(size);
        let (start, n) = {
            let mut c = self.central[class].0.lock();
            if !c.batches.is_null() {
                let head = c.batches;
                c.batches = *link(head.add(8));
                return (head, b);
            }
            if !c.loose.is_null() {
                let mut head = std::ptr::null_mut();
                let mut n = 0;
                while n < b && !c.loose.is_null() {
                    let p = c.loose;
                    c.loose = *link(p);
                    *link(p) = head;
                    head = p;
                    n += 1;
                }
                return (head, n);
            }
            self.carve(&mut c, size, b)
        };
        if n == 0 {
            return (std::ptr::null_mut(), 0);
        }
        // Link the fresh objects after the lock is released: this touches their memory. (The terminator
        // below is already null, since fresh slab memory is zeroed by the system; it is written
        // anyway so that the list does not depend on that.)
        for i in 0..n - 1 {
            *link(start.add(i * size)) = start.add((i + 1) * size);
        }
        *link(start.add((n - 1) * size)) = std::ptr::null_mut();
        (start, n)
    }

    /// Gives back exactly one batch: a null-terminated list of `batch(size)` objects.
    #[cold]
    unsafe fn give_batch(&self, class: usize, head: *mut u8) {
        let mut c = self.central[class].0.lock();
        *link(head.add(8)) = c.batches;
        c.batches = head;
    }

    /// Takes a single object.
    unsafe fn take_one(&self, class: usize) -> *mut u8 {
        let size = SIZES[class];
        let mut c = self.central[class].0.lock();
        if !c.loose.is_null() {
            let p = c.loose;
            c.loose = *link(p);
            return p;
        }
        let (start, n) = self.carve(&mut c, size, 1);
        if n == 1 {
            return start;
        }
        if !c.batches.is_null() {
            // Break a batch up: use its first object, keep the rest as loose objects.
            let head = c.batches;
            c.batches = *link(head.add(8));
            let rest = *link(head);
            if !rest.is_null() {
                let mut tail = rest;
                while !(*link(tail)).is_null() {
                    tail = *link(tail);
                }
                *link(tail) = c.loose;
                c.loose = rest;
            }
            return head;
        }
        std::ptr::null_mut()
    }

    /// Gives back a list of objects (`head` to `tail`, any length) as loose objects.
    unsafe fn give_loose(&self, class: usize, head: *mut u8, tail: *mut u8) {
        let mut c = self.central[class].0.lock();
        *link(tail) = c.loose;
        c.loose = head;
    }

    /// Hands everything in one cache class to the pool: whole batches as batches, the rest loose.
    unsafe fn flush_class(&self, class: usize, c: &mut CacheClass) {
        let b = batch(SIZES[class]);
        while c.count >= b {
            self.flush_one_batch(c, class);
        }
        if c.count > 0 {
            let head = c.head;
            let mut tail = head;
            for _ in 1..c.count {
                tail = *link(tail);
            }
            self.give_loose(class, head, tail);
            c.head = std::ptr::null_mut();
            c.count = 0;
        }
    }

    /// Moves the `batch` newest objects of a cache class to the pool.
    #[cold]
    #[inline(never)]
    unsafe fn flush_one_batch(&self, c: &mut CacheClass, class: usize) {
        let b = batch(SIZES[class]);
        let head = c.head;
        let mut tail = head;
        for _ in 1..b {
            tail = *link(tail);
        }
        c.head = *link(tail);
        *link(tail) = std::ptr::null_mut();
        c.count -= b;
        self.give_batch(class, head);
    }

    // ---- thread caches ----------------------------------------------------------------------

    fn ensure_id(&self) -> usize {
        let id = self.id.load(Ordering::Relaxed);
        if id != 0 {
            return id;
        }
        let new = NEXT_INSTANCE.fetch_add(1, Ordering::Relaxed);
        match self.id.compare_exchange(0, new, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => new,
            Err(existing) => existing,
        }
    }

    /// The calling thread's cache for this allocator (the array of `NUM_CLASSES` entries), or null if
    /// it has none.
    #[inline(always)]
    fn cache(&self) -> *mut CacheClass {
        let e = TLS.with(|t| t.get());
        if e.id == self.id.load(Ordering::Relaxed) && e.id != 0 {
            if e.slot == NO_SLOT {
                return std::ptr::null_mut();
            }
            return unsafe { (*e.slot).classes.get() as *mut CacheClass };
        }
        self.cache_slow(e)
    }

    /// The thread has no slot for this allocator yet (or never will): leave any other allocator's
    /// slot, claim one here and arrange for it to be released at thread exit.
    #[cold]
    #[inline(never)]
    fn cache_slow(&self, e: Tls) -> *mut CacheClass {
        if e.id == DEAD {
            return std::ptr::null_mut();
        }
        unsafe {
            if e.id != 0 && e.slot != NO_SLOT && !e.slot.is_null() {
                // This thread is moving to a different allocator instance: give its old slot back.
                let release = (*(*e.slot).release.get()).unwrap();
                release(e.slot);
            }
            let id = self.ensure_id();
            let start = NEXT_START.fetch_add(1, Ordering::Relaxed);
            for i in 0..SLOTS {
                let s = &self.slots[(start + i) % SLOTS];
                if !s.owned.load(Ordering::Relaxed)
                    && s.owned.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_ok()
                {
                    *s.instance.get() = self as *const Self as *const ();
                    *s.release.get() = Some(release_slot::<CACHE>);
                    let slot = s as *const Slot as *mut Slot;
                    if !threads::set_exit_value(slot as *mut u8) {
                        // Without an exit hook the slot could never be released: do not use it.
                        s.owned.store(false, Ordering::Release);
                        break;
                    }
                    TLS.with(|t| t.set(Tls { id, slot }));
                    return (*slot).classes.get() as *mut CacheClass;
                }
            }
            TLS.with(|t| t.set(Tls { id, slot: NO_SLOT }));
        }
        std::ptr::null_mut()
    }

    // ---- small ------------------------------------------------------------------------------

    #[inline(always)]
    unsafe fn small_alloc(&self, class: usize) -> *mut u8 {
        if CACHE {
            let classes = self.cache();
            if !classes.is_null() {
                let c = &mut *classes.add(class);
                let p = c.head;
                if !p.is_null() {
                    c.head = *link(p);
                    c.count -= 1;
                    return p;
                }
                return self.refill(c, class);
            }
        }
        self.take_one(class)
    }

    #[cold]
    #[inline(never)]
    unsafe fn refill(&self, c: &mut CacheClass, class: usize) -> *mut u8 {
        let (head, n) = self.take_batch(class);
        if n == 0 {
            return std::ptr::null_mut();
        }
        c.head = *link(head);
        c.count = n - 1;
        head
    }

    #[inline(always)]
    unsafe fn small_free(&self, p: *mut u8, class: usize) {
        if CACHE {
            let classes = self.cache();
            if !classes.is_null() {
                let c = &mut *classes.add(class);
                *link(p) = c.head;
                c.head = p;
                c.count += 1;
                if c.count >= 2 * batch(SIZES[class]) {
                    self.flush_one_batch(c, class);
                }
                return;
            }
        }
        self.give_loose(class, p, p);
    }

    // ---- huge -------------------------------------------------------------------------------

    unsafe fn huge_alloc(&self, size: usize) -> *mut u8 {
        let bytes = os::round_up_page(size);
        let p = os::map(bytes);
        if !p.is_null() {
            self.huge_mapped.fetch_add(bytes, Ordering::Relaxed);
        }
        p
    }

    unsafe fn huge_free(&self, p: *mut u8, size: usize) {
        let bytes = os::round_up_page(size);
        self.huge_mapped.fetch_sub(bytes, Ordering::Relaxed);
        os::unmap(p, bytes);
    }
}

unsafe impl<const CACHE: bool> GlobalAlloc for Tmalloc<CACHE> {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        match route(layout.size(), layout.align()) {
            Route::Small(c) => self.small_alloc(c),
            Route::Heap => self.heap.lock().alloc(layout.size(), layout.align()),
            Route::Huge => self.huge_alloc(layout.size()),
            Route::System => System.alloc(layout),
        }
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        match route(layout.size(), layout.align()) {
            Route::Small(c) => self.small_free(ptr, c),
            Route::Heap => self.heap.lock().free(ptr),
            Route::Huge => self.huge_free(ptr, layout.size()),
            Route::System => System.dealloc(ptr, layout),
        }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        match route(layout.size(), layout.align()) {
            // Fresh pages from the system are already zero.
            Route::Huge => self.huge_alloc(layout.size()),
            Route::System => System.alloc_zeroed(layout),
            _ => {
                let p = self.alloc(layout);
                if !p.is_null() {
                    std::ptr::write_bytes(p, 0, layout.size());
                }
                p
            }
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        match (route(layout.size(), layout.align()), route(new_size, layout.align())) {
            (Route::Small(a), Route::Small(b)) if a == b => return ptr,
            (Route::Heap, Route::Heap) => {
                if self.heap.lock().realloc_in_place(ptr, new_size) {
                    return ptr;
                }
            }
            (Route::Huge, Route::Huge) if os::round_up_page(layout.size()) == os::round_up_page(new_size) => return ptr,
            (Route::System, Route::System) => return System.realloc(ptr, layout, new_size),
            _ => {}
        }
        let new_layout = Layout::from_size_align_unchecked(new_size, layout.align());
        let np = self.alloc(new_layout);
        if !np.is_null() {
            std::ptr::copy_nonoverlapping(ptr, np, layout.size().min(new_size));
            self.dealloc(ptr, layout);
        }
        np
    }
}

#[cfg(test)]
mod tests;
