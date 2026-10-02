use super::*;

/// Waits until threads that have finished their work have also finished exiting (their cache slots
/// are released by the system's thread-exit callback, which can run just after a scope ends).
fn quiesce(a: &Tmalloc, baseline: usize) {
    let t = std::time::Instant::now();
    while a.live_caches() > baseline {
        assert!(t.elapsed().as_secs() < 10, "{} cache slots are still owned", a.live_caches());
        std::thread::yield_now();
    }
}

#[test]
fn routing_by_size_and_alignment() {
    assert_eq!(route(1, 1), Route::Small(0));
    assert_eq!(route(16, 16), Route::Small(0));
    assert_eq!(route(17, 8), Route::Small(1));
    assert_eq!(route(SMALL_MAX, 16), Route::Small(NUM_CLASSES - 1));
    assert_eq!(route(SMALL_MAX + 1, 16), Route::Heap);
    assert_eq!(route(HUGE_MIN - 1, 16), Route::Heap);
    assert_eq!(route(HUGE_MIN, 16), Route::Huge);
    // Larger alignments round a small request up to a power-of-two class.
    assert_eq!(route(1, 64), Route::Small(class_of(64)));
    assert_eq!(route(100, 64), Route::Small(class_of(128)));
    assert_eq!(route(100, 4096), Route::Small(class_of(4096)));
    assert_eq!(route(5000, 4096), Route::Small(class_of(8192)));
    assert_eq!(route(5000, 16384), Route::Heap);
    assert_eq!(route(HUGE_MIN, 65536), Route::Huge);
    assert_eq!(route(100, 65536 * 2), Route::System);
    assert_eq!(route(0, 1), Route::Small(0));
}

#[test]
fn the_same_layout_always_routes_the_same_way() {
    for size in [1usize, 15, 16, 17, 100, 4096, 8192, 8193, 100_000, HUGE_MIN, 10_000_000] {
        for align in [1usize, 8, 16, 32, 64, 4096, 65536, 1 << 20] {
            assert_eq!(route(size, align), route(size, align));
        }
    }
}

#[test]
fn small_allocations_are_distinct_aligned_and_reused() {
    let a: Tmalloc = Tmalloc::new();
    unsafe {
        let l = Layout::from_size_align(24, 8).unwrap();
        let ps: Vec<_> = (0..1000).map(|_| a.alloc(l)).collect();
        let mut sorted: Vec<_> = ps.iter().map(|&p| p as usize).collect();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 1000);
        assert!(ps.iter().all(|&p| (p as usize).is_multiple_of(16)));
        let before = a.stats().small_mapped;
        for &p in &ps {
            a.dealloc(p, l);
        }
        for _ in 0..1000 {
            assert!(!a.alloc(l).is_null());
        }
        assert_eq!(a.stats().small_mapped, before, "freed objects are reused, not newly mapped");
    }
    a.check_pools().unwrap();
}

#[test]
fn cache_counts_stay_exact_through_refills_and_flushes() {
    let a: Tmalloc = Tmalloc::new();
    unsafe {
        for size in [8usize, 100, 1000, 8000] {
            let l = Layout::from_size_align(size, 8).unwrap();
            // Enough to refill and flush several times at the largest and smallest batch sizes.
            let ps: Vec<_> = (0..700).map(|_| a.alloc(l)).collect();
            a.check_pools().unwrap();
            for (i, p) in ps.into_iter().enumerate() {
                a.dealloc(p, l);
                if i % 50 == 0 {
                    a.check_pools().unwrap();
                }
            }
            a.check_pools().unwrap();
        }
    }
}

#[test]
fn every_class_boundary_works() {
    let a: Tmalloc = Tmalloc::new();
    unsafe {
        for size in (1..=SMALL_MAX + 100).step_by(7).chain([SMALL_MAX, SMALL_MAX + 1, HUGE_MIN - 1, HUGE_MIN, HUGE_MIN + 1]) {
            let l = Layout::from_size_align(size, 8).unwrap();
            let p = a.alloc(l);
            assert!(!p.is_null(), "size {size}");
            std::ptr::write_bytes(p, 0xC3, size);
            a.dealloc(p, l);
        }
    }
    a.check_heap().unwrap();
}

#[test]
fn realloc_keeps_contents_across_every_path() {
    let a: Tmalloc = Tmalloc::new();
    unsafe {
        let sizes = [1usize, 20, 100, 5000, 8192, 8193, 30_000, HUGE_MIN - 8, HUGE_MIN, 1 << 20, 5000, 100, 1];
        let mut size = sizes[0];
        let mut l = Layout::from_size_align(size, 16).unwrap();
        let mut p = a.alloc(l);
        std::ptr::write_bytes(p, 0x77, size);
        for &ns in &sizes[1..] {
            let np = a.realloc(p, l, ns);
            assert!(!np.is_null(), "{size} -> {ns}");
            let keep = size.min(ns);
            assert!(std::slice::from_raw_parts(np, keep).iter().all(|&b| b == 0x77), "{size} -> {ns}");
            std::ptr::write_bytes(np, 0x77, ns);
            size = ns;
            l = Layout::from_size_align(size, 16).unwrap();
            p = np;
        }
        a.dealloc(p, l);
    }
    assert_eq!(a.stats().huge_mapped, 0);
    a.check_heap().unwrap();
}

#[test]
fn alloc_zeroed_is_zero_even_for_recycled_memory() {
    let a: Tmalloc = Tmalloc::new();
    unsafe {
        for size in [8usize, 100, 5000, 20_000, HUGE_MIN, 1 << 21] {
            let l = Layout::from_size_align(size, 16).unwrap();
            let p = a.alloc(l);
            std::ptr::write_bytes(p, 0xFF, size);
            a.dealloc(p, l);
            let z = a.alloc_zeroed(l);
            assert!(std::slice::from_raw_parts(z, size).iter().all(|&b| b == 0), "size {size}");
            a.dealloc(z, l);
        }
    }
}

#[test]
fn aligned_allocations_on_every_path() {
    let a: Tmalloc = Tmalloc::new();
    unsafe {
        for align in [16usize, 32, 64, 256, 4096, 16384, 65536, 131072] {
            for size in [1usize, 100, 5000, 9000, 300_000, 2_000_000] {
                let l = Layout::from_size_align(size, align).unwrap();
                let p = a.alloc(l);
                assert!(!p.is_null(), "size {size} align {align}");
                assert_eq!(p as usize % align, 0, "size {size} align {align}");
                std::ptr::write_bytes(p, 1, size);
                a.dealloc(p, l);
            }
        }
    }
    a.check_heap().unwrap();
}

#[test]
fn huge_allocations_are_returned_to_the_system() {
    let a: Tmalloc = Tmalloc::new();
    unsafe {
        let l = Layout::from_size_align(5 << 20, 16).unwrap();
        let p = a.alloc(l);
        assert_eq!(a.stats().huge_mapped, 5 << 20);
        a.dealloc(p, l);
        assert_eq!(a.stats().huge_mapped, 0);
    }
}

#[test]
fn without_caches_it_still_works() {
    let a: Tmalloc<false> = Tmalloc::new();
    unsafe {
        let l = Layout::from_size_align(64, 8).unwrap();
        let ps: Vec<_> = (0..500).map(|_| a.alloc(l)).collect();
        for p in ps {
            a.dealloc(p, l);
        }
    }
}

/// Runs `f` on a new thread and waits until that thread has fully exited.
fn on_new_thread<R: Send>(a: &Tmalloc, f: impl FnOnce() -> R + Send) -> R {
    let r = std::thread::scope(|s| s.spawn(f).join().unwrap());
    quiesce(a, 0);
    r
}

#[test]
fn a_thread_that_exits_hands_everything_in_its_cache_to_the_pool() {
    let l = Layout::from_size_align(48, 8).unwrap();
    let class = class_of(48);
    let b = batch(SIZES[class]);
    // (objects allocated, objects then freed by a different thread). The freeing thread's cache
    // holds `m` objects at exit, which is rarely a whole number of batches: the remainder must go
    // back to the pool too.
    for (n, m) in [(1030usize, 100usize), (1000, 31), (500, 200), (64, 64), (300, 1), (130, 129)] {
        let a: Tmalloc = Tmalloc::new();
        assert_eq!(a.live_caches(), 0);
        let rounded = n.div_ceil(b) * b;
        // Thread 1 allocates `n` objects and exits holding the unused rest of its last batch.
        let ptrs: Vec<usize> = on_new_thread(&a, || unsafe { (0..n).map(|_| a.alloc(l) as usize).collect() });
        a.check_pools().unwrap();
        assert_eq!(a.pooled_objects(class), rounded - n, "n = {n}: the unused part of the last batch was lost");
        // Thread 2 frees `m` of them (they were allocated elsewhere) and exits.
        on_new_thread(&a, || unsafe {
            for &p in &ptrs[..m] {
                a.dealloc(p as *mut u8, l);
            }
        });
        a.check_pools().unwrap();
        assert_eq!(a.pooled_objects(class), rounded - n + m, "n = {n}, m = {m}: objects were lost when the thread exited");
        // Thread 3 frees the rest: everything is back.
        on_new_thread(&a, || unsafe {
            for &p in &ptrs[m..] {
                a.dealloc(p as *mut u8, l);
            }
        });
        assert_eq!(a.pooled_objects(class), rounded, "n = {n}, m = {m}");
        // And all of it is reusable without mapping anything new.
        let mapped = a.stats().small_mapped;
        on_new_thread(&a, || unsafe {
            let ps: Vec<_> = (0..n).map(|_| a.alloc(l)).collect();
            for p in ps {
                a.dealloc(p, l);
            }
        });
        assert_eq!(a.stats().small_mapped, mapped, "n = {n}: pooled objects were reused");
        a.check_pools().unwrap();
    }
}

#[test]
fn more_threads_than_slots_still_work() {
    let a: Tmalloc = Tmalloc::new();
    let n = SLOTS + 60;
    let barrier = std::sync::Barrier::new(n);
    std::thread::scope(|s| {
        for t in 0..n {
            let (a, barrier) = (&a, &barrier);
            s.spawn(move || unsafe {
                // Every thread is alive at once, so the later ones find no slot free.
                let l = Layout::from_size_align(16 + t % 200, 8).unwrap();
                let ps: Vec<_> = (0..300).map(|_| a.alloc(l)).collect();
                for &p in &ps {
                    std::ptr::write_bytes(p, t as u8, l.size());
                }
                barrier.wait();
                for p in ps {
                    assert_eq!(*p, t as u8);
                    a.dealloc(p, l);
                }
            });
        }
    });
    quiesce(&a, 0);
    a.check_pools().unwrap();
    assert_eq!(a.live_caches(), 0);
}

#[test]
fn one_thread_can_use_two_instances_in_turn() {
    let a: Tmalloc = Tmalloc::new();
    let b: Tmalloc = Tmalloc::new();
    unsafe {
        let l = Layout::from_size_align(80, 8).unwrap();
        let mut held = Vec::new();
        for round in 0..50 {
            let (x, y) = if round % 2 == 0 { (&a, &b) } else { (&b, &a) };
            let p = x.alloc(l);
            let q = y.alloc(l);
            std::ptr::write_bytes(p, 1, 80);
            std::ptr::write_bytes(q, 2, 80);
            held.push((x as *const Tmalloc, p, y as *const Tmalloc, q));
        }
        for (x, p, y, q) in held {
            assert_eq!(*p, 1);
            assert_eq!(*q, 2);
            (*x).dealloc(p, l);
            (*y).dealloc(q, l);
        }
    }
    a.check_pools().unwrap();
    b.check_pools().unwrap();
    // Each instance served its own requests from its own slabs, not through the other's cache.
    assert!(a.stats().small_mapped > 0 && b.stats().small_mapped > 0);
    // Alternating gave the slot back each time the thread moved on: at most one is held now.
    assert!(a.live_caches() + b.live_caches() <= 1);
}

#[test]
fn dropping_an_allocator_releases_this_threads_slot() {
    // A thread that used an instance and then drops it must not leave a dangling cache behind: a
    // later instance, possibly at the same address, starts clean.
    for _ in 0..20 {
        let a: Tmalloc = Tmalloc::new();
        unsafe {
            let l = Layout::from_size_align(32, 8).unwrap();
            let p = a.alloc(l);
            a.dealloc(p, l);
        }
        a.check_pools().unwrap();
    }
}

#[test]
fn pooled_batches_always_hold_exactly_one_batch() {
    let a: Tmalloc = Tmalloc::new();
    unsafe {
        for size in [16usize, 100, 1000, 8000] {
            let l = Layout::from_size_align(size, 8).unwrap();
            // Fill and empty the cache several times over so batches flow in both directions.
            let ps: Vec<_> = (0..2000).map(|_| a.alloc(l)).collect();
            for p in &ps {
                a.dealloc(*p, l);
            }
            a.check_pools().unwrap();
            let again: Vec<_> = (0..1500).map(|_| a.alloc(l)).collect();
            a.check_pools().unwrap();
            for p in again {
                a.dealloc(p, l);
            }
        }
    }
    a.check_pools().unwrap();
}

#[test]
fn single_object_paths_and_batches_mix() {
    // The uncached allocator moves one object at a time; objects it frees are loose, never batches.
    let nc: Tmalloc<false> = Tmalloc::new();
    unsafe {
        let l = Layout::from_size_align(64, 8).unwrap();
        let ps: Vec<_> = (0..500).map(|_| nc.alloc(l)).collect();
        for p in ps {
            nc.dealloc(p, l);
        }
        let ps: Vec<_> = (0..500).map(|_| nc.alloc(l)).collect();
        assert_eq!(nc.stats().small_mapped, 64 * 1024, "all 500 came from the loose list and one slab");
        for p in ps {
            nc.dealloc(p, l);
        }
    }
    nc.check_pools().unwrap();
}
