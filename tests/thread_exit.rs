//! Thread exit with `Tmalloc` as the global allocator.
//!
//! When a thread ends, the system calls back to flush the thread's cache. Other destructors run
//! around that callback, and some of them allocate and free. Anything freed after the cache has been
//! handed back must go to the shared pools, because the slot may already belong to a new thread.
//!
//! This is the only test in its binary, so nothing else owns a cache while it checks the pools.

use std::cell::RefCell;
use std::thread;

use tmalloc::Tmalloc;

#[global_allocator]
static ALLOC: Tmalloc = Tmalloc::new();

/// Holds allocations that are freed, and new ones made and freed, by the thread-local destructor.
struct Bag {
    // One allocation per element is the point, so the boxes stay boxes.
    #[allow(clippy::vec_box)]
    boxes: Vec<Box<[u8; 100]>>,
    strings: Vec<String>,
}

impl Drop for Bag {
    fn drop(&mut self) {
        let extra: Vec<Vec<u8>> = (0..64).map(|i| vec![i as u8; 16 + i * 7]).collect();
        assert!(extra.iter().enumerate().all(|(i, v)| v.iter().all(|&b| b == i as u8)));
        drop(extra);
        self.boxes.clear();
        self.strings.clear();
        let _late = vec![0u8; 300];
    }
}

thread_local! {
    static BAG: RefCell<Bag> = const { RefCell::new(Bag { boxes: Vec::new(), strings: Vec::new() }) };
}

#[test]
fn destructors_that_allocate_during_thread_exit_are_safe() {
    // The harness's main thread and this test's thread each own a cache slot already.
    let baseline = ALLOC.live_caches();
    for round in 0..150 {
        let hs: Vec<_> = (0..24)
            .map(|t| {
                thread::spawn(move || {
                    BAG.with(|b| {
                        let mut b = b.borrow_mut();
                        for i in 0..400 {
                            b.boxes.push(Box::new([(i + t) as u8; 100]));
                            b.strings.push(format!("{round}-{t}-{i}"));
                        }
                    });
                    // Also allocate through the allocator directly, outside the bag.
                    let v: Vec<Vec<u8>> = (0..300).map(|i| vec![t as u8; 1 + i % 500]).collect();
                    v.iter().map(|x| x.len()).sum::<usize>()
                })
            })
            .collect();
        for h in hs {
            assert!(h.join().unwrap() > 0);
        }
    }
    // Every thread has exited: all their cache slots were released, and what they held is back in the
    // shared pools in the right shape.
    let t = std::time::Instant::now();
    while ALLOC.live_caches() > baseline {
        assert!(t.elapsed().as_secs() < 10, "{} cache slots still owned", ALLOC.live_caches());
        thread::yield_now();
    }
    ALLOC.check_pools().unwrap();
    ALLOC.check_heap().unwrap();
    // Slabs are bounded by what 24 threads held at once, not by 150 rounds of them: slots are
    // reused and so is the memory in the pools.
    let mapped = ALLOC.stats().small_mapped;
    assert!(mapped < 64 << 20, "{mapped} bytes of slabs after 3,600 short-lived threads");
}
