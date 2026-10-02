//! `Tmalloc` as the process's global allocator, under ordinary programs.
//!
//! The test harness, every thread it starts and everything below allocate through it.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

use tmalloc::Tmalloc;

#[global_allocator]
static ALLOC: Tmalloc = Tmalloc::new();

#[test]
fn collections() {
    let mut map = HashMap::new();
    for i in 0..200_000u64 {
        map.insert(i, format!("value-{i}"));
    }
    assert_eq!(map.len(), 200_000);
    assert_eq!(map[&123_456], "value-123456");
    for i in (0..200_000u64).step_by(2) {
        map.remove(&i);
    }
    assert_eq!(map.len(), 100_000);

    let mut tree = BTreeMap::new();
    for i in 0..100_000u32 {
        tree.insert(i.wrapping_mul(2_654_435_761), vec![i; (i % 17) as usize]);
    }
    let sum: u64 = tree.values().map(|v| v.len() as u64).sum();
    assert_eq!(sum, (0..100_000u32).map(|i| (i % 17) as u64).sum::<u64>());

    let mut dq = VecDeque::new();
    for i in 0..500_000 {
        dq.push_back(i);
        if i % 3 == 0 {
            dq.pop_front();
        }
    }
    assert_eq!(dq.len(), 500_000 - 166_667);
}

#[test]
fn a_vec_that_grows_through_every_size_path() {
    // Doubling passes the class boundary (8 KiB), the huge threshold (256 KiB) and ends in the
    // tens of megabytes: each growth is a realloc that changes path or maps new pages.
    let mut v: Vec<u64> = Vec::new();
    for i in 0..6_000_000u64 {
        v.push(i * 3);
    }
    assert!(v.iter().enumerate().all(|(i, &x)| x == i as u64 * 3));
    v.shrink_to_fit();
    v.truncate(1000);
    v.shrink_to_fit();
    assert_eq!(v.len(), 1000);
    assert_eq!(v[999], 2997);
}

#[test]
fn zeroed_memory_is_zero() {
    for n in [1usize, 17, 4000, 9000, 70_000, 300_000, 5_000_000] {
        let v = vec![0u8; n];
        assert!(v.iter().all(|&b| b == 0), "{n}");
        // And again after dirtying and freeing a block of the same size.
        let mut d = vec![0xFFu8; n];
        d[0] = 1;
        drop(d);
        let v = vec![0u8; n];
        assert!(v.iter().all(|&b| b == 0), "{n} after reuse");
    }
}

#[test]
fn strings_and_sorting() {
    let mut words: Vec<String> = (0..300_000u32).map(|i| format!("{:08x}", i.wrapping_mul(2_246_822_519))).collect();
    words.sort();
    assert!(words.windows(2).all(|w| w[0] <= w[1]));
    let joined = words.join(",");
    assert_eq!(joined.len(), 300_000 * 9 - 1);
    let parts = joined.split(',').count();
    assert_eq!(parts, 300_000);
    let big = "ab".repeat(10_000_000);
    assert_eq!(big.len(), 20_000_000);
}

#[test]
fn threads_start_run_and_exit() {
    // Starting a thread allocates its stack bookkeeping and its name; exiting frees them. Many short
    // threads also reuse the cache slots by turns.
    for round in 0..40 {
        let hs: Vec<_> = (0..16)
            .map(|t| {
                thread::spawn(move || {
                    let mut v = Vec::new();
                    for i in 0..5000 {
                        v.push(format!("{round}-{t}-{i}"));
                    }
                    v.iter().map(|s| s.len()).sum::<usize>()
                })
            })
            .collect();
        for h in hs {
            assert!(h.join().unwrap() > 0);
        }
    }
}

#[test]
fn channels_move_memory_between_threads() {
    // Every vector is allocated on the producer's thread and freed on the consumer's.
    let (tx, rx) = mpsc::channel::<Vec<u8>>();
    let producers: Vec<_> = (0..4)
        .map(|p| {
            let tx = tx.clone();
            thread::spawn(move || {
                for i in 0..50_000usize {
                    tx.send(vec![(i % 251) as u8; 1 + (i + p) % 3000]).unwrap();
                }
            })
        })
        .collect();
    drop(tx);
    let total = Arc::new(Mutex::new(0usize));
    let consumer = {
        let total = total.clone();
        thread::spawn(move || {
            for v in rx {
                assert!(v.iter().all(|&b| b == v[0]));
                *total.lock().unwrap() += v.len();
            }
        })
    };
    for p in producers {
        p.join().unwrap();
    }
    consumer.join().unwrap();
    let expect: usize = (0..4).map(|p| (0..50_000usize).map(|i| 1 + (i + p) % 3000).sum::<usize>()).sum();
    assert_eq!(*total.lock().unwrap(), expect);
}

#[test]
fn the_allocator_reports_and_stays_consistent() {
    let before = ALLOC.stats();
    {
        let big: Vec<Vec<u8>> = (0..64).map(|i| vec![i as u8; 20_000 + i * 100]).collect();
        let s = ALLOC.stats();
        assert!(s.heap.in_use > before.heap.in_use);
        drop(big);
    }
    ALLOC.check_heap().unwrap();
    let huge = vec![1u8; 4 << 20];
    assert!(ALLOC.stats().huge_mapped >= 4 << 20);
    drop(huge);
}
