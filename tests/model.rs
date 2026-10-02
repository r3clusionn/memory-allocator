//! Random operations on a `Tmalloc` instance from many threads, with content checks.
//!
//! Every block is filled with a pattern that depends on its address and a per-block key. Two live
//! blocks that overlap, a block handed out twice, a free that lands in the wrong pool or a write
//! through a stale pointer all show up as a pattern that is not what was written.

use std::alloc::{GlobalAlloc, Layout};
use std::sync::{mpsc, Barrier};
use std::thread;

use tmalloc::{Tmalloc, HUGE_MIN};

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

#[derive(Clone, Copy)]
struct Block {
    ptr: usize,
    size: usize,
    align: usize,
    key: u8,
}

fn pattern(ptr: usize, i: usize, key: u8) -> u8 {
    (ptr >> 4) as u8 ^ (i as u8).wrapping_mul(31) ^ key
}

unsafe fn fill(b: &Block, from: usize, to: usize) {
    for i in from..to {
        *(b.ptr as *mut u8).add(i) = pattern(b.ptr, i, b.key);
    }
}

unsafe fn verify(b: &Block, upto: usize, what: &str) {
    // Checking every byte of a 2 MB block is slow; check the ends, the middle and a sample.
    let n = upto.min(b.size);
    let mut i = 0;
    while i < n {
        let got = *(b.ptr as *const u8).add(i);
        assert_eq!(
            got,
            pattern(b.ptr, i, b.key),
            "{what}: block {:#x} size {} align {} damaged at byte {i}",
            b.ptr,
            b.size,
            b.align
        );
        i += if n > 4096 { 61 } else { 1 };
    }
    if n > 0 {
        let last = n - 1;
        assert_eq!(*(b.ptr as *const u8).add(last), pattern(b.ptr, last, b.key), "{what}: last byte of {:#x}", b.ptr);
    }
}

/// Threads that finished their work may not have finished exiting; their caches are flushed when
/// they do. Wait for that before inspecting the pools.
fn quiesce(a: &Tmalloc, baseline: usize) {
    let t = std::time::Instant::now();
    while a.live_caches() > baseline {
        assert!(t.elapsed().as_secs() < 10, "{} cache slots are still owned", a.live_caches());
        thread::yield_now();
    }
}

fn pick_size(rng: &mut Rng) -> usize {
    match rng.below(100) {
        0..=59 => 1 + rng.below(256) as usize,
        60..=84 => 1 + rng.below(8192) as usize,
        85..=96 => 8193 + rng.below(100_000) as usize,
        97..=98 => 100_000 + rng.below(200_000) as usize,
        _ => HUGE_MIN + rng.below(1_500_000) as usize,
    }
}

fn layout(b: &Block) -> Layout {
    Layout::from_size_align(b.size, b.align).unwrap()
}

/// One thread's random life: alloc, free, realloc, and hand blocks to other threads.
fn worker(
    a: &Tmalloc,
    seed: u64,
    ops: usize,
    send: Option<&mpsc::Sender<Block>>,
    recv: Option<&mpsc::Receiver<Block>>,
    done: Option<&Barrier>,
) {
    let mut rng = Rng(seed);
    let mut live: Vec<Block> = Vec::new();
    unsafe {
        for step in 0..ops {
            // Free anything other threads sent us.
            if let Some(r) = recv {
                while let Ok(b) = r.try_recv() {
                    verify(&b, b.size, "received");
                    a.dealloc(b.ptr as *mut u8, layout(&b));
                }
            }
            let r = rng.below(100);
            if r < 40 || live.is_empty() {
                let size = pick_size(&mut rng);
                let align = if rng.below(16) == 0 { 16usize << rng.below(8) } else { [1usize, 8, 16][rng.below(3) as usize] };
                let l = Layout::from_size_align(size, align).unwrap();
                let zeroed = rng.below(8) == 0;
                let p = if zeroed { a.alloc_zeroed(l) } else { a.alloc(l) };
                assert!(!p.is_null(), "step {step}: alloc({size}, {align}) failed");
                assert_eq!(p as usize % align, 0, "step {step}: misaligned ({size}, {align})");
                if zeroed {
                    let z = std::slice::from_raw_parts(p, size);
                    assert!(z.iter().all(|&x| x == 0), "step {step}: alloc_zeroed({size}, {align}) returned dirty memory");
                }
                let b = Block { ptr: p as usize, size, align, key: rng.next() as u8 };
                fill(&b, 0, size);
                live.push(b);
            } else if r < 75 {
                let i = rng.below(live.len() as u64) as usize;
                let b = live.swap_remove(i);
                verify(&b, b.size, "free");
                if let (Some(s), true) = (send, rng.below(3) == 0) {
                    // Freed by another thread.
                    s.send(b).unwrap();
                } else {
                    a.dealloc(b.ptr as *mut u8, layout(&b));
                }
            } else {
                let i = rng.below(live.len() as u64) as usize;
                let b = live[i];
                let new_size = pick_size(&mut rng);
                verify(&b, b.size, "before realloc");
                let np = a.realloc(b.ptr as *mut u8, layout(&b), new_size);
                assert!(!np.is_null(), "step {step}: realloc to {new_size} failed");
                assert_eq!(np as usize % b.align, 0, "realloc broke alignment");
                let nb = Block { ptr: np as usize, size: new_size, align: b.align, key: b.key };
                // The kept prefix must have moved with the block; the old pattern depended on the old
                // address, so compare against bytes written for the old block.
                let keep = b.size.min(new_size);
                for i in (0..keep).step_by(if keep > 4096 { 61 } else { 1 }) {
                    assert_eq!(
                        *np.add(i),
                        pattern(b.ptr, i, b.key),
                        "step {step}: realloc lost byte {i} ({} -> {new_size})",
                        b.size
                    );
                }
                fill(&nb, 0, new_size);
                live[i] = nb;
            }
        }
        for b in live {
            verify(&b, b.size, "final");
            a.dealloc(b.ptr as *mut u8, layout(&b));
        }
        // Once every thread has stopped sending, free what is left in this thread's channel.
        if let (Some(bar), Some(r)) = (done, recv) {
            bar.wait();
            for b in r.try_iter() {
                verify(&b, b.size, "drained");
                a.dealloc(b.ptr as *mut u8, layout(&b));
            }
        }
    }
}

#[test]
fn one_thread_every_path() {
    let a: Tmalloc = Tmalloc::new();
    worker(&a, 0xABCD, 30_000, None, None, None);
    quiesce(&a, 1);
    a.check_heap().unwrap();
    a.check_pools().unwrap();
    let s = a.stats();
    assert_eq!(s.huge_mapped, 0);
    assert_eq!(s.heap.in_use, 0);
}

#[test]
fn many_threads_with_cross_thread_frees() {
    let a: Tmalloc = Tmalloc::new();
    const THREADS: usize = 8;
    // A ring: thread i sends some of its blocks to thread i+1, which frees them.
    let (senders, receivers): (Vec<_>, Vec<_>) = (0..THREADS).map(|_| mpsc::channel::<Block>()).unzip();
    let mut receivers: Vec<_> = receivers.into_iter().map(Some).collect();
    let done = Barrier::new(THREADS);
    thread::scope(|s| {
        for t in 0..THREADS {
            let a = &a;
            let send = senders[(t + 1) % THREADS].clone();
            let recv = receivers[t].take().unwrap();
            let done = &done;
            s.spawn(move || worker(a, 0x1000 + t as u64 * 7919, 15_000, Some(&send), Some(&recv), Some(done)));
        }
        drop(senders);
    });
    quiesce(&a, 0);
    a.check_heap().unwrap();
    a.check_pools().unwrap();
    let s = a.stats();
    assert_eq!((s.huge_mapped, s.heap.in_use), (0, 0), "everything was freed");
}

#[test]
fn the_per_thread_caches_can_be_turned_off() {
    let a: Tmalloc<false> = Tmalloc::new();
    thread::scope(|s| {
        for t in 0..4 {
            let a = &a;
            s.spawn(move || {
                let mut rng = Rng(99 + t);
                let mut live = Vec::new();
                unsafe {
                    for _ in 0..20_000 {
                        if rng.below(2) == 0 || live.is_empty() {
                            let l = Layout::from_size_align(1 + rng.below(2000) as usize, 8).unwrap();
                            let p = a.alloc(l);
                            let b = Block { ptr: p as usize, size: l.size(), align: 8, key: t as u8 };
                            fill(&b, 0, b.size);
                            live.push(b);
                        } else {
                            let i = rng.below(live.len() as u64) as usize;
                            let b = live.swap_remove(i);
                            verify(&b, b.size, "no-cache");
                            a.dealloc(b.ptr as *mut u8, layout(&b));
                        }
                    }
                    for b in live {
                        a.dealloc(b.ptr as *mut u8, layout(&b));
                    }
                }
            });
        }
    });
}

#[test]
fn freed_small_objects_are_recycled_across_threads() {
    let a: Tmalloc = Tmalloc::new();
    let (tx, rx) = mpsc::sync_channel::<(usize, usize)>(1024);
    thread::scope(|s| {
        let a = &a;
        s.spawn(move || {
            let mut rng = Rng(11);
            for _ in 0..2_000_000 {
                let size = 16 + rng.below(240) as usize;
                let l = Layout::from_size_align(size, 16).unwrap();
                let p = unsafe { a.alloc(l) };
                assert!(!p.is_null());
                tx.send((p as usize, size)).unwrap();
            }
        });
        s.spawn(move || {
            for (p, size) in rx {
                let l = Layout::from_size_align(size, 16).unwrap();
                unsafe { a.dealloc(p as *mut u8, l) };
            }
        });
    });
    // The instance is on this test's stack: the worker threads must be fully gone (their exit
    // callbacks touch it) before it is dropped.
    quiesce(&a, 0);
    let mapped = a.stats().small_mapped;
    // At most about 1024 blocks are in flight (plus what the caches hold): a few MB, not the
    // 2,000,000 * 136 bytes = 272 MB that leaking would need.
    assert!(mapped < 32 << 20, "small_mapped is {mapped} bytes after 2,000,000 cross-thread alloc/free pairs");
}
