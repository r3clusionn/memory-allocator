//! Allocator benchmarks: `cargo run --release --example bench [-- --runs N] [--quick]`.
//!
//! Every allocator is driven through the same `GlobalAlloc` calls, so the comparison is of the
//! allocators and not of Rust collections. Each time is the median of `--runs` runs. Single-thread
//! rows are nanoseconds per allocate-plus-free pair (lower is better); multi-thread rows are millions
//! of pairs per second over all threads (higher is better). The memory section starts one process
//! per allocator and compares working sets.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::sync::Barrier;
use std::time::Instant;

use mimalloc::MiMalloc;
use tmalloc::Tmalloc;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next() % (hi - lo + 1) as u64) as usize
    }
}

fn layout(size: usize) -> Layout {
    Layout::from_size_align(black_box(size), 16).unwrap()
}

// Every allocation and free goes through one of these, never inlined, with the layout hidden from
// the optimiser. A program using `#[global_allocator]` calls the allocator across a function
// boundary with sizes the compiler usually cannot see; without this, an allocator written in Rust
// would have its size-class lookup folded away at compile time while mimalloc and the system heap,
// which are opaque calls, could not.
#[inline(never)]
unsafe fn al<A: GlobalAlloc>(a: &A, l: Layout) -> *mut u8 {
    a.alloc(l)
}

#[inline(never)]
unsafe fn de<A: GlobalAlloc>(a: &A, p: *mut u8, l: Layout) {
    a.dealloc(p, l)
}

/// Touches the first byte of each page so the system really provides the memory.
#[inline(always)]
unsafe fn touch(p: *mut u8, size: usize) {
    let mut i = 0;
    while i < size {
        p.add(i).write_volatile(1);
        i += 4096;
    }
    p.add(size - 1).write_volatile(1);
}

// ---- single-thread workloads: return ns per alloc+free pair ------------------------------------

fn small_churn<A: GlobalAlloc>(a: &A, scale: usize) -> f64 {
    let n = 5_000_000 * scale;
    let l = layout(64);
    let t = Instant::now();
    unsafe {
        for _ in 0..n {
            let p = al(a, l);
            black_box(p);
            p.write_volatile(1);
            de(a, p, l);
        }
    }
    t.elapsed().as_secs_f64() * 1e9 / n as f64
}

fn lifo_batches<A: GlobalAlloc>(a: &A, scale: usize) -> f64 {
    let mut rng = Rng(1);
    let sizes: Vec<usize> = (0..1000).map(|_| rng.range(16, 256)).collect();
    let rounds = 2_000 * scale;
    let mut ptrs = vec![std::ptr::null_mut::<u8>(); sizes.len()];
    let t = Instant::now();
    unsafe {
        for _ in 0..rounds {
            for (i, &s) in sizes.iter().enumerate() {
                let p = al(a, layout(s));
                p.write_volatile(1);
                ptrs[i] = p;
            }
            for (i, &s) in sizes.iter().enumerate().rev() {
                de(a, ptrs[i], layout(s));
            }
        }
    }
    t.elapsed().as_secs_f64() * 1e9 / (rounds * sizes.len()) as f64
}

fn random_order_free<A: GlobalAlloc>(a: &A, scale: usize) -> f64 {
    let mut rng = Rng(2);
    let n = 100_000;
    let sizes: Vec<usize> = (0..n).map(|_| rng.range(16, 1024)).collect();
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        order.swap(i, rng.range(0, i));
    }
    let rounds = 10 * scale;
    let mut ptrs = vec![std::ptr::null_mut::<u8>(); n];
    let t = Instant::now();
    unsafe {
        for _ in 0..rounds {
            for i in 0..n {
                let p = al(a, layout(sizes[i]));
                p.write_volatile(1);
                ptrs[i] = p;
            }
            for &i in &order {
                de(a, ptrs[i], layout(sizes[i]));
            }
        }
    }
    t.elapsed().as_secs_f64() * 1e9 / (rounds * n) as f64
}

fn mixed_steady_state<A: GlobalAlloc>(a: &A, scale: usize) -> f64 {
    let mut rng = Rng(3);
    let slots = 50_000;
    let ops = 2_000_000 * scale;
    let mut live: Vec<(*mut u8, usize)> = vec![(std::ptr::null_mut(), 0); slots];
    let size = |rng: &mut Rng| match rng.next() % 1000 {
        0..=639 => rng.range(16, 256),
        640..=939 => rng.range(257, 4096),
        940..=994 => rng.range(4097, 65_536),
        _ => rng.range(65_537, 400_000),
    };
    unsafe {
        for s in live.iter_mut() {
            let n = size(&mut rng);
            let p = al(a, layout(n));
            p.write_volatile(1);
            *s = (p, n);
        }
        let t = Instant::now();
        for _ in 0..ops {
            let i = rng.range(0, slots - 1);
            let (p, n) = live[i];
            de(a, p, layout(n));
            let n = size(&mut rng);
            let p = al(a, layout(n));
            p.write_volatile(1);
            live[i] = (p, n);
        }
        let dt = t.elapsed().as_secs_f64() * 1e9 / ops as f64;
        for (p, n) in live {
            de(a, p, layout(n));
        }
        dt
    }
}

fn medium_blocks<A: GlobalAlloc>(a: &A, scale: usize) -> f64 {
    let mut rng = Rng(4);
    let slots = 2_000;
    let ops = 300_000 * scale;
    let mut live: Vec<(*mut u8, usize)> = vec![(std::ptr::null_mut(), 0); slots];
    unsafe {
        for s in live.iter_mut() {
            let n = rng.range(16_384, 131_072);
            let p = al(a, layout(n));
            touch(p, n);
            *s = (p, n);
        }
        let t = Instant::now();
        for _ in 0..ops {
            let i = rng.range(0, slots - 1);
            let (p, n) = live[i];
            de(a, p, layout(n));
            let n = rng.range(16_384, 131_072);
            let p = al(a, layout(n));
            touch(p, n);
            live[i] = (p, n);
        }
        let dt = t.elapsed().as_secs_f64() * 1e9 / ops as f64;
        for (p, n) in live {
            de(a, p, layout(n));
        }
        dt
    }
}

fn large_blocks<A: GlobalAlloc>(a: &A, scale: usize) -> f64 {
    let mut rng = Rng(5);
    let ops = 5_000 * scale;
    let t = Instant::now();
    unsafe {
        for _ in 0..ops {
            let n = rng.range(1 << 20, 4 << 20);
            let p = al(a, layout(n));
            touch(p, n);
            de(a, p, layout(n));
        }
    }
    t.elapsed().as_secs_f64() * 1e9 / ops as f64
}

// ---- multi-thread workloads: return millions of pairs per second -------------------------------

fn thread_test<A: GlobalAlloc + Sync>(a: &A, threads: usize, scale: usize) -> f64 {
    let per_thread = 50_000;
    let rounds = 10 * scale;
    let barrier = Barrier::new(threads + 1);
    let mut secs = 0.0;
    std::thread::scope(|s| {
        for t in 0..threads {
            let barrier = &barrier;
            s.spawn(move || {
                let mut rng = Rng(100 + t as u64);
                let sizes: Vec<usize> = (0..per_thread).map(|_| rng.range(16, 128)).collect();
                let mut ptrs = vec![std::ptr::null_mut::<u8>(); per_thread];
                barrier.wait();
                unsafe {
                    for _ in 0..rounds {
                        for i in 0..per_thread {
                            let p = al(a, layout(sizes[i]));
                            p.write_volatile(1);
                            ptrs[i] = p;
                        }
                        for i in 0..per_thread {
                            de(a, ptrs[i], layout(sizes[i]));
                        }
                    }
                }
                barrier.wait();
            });
        }
        barrier.wait();
        let t = Instant::now();
        barrier.wait();
        secs = t.elapsed().as_secs_f64();
    });
    (threads * per_thread * rounds) as f64 / secs / 1e6
}

/// One thread allocates and another frees, through a single-producer ring of pointers.
fn cross_thread<A: GlobalAlloc + Sync>(a: &A, scale: usize) -> f64 {
    const RING: usize = 4096;
    let n = 2_000_000 * scale;
    let ring: Vec<AtomicPtr<u8>> = (0..RING).map(|_| AtomicPtr::new(std::ptr::null_mut())).collect();
    let head = AtomicUsize::new(0);
    let tail = AtomicUsize::new(0);
    let l = layout(64);
    let t = Instant::now();
    std::thread::scope(|s| {
        s.spawn(|| unsafe {
            for i in 0..n {
                let p = al(a, l);
                p.write_volatile(1);
                while i - head.load(Ordering::Acquire) >= RING {
                    std::hint::spin_loop();
                }
                ring[i % RING].store(p, Ordering::Relaxed);
                tail.store(i + 1, Ordering::Release);
            }
        });
        s.spawn(|| unsafe {
            for i in 0..n {
                while tail.load(Ordering::Acquire) <= i {
                    std::hint::spin_loop();
                }
                let p = ring[i % RING].load(Ordering::Relaxed);
                de(a, p, l);
                head.store(i + 1, Ordering::Release);
            }
        });
    });
    n as f64 / t.elapsed().as_secs_f64() / 1e6
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

struct Row {
    name: String,
    unit: &'static str,
    values: Vec<f64>,
}

fn all<A: GlobalAlloc + Sync>(a: &A, runs: usize, scale: usize, threads: &[usize]) -> Vec<(String, &'static str, f64)> {
    let mut out = Vec::new();
    macro_rules! single {
        ($name:expr, $f:expr) => {{
            let v = median((0..runs).map(|_| $f).collect());
            out.push(($name.to_string(), "ns", v));
        }};
    }
    single!("64-byte alloc and free, one at a time", small_churn(a, scale));
    single!("1,000 blocks of 16-256 bytes, freed in reverse", lifo_batches(a, scale));
    single!("100,000 blocks of 16-1,024 bytes, freed in random order", random_order_free(a, scale));
    single!("mixed sizes to 400 KB, 50,000 live, steady state", mixed_steady_state(a, scale));
    single!("16-128 KB blocks, 2,000 live, touched", medium_blocks(a, scale));
    single!("1-4 MB blocks, touched", large_blocks(a, scale));
    for &t in threads {
        let v = median((0..runs).map(|_| thread_test(a, t, scale)).collect());
        out.push((format!("{t} threads, each allocating 50,000 then freeing"), "M/s", v));
    }
    let v = median((0..runs).map(|_| cross_thread(a, scale)).collect());
    out.push(("allocated on one thread, freed on another".to_string(), "M/s", v));
    out
}

// ---- memory use --------------------------------------------------------------------------------

#[cfg(windows)]
fn working_set() -> (usize, usize) {
    use windows_sys::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    unsafe {
        let mut c: PROCESS_MEMORY_COUNTERS = std::mem::zeroed();
        c.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
        GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb);
        (c.WorkingSetSize, c.PeakWorkingSetSize)
    }
}

#[cfg(not(windows))]
fn working_set() -> (usize, usize) {
    (0, 0)
}

/// Builds and churns a heap, printing the working set after each phase:
/// `phase1 live_bytes working_set ...` on one line, for the parent to read.
fn memory_workload<A: GlobalAlloc>(a: &A) {
    let mut rng = Rng(77);
    let (base, _) = working_set();
    let mut live: Vec<(*mut u8, usize)> = Vec::new();
    let mut live_bytes = 0usize;
    let mut peak_live = 0usize;
    let report = |name: &str, live_bytes: usize, peak_live: &mut usize| {
        *peak_live = (*peak_live).max(live_bytes);
        let (ws, _) = working_set();
        println!("{name} {live_bytes} {}", ws.saturating_sub(base));
    };
    unsafe {
        // Phase 1: 300,000 blocks, sizes log-uniform from 16 B to 32 KB.
        for _ in 0..300_000 {
            let n = (16.0 * 2f64.powf((rng.next() % 1000) as f64 / 1000.0 * 11.0)) as usize;
            let p = al(a, layout(n));
            std::ptr::write_bytes(p, 1, n);
            live.push((p, n));
            live_bytes += n;
        }
        report("built", live_bytes, &mut peak_live);
        // Phase 2: free a random 80 percent.
        for i in (1..live.len()).rev() {
            let j = rng.range(0, i);
            live.swap(i, j);
        }
        let keep = live.len() / 5;
        for (p, n) in live.drain(keep..) {
            de(a, p, layout(n));
            live_bytes -= n;
        }
        report("freed 80%", live_bytes, &mut peak_live);
        // Phase 3: 300,000 new blocks, all small (up to 2 KB), into the holes.
        for _ in 0..300_000 {
            let n = rng.range(16, 2048);
            let p = al(a, layout(n));
            std::ptr::write_bytes(p, 1, n);
            live.push((p, n));
            live_bytes += n;
        }
        report("refilled", live_bytes, &mut peak_live);
        // Phase 4: free everything but a tenth.
        for i in (1..live.len()).rev() {
            let j = rng.range(0, i);
            live.swap(i, j);
        }
        let keep = live.len() / 10;
        for (p, n) in live.drain(keep..) {
            de(a, p, layout(n));
            live_bytes -= n;
        }
        report("freed 90%", live_bytes, &mut peak_live);
        for (p, n) in live {
            de(a, p, layout(n));
        }
    }
    let (_, peak) = working_set();
    println!("peak {peak_live} {}", peak.saturating_sub(base));
}

static TM: Tmalloc = Tmalloc::new();
static TM_NOCACHE: Tmalloc<false> = Tmalloc::new();
static MI: MiMalloc = MiMalloc;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    if let Some(which) = arg("--mem") {
        match which.as_str() {
            "tmalloc" => memory_workload(&TM),
            "system" => memory_workload(&System),
            "mimalloc" => memory_workload(&MI),
            other => panic!("unknown allocator {other}"),
        }
        return;
    }
    let quick = args.iter().any(|a| a == "--quick");
    let runs: usize = arg("--runs").and_then(|v| v.parse().ok()).unwrap_or(if quick { 1 } else { 5 });
    let scale = if quick { 1 } else { 2 };
    let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let threads: Vec<usize> = [1, 4, 8, 16].into_iter().filter(|&t| t <= cpus).collect();

    println!("median of {runs} runs, {cpus} logical processors");
    println!("single-thread rows: ns per allocate+free pair (lower is better); multi-thread rows: millions of pairs per second (higher is better)\n");
    let results = [
        ("tmalloc", all(&TM, runs, scale, &threads)),
        ("tmalloc, no thread caches", all(&TM_NOCACHE, runs, scale, &threads)),
        ("system", all(&System, runs, scale, &threads)),
        ("mimalloc", all(&MI, runs, scale, &threads)),
    ];
    let rows: Vec<Row> = (0..results[0].1.len())
        .map(|i| Row {
            name: results[0].1[i].0.clone(),
            unit: results[0].1[i].1,
            values: results.iter().map(|r| r.1[i].2).collect(),
        })
        .collect();
    print!("| Workload (unit) |");
    for (n, _) in &results {
        print!(" {n} |");
    }
    println!();
    print!("|---|");
    for _ in &results {
        print!("---|");
    }
    println!();
    for r in &rows {
        print!("| {} ({}) |", r.name, r.unit);
        for v in &r.values {
            if *v >= 100.0 {
                print!(" {v:.0} |");
            } else {
                print!(" {v:.1} |");
            }
        }
        println!();
    }

    if cfg!(windows) {
        println!(
            "\nMemory: 300,000 blocks of 16 B-32 KB, free 80% at random, 300,000 more blocks of up to 2 KB, free 90% at random."
        );
        println!("Working set above the process's size at start, in MB, and live bytes in MB (system = Windows heap).\n");
        let exe = std::env::current_exe().unwrap();
        // Per allocator: (phase name, live MB, working set MB) for each phase.
        type Phases = Vec<(String, f64, f64)>;
        let mut table: Vec<(String, Phases)> = Vec::new();
        for name in ["tmalloc", "system", "mimalloc"] {
            let out = std::process::Command::new(&exe).args(["--mem", name]).output().unwrap();
            let text = String::from_utf8_lossy(&out.stdout);
            let phases: Vec<(String, f64, f64)> = text
                .lines()
                .filter_map(|l| {
                    let p: Vec<&str> = l.split(' ').collect();
                    let n = p.len();
                    if n < 3 {
                        return None;
                    }
                    Some((p[..n - 2].join(" "), p[n - 2].parse::<f64>().ok()? / 1e6, p[n - 1].parse::<f64>().ok()? / 1e6))
                })
                .collect();
            table.push((name.to_string(), phases));
        }
        println!("| Phase | live (MB) | tmalloc (MB) | system (MB) | mimalloc (MB) |");
        println!("|---|---|---|---|---|");
        for i in 0..table[0].1.len() {
            let (phase, live, _) = &table[0].1[i];
            let label = if phase == "peak" { "peak working set over the whole run" } else { phase.as_str() };
            print!("| {label} | {live:.0} |");
            for (_, p) in &table {
                print!(" {:.0} |", p[i].2);
            }
            println!();
        }
    }
}
