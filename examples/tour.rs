//! What the allocator does with requests of different sizes: `cargo run --release --example tour`.

use std::alloc::{GlobalAlloc, Layout};

use tmalloc::{route, Route, Tmalloc};

static ALLOC: Tmalloc = Tmalloc::new();

fn describe(r: Route) -> String {
    match r {
        Route::Small(c) => format!("size class {c} ({} B)", tmalloc::classes::SIZES[c]),
        Route::Heap => "boundary-tag heap".into(),
        Route::Huge => "direct mapping".into(),
        Route::System => "system allocator".into(),
    }
}

fn main() {
    println!("{:>10}  {:>6}  {:<26}  {:>9}  {:>8}", "size", "align", "served by", "at least", "aligned");
    let mut kept = Vec::new();
    for (size, align) in [
        (1, 1),
        (24, 8),
        (100, 16),
        (4000, 16),
        (8192, 16),
        (8193, 16),
        (60_000, 16),
        (262_143, 16),
        (262_144, 16),
        (5_000_000, 16),
        (100, 64),
        (100, 4096),
        (20_000, 4096),
    ] {
        let layout = Layout::from_size_align(size, align).unwrap();
        let p = unsafe { ALLOC.alloc(layout) };
        let r = route(size, align);
        let aligned = if (p as usize).is_multiple_of(align) { "yes" } else { "NO" };
        println!("{size:>10}  {align:>6}  {:<26}  {:>9}  {aligned:>8}", describe(r), ALLOC.usable_size(layout));
        kept.push((p, layout));
    }
    let s = ALLOC.stats();
    println!("\nwhile all of those are live:");
    println!("  size-class slabs mapped  {:>10} bytes (never returned)", s.small_mapped);
    println!("  heap: mapped             {:>10} bytes in {} chunk(s)", s.heap.mapped, s.heap.chunks);
    println!("  heap: in use             {:>10} bytes in blocks", s.heap.in_use);
    println!("  direct mappings          {:>10} bytes", s.huge_mapped);
    for (p, l) in kept {
        unsafe { ALLOC.dealloc(p, l) };
    }
    let s = ALLOC.stats();
    println!("\nafter freeing them all:");
    println!("  size-class slabs mapped  {:>10} bytes (kept for reuse)", s.small_mapped);
    println!("  heap: mapped             {:>10} bytes in {} chunk(s) (one spare chunk is kept)", s.heap.mapped, s.heap.chunks);
    println!("  heap: in use             {:>10} bytes", s.heap.in_use);
    println!("  direct mappings          {:>10} bytes (returned to the system)", s.huge_mapped);
    ALLOC.check_heap().expect("heap invariants hold");
    println!("\nheap invariants checked: ok");
}
