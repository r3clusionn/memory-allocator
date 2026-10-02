//! A general-purpose memory allocator that can be installed as `#[global_allocator]`.
//!
//! * [`Tmalloc`]: the allocator. Small requests use size classes with per-thread caches, medium ones
//!   a boundary-tag heap with coalescing, large ones a mapping of their own.
//! * [`heap::Heap`]: the heap on its own, single-threaded, with an invariant checker.
//! * [`os`]: mapping pages from the system.

pub mod classes;
pub mod heap;
pub mod os;
mod spin;
mod threads;
mod tmalloc;

pub use tmalloc::{route, Route, Stats, Tmalloc, HUGE_MIN};
