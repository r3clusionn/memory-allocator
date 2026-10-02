//! A small spin lock. The allocator cannot use `std::sync::Mutex`: on some platforms its slow path
//! allocates, and an allocator that allocates while holding its own lock deadlocks itself.

use std::cell::UnsafeCell;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, Ordering};

pub struct SpinLock<T> {
    locked: AtomicBool,
    data: UnsafeCell<T>,
}

unsafe impl<T: Send> Send for SpinLock<T> {}
unsafe impl<T: Send> Sync for SpinLock<T> {}

pub struct Guard<'a, T> {
    lock: &'a SpinLock<T>,
}

impl<T> SpinLock<T> {
    pub const fn new(v: T) -> Self {
        SpinLock { locked: AtomicBool::new(false), data: UnsafeCell::new(v) }
    }

    /// Takes the lock if it is free right now.
    #[inline(always)]
    pub fn try_lock(&self) -> Option<Guard<'_, T>> {
        // Test first so a contended lock is not hammered with read-modify-write operations.
        if !self.locked.load(Ordering::Relaxed)
            && self.locked.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_ok()
        {
            Some(Guard { lock: self })
        } else {
            None
        }
    }

    pub fn lock(&self) -> Guard<'_, T> {
        let mut spins = 0u32;
        loop {
            if let Some(g) = self.try_lock() {
                return g;
            }
            while self.locked.load(Ordering::Relaxed) {
                spins += 1;
                if spins < 100 {
                    std::hint::spin_loop();
                } else {
                    // The holder may have been descheduled; let it run.
                    std::thread::yield_now();
                    spins = 0;
                }
            }
        }
    }
}

impl<T> Deref for Guard<'_, T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &T {
        unsafe { &*self.lock.data.get() }
    }
}

impl<T> DerefMut for Guard<'_, T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.lock.data.get() }
    }
}

impl<T> Drop for Guard<'_, T> {
    #[inline(always)]
    fn drop(&mut self) {
        self.lock.locked.store(false, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_lock_fails_while_held_and_succeeds_after() {
        let l = SpinLock::new(5);
        let g = l.try_lock().unwrap();
        assert!(l.try_lock().is_none());
        drop(g);
        assert_eq!(*l.try_lock().unwrap(), 5);
    }

    #[test]
    fn it_excludes() {
        let l = SpinLock::new(0u64);
        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| {
                    for _ in 0..50_000 {
                        *l.lock() += 1;
                    }
                });
            }
        });
        assert_eq!(*l.lock(), 400_000);
    }
}
