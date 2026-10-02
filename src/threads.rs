//! A callback that runs when a thread exits, without allocating.
//!
//! The thread caches must hand their contents back when their thread ends. Rust's own
//! `thread_local!` destructors are not usable for that: registering one can allocate, and the
//! allocator is what is asking. The operating system has primitives for this that do not go through
//! the Rust allocator: fiber-local storage with a destructor callback on Windows, a pthread key with
//! a destructor on Unix.
//!
//! One key is shared by the whole process. Each thread stores the cache slot it owns as the key's
//! value; at thread exit the system calls [`crate::tmalloc::on_thread_exit`] with that value.

use std::sync::atomic::{AtomicU32, Ordering};

const NO_KEY: u32 = u32::MAX;

static KEY: AtomicU32 = AtomicU32::new(NO_KEY);

/// Makes the calling thread's exit callback receive `value` (null cancels it). Returns false if no
/// key could be created, in which case the caller must not rely on the callback.
///
/// # Safety
/// `value`, if not null, must stay valid until the callback runs.
#[cfg(windows)]
pub unsafe fn set_exit_value(value: *mut u8) -> bool {
    use std::ffi::c_void;
    use windows_sys::Win32::System::Threading::{FlsAlloc, FlsFree, FlsSetValue};

    unsafe extern "system" fn callback(p: *const c_void) {
        if !p.is_null() {
            crate::tmalloc::on_thread_exit(p as *mut u8);
        }
    }

    let mut key = KEY.load(Ordering::Acquire);
    if key == NO_KEY {
        let new = FlsAlloc(Some(callback));
        if new == NO_KEY {
            return false;
        }
        match KEY.compare_exchange(NO_KEY, new, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => key = new,
            Err(existing) => {
                FlsFree(new);
                key = existing;
            }
        }
    }
    FlsSetValue(key, value as *const c_void) != 0
}

/// # Safety
/// As for the Windows version.
#[cfg(unix)]
pub unsafe fn set_exit_value(value: *mut u8) -> bool {
    use std::ffi::c_void;

    extern "C" fn destructor(p: *mut c_void) {
        if !p.is_null() {
            unsafe { crate::tmalloc::on_thread_exit(p as *mut u8) };
        }
    }

    let mut key = KEY.load(Ordering::Acquire);
    if key == NO_KEY {
        let mut new: libc::pthread_key_t = 0;
        if libc::pthread_key_create(&mut new, Some(destructor)) != 0 {
            return false;
        }
        match KEY.compare_exchange(NO_KEY, new as u32, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => key = new as u32,
            Err(existing) => {
                libc::pthread_key_delete(new);
                key = existing;
            }
        }
    }
    libc::pthread_setspecific(key as libc::pthread_key_t, value as *const c_void) == 0
}
