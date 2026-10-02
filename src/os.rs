//! Memory from the operating system: whole pages, never smaller, aligned to [`GRANULE`].
//!
//! Everything above this layer asks for address space in multiples of [`PAGE`] and gives it back
//! with the same size. Fresh pages are zero-filled by the system; callers may rely on that for
//! memory that has never been used (the large-allocation path does, to skip clearing for
//! `alloc_zeroed`).

/// Every mapping starts at a multiple of this.
pub const GRANULE: usize = 64 * 1024;
/// The unit of mapping.
pub const PAGE: usize = 4096;

pub const fn round_up_page(n: usize) -> usize {
    (n + PAGE - 1) & !(PAGE - 1)
}

/// Maps `size` bytes (a multiple of [`PAGE`]) of zeroed read-write memory, aligned to [`GRANULE`].
/// Returns null if the system refuses.
///
/// # Safety
/// `size` must be a non-zero multiple of [`PAGE`].
#[cfg(windows)]
pub unsafe fn map(size: usize) -> *mut u8 {
    use windows_sys::Win32::System::Memory::{VirtualAlloc, MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE};
    debug_assert!(size != 0 && size.is_multiple_of(PAGE));
    // The base address of a reservation is always a multiple of the 64 KiB allocation granularity.
    VirtualAlloc(std::ptr::null(), size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) as *mut u8
}

/// Returns a mapping made by [`map`]. `size` must be the size it was mapped with.
///
/// # Safety
/// `p` and `size` are exactly what [`map`] was given and returned, and nothing uses the memory again.
#[cfg(windows)]
pub unsafe fn unmap(p: *mut u8, _size: usize) {
    use windows_sys::Win32::System::Memory::{VirtualFree, MEM_RELEASE};
    // MEM_RELEASE frees the whole reservation and requires a size of zero.
    VirtualFree(p as *mut _, 0, MEM_RELEASE);
}

/// # Safety
/// `size` must be a non-zero multiple of [`PAGE`].
#[cfg(unix)]
pub unsafe fn map(size: usize) -> *mut u8 {
    debug_assert!(size != 0 && size % PAGE == 0);
    // mmap only promises page alignment, so map GRANULE extra and give back the unaligned ends.
    let total = size + GRANULE;
    let p = libc::mmap(
        std::ptr::null_mut(),
        total,
        libc::PROT_READ | libc::PROT_WRITE,
        libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
        -1,
        0,
    );
    if p == libc::MAP_FAILED {
        return std::ptr::null_mut();
    }
    let start = p as usize;
    let aligned = (start + GRANULE - 1) & !(GRANULE - 1);
    if aligned > start {
        libc::munmap(start as *mut _, aligned - start);
    }
    let end = start + total;
    if end > aligned + size {
        libc::munmap((aligned + size) as *mut _, end - (aligned + size));
    }
    aligned as *mut u8
}

/// # Safety
/// `p` and `size` are exactly what [`map`] was given and returned, and nothing uses the memory again.
#[cfg(unix)]
pub unsafe fn unmap(p: *mut u8, size: usize) {
    libc::munmap(p as *mut _, size);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mappings_are_aligned_zeroed_and_usable() {
        unsafe {
            for pages in [1, 2, 16, 100] {
                let size = pages * PAGE;
                let p = map(size);
                assert!(!p.is_null());
                assert_eq!(p as usize % GRANULE, 0);
                assert!(std::slice::from_raw_parts(p, size).iter().all(|&b| b == 0));
                p.write(1);
                p.add(size - 1).write(2);
                unmap(p, size);
            }
        }
    }

    #[test]
    fn an_absurd_request_fails_cleanly() {
        unsafe {
            assert!(map(1 << 60).is_null());
        }
    }

    #[test]
    fn rounding() {
        assert_eq!(round_up_page(1), PAGE);
        assert_eq!(round_up_page(PAGE), PAGE);
        assert_eq!(round_up_page(PAGE + 1), 2 * PAGE);
    }
}
