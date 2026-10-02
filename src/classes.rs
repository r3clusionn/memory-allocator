//! Size classes for small allocations.
//!
//! Sizes up to 128 bytes step by 16. Above that there are four classes per doubling, so a request
//! is rounded up by at most 25 percent: 160, 192, 224, 256, 320, 384, 448, 512, 640, ... 8192.
//! Every power of two from 16 to 8192 is a class, which the aligned-allocation path relies on.

/// The largest size served from a class; anything bigger goes to the heap.
pub const SMALL_MAX: usize = 8192;
pub const NUM_CLASSES: usize = 32;

const fn build() -> [usize; NUM_CLASSES] {
    let mut t = [0; NUM_CLASSES];
    let mut i = 0;
    while i < 8 {
        t[i] = 16 * (i + 1);
        i += 1;
    }
    let mut base = 128;
    while base < SMALL_MAX {
        let mut j = 1;
        while j <= 4 {
            t[i] = base + base / 4 * j;
            i += 1;
            j += 1;
        }
        base *= 2;
    }
    t
}

/// The size of each class in bytes.
pub const SIZES: [usize; NUM_CLASSES] = build();

/// The index of the smallest class that holds `size` bytes. `size` must be in `1..=SMALL_MAX`.
#[inline(always)]
pub fn class_of(size: usize) -> usize {
    debug_assert!((1..=SMALL_MAX).contains(&size));
    if size <= 128 {
        size.div_ceil(16) - 1
    } else {
        // size - 1 lies in [base, 2 * base) where base is a power of two of at least 128.
        let k = (usize::BITS - 1 - (size - 1).leading_zeros()) as usize;
        let step_shift = k - 2;
        8 + (k - 7) * 4 + ((size - 1 - (1 << k)) >> step_shift)
    }
}

/// How many objects move between a thread cache and the shared pool at once.
#[inline(always)]
pub fn batch(class_size: usize) -> usize {
    (32 * 1024 / class_size).clamp(2, 64)
}

/// Bytes mapped at a time to carve objects of one class from.
#[inline(always)]
pub fn slab_bytes(class_size: usize) -> usize {
    let want = (class_size * 32).max(64 * 1024);
    (want + 65535) & !65535
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_is_what_the_comment_says() {
        assert_eq!(&SIZES[..12], &[16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256]);
        assert_eq!(&SIZES[28..], &[5120, 6144, 7168, 8192]);
        assert!(SIZES.windows(2).all(|w| w[0] < w[1]));
        assert!(SIZES.iter().all(|s| s % 16 == 0));
        // Every power of two from 16 to the maximum is a class.
        let mut p = 16;
        while p <= SMALL_MAX {
            assert!(SIZES.contains(&p), "{p}");
            p *= 2;
        }
    }

    #[test]
    fn every_size_maps_to_the_smallest_class_that_holds_it() {
        for size in 1..=SMALL_MAX {
            let c = class_of(size);
            assert!(SIZES[c] >= size, "size {size} class {c}");
            assert!(c == 0 || SIZES[c - 1] < size, "size {size} could use a smaller class than {c}");
            // Waste is bounded: under 25 percent beyond the 16-byte steps.
            assert!(SIZES[c] - size < 16 || (SIZES[c] - size) * 4 <= SIZES[c], "size {size}");
        }
    }

    #[test]
    fn batch_and_slab_sizes_are_sane() {
        for &s in &SIZES {
            assert!((2..=64).contains(&batch(s)));
            let b = slab_bytes(s);
            assert!(b.is_multiple_of(65536) && b >= 32 * s);
        }
        assert_eq!(batch(16), 64);
        assert_eq!(batch(8192), 4);
        // The clamp matters for sizes larger than any class.
        assert_eq!(batch(65536), 2);
        assert_eq!(batch(1 << 20), 2);
        assert_eq!(slab_bytes(16), 64 * 1024);
        assert_eq!(slab_bytes(8192), 256 * 1024);
    }
}
