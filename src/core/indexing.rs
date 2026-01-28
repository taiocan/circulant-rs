// @module: crate::core::indexing
// @status: stable
// @owner: code_expert
// @feature: none
// @depends: []
// @tests: [unit]

//! Circular indexing utilities.

/// Compute the circular (modular) index.
///
/// Maps any integer (including negative) to [0, size).
#[inline]
pub fn circular_index(i: isize, size: usize) -> usize {
    let size = size as isize;
    ((i % size + size) % size) as usize
}

/// Get an element from a slice with circular indexing.
#[inline]
pub fn circular_get<T: Copy>(slice: &[T], i: isize) -> T {
    slice[circular_index(i, slice.len())]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circular_index_positive() {
        assert_eq!(circular_index(0, 4), 0);
        assert_eq!(circular_index(1, 4), 1);
        assert_eq!(circular_index(3, 4), 3);
        assert_eq!(circular_index(4, 4), 0);
        assert_eq!(circular_index(5, 4), 1);
        assert_eq!(circular_index(8, 4), 0);
    }

    #[test]
    fn test_circular_index_negative() {
        assert_eq!(circular_index(-1, 4), 3);
        assert_eq!(circular_index(-2, 4), 2);
        assert_eq!(circular_index(-4, 4), 0);
        assert_eq!(circular_index(-5, 4), 3);
    }

    #[test]
    fn test_circular_get() {
        let arr = [10, 20, 30, 40];
        assert_eq!(circular_get(&arr, 0), 10);
        assert_eq!(circular_get(&arr, 4), 10);
        assert_eq!(circular_get(&arr, -1), 40);
        assert_eq!(circular_get(&arr, -4), 10);
    }
}
