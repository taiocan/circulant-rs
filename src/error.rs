//! Error types for circulant-rs operations.

use thiserror::Error;

/// Errors that can occur during circulant matrix operations.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum CirculantError {
    /// The generator vector is empty.
    #[error("generator vector cannot be empty")]
    EmptyGenerator,

    /// Dimension mismatch between matrix and vector.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        got: usize,
    },

    /// Invalid size for FFT (must be positive).
    #[error("invalid FFT size: {0}")]
    InvalidFftSize(usize),

    /// Block structure is invalid.
    #[error("invalid block structure: {0}")]
    InvalidBlockStructure(String),

    /// Quantum state is not normalized.
    #[error("quantum state not normalized: norm squared = {0}")]
    NotNormalized(String),

    /// Invalid coin dimension.
    #[error("invalid coin dimension: expected {expected}, got {got}")]
    InvalidCoinDimension {
        /// Expected coin dimension.
        expected: usize,
        /// Actual coin dimension.
        got: usize,
    },

    /// Position out of bounds.
    #[error("position {position} out of bounds for size {size}")]
    PositionOutOfBounds {
        /// The invalid position.
        position: usize,
        /// The valid size.
        size: usize,
    },
}

/// A specialized Result type for circulant operations.
pub type Result<T> = std::result::Result<T, CirculantError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display_messages() {
        let err = CirculantError::EmptyGenerator;
        assert_eq!(err.to_string(), "generator vector cannot be empty");

        let err = CirculantError::DimensionMismatch {
            expected: 4,
            got: 3,
        };
        assert_eq!(err.to_string(), "dimension mismatch: expected 4, got 3");

        let err = CirculantError::InvalidFftSize(0);
        assert_eq!(err.to_string(), "invalid FFT size: 0");

        let err = CirculantError::InvalidBlockStructure("non-square blocks".to_string());
        assert_eq!(
            err.to_string(),
            "invalid block structure: non-square blocks"
        );

        let err = CirculantError::NotNormalized("1.5".to_string());
        assert_eq!(
            err.to_string(),
            "quantum state not normalized: norm squared = 1.5"
        );

        let err = CirculantError::InvalidCoinDimension {
            expected: 2,
            got: 3,
        };
        assert_eq!(
            err.to_string(),
            "invalid coin dimension: expected 2, got 3"
        );

        let err = CirculantError::PositionOutOfBounds {
            position: 10,
            size: 5,
        };
        assert_eq!(
            err.to_string(),
            "position 10 out of bounds for size 5"
        );
    }

    #[test]
    fn test_error_equality() {
        let err1 = CirculantError::EmptyGenerator;
        let err2 = CirculantError::EmptyGenerator;
        assert_eq!(err1, err2);

        let err3 = CirculantError::DimensionMismatch {
            expected: 4,
            got: 3,
        };
        let err4 = CirculantError::DimensionMismatch {
            expected: 4,
            got: 3,
        };
        assert_eq!(err3, err4);
    }
}
