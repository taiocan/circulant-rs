// @module: crate::error
// @status: stable
// @owner: code_expert
// @feature: none
// @depends: [thiserror]
// @tests: [unit]

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

    /// Invalid kernel dimensions.
    #[error("invalid kernel: {0}")]
    InvalidKernel(String),

    /// Image dimension mismatch.
    #[error("image dimensions {got:?} incompatible with filter {expected:?}")]
    ImageDimensionMismatch {
        /// Expected dimensions.
        expected: (usize, usize),
        /// Actual dimensions.
        got: (usize, usize),
    },

    /// Hamiltonian is not Hermitian.
    #[error("Hamiltonian must be Hermitian")]
    NotHermitian,

    /// Time parameter invalid.
    #[error("invalid time parameter: {0}")]
    InvalidTime(String),

    /// Visualization error.
    #[error("visualization error: {0}")]
    VisualizationError(String),

    /// Invalid walk parameters.
    #[error("invalid walk parameters: {0}")]
    InvalidWalkParameters(String),

    /// Invalid tensor shape.
    #[error("invalid tensor shape: expected {expected:?}, got {got:?}")]
    InvalidTensorShape {
        /// Expected shape.
        expected: Vec<usize>,
        /// Actual shape.
        got: Vec<usize>,
    },

    /// Invalid tensor dimension.
    #[error("invalid tensor dimension: expected {expected}, got {got}")]
    InvalidTensorDimension {
        /// Expected dimension count.
        expected: usize,
        /// Actual dimension count.
        got: usize,
    },

    /// Reshape operation failed.
    #[error("reshape failed: {0}")]
    ReshapeFailed(String),
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
        assert_eq!(err.to_string(), "invalid coin dimension: expected 2, got 3");

        let err = CirculantError::PositionOutOfBounds {
            position: 10,
            size: 5,
        };
        assert_eq!(err.to_string(), "position 10 out of bounds for size 5");

        let err = CirculantError::InvalidKernel("kernel must be odd-sized".to_string());
        assert_eq!(err.to_string(), "invalid kernel: kernel must be odd-sized");

        let err = CirculantError::ImageDimensionMismatch {
            expected: (64, 64),
            got: (32, 32),
        };
        assert_eq!(
            err.to_string(),
            "image dimensions (32, 32) incompatible with filter (64, 64)"
        );

        let err = CirculantError::NotHermitian;
        assert_eq!(err.to_string(), "Hamiltonian must be Hermitian");

        let err = CirculantError::InvalidTime("time must be non-negative".to_string());
        assert_eq!(
            err.to_string(),
            "invalid time parameter: time must be non-negative"
        );

        let err = CirculantError::VisualizationError("failed to create plot".to_string());
        assert_eq!(
            err.to_string(),
            "visualization error: failed to create plot"
        );

        let err = CirculantError::InvalidWalkParameters("positions must be positive".to_string());
        assert_eq!(
            err.to_string(),
            "invalid walk parameters: positions must be positive"
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
