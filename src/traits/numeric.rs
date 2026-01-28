// @module: crate::traits::numeric
// @status: stable
// @owner: math_expert
// @feature: none
// @depends: [num_complex, num_traits]
// @tests: [unit]

//! Numeric trait bounds for scalar and complex types.

use num_complex::Complex;
use num_traits::{Float, FloatConst, NumAssign, One, Zero};

/// A real scalar type suitable for circulant matrix operations.
///
/// This trait is implemented for `f32` and `f64`.
pub trait Scalar: Float + FloatConst + NumAssign + Send + Sync + Copy + Default + 'static {
    /// The name of this scalar type (for debugging).
    fn type_name() -> &'static str;
}

impl Scalar for f32 {
    fn type_name() -> &'static str {
        "f32"
    }
}

impl Scalar for f64 {
    fn type_name() -> &'static str {
        "f64"
    }
}

/// A complex scalar type suitable for quantum and FFT operations.
///
/// This trait is implemented for `Complex<f32>` and `Complex<f64>`.
pub trait ComplexScalar<T: Scalar>:
    Clone
    + Copy
    + Send
    + Sync
    + Zero
    + One
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::AddAssign
    + std::ops::SubAssign
    + std::ops::MulAssign
    + std::ops::DivAssign
    + 'static
{
    /// Create a complex number from real and imaginary parts.
    fn from_parts(re: T, im: T) -> Self;

    /// Get the real part.
    fn re(&self) -> T;

    /// Get the imaginary part.
    fn im(&self) -> T;

    /// Create a complex number from a real value.
    fn from_real(re: T) -> Self;

    /// Compute the complex conjugate.
    fn conj(&self) -> Self;

    /// Compute the squared magnitude (norm squared).
    fn norm_sqr(&self) -> T;

    /// Compute the magnitude (absolute value).
    fn norm(&self) -> T;

    /// Compute the unit complex number for angle theta: e^(i*theta).
    fn from_polar(r: T, theta: T) -> Self;
}

impl<T: Scalar> ComplexScalar<T> for Complex<T> {
    #[inline]
    fn from_parts(re: T, im: T) -> Self {
        Complex::new(re, im)
    }

    #[inline]
    fn re(&self) -> T {
        self.re
    }

    #[inline]
    fn im(&self) -> T {
        self.im
    }

    #[inline]
    fn from_real(re: T) -> Self {
        Complex::new(re, T::zero())
    }

    #[inline]
    fn conj(&self) -> Self {
        Complex::new(self.re, -self.im)
    }

    #[inline]
    fn norm_sqr(&self) -> T {
        self.re * self.re + self.im * self.im
    }

    #[inline]
    fn norm(&self) -> T {
        self.norm_sqr().sqrt()
    }

    #[inline]
    fn from_polar(r: T, theta: T) -> Self {
        Complex::new(r * theta.cos(), r * theta.sin())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_scalar_trait_bounds_f32() {
        fn requires_scalar<T: Scalar>(_: T) {}
        requires_scalar(1.0f32);
    }

    #[test]
    fn test_scalar_trait_bounds_f64() {
        fn requires_scalar<T: Scalar>(_: T) {}
        requires_scalar(1.0f64);
    }

    #[test]
    fn test_scalar_type_name() {
        assert_eq!(f32::type_name(), "f32");
        assert_eq!(f64::type_name(), "f64");
    }

    #[test]
    fn test_complex_scalar_from_parts() {
        let c: Complex<f64> = ComplexScalar::from_parts(3.0, 4.0);
        assert_eq!(c.re(), 3.0);
        assert_eq!(c.im(), 4.0);
    }

    #[test]
    fn test_complex_scalar_from_real() {
        let c: Complex<f64> = ComplexScalar::from_real(5.0);
        assert_eq!(c.re(), 5.0);
        assert_eq!(c.im(), 0.0);
    }

    #[test]
    fn test_complex_scalar_conjugate() {
        let c: Complex<f64> = ComplexScalar::from_parts(3.0, 4.0);
        let conj = c.conj();
        assert_eq!(conj.re(), 3.0);
        assert_eq!(conj.im(), -4.0);
    }

    #[test]
    fn test_complex_scalar_norm() {
        let c: Complex<f64> = ComplexScalar::from_parts(3.0, 4.0);
        assert_relative_eq!(c.norm_sqr(), 25.0);
        assert_relative_eq!(c.norm(), 5.0);
    }

    #[test]
    fn test_complex_scalar_operations() {
        let a: Complex<f64> = ComplexScalar::from_parts(1.0, 2.0);
        let b: Complex<f64> = ComplexScalar::from_parts(3.0, 4.0);

        let sum = a + b;
        assert_eq!(sum.re(), 4.0);
        assert_eq!(sum.im(), 6.0);

        let diff = b - a;
        assert_eq!(diff.re(), 2.0);
        assert_eq!(diff.im(), 2.0);

        let prod = a * b;
        // (1+2i)(3+4i) = 3 + 4i + 6i + 8i^2 = 3 + 10i - 8 = -5 + 10i
        assert_eq!(prod.re(), -5.0);
        assert_eq!(prod.im(), 10.0);
    }

    #[test]
    fn test_complex_scalar_from_polar() {
        use std::f64::consts::PI;

        // e^(i*0) = 1
        let c: Complex<f64> = ComplexScalar::from_polar(1.0, 0.0);
        assert_relative_eq!(c.re(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(c.im(), 0.0, epsilon = 1e-10);

        // e^(i*pi/2) = i
        let c: Complex<f64> = ComplexScalar::from_polar(1.0, PI / 2.0);
        assert_relative_eq!(c.re(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(c.im(), 1.0, epsilon = 1e-10);

        // e^(i*pi) = -1
        let c: Complex<f64> = ComplexScalar::from_polar(1.0, PI);
        assert_relative_eq!(c.re(), -1.0, epsilon = 1e-10);
        assert_relative_eq!(c.im(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_complex_scalar_zero_one() {
        let zero: Complex<f64> = Zero::zero();
        let one: Complex<f64> = One::one();

        assert_eq!(zero.re(), 0.0);
        assert_eq!(zero.im(), 0.0);
        assert_eq!(one.re(), 1.0);
        assert_eq!(one.im(), 0.0);
    }
}
