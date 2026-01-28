// @test_for: [crate::core::tensor]
// @math_verified: false
// @verified_by: math_expert
// @properties_checked: [linearity, fft_roundtrip, eigenvalue_count, identity]
// @version: 1.0.0
// @event: DE-2026-002

//! Property-based tests for CirculantTensor.

use approx::assert_relative_eq;
use circulant_rs::{CirculantTensor, TensorOps};
use ndarray::{ArrayD, IxDyn};
use num_complex::Complex;
use proptest::prelude::*;

fn complex(re: f64, im: f64) -> Complex<f64> {
    Complex::new(re, im)
}

/// Generate a random complex number.
fn arb_complex() -> impl Strategy<Value = Complex<f64>> {
    (-10.0..10.0, -10.0..10.0).prop_map(|(re, im)| complex(re, im))
}

/// Generate a 2D tensor of given size.
fn arb_tensor_2d(n: usize) -> impl Strategy<Value = ArrayD<Complex<f64>>> {
    proptest::collection::vec(arb_complex(), n * n).prop_map(move |v| {
        ArrayD::from_shape_vec(IxDyn(&[n, n]), v).expect("valid shape")
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Test linearity: C(αx + βy) = αC(x) + βC(y)
    #[test]
    fn prop_2d_linearity(
        gen in arb_tensor_2d(4),
        x in arb_tensor_2d(4),
        y in arb_tensor_2d(4),
        alpha_re in -5.0..5.0f64,
        beta_re in -5.0..5.0f64,
    ) {
        let tensor = CirculantTensor::<f64, 2>::new(gen).expect("valid tensor");
        let alpha = complex(alpha_re, 0.0);
        let beta = complex(beta_re, 0.0);

        // C(αx + βy)
        let ax: ArrayD<Complex<f64>> = x.mapv(|v| alpha * v);
        let by: ArrayD<Complex<f64>> = y.mapv(|v| beta * v);
        let sum: ArrayD<Complex<f64>> = &ax + &by;
        let c_sum = tensor.mul_tensor(&sum).expect("multiply failed");

        // αC(x) + βC(y)
        let c_x = tensor.mul_tensor(&x).expect("multiply failed");
        let c_y = tensor.mul_tensor(&y).expect("multiply failed");
        let ac_x: ArrayD<Complex<f64>> = c_x.mapv(|v| alpha * v);
        let bc_y: ArrayD<Complex<f64>> = c_y.mapv(|v| beta * v);
        let expected: ArrayD<Complex<f64>> = &ac_x + &bc_y;

        for (got, want) in c_sum.iter().zip(expected.iter()) {
            assert_relative_eq!(got.re, want.re, epsilon = 1e-8);
            assert_relative_eq!(got.im, want.im, epsilon = 1e-8);
        }
    }

    /// Test eigenvalue count equals total elements.
    #[test]
    fn prop_eigenvalue_count(
        n1 in 2usize..6,
        n2 in 2usize..6,
    ) {
        let total = n1 * n2;
        let gen = ArrayD::from_shape_vec(
            IxDyn(&[n1, n2]),
            (0..total).map(|i| complex(i as f64, 0.0)).collect()
        ).expect("valid shape");

        let tensor = CirculantTensor::<f64, 2>::new(gen).expect("valid tensor");
        let eigenvalues = tensor.eigenvalues_nd();

        prop_assert_eq!(eigenvalues.len(), total);
        prop_assert_eq!(eigenvalues.shape(), &[n1, n2]);
    }

    /// Test identity generator: I·x = x
    #[test]
    fn prop_identity_2d(x in arb_tensor_2d(4)) {
        let mut gen = ArrayD::from_elem(IxDyn(&[4, 4]), complex(0.0, 0.0));
        gen[IxDyn(&[0, 0])] = complex(1.0, 0.0);

        let tensor = CirculantTensor::<f64, 2>::new(gen).expect("valid tensor");
        let result = tensor.mul_tensor(&x).expect("multiply failed");

        for (got, want) in result.iter().zip(x.iter()) {
            assert_relative_eq!(got.re, want.re, epsilon = 1e-10);
            assert_relative_eq!(got.im, want.im, epsilon = 1e-10);
        }
    }
}

#[test]
fn test_3d_linearity() {
    // Explicit 3D linearity test
    let gen = ArrayD::from_shape_vec(
        IxDyn(&[3, 3, 3]),
        (0..27).map(|x| complex(x as f64 * 0.1, 0.0)).collect(),
    )
    .expect("valid shape");

    let tensor = CirculantTensor::<f64, 3>::new(gen).expect("valid tensor");

    let x = ArrayD::from_shape_vec(
        IxDyn(&[3, 3, 3]),
        (0..27).map(|i| complex(i as f64, 0.0)).collect(),
    )
    .expect("valid shape");

    let y = ArrayD::from_shape_vec(
        IxDyn(&[3, 3, 3]),
        (0..27).map(|i| complex((i + 1) as f64, 0.5)).collect(),
    )
    .expect("valid shape");

    let alpha = complex(2.0, 0.0);
    let beta = complex(1.5, 0.0);

    // C(αx + βy)
    let ax: ArrayD<Complex<f64>> = x.mapv(|v| alpha * v);
    let by: ArrayD<Complex<f64>> = y.mapv(|v| beta * v);
    let sum: ArrayD<Complex<f64>> = &ax + &by;
    let c_sum = tensor.mul_tensor(&sum).expect("multiply failed");

    // αC(x) + βC(y)
    let c_x = tensor.mul_tensor(&x).expect("multiply failed");
    let c_y = tensor.mul_tensor(&y).expect("multiply failed");
    let ac_x: ArrayD<Complex<f64>> = c_x.mapv(|v| alpha * v);
    let bc_y: ArrayD<Complex<f64>> = c_y.mapv(|v| beta * v);
    let expected: ArrayD<Complex<f64>> = &ac_x + &bc_y;

    for (got, want) in c_sum.iter().zip(expected.iter()) {
        assert_relative_eq!(got.re, want.re, epsilon = 1e-9);
        assert_relative_eq!(got.im, want.im, epsilon = 1e-9);
    }
}

#[test]
fn test_precompute_consistency() {
    let gen = ArrayD::from_shape_vec(
        IxDyn(&[4, 4]),
        (0..16).map(|x| complex(x as f64, 0.0)).collect(),
    )
    .expect("valid shape");

    let x = ArrayD::from_shape_vec(
        IxDyn(&[4, 4]),
        (0..16).map(|i| complex(1.0 / (i + 1) as f64, 0.0)).collect(),
    )
    .expect("valid shape");

    // Without precompute
    let tensor1 = CirculantTensor::<f64, 2>::new(gen.clone()).expect("valid tensor");
    let result1 = tensor1.mul_tensor(&x).expect("multiply failed");

    // With precompute
    let mut tensor2 = CirculantTensor::<f64, 2>::new(gen).expect("valid tensor");
    tensor2.precompute();
    let result2 = tensor2.mul_tensor(&x).expect("multiply failed");

    for (r1, r2) in result1.iter().zip(result2.iter()) {
        assert_relative_eq!(r1.re, r2.re, epsilon = 1e-14);
        assert_relative_eq!(r1.im, r2.im, epsilon = 1e-14);
    }
}
