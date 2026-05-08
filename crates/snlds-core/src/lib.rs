pub mod hmm;

#[cfg(test)]
mod tests {
    use burn::backend::ndarray::NdArrayDevice;
    use burn::backend::Autodiff;
    use burn::backend::NdArray;
    use burn::tensor::Tensor;

    type B = NdArray<f32>;
    type AB = Autodiff<NdArray<f32>>;

    fn cpu() -> NdArrayDevice {
        NdArrayDevice::Cpu
    }

    #[test]
    fn tensor_ops_finite() {
        let dev = cpu();
        let a = Tensor::<B, 1>::from_floats([1.0_f32, 2.0, 3.0, 4.0], &dev);
        let b = Tensor::<B, 1>::from_floats([0.5_f32, 1.5, 2.5, 3.5], &dev);

        let sum = (a.clone() + b.clone()).into_data().to_vec::<f32>().unwrap();
        let product = (a * b).into_data().to_vec::<f32>().unwrap();

        assert!(
            sum.iter().all(|v| v.is_finite()),
            "sum contains non-finite values"
        );
        assert!(
            product.iter().all(|v| v.is_finite()),
            "product contains non-finite values"
        );
        assert_eq!(sum.len(), 4);
    }

    #[test]
    fn matmul_finite() {
        let dev = cpu();
        // 2x3 @ 3x2 → 2x2
        let a = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &dev);
        let b = Tensor::<B, 2>::from_floats([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], &dev);
        let c = a.matmul(b).into_data().to_vec::<f32>().unwrap();

        assert_eq!(c.len(), 4);
        assert!(
            c.iter().all(|v| v.is_finite()),
            "matmul result contains non-finite values"
        );
    }

    #[test]
    fn autograd_smoke() {
        let dev = cpu();
        // y = x^2, dy/dx = 2x; at x=[1,2,3] expect grads=[2,4,6]
        let x = Tensor::<AB, 1>::from_floats([1.0_f32, 2.0, 3.0], &dev).require_grad();
        let y = x.clone().powf_scalar(2.0_f32).sum();
        let grads = y.backward();

        let dx = x
            .grad(&grads)
            .expect("gradient missing for x")
            .into_data()
            .to_vec::<f32>()
            .unwrap();

        assert!(
            dx.iter().all(|v| v.is_finite()),
            "gradients contain non-finite values"
        );
        assert_eq!(dx.len(), 3);
        // dy/dx = 2x
        for (i, (&got, expected)) in dx.iter().zip([2.0_f32, 4.0, 6.0]).enumerate() {
            assert!(
                (got - expected).abs() < 1e-5,
                "grad[{i}]: got {got}, expected {expected}"
            );
        }
    }
}
