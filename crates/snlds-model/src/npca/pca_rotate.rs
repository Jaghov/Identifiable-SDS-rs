//! SVD rotation layer for Neural PCA.

use burn::backend::autodiff::checkpoint::strategy::CheckpointStrategy;
use burn::backend::{Autodiff, LibTorch, NdArray};
use burn::module::{Module, Param};
use burn::prelude::Backend;
use burn::tensor::Tensor;
use linfa_linalg::svd::SVD;

/// Neural PCA training-mode SVD (`V` from `Z`); implemented per concrete backend.
///
/// Uses [`linfa-linalg`] on CPU via `into_data()` round-trip.
pub trait PcaSvdBackend: Backend {
    /// Right singular factor `V` with orthonormal columns, shape `[D, D]`, matching the
    /// [`linfa_linalg`] / economy SVD convention used for `z @ V`.
    fn svd_right_vectors(z: Tensor<Self, 2>) -> Tensor<Self, 2>;
}

impl PcaSvdBackend for NdArray<f32> {
    fn svd_right_vectors(z: Tensor<Self, 2>) -> Tensor<Self, 2> {
        cpu_svd_right_vectors(z)
    }
}

impl PcaSvdBackend for LibTorch<f32> {
    fn svd_right_vectors(z: Tensor<Self, 2>) -> Tensor<Self, 2> {
        cpu_svd_right_vectors(z)
    }
}

impl<B, C> PcaSvdBackend for Autodiff<B, C>
where
    B: PcaSvdBackend,
    C: CheckpointStrategy,
{
    fn svd_right_vectors(z: Tensor<Self, 2>) -> Tensor<Self, 2> {
        let z_inner = z.detach().inner();
        let v = B::svd_right_vectors(z_inner);
        Tensor::from_inner(v)
    }
}

#[derive(Module, Debug)]
pub struct PcaRotate<B: Backend> {
    /// Placeholder identity until [`Self::freeze`]; always persisted in checkpoints.
    pub v_tilde: Param<Tensor<B, 2>>,
    /// Not always restored by [`burn::record::Recorder`]; call [`crate::NeuralPca::sync_training_mode_after_load`] after load.
    pub training: bool,
}

pub struct RotateOutput<B: Backend> {
    pub z_pca: Tensor<B, 2>,
    pub v_matrix: Option<Tensor<B, 2>>,
}

impl<B: Backend> PcaRotate<B> {
    pub fn new(dim: usize, device: &B::Device) -> Self {
        Self {
            v_tilde: Param::from_tensor(Tensor::eye(dim, device)).no_grad(),
            training: true,
        }
    }

    pub fn inverse(&self, z_pca: Tensor<B, 2>) -> Tensor<B, 2> {
        let v = self.v_tilde.val();
        z_pca.matmul(v.transpose())
    }

    pub fn freeze(&mut self, v_tilde: Tensor<B, 2>) {
        self.v_tilde = Param::from_tensor(v_tilde).no_grad();
        self.training = false;
    }

    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

impl<B: Backend + PcaSvdBackend> PcaRotate<B> {
    pub fn forward(&self, z: Tensor<B, 2>) -> RotateOutput<B> {
        if !self.training {
            let v = self.v_tilde.val();
            return RotateOutput {
                z_pca: z.matmul(v),
                v_matrix: None,
            };
        }

        let v = B::svd_right_vectors(z.clone());
        let v_detached = v.clone().detach();
        RotateOutput {
            z_pca: z.matmul(v_detached),
            v_matrix: Some(v),
        }
    }
}

fn cpu_svd_right_vectors<B: Backend>(z: Tensor<B, 2>) -> Tensor<B, 2> {
    let [rows, cols] = z.dims();
    let device = z.device();

    let data: Vec<f32> = z.into_data().to_vec().unwrap();
    let v = ndarray_svd_full_v_from_rows_cols(rows, cols, &data);
    ndarray_to_tensor(&v, &device)
}

/// Full **[D,D]** right orthogonal factor **`V`** with **`Z ≈ U Σ Vᵀ`** (`Z` is `[B, D]`, **`B < D`**).
///
/// Avoids an **O(D³)** SVD of **`ZᵀZ`**: uses **`Z Zᵀ` [B×B]**, then completes an orthonormal basis.
fn ndarray_svd_full_v_from_rows_cols(
    rows: usize,
    cols: usize,
    data: &[f32],
) -> ndarray::Array2<f32> {
    let mat = ndarray::Array2::<f32>::from_shape_vec((rows, cols), data.to_vec())
        .expect("shape mismatch");
    ndarray_svd_full_v(&mat)
}

fn ndarray_svd_full_v(mat: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
    let (rows, cols) = mat.dim();
    if rows >= cols {
        let (_u, _sigma, vt) = mat.svd(false, true).expect("SVD failed");
        let vt = vt.expect("SVD did not return Vt");
        return vt.t().to_owned();
    }

    wide_svd_full_v(mat)
}

/// **`Z`** shape **[B, D]**, **`B < D`**. Singular vectors from **`Z Zᵀ` [B,B]**; Gram–Schmidt completes **`V ∈ ℝ^{D×D}`**.
fn wide_svd_full_v(z: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
    let (b, d) = z.dim();
    debug_assert!(b < d);
    if b == 0 {
        return ndarray::Array2::from_diag(&ndarray::Array1::<f32>::ones(d));
    }

    let zzt = z.dot(&z.t());
    let (u_bb, sigma_z_sq, _vt) = zzt.svd(true, true).expect("svd(Z Z^T)");
    let u = u_bb.expect("U for Z Z^T");
    let z_t = z.t();
    let mut v_lead: ndarray::Array2<f32> = z_t.dot(&u);
    for j in 0..b {
        let sig = sigma_z_sq[j].max(1e-20_f32).sqrt();
        for i in 0..d {
            v_lead[[i, j]] /= sig;
        }
    }

    let mut basis: Vec<ndarray::Array1<f32>> =
        (0..b).map(|j| v_lead.column(j).to_owned()).collect();
    orthonormalize_basis(&mut basis);

    for k in 0..d {
        if basis.len() >= d {
            break;
        }
        let mut r = ndarray::Array1::<f32>::zeros(d);
        r[k] = 1.0;
        for col in &basis {
            let c = r.dot(col);
            r -= &(col * c);
        }
        let n = r.dot(&r).sqrt();
        if n > 1e-5_f32 {
            r /= n;
            basis.push(r);
        }
    }

    let mut attempt = 0u32;
    while basis.len() < d {
        let mut r = ndarray::Array1::<f32>::zeros(d);
        let seed = basis.len() + attempt as usize;
        for i in 0..d {
            let u = (i as u32)
                .wrapping_mul(1103515245)
                .wrapping_add(seed as u32)
                .wrapping_add(attempt.wrapping_mul(7919));
            r[i] = ((u & 0x7fff) as f32) / 16384.0 - 1.0;
        }
        attempt = attempt.wrapping_add(1);
        for col in &basis {
            let c = r.dot(col);
            r -= &(col * c);
        }
        let n = r.dot(&r).sqrt();
        if n > 1e-8_f32 {
            r /= n;
            basis.push(r);
        }
        if attempt > 1_000_000 {
            panic!(
                "wide_svd_full_v: could not complete {d}x{d} basis (got {})",
                basis.len()
            );
        }
    }

    let mut out = ndarray::Array2::<f32>::zeros((d, d));
    for (j, col) in basis.iter().enumerate().take(d) {
        out.column_mut(j).assign(col);
    }
    out
}

fn orthonormalize_basis(basis: &mut [ndarray::Array1<f32>]) {
    for i in 0..basis.len() {
        for j in 0..i {
            let d = basis[i].dot(&basis[j]);
            let bj = basis[j].clone();
            basis[i] -= &(bj * d);
        }
        let n = basis[i].dot(&basis[i]).sqrt();
        if n > 1e-10_f32 {
            basis[i] /= n;
        }
    }
}

fn ndarray_to_tensor<B: Backend>(arr: &ndarray::Array2<f32>, device: &B::Device) -> Tensor<B, 2> {
    let (r, c) = arr.dim();
    let flat: Vec<f32> = arr.as_standard_layout().as_slice().unwrap().to_vec();
    Tensor::<B, 1>::from_floats(flat.as_slice(), device).reshape([r, c])
}

/// Determinant via Gaussian elimination with partial pivoting (sign + magnitude).
fn matrix_det_f32(a: &ndarray::Array2<f32>) -> f32 {
    let n = a.nrows();
    debug_assert_eq!(n, a.ncols());
    let mut m = a.to_owned();
    let mut det_sign = 1.0f32;
    let mut det_prod = 1.0f32;

    for k in 0..n {
        let mut pivot_row = k;
        let mut best = m[[k, k]].abs();
        for i in (k + 1)..n {
            let v = m[[i, k]].abs();
            if v > best {
                best = v;
                pivot_row = i;
            }
        }
        if best < 1e-20 {
            return 0.0;
        }
        if pivot_row != k {
            swap_rows_f32(&mut m, k, pivot_row);
            det_sign = -det_sign;
        }
        let pivot_val = m[[k, k]];
        det_prod *= pivot_val;
        let inv_pivot = 1.0 / pivot_val;
        for i in (k + 1)..n {
            let f = m[[i, k]] * inv_pivot;
            if f != 0.0 {
                for j in (k + 1)..n {
                    m[[i, j]] -= f * m[[k, j]];
                }
            }
        }
    }

    det_sign * det_prod
}

fn swap_rows_f32(m: &mut ndarray::Array2<f32>, i: usize, j: usize) {
    if i == j {
        return;
    }
    let n = m.ncols();
    for c in 0..n {
        let t = m[[i, c]];
        m[[i, c]] = m[[j, c]];
        m[[j, c]] = t;
    }
}

pub fn project_mean_rotation<B: Backend>(vs: Tensor<B, 3>, device: &B::Device) -> Tensor<B, 2> {
    let [n, d, d2] = vs.dims();
    assert_ne!(n, 0, "V must not be empty");
    assert_eq!(d, d2, "V must be square");

    let mean_rotation = vs.mean_dim(0).squeeze::<2>().into_data().to_vec().unwrap();
    let v_bar = ndarray::Array2::<f32>::from_shape_vec((d, d), mean_rotation).unwrap();
    let (u, _sigma, vt) = v_bar.svd(true, true).expect("SVD of V_bar failed");
    let mut u = u.expect("no U");
    let vt = vt.expect("no Vt");
    let mut proj = u.dot(&vt);
    if matrix_det_f32(&proj) < 0.0 {
        for r in 0..d {
            u[[r, d - 1]] = -u[[r, d - 1]];
        }
        proj = u.dot(&vt);
    }
    ndarray_to_tensor(&proj, device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    type B = NdArray<f32>;

    #[test]
    fn svd_v_is_orthogonal() {
        let device = Default::default();
        let z = Tensor::<B, 2>::random([16, 4], Distribution::Normal(0.0, 1.0), &device);
        let v = cpu_svd_right_vectors::<B>(z);
        let [d, d2] = v.dims();
        assert_eq!(d, d2);
        assert_eq!(d, 4);
        let vvt = v.clone().matmul(v.transpose());
        let eye = Tensor::<B, 2>::eye(d, &device);
        let max_err: f32 = (vvt - eye).abs().max().into_scalar();
        assert!(max_err < 1e-4, "V should be orthogonal; max_err={max_err}");
    }

    #[test]
    fn svd_v_is_orthogonal_when_batch_lt_dim() {
        let device = Default::default();
        let z = Tensor::<B, 2>::random([4, 16], Distribution::Normal(0.0, 1.0), &device);
        let v = cpu_svd_right_vectors::<B>(z);
        let [d, d2] = v.dims();
        assert_eq!(d, 16);
        assert_eq!(d2, 16);
        let vvt = v.clone().matmul(v.transpose());
        let eye = Tensor::<B, 2>::eye(d, &device);
        let max_err: f32 = (vvt - eye).abs().max().into_scalar();
        assert!(
            max_err < 1e-4,
            "V should be orthogonal even when B < D; max_err={max_err}"
        );
    }

    #[test]
    fn rotate_v_matrix_matches_cpu_svd() {
        let device = Default::default();
        for (b, d) in [(16usize, 4usize), (4usize, 16usize)] {
            let rot = PcaRotate::<B>::new(d, &device);
            let z = Tensor::<B, 2>::random([b, d], Distribution::Normal(0.0, 1.0), &device);
            let out = rot.forward(z.clone());
            let v_exp = cpu_svd_right_vectors(z);
            let v_act = out.v_matrix.expect("training mode yields V");
            let max_err: f32 = (v_act - v_exp).abs().max().into_scalar();
            assert!(
                max_err < 1e-5,
                "V from forward should match cpu_svd_right_vectors; B={b} D={d} max_err={max_err}"
            );
        }
    }

    #[test]
    fn project_mean_rotation_is_orthogonal() {
        let device = Default::default();
        let mut vs = Vec::new();
        for _ in 0..5 {
            let z = Tensor::<B, 2>::random([16, 4], Distribution::Normal(0.0, 1.0), &device);
            vs.push(cpu_svd_right_vectors::<B>(z));
        }
        let vs = Tensor::stack(vs, 0);
        let v_tilde = project_mean_rotation::<B>(vs, &device);
        let vvt = v_tilde.clone().matmul(v_tilde.clone().transpose());
        let eye = Tensor::<B, 2>::eye(4, &device);
        let max_err: f32 = (vvt - eye).abs().max().into_scalar();
        assert!(
            max_err < 1e-4,
            "V_tilde should be orthogonal; max_err={max_err}"
        );
        let data: Vec<f32> = v_tilde.into_data().to_vec().unwrap();
        let m = ndarray::Array2::<f32>::from_shape_vec((4, 4), data).unwrap();
        let det = matrix_det_f32(&m);
        assert!(
            det > 0.0 && (det - 1.0).abs() < 1e-3,
            "V_tilde should be SO(4), det={det}"
        );
    }

    #[test]
    fn project_mean_rotation_flips_improper_mean_to_so2() {
        let device = Default::default();
        // Mean of these orthogonals is a reflection (det −1); Procrustes U Vᵀ can be improper;
        // we flip the last column of U so the result lies in SO(2).
        let f = Tensor::<B, 2>::from_floats([[1.0f32, 0.0], [0.0, -1.0]], &device);
        let vs = Tensor::stack(vec![f.clone(), f.clone(), f.clone()], 0);
        let v_tilde = project_mean_rotation::<B>(vs, &device);
        let data: Vec<f32> = v_tilde.into_data().to_vec().unwrap();
        let m = ndarray::Array2::<f32>::from_shape_vec((2, 2), data).unwrap();
        let det = matrix_det_f32(&m);
        assert!((det - 1.0).abs() < 1e-4, "expected SO(2), det={det}");
        let eye = ndarray::Array2::<f32>::eye(2);
        assert!((m.dot(&m.t()) - &eye).mapv(f32::abs).sum() < 1e-3);
    }
}
