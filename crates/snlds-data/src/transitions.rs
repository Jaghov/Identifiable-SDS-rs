//! Port of [identifiable-SDS/utils/transitions.py](../identifiable-SDS/utils/transitions.py).
#![allow(clippy::needless_range_loop)]

use crate::polynomial::{expand_polynomial_row, sklearn_powers};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2};
use rand::Rng;

/// Hidden width for leaky-ReLU emission (`params_leaky`) and cosine feature maps (matches Python `generate_data`).
pub const EMISSION_HIDDEN_DIM: usize = 8;

/// Cyclic Markov transition with 0.9 self and 0.1 to the next state.
pub fn get_trans_mat(size: usize) -> Array2<f32> {
    let mut q = Array2::<f32>::zeros((size, size));
    for i in 0..size.saturating_sub(1) {
        q[[i, i]] = 0.9;
        q[[i, i + 1]] = 0.1;
    }
    if size > 0 {
        q[[size - 1, size - 1]] = 0.9;
        q[[size - 1, 0]] = 0.1;
    }
    q
}

/// Bernoulli edges on off-diagonal; diagonal forced to 1.
pub fn sample_adj_mat<R: Rng + ?Sized>(rng: &mut R, sparsity_prob: f32, dim: usize) -> Array2<f32> {
    let mut m = Array2::<f32>::zeros((dim, dim));
    let p_edge = (1.0 - sparsity_prob) as f64;
    for i in 0..dim {
        for j in 0..dim {
            if i != j {
                m[[i, j]] = if rng.random_bool(p_edge) { 1.0 } else { 0.0 };
            }
        }
        m[[i, i]] = 1.0;
    }
    m
}

/// `alphas`: `[dim_obs, H]`, `omegas`: `[H, dim_latent]`, `betas`: `[H]` with `H = EMISSION_HIDDEN_DIM`
/// (matches `params_leaky` in Python `generate_data`).
#[derive(Clone, Debug)]
pub struct LeakyParams {
    pub alphas: Array2<f32>,
    pub omegas: Array2<f32>,
    pub betas: Array1<f32>,
}

/// Batched leaky-ReLU emission: **`z`** is **`[N, dim_latent]`**, output **`[N, dim_obs]`**
/// (**vectorized**, matches Python **`func_leaky_relu(latents[:, t, :], params_leaky)`**).
pub fn func_leaky_relu_batch(z: ArrayView2<f32>, params: &LeakyParams) -> Array2<f32> {
    let mut h = z.dot(&params.omegas.t());
    for mut row in h.rows_mut() {
        row += &params.betas;
    }
    let h_relu = h.mapv(|v| v.max(0.2 * v));
    h_relu.dot(&params.alphas.t())
}

/// Cosine dynamics for one discrete state (`params[k]` tuple in Python).
#[derive(Clone, Debug)]
pub struct CosineStateParams {
    /// `(1, dim_latent, H)`
    pub alphas: Array3<f32>,
    /// `(H, dim_latent, dim_latent)`
    pub omegas: Array3<f32>,
    /// `(dim_latent, H)`
    pub betas: Array2<f32>,
    pub adj: Array2<f32>,
}

pub fn func_cosine_with_sparsity(x: ArrayView1<f32>, feat: &CosineStateParams) -> Array1<f32> {
    let dim_lat = x.len();
    let mut out = Array1::<f32>::zeros(dim_lat);
    for i in 0..dim_lat {
        let mut inp = Array1::<f32>::zeros(dim_lat);
        for j in 0..dim_lat {
            inp[j] = x[j] * feat.adj[[i, j]];
        }
        let mut hcos = [0f32; EMISSION_HIDDEN_DIM];
        for k in 0..EMISSION_HIDDEN_DIM {
            let mut pre = 0f32;
            for j in 0..dim_lat {
                pre += feat.omegas[[k, i, j]] * inp[j];
            }
            hcos[k] = (pre + feat.betas[[i, k]]).cos();
        }
        let mut s = 0f32;
        for k in 0..EMISSION_HIDDEN_DIM {
            s += feat.alphas[[0, i, k]] * hcos[k];
        }
        out[i] = s;
    }
    out
}

#[derive(Clone, Debug)]
pub struct PolynomialStateParams {
    /// `[num_states, dim_latent, num_params]`
    pub coeffs: Array3<f32>,
    powers: Vec<Vec<u32>>,
}

impl PolynomialStateParams {
    pub fn new(coeffs: Array3<f32>, dim_latent: usize, degree: usize) -> Self {
        let powers = sklearn_powers(dim_latent, degree);
        debug_assert_eq!(coeffs.dim().2, powers.len());
        Self { coeffs, powers }
    }

    /// Polynomial transition mean for one latent vector and discrete state (no batch dimension).
    pub fn poly_mean_for_state(&self, z: ArrayView1<f32>, state: usize) -> Array1<f32> {
        let dim_latent = z.len();
        let mut mean = Array1::<f32>::zeros(dim_latent);
        let x: Vec<f32> = z.iter().copied().collect();
        let obs_one = expand_polynomial_row(&x, &self.powers);
        for d in 0..dim_latent {
            let mut acc = 0f32;
            for p in 0..obs_one.len() {
                acc += self.coeffs[[state, d, p]] * obs_one[p];
            }
            mean[d] = acc;
        }
        mean
    }

    pub fn poly_means_rows(
        &self,
        z_prev: ArrayView2<f32>,
        state_idx: ArrayView1<usize>,
    ) -> ndarray::Array2<f32> {
        let (_, dim_latent) = z_prev.dim();
        let n = z_prev.nrows();
        let mut means = ndarray::Array2::<f32>::zeros((n, dim_latent));
        for ni in 0..n {
            let row = self.poly_mean_for_state(z_prev.row(ni), state_idx[ni]);
            means.row_mut(ni).assign(&row);
        }
        means
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, s};

    #[test]
    fn func_leaky_relu_batch_single_row_slices_match_full_batch() {
        let dim_obs = 2usize;
        let dl = 3usize;
        let h = EMISSION_HIDDEN_DIM;
        let leaky = LeakyParams {
            alphas: Array2::from_elem((dim_obs, h), 0.12f32),
            omegas: Array2::from_elem((h, dl), 0.07f32),
            betas: Array1::from_elem(h, -0.03f32),
        };
        let z = array![
            [1.0f32, -0.25, 0.5],
            [-1.0f32, 2.0, 0.1],
            [0.0f32, 0.0, 0.0],
        ];
        let batch = func_leaky_relu_batch(z.view(), &leaky);
        for i in 0..3 {
            let one_row = func_leaky_relu_batch(z.slice(s![i..i + 1, ..]), &leaky);
            assert!(
                batch
                    .row(i)
                    .iter()
                    .zip(one_row.row(0).iter())
                    .all(|(a, b)| (*a - *b).abs() < 1e-5),
                "row {}",
                i
            );
        }
    }

    #[test]
    fn func_leaky_relu_batch_negative_pre_activation_expected_output() {
        let h = EMISSION_HIDDEN_DIM;
        let mut alphas = Array2::<f32>::zeros((1, h));
        alphas[[0, 0]] = 1.0;
        let mut omegas = Array2::<f32>::zeros((h, 1));
        omegas[[0, 0]] = 1.0;
        let betas = Array1::<f32>::zeros(h);
        let leaky = LeakyParams {
            alphas,
            omegas,
            betas,
        };
        let z = array![[-1.0f32]];
        let out = func_leaky_relu_batch(z.view(), &leaky);
        assert!((out[[0, 0]] - (-0.2f32)).abs() < 1e-5);
    }

    #[test]
    fn func_cosine_with_sparsity_matches_cos_on_single_channel() {
        let h = EMISSION_HIDDEN_DIM;
        let dl = 1usize;
        let mut alphas = Array3::<f32>::zeros((1, dl, h));
        alphas[[0, 0, 0]] = 1.0;
        let mut omegas = Array3::<f32>::zeros((h, dl, dl));
        omegas[[0, 0, 0]] = 1.0;
        let betas = Array2::<f32>::zeros((dl, h));
        let adj = Array2::<f32>::ones((dl, dl));
        let feat = CosineStateParams {
            alphas,
            omegas,
            betas,
            adj,
        };
        let x = 0.37f32;
        let out = func_cosine_with_sparsity(array![x].view(), &feat);
        assert!((out[0] - x.cos()).abs() < 1e-5);
    }

    #[test]
    fn func_cosine_with_sparsity_identity_adj_differs_from_dense_adj() {
        let h = EMISSION_HIDDEN_DIM;
        let dl = 2usize;
        let mut alphas = Array3::<f32>::zeros((1, dl, h));
        alphas[[0, 0, 0]] = 1.0;
        alphas[[0, 1, 0]] = 1.0;
        let mut omegas = Array3::<f32>::zeros((h, dl, dl));
        omegas[[0, 0, 0]] = 1.0;
        omegas[[0, 0, 1]] = 0.5;
        omegas[[0, 1, 0]] = 0.25;
        omegas[[0, 1, 1]] = 1.0;
        let mut betas = Array2::<f32>::zeros((dl, h));
        betas[[0, 0]] = 0.1;
        betas[[1, 0]] = -0.2;
        let adj_i = Array2::<f32>::eye(dl);
        let adj_ones = Array2::<f32>::ones((dl, dl));
        let base = CosineStateParams {
            alphas,
            omegas,
            betas,
            adj: adj_i,
        };
        let x = array![1.0f32, 1.0f32];
        let out_i = func_cosine_with_sparsity(x.view(), &base);
        let dense = CosineStateParams {
            adj: adj_ones,
            ..base
        };
        let out_d = func_cosine_with_sparsity(x.view(), &dense);
        assert!(out_i
            .iter()
            .zip(out_d.iter())
            .any(|(a, b)| (*a - *b).abs() > 1e-4));
    }

    #[test]
    fn sample_adj_mat_sparsity_zero_all_ones() {
        let mut rng = rand::rng();
        let m = sample_adj_mat(&mut rng, 0.0, 4);
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(m[[i, j]], 1.0);
            }
        }
    }

    #[test]
    fn sample_adj_mat_sparsity_one_off_diagonal_zero() {
        let mut rng = rand::rng();
        let m = sample_adj_mat(&mut rng, 1.0, 4);
        for i in 0..4 {
            assert_eq!(m[[i, i]], 1.0);
            for j in 0..4 {
                if i != j {
                    assert_eq!(m[[i, j]], 0.0);
                }
            }
        }
    }

    #[test]
    fn sample_adj_mat_diagonal_always_one() {
        let mut rng = rand::rng();
        let m = sample_adj_mat(&mut rng, 0.4, 5);
        for i in 0..5 {
            assert_eq!(m[[i, i]], 1.0);
        }
    }

    #[test]
    fn trans_mat_matches_python_doc_example() {
        let q = get_trans_mat(3);
        assert!((q[[0, 0]] - 0.9).abs() < 1e-6);
        assert!((q[[0, 1]] - 0.1).abs() < 1e-6);
        assert!((q[[1, 1]] - 0.9).abs() < 1e-6);
        assert!((q[[2, 2]] - 0.9).abs() < 1e-6);
        assert!((q[[2, 0]] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn poly_mean_for_state_matches_poly_means_rows_one_row() {
        let k = 2usize;
        let dl = 2usize;
        let degree = 2usize;
        let num_p = crate::polynomial::sklearn_poly_output_count(dl, degree);
        let coeffs = Array3::from_elem((k, dl, num_p), 0.25f32);
        let p = PolynomialStateParams::new(coeffs, dl, degree);
        let z2 = array![[0.5f32, -1.0f32]];
        let states = ndarray::Array1::from_elem(1, 0usize);
        let batch = p.poly_means_rows(z2.view(), states.view());
        let scalar = p.poly_mean_for_state(z2.row(0), 0);
        assert!(batch
            .row(0)
            .iter()
            .zip(scalar.iter())
            .all(|(a, b)| (*a - *b).abs() < 1e-6));
    }
}
