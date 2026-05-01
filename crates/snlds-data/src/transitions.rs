//! Port of [identifiable-SDS/utils/transitions.py](../identifiable-SDS/utils/transitions.py).
#![allow(clippy::needless_range_loop)]

use crate::polynomial::{expand_polynomial_row, sklearn_powers};
use anyhow::ensure;
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2};
use rand::Rng;

/// Hidden width for leaky-ReLU emission (`params_leaky`) and cosine feature maps (matches Python `generate_data`).
pub const EMISSION_HIDDEN_DIM: usize = 8;

/// Default self-transition probability for [`TransitionPattern::Cyclic`] (historical 0.9 / 0.1 chain).
pub(crate) const DEFAULT_CYCLIC_SELF_PROB: f32 = 0.9;

/// Row-sum tolerance for [`TransitionPattern::Provided`] validation.
const TRANSITION_ROW_SUM_TOL: f32 = 1e-6;

/// Markov transition topology for the synthetic simulator.
#[derive(Clone, Debug, PartialEq)]
pub enum TransitionPattern {
    /// Cyclic chain: diagonal `self_prob`, forward edge (with wrap) `1 - self_prob`.
    /// `self_prob` must be finite and in `[0, 1]`.
    Cyclic {
        /// Diagonal entry when `size >= 2`. For `size == 1`, [`get_trans_mat`] returns `[[1.0]]`
        /// regardless of this field (the only row-stochastic `1×1` matrix).
        self_prob: f32,
    },
    /// Caller-supplied row-stochastic matrix: shape `[size, size]`, finite non-negative entries,
    /// rows summing to 1 within `1e-6` (per-entry upper bounds follow from row-stochasticity).
    Provided(Array2<f32>),
}

impl Default for TransitionPattern {
    fn default() -> Self {
        Self::Cyclic {
            self_prob: DEFAULT_CYCLIC_SELF_PROB,
        }
    }
}

/// Build the Markov transition matrix for `pattern` at dimension `size` (`num_states`).
///
/// `size` must be positive. For cyclic topology with `size == 1`, returns `[[1.0]]`.
pub fn get_trans_mat(pattern: &TransitionPattern, size: usize) -> anyhow::Result<Array2<f32>> {
    ensure!(
        size > 0,
        "transition matrix size must be > 0 (num_states) (got {size})"
    );
    match pattern {
        TransitionPattern::Cyclic { self_prob } => {
            ensure!(
                self_prob.is_finite() && (0.0..=1.0).contains(self_prob),
                "TransitionPattern::Cyclic.self_prob must be finite and in [0, 1] (got {self_prob})"
            );
            if size == 1 {
                let mut trans_mat = Array2::<f32>::zeros((1, 1));
                trans_mat[[0, 0]] = 1.0;
                return Ok(trans_mat);
            }
            let mut trans_mat = Array2::<f32>::zeros((size, size));
            let next_prob = 1.0 - self_prob;
            for row_idx in 0..size {
                let next_idx = (row_idx + 1) % size;
                trans_mat[[row_idx, row_idx]] = *self_prob;
                if next_idx != row_idx {
                    trans_mat[[row_idx, next_idx]] = next_prob;
                }
            }
            Ok(trans_mat)
        }
        TransitionPattern::Provided(matrix) => {
            ensure!(
                matrix.shape() == [size, size],
                "TransitionPattern::Provided expected shape [{size}, {size}], got {:?}",
                matrix.shape()
            );
            for ((row_idx, col_idx), entry) in matrix.indexed_iter() {
                ensure!(
                    entry.is_finite() && *entry >= 0.0,
                    "TransitionPattern::Provided entry at ({row_idx}, {col_idx}) must be finite and non-negative (got {entry})"
                );
            }
            for (row_idx, row) in matrix.rows().into_iter().enumerate() {
                let row_sum: f32 = row.iter().sum();
                let deviation = (row_sum - 1.0).abs();
                ensure!(
                    deviation <= TRANSITION_ROW_SUM_TOL,
                    "TransitionPattern::Provided row {row_idx} must sum to 1 within {} (got {row_sum}, deviation {deviation})",
                    TRANSITION_ROW_SUM_TOL
                );
            }
            Ok(matrix.clone())
        }
    }
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
    let mut pre_act = z.dot(&params.omegas.t());
    for mut row in pre_act.rows_mut() {
        row += &params.betas;
    }
    let activated = pre_act.mapv(|v| v.max(0.2 * v));
    activated.dot(&params.alphas.t())
}

/// Cosine dynamics for one discrete state (`params[k]` tuple in Python).
///
/// **Invariant:** the hidden-dim axis must agree across all three weight tensors —
/// `alphas.dim().2 == omegas.dim().0 == betas.dim().1`. `func_cosine_with_sparsity`
/// reads `H` from `alphas` and indexes `omegas` on axis 0 and `betas` on axis 1
/// using that same value; mismatched shapes panic at index time.
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
    let hidden_dim = feat.alphas.dim().2;
    debug_assert_eq!(
        feat.alphas.dim().2,
        feat.omegas.dim().0,
        "CosineStateParams hidden dim mismatch: alphas axis 2 vs omegas axis 0"
    );
    debug_assert_eq!(
        feat.alphas.dim().2,
        feat.betas.dim().1,
        "CosineStateParams hidden dim mismatch: alphas axis 2 vs betas axis 1"
    );
    let mut out = Array1::<f32>::zeros(dim_lat);
    for i in 0..dim_lat {
        let mut masked_x = Array1::<f32>::zeros(dim_lat);
        for j in 0..dim_lat {
            masked_x[j] = x[j] * feat.adj[[i, j]];
        }
        let mut cosine_features = vec![0f32; hidden_dim];
        for k in 0..hidden_dim {
            let mut pre_act = 0f32;
            for j in 0..dim_lat {
                pre_act += feat.omegas[[k, i, j]] * masked_x[j];
            }
            cosine_features[k] = (pre_act + feat.betas[[i, k]]).cos();
        }
        let mut weighted_sum = 0f32;
        for k in 0..hidden_dim {
            weighted_sum += feat.alphas[[0, i, k]] * cosine_features[k];
        }
        out[i] = weighted_sum;
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
        let z_vec: Vec<f32> = z.iter().copied().collect();
        let poly_features = expand_polynomial_row(&z_vec, &self.powers);
        for d in 0..dim_latent {
            let mut acc = 0f32;
            for feat_idx in 0..poly_features.len() {
                acc += self.coeffs[[state, d, feat_idx]] * poly_features[feat_idx];
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
        let num_seqs = z_prev.nrows();
        let mut means = ndarray::Array2::<f32>::zeros((num_seqs, dim_latent));
        for ni in 0..num_seqs {
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
        let dim_latent = 3usize;
        let h = EMISSION_HIDDEN_DIM;
        let leaky = LeakyParams {
            alphas: Array2::from_elem((dim_obs, h), 0.12f32),
            omegas: Array2::from_elem((h, dim_latent), 0.07f32),
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
        let dim_latent = 1usize;
        let mut alphas = Array3::<f32>::zeros((1, dim_latent, h));
        alphas[[0, 0, 0]] = 1.0;
        let mut omegas = Array3::<f32>::zeros((h, dim_latent, dim_latent));
        omegas[[0, 0, 0]] = 1.0;
        let betas = Array2::<f32>::zeros((dim_latent, h));
        let adj = Array2::<f32>::ones((dim_latent, dim_latent));
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
        let dim_latent = 2usize;
        let mut alphas = Array3::<f32>::zeros((1, dim_latent, h));
        alphas[[0, 0, 0]] = 1.0;
        alphas[[0, 1, 0]] = 1.0;
        let mut omegas = Array3::<f32>::zeros((h, dim_latent, dim_latent));
        omegas[[0, 0, 0]] = 1.0;
        omegas[[0, 0, 1]] = 0.5;
        omegas[[0, 1, 0]] = 0.25;
        omegas[[0, 1, 1]] = 1.0;
        let mut betas = Array2::<f32>::zeros((dim_latent, h));
        betas[[0, 0]] = 0.1;
        betas[[1, 0]] = -0.2;
        let adj_i = Array2::<f32>::eye(dim_latent);
        let adj_ones = Array2::<f32>::ones((dim_latent, dim_latent));
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
        let q = get_trans_mat(&TransitionPattern::default(), 3).unwrap();
        assert!((q[[0, 0]] - 0.9).abs() < 1e-6);
        assert!((q[[0, 1]] - 0.1).abs() < 1e-6);
        assert!((q[[1, 1]] - 0.9).abs() < 1e-6);
        assert!((q[[2, 2]] - 0.9).abs() < 1e-6);
        assert!((q[[2, 0]] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn cyclic_size_one_is_identity() {
        let q = get_trans_mat(&TransitionPattern::Cyclic { self_prob: 0.9 }, 1).unwrap();
        assert_eq!(q.shape(), [1, 1]);
        assert!((q[[0, 0]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cyclic_self_prob_zero_three_states() {
        let q = get_trans_mat(&TransitionPattern::Cyclic { self_prob: 0.0 }, 3).unwrap();
        assert!((q[[0, 1]] - 1.0).abs() < 1e-6);
        assert!((q[[1, 2]] - 1.0).abs() < 1e-6);
        assert!((q[[2, 0]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cyclic_self_prob_one_three_states() {
        let q = get_trans_mat(&TransitionPattern::Cyclic { self_prob: 1.0 }, 3).unwrap();
        for i in 0..3 {
            assert!((q[[i, i]] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn cyclic_rejects_negative_self_prob() {
        let err = get_trans_mat(&TransitionPattern::Cyclic { self_prob: -0.5 }, 3).unwrap_err();
        assert!(err.to_string().contains("self_prob"));
    }

    #[test]
    fn provided_rejects_infinity() {
        let mut m = Array2::<f32>::zeros((2, 2));
        m[[0, 0]] = f32::INFINITY;
        m[[0, 1]] = 0.0;
        m[[1, 0]] = 0.5;
        m[[1, 1]] = 0.5;
        let err = get_trans_mat(&TransitionPattern::Provided(m), 2).unwrap_err();
        assert!(err.to_string().contains("(0, 0)"));
    }

    #[test]
    fn provided_rejects_negative_entry() {
        let mut m = Array2::<f32>::zeros((2, 2));
        m[[0, 0]] = 1.0;
        m[[0, 1]] = 0.0;
        m[[1, 0]] = -0.1;
        m[[1, 1]] = 1.1;
        let err = get_trans_mat(&TransitionPattern::Provided(m), 2).unwrap_err();
        assert!(err.to_string().contains("(1, 0)"));
    }

    #[test]
    fn get_trans_mat_rejects_size_zero() {
        let err = get_trans_mat(&TransitionPattern::default(), 0).unwrap_err();
        assert!(err.to_string().contains("size"));
    }

    #[test]
    fn poly_mean_for_state_matches_poly_means_rows_one_row() {
        let num_states = 2usize;
        let dim_latent = 2usize;
        let degree = 2usize;
        let num_p = crate::polynomial::sklearn_poly_output_count(dim_latent, degree);
        let coeffs = Array3::from_elem((num_states, dim_latent, num_p), 0.25f32);
        let poly = PolynomialStateParams::new(coeffs, dim_latent, degree);
        let z_batch = array![[0.5f32, -1.0f32]];
        let states = ndarray::Array1::from_elem(1, 0usize);
        let batch = poly.poly_means_rows(z_batch.view(), states.view());
        let scalar = poly.poly_mean_for_state(z_batch.row(0), 0);
        assert!(batch
            .row(0)
            .iter()
            .zip(scalar.iter())
            .all(|(a, b)| (*a - *b).abs() < 1e-6));
    }
}
