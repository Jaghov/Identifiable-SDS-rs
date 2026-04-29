//! Synthetic sequence generation (aligned with `generate_data` in Python).

use crate::polynomial::sklearn_poly_output_count;
use crate::transitions::{
    func_cosine_with_sparsity, func_leaky_relu_batch, get_trans_mat, sample_adj_mat,
    CosineStateParams, LeakyParams, PolynomialStateParams, EMISSION_HIDDEN_DIM,
};
use ndarray::{s, Array1, Array2, Array3};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimulatorKind {
    Cosine,
    Poly,
}

#[derive(Clone, Debug)]
pub struct GenConfig {
    pub seed: u64,
    pub num_states: usize,
    pub dim_obs: usize,
    pub dim_latent: usize,
    pub seq_length: usize,
    /// Train split size (`N`); test uses `max(1, num_samples / 10)`.
    pub num_samples: usize,
    pub sparsity_prob: f32,
    pub kind: SimulatorKind,
    /// Used when `SimulatorKind::Poly`; default 3 (Python).
    pub poly_degree: usize,
}

impl Default for GenConfig {
    fn default() -> Self {
        Self {
            seed: 24,
            num_states: 3,
            dim_obs: 2,
            dim_latent: 2,
            seq_length: 200,
            num_samples: 5000,
            sparsity_prob: 0.0,
            kind: SimulatorKind::Cosine,
            poly_degree: 3,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TrainTest {
    pub latents_train: Array3<f32>,
    pub obs_train: Array3<f32>,
    /// Discrete state indices per timestep (SafeTensors **I32** on export).
    pub states_train: Array2<i32>,
    pub latents_test: Array3<f32>,
    pub obs_test: Array3<f32>,
    pub states_test: Array2<i32>,
    /// Ground-truth Markov transition matrix `[K, K]` produced by
    /// [`crate::transitions::get_trans_mat`]. Persisted as `q_true` (F32) since schema v3 so
    /// downstream viz / eval can compare against a learned `Q`.
    pub q_true: Array2<f32>,
    /// Ground-truth initial state distribution `[K]`. The Python reference samples the first
    /// state uniformly across `K`, so this is `1/K` everywhere unless the simulator is changed.
    pub pi_true: Array1<f32>,
}

/// Generate train (`N=num_samples`) and test (`max(1, num_samples // 10)`) splits.
pub fn generate_train_test(cfg: &GenConfig) -> TrainTest {
    let n_train = cfg.num_samples;
    let n_test = (cfg.num_samples / 10).max(1);
    let rng = ChaCha8Rng::seed_from_u64(cfg.seed);
    generate_split(rng, cfg, n_train, n_test)
}

fn generate_split(
    mut rng: ChaCha8Rng,
    cfg: &GenConfig,
    n_train: usize,
    n_test: usize,
) -> TrainTest {
    let (latents_train, obs_train, states_train) = roll_sequences(&mut rng, cfg, n_train);
    let (latents_test, obs_test, states_test) = roll_sequences(&mut rng, cfg, n_test);
    let q_true = crate::transitions::get_trans_mat(cfg.num_states);
    let pi_true = Array1::from_elem(cfg.num_states, 1.0 / cfg.num_states as f32);
    TrainTest {
        latents_train,
        obs_train,
        states_train,
        latents_test,
        obs_test,
        states_test,
        q_true,
        pi_true,
    }
}

fn rand_leaky(rng: &mut ChaCha8Rng, dim_obs: usize, dim_latent: usize) -> LeakyParams {
    let weight_dist = Normal::new(0.0f32, 0.5).unwrap();
    let mut alphas = Array2::<f32>::zeros((dim_obs, EMISSION_HIDDEN_DIM));
    let mut omegas = Array2::<f32>::zeros((EMISSION_HIDDEN_DIM, dim_latent));
    let mut betas = Array1::<f32>::zeros(EMISSION_HIDDEN_DIM);
    for a in alphas.iter_mut() {
        *a = weight_dist.sample(rng);
    }
    for o in omegas.iter_mut() {
        *o = weight_dist.sample(rng);
    }
    for b in betas.iter_mut() {
        *b = weight_dist.sample(rng);
    }
    LeakyParams {
        alphas,
        omegas,
        betas,
    }
}

enum DynSim {
    Cos(Vec<CosineStateParams>),
    Poly(PolynomialStateParams),
}

fn roll_sequences(
    rng: &mut ChaCha8Rng,
    cfg: &GenConfig,
    n: usize,
) -> (Array3<f32>, Array3<f32>, Array2<i32>) {
    let seq_len = cfg.seq_length;
    let k = cfg.num_states;
    let dim_latent = cfg.dim_latent;
    let q = get_trans_mat(k);
    let leaky = rand_leaky(rng, cfg.dim_obs, cfg.dim_latent);

    let sim = match cfg.kind {
        SimulatorKind::Cosine => {
            let mut state_params = Vec::with_capacity(k);
            for _ in 0..k {
                state_params.push(rand_cosine_state(rng, cfg.dim_latent, cfg.sparsity_prob));
            }
            DynSim::Cos(state_params)
        }
        SimulatorKind::Poly => {
            let num_p = sklearn_poly_output_count(cfg.dim_latent, cfg.poly_degree);
            let mut coeffs = Array3::<f32>::zeros((k, dim_latent, num_p));
            for s in 0..k {
                for i in 0..dim_latent {
                    for j in 0..num_p {
                        let u: f32 = rng.random_range(0.0..1.0);
                        let mut v = u - 0.5;
                        if j > 0 {
                            v *= 0.05;
                        }
                        coeffs[[s, i, j]] = v;
                    }
                }
            }
            DynSim::Poly(PolynomialStateParams::new(
                coeffs,
                cfg.dim_latent,
                cfg.poly_degree,
            ))
        }
    };

    let mut latents = Array3::<f32>::zeros((n, seq_len, dim_latent));
    let mut obs = Array3::<f32>::zeros((n, seq_len, cfg.dim_obs));
    let mut states = Array2::<i32>::zeros((n, seq_len));

    let init_noise = Normal::new(0.0f32, 0.1).unwrap();
    let init_means: Array2<f32> = {
        let normal = Normal::new(0.0f32, 0.7).unwrap();
        let mut m = Array2::<f32>::zeros((k, dim_latent));
        for s in 0..k {
            for j in 0..dim_latent {
                m[[s, j]] = normal.sample(rng);
            }
        }
        m
    };

    // Per-sequence initial state and t=0
    for ni in 0..n {
        let init_state = rng.random_range(0..k);
        states[[ni, 0]] = init_state as i32;
        for j in 0..dim_latent {
            latents[[ni, 0, j]] = init_means[[init_state, j]] + init_noise.sample(rng);
        }
    }
    {
        let obs_t0 = func_leaky_relu_batch(latents.slice(s![.., 0, ..]), &leaky);
        obs.slice_mut(s![.., 0, ..]).assign(&obs_t0);
    }

    let scale_var = 0.05f32;
    let scale_std = scale_var.sqrt();
    let step_noise = Normal::new(0.0f32, scale_std).unwrap();

    for ti in 1..seq_len {
        for ni in 0..n {
            let prev_state = states[[ni, ti - 1]] as usize;
            let trans_probs = q.row(prev_state);
            let next_state = sample_from_probs(rng, trans_probs.as_slice().unwrap());
            states[[ni, ti]] = next_state as i32;

            let z_prev = latents.slice(s![ni, ti - 1, ..]);

            let dyn_mean: Array1<f32> = match &sim {
                DynSim::Cos(state_params) => {
                    func_cosine_with_sparsity(z_prev, &state_params[next_state])
                }
                DynSim::Poly(p) => p.poly_mean_for_state(z_prev, next_state),
            };

            for j in 0..dim_latent {
                latents[[ni, ti, j]] = dyn_mean[j] + step_noise.sample(rng);
            }
        }
        let obs_t = func_leaky_relu_batch(latents.slice(s![.., ti, ..]), &leaky);
        obs.slice_mut(s![.., ti, ..]).assign(&obs_t);
    }

    (latents, obs, states)
}

fn rand_cosine_state(
    rng: &mut ChaCha8Rng,
    dim_latent: usize,
    sparsity_prob: f32,
) -> CosineStateParams {
    let weight_dist = Normal::new(0.0f32, 0.5).unwrap();
    let mut alphas = Array3::<f32>::zeros((1, dim_latent, EMISSION_HIDDEN_DIM));
    let mut omegas = Array3::<f32>::zeros((EMISSION_HIDDEN_DIM, dim_latent, dim_latent));
    let mut betas = Array2::<f32>::zeros((dim_latent, EMISSION_HIDDEN_DIM));
    for x in alphas.iter_mut() {
        *x = weight_dist.sample(rng);
    }
    for x in omegas.iter_mut() {
        *x = weight_dist.sample(rng);
    }
    for x in betas.iter_mut() {
        *x = weight_dist.sample(rng);
    }
    let adj = sample_adj_mat(rng, sparsity_prob, dim_latent);
    CosineStateParams {
        alphas,
        omegas,
        betas,
        adj,
    }
}

fn sample_from_probs<R: Rng + ?Sized>(rng: &mut R, p: &[f32]) -> usize {
    let u = rng.random::<f32>();
    let mut cum_prob = 0f32;
    for (i, prob) in p.iter().enumerate() {
        cum_prob += *prob;
        if u < cum_prob {
            return i;
        }
    }
    p.len().saturating_sub(1)
}
