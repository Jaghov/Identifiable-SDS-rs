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
    let (lat_tr, obs_tr, st_tr) = roll_sequences(&mut rng, cfg, n_train);
    let (lat_te, obs_te, st_te) = roll_sequences(&mut rng, cfg, n_test);
    TrainTest {
        latents_train: lat_tr,
        obs_train: obs_tr,
        states_train: st_tr,
        latents_test: lat_te,
        obs_test: obs_te,
        states_test: st_te,
    }
}

fn rand_leaky(rng: &mut ChaCha8Rng, dim_obs: usize, dim_latent: usize) -> LeakyParams {
    let n = Normal::new(0.0f32, 0.5).unwrap();
    let mut alphas = Array2::<f32>::zeros((dim_obs, EMISSION_HIDDEN_DIM));
    let mut omegas = Array2::<f32>::zeros((EMISSION_HIDDEN_DIM, dim_latent));
    let mut betas = Array1::<f32>::zeros(EMISSION_HIDDEN_DIM);
    for a in alphas.iter_mut() {
        *a = n.sample(rng);
    }
    for o in omegas.iter_mut() {
        *o = n.sample(rng);
    }
    for b in betas.iter_mut() {
        *b = n.sample(rng);
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
    let t = cfg.seq_length;
    let k = cfg.num_states;
    let dl = cfg.dim_latent;
    let q = get_trans_mat(k);
    let leaky = rand_leaky(rng, cfg.dim_obs, cfg.dim_latent);

    let sim = match cfg.kind {
        SimulatorKind::Cosine => {
            let mut v = Vec::with_capacity(k);
            for _ in 0..k {
                v.push(rand_cosine_state(rng, cfg.dim_latent, cfg.sparsity_prob));
            }
            DynSim::Cos(v)
        }
        SimulatorKind::Poly => {
            let num_p = sklearn_poly_output_count(cfg.dim_latent, cfg.poly_degree);
            let mut coeffs = Array3::<f32>::zeros((k, dl, num_p));
            for s in 0..k {
                for i in 0..dl {
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

    let mut lat = Array3::<f32>::zeros((n, t, dl));
    let mut obs = Array3::<f32>::zeros((n, t, cfg.dim_obs));
    let mut disc = Array2::<i32>::zeros((n, t));

    let init_std = Normal::new(0.0f32, 0.1).unwrap();
    let means_init: Array2<f32> = {
        let nrm = Normal::new(0.0f32, 0.7).unwrap();
        let mut m = Array2::<f32>::zeros((k, dl));
        for s in 0..k {
            for j in 0..dl {
                m[[s, j]] = nrm.sample(rng);
            }
        }
        m
    };

    // Per-sequence initial state and t=0
    for ni in 0..n {
        let s0 = rng.random_range(0..k);
        disc[[ni, 0]] = s0 as i32;
        for j in 0..dl {
            lat[[ni, 0, j]] = means_init[[s0, j]] + init_std.sample(rng);
        }
    }
    {
        let o0 = func_leaky_relu_batch(lat.slice(s![.., 0, ..]), &leaky);
        obs.slice_mut(s![.., 0, ..]).assign(&o0);
    }

    let scale_var = 0.05f32;
    let scale_std = scale_var.sqrt();
    let step_noise = Normal::new(0.0f32, scale_std).unwrap();

    for ti in 1..t {
        for ni in 0..n {
            let prev_s = disc[[ni, ti - 1]] as usize;
            let row = q.row(prev_s);
            let new_s = sample_from_probs(rng, row.as_slice().unwrap());
            disc[[ni, ti]] = new_s as i32;

            let z_prev = lat.slice(s![ni, ti - 1, ..]);

            let mean_row: Array1<f32> = match &sim {
                DynSim::Cos(v) => func_cosine_with_sparsity(z_prev, &v[new_s]),
                DynSim::Poly(p) => p.poly_mean_for_state(z_prev, new_s),
            };

            for j in 0..dl {
                lat[[ni, ti, j]] = mean_row[j] + step_noise.sample(rng);
            }
        }
        let o_t = func_leaky_relu_batch(lat.slice(s![.., ti, ..]), &leaky);
        obs.slice_mut(s![.., ti, ..]).assign(&o_t);
    }

    (lat, obs, disc)
}

fn rand_cosine_state(
    rng: &mut ChaCha8Rng,
    dim_latent: usize,
    sparsity_prob: f32,
) -> CosineStateParams {
    let n = Normal::new(0.0f32, 0.5).unwrap();
    let mut alphas = Array3::<f32>::zeros((1, dim_latent, EMISSION_HIDDEN_DIM));
    let mut omegas = Array3::<f32>::zeros((EMISSION_HIDDEN_DIM, dim_latent, dim_latent));
    let mut betas = Array2::<f32>::zeros((dim_latent, EMISSION_HIDDEN_DIM));
    for x in alphas.iter_mut() {
        *x = n.sample(rng);
    }
    for x in omegas.iter_mut() {
        *x = n.sample(rng);
    }
    for x in betas.iter_mut() {
        *x = n.sample(rng);
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
    let mut c = 0f32;
    for (i, pi) in p.iter().enumerate() {
        c += *pi;
        if u < c {
            return i;
        }
    }
    p.len().saturating_sub(1)
}
