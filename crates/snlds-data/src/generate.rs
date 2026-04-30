//! Synthetic sequence generation (aligned with `generate_data` in Python).

use crate::polynomial::sklearn_poly_output_count;
use crate::transitions::{
    func_cosine_with_sparsity, func_leaky_relu_batch, get_trans_mat, sample_adj_mat,
    CosineStateParams, LeakyParams, PolynomialStateParams, EMISSION_HIDDEN_DIM,
};
use anyhow::ensure;
use ndarray::{s, Array1, Array2, Array3};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};

/// Default std-dev for `z_0` jitter in the simulator (matches Python `generate_data`).
pub const DEFAULT_INIT_NOISE_STD: f32 = 0.1;
/// Default std-dev for the per-state init-mean prior (matches Python `generate_data`).
pub const DEFAULT_INIT_MEAN_STD: f32 = 0.7;
/// Default variance of the transition step noise added to `z_t` each step.
pub const DEFAULT_TRANSITION_STEP_VAR: f32 = 0.05;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimulatorKind {
    Cosine,
    Poly,
}

/// Observation channel for the simulator.
///
/// The default `Vector` path emits via the leaky-ReLU MLP (`func_leaky_relu_batch`)
/// matching the Python `factored` setup. The `Image { res }` path renders each 2-D
/// latent into a flat `[res*res*3]` RGB frame using
/// [`crate::render::draw_sequence`] and discards the leaky-ReLU emission entirely;
/// `dim_latent` must be `2` and `dim_obs` must be `res * res * 3`. The model side
/// re-derives the spatial shape via `EncoderKind::Cnn { res }` (see `snlds-model`),
/// so the obs tensor stays flat on disk and no manifest schema bump is needed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ObservationKind {
    /// Leaky-ReLU emission MLP from latent to `dim_obs` (Python parity).
    #[default]
    Vector,
    /// Render 2-D latents to flat RGB images via `draw_sequence`.
    Image {
        /// Spatial side length in pixels (frame is `res × res × 3`).
        res: usize,
    },
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
    /// Std-dev of the Gaussian jitter added to `z_0` around each per-state init mean
    /// (default [`DEFAULT_INIT_NOISE_STD`] = 0.1, Python parity).
    /// Invariant (validated by [`generate_train_test`]): finite and `> 0`.
    pub init_noise_std: f32,
    /// Std-dev of the per-state init-mean prior used to draw `init_means[k, :]`
    /// (default [`DEFAULT_INIT_MEAN_STD`] = 0.7, Python parity).
    /// Invariant (validated by [`generate_train_test`]): finite and `> 0`.
    pub init_mean_std: f32,
    /// Variance of the transition step noise added to `z_t` each step
    /// (default [`DEFAULT_TRANSITION_STEP_VAR`] = 0.05, Python parity).
    /// (variance, not std-dev — fed to `Normal::new` as `sqrt(var)`.)
    /// Invariant (validated by [`generate_train_test`]): finite and `>= 0`
    /// (`0` is the degenerate "no step noise" case).
    pub transition_step_var: f32,
    /// Hidden dimension of the leaky-ReLU emission network in the simulator
    /// (default [`EMISSION_HIDDEN_DIM`] = 8). Persisted in the manifest since v4.
    pub emission_hidden_dim: usize,
    /// Initial-state distribution. `None` = uniform over `0..num_states`. `Some(arr)` must
    /// have length `num_states`, be all finite and non-negative, and sum to 1 within
    /// `1e-6` (validated by [`generate_train_test`]). Drives both the per-sequence draw
    /// of `s_0` and the persisted `pi_true` tensor.
    pub initial_distribution: Option<Vec<f32>>,
    /// Observation channel. Default `Vector` (leaky-ReLU emission, Python parity);
    /// `Image { res }` renders 2-D latents into flat `[res*res*3]` RGB frames via
    /// [`crate::render::draw_sequence`]. When `Image`, [`generate_train_test`] enforces
    /// `dim_latent == 2` and `dim_obs == res * res * 3` and `res >= 1`.
    pub observation: ObservationKind,
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
            init_noise_std: DEFAULT_INIT_NOISE_STD,
            init_mean_std: DEFAULT_INIT_MEAN_STD,
            transition_step_var: DEFAULT_TRANSITION_STEP_VAR,
            emission_hidden_dim: EMISSION_HIDDEN_DIM,
            initial_distribution: None,
            observation: ObservationKind::Vector,
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
    /// Ground-truth initial state distribution `[K]`. Matches `cfg.initial_distribution`
    /// after validation, or uniform `1/K` when `None`.
    pub pi_true: Array1<f32>,
}

/// Generate train (`N=num_samples`) and test (`max(1, num_samples // 10)`) splits.
///
/// Returns `Err` if any simulator hyperparameter is invalid:
/// - `init_noise_std` / `init_mean_std` not finite or not `> 0`,
/// - `transition_step_var` not finite or not `>= 0`,
/// - `initial_distribution` is `Some(_)` but wrong length, contains non-finite or
///   negative entries, or doesn't sum to 1 within `1e-6`,
/// - `observation == Image { res }` but `res == 0`, `dim_latent != 2`, or
///   `dim_obs != res * res * 3`.
pub fn generate_train_test(cfg: &GenConfig) -> anyhow::Result<TrainTest> {
    let n_train = cfg.num_samples;
    let n_test = (cfg.num_samples / 10).max(1);
    let rng = ChaCha8Rng::seed_from_u64(cfg.seed);
    generate_split(rng, cfg, n_train, n_test)
}

/// Validate the simulator scalar invariants documented on each `GenConfig` field.
fn validate_simulator_hparams(cfg: &GenConfig) -> anyhow::Result<()> {
    ensure!(
        cfg.init_noise_std.is_finite() && cfg.init_noise_std > 0.0,
        "init_noise_std must be finite and > 0 (got {})",
        cfg.init_noise_std
    );
    ensure!(
        cfg.init_mean_std.is_finite() && cfg.init_mean_std > 0.0,
        "init_mean_std must be finite and > 0 (got {})",
        cfg.init_mean_std
    );
    ensure!(
        cfg.transition_step_var.is_finite() && cfg.transition_step_var >= 0.0,
        "transition_step_var must be finite and >= 0 (got {})",
        cfg.transition_step_var
    );
    if let ObservationKind::Image { res } = cfg.observation {
        ensure!(res > 0, "observation Image.res must be > 0");
        ensure!(
            cfg.dim_latent == 2,
            "observation Image requires dim_latent == 2 (got {})",
            cfg.dim_latent
        );
        let expected_obs_dim = res * res * 3;
        ensure!(
            cfg.dim_obs == expected_obs_dim,
            "observation Image {{ res: {res} }} requires dim_obs == {expected_obs_dim} (got {})",
            cfg.dim_obs
        );
    }
    Ok(())
}

/// Validate `cfg.initial_distribution` and return the effective distribution used
/// for sampling `s_0` and persisted as `pi_true`. Also validates the simulator
/// scalar invariants (`init_noise_std`, `init_mean_std`, `transition_step_var`).
fn resolved_initial_distribution(cfg: &GenConfig) -> anyhow::Result<Vec<f32>> {
    validate_simulator_hparams(cfg)?;
    match &cfg.initial_distribution {
        None => Ok(vec![1.0 / cfg.num_states as f32; cfg.num_states]),
        Some(probs) => {
            ensure!(
                probs.len() == cfg.num_states,
                "initial_distribution length {} != num_states {}",
                probs.len(),
                cfg.num_states
            );
            ensure!(
                probs.iter().all(|prob| prob.is_finite() && *prob >= 0.0),
                "initial_distribution must be all finite and non-negative"
            );
            let sum: f32 = probs.iter().sum();
            ensure!(
                (sum - 1.0).abs() <= 1e-6,
                "initial_distribution must sum to 1 (got {sum})"
            );
            Ok(probs.clone())
        }
    }
}

fn generate_split(
    mut rng: ChaCha8Rng,
    cfg: &GenConfig,
    n_train: usize,
    n_test: usize,
) -> anyhow::Result<TrainTest> {
    let pi = resolved_initial_distribution(cfg)?;
    let (latents_train, obs_train_raw, states_train) = roll_sequences(&mut rng, cfg, n_train, &pi);
    let (latents_test, obs_test_raw, states_test) = roll_sequences(&mut rng, cfg, n_test, &pi);

    // For ObservationKind::Image the leaky-ReLU emission output is discarded and
    // replaced with rendered RGB frames flattened to [N, T, res*res*3]. The model
    // side will reshape back to [N*T, 3, res, res] inside the CNN encoder.
    let (obs_train, obs_test) = match cfg.observation {
        ObservationKind::Vector => (obs_train_raw, obs_test_raw),
        ObservationKind::Image { res } => (
            render_obs_batch(&latents_train, res),
            render_obs_batch(&latents_test, res),
        ),
    };

    let q_true = crate::transitions::get_trans_mat(cfg.num_states);
    let pi_true = Array1::from_vec(pi);
    Ok(TrainTest {
        latents_train,
        obs_train,
        states_train,
        latents_test,
        obs_test,
        states_test,
        q_true,
        pi_true,
    })
}

/// Render every per-timestep 2-D latent in `latents` (shape `[N, T, 2]`) to a
/// flat RGB frame, returning `[N, T, res*res*3]`. Pixel ordering matches
/// `draw_sequence`'s `[T, res, res, 3]` (NHWC) flattened in row-major order.
fn render_obs_batch(latents: &Array3<f32>, res: usize) -> Array3<f32> {
    let n = latents.shape()[0];
    let t_len = latents.shape()[1];
    let flat_dim = res * res * 3;
    let mut obs = Array3::<f32>::zeros((n, t_len, flat_dim));
    for ni in 0..n {
        let traj = latents.slice(s![ni, .., ..]);
        let frames = crate::render::draw_sequence(traj, res);
        // frames shape: [T, res, res, 3]. Flatten the trailing three axes per timestep.
        for ti in 0..t_len {
            let frame_t = frames.slice(s![ti, .., .., ..]);
            let mut flat_idx = 0usize;
            for row in 0..res {
                for col in 0..res {
                    for ch in 0..3 {
                        obs[[ni, ti, flat_idx]] = frame_t[[row, col, ch]];
                        flat_idx += 1;
                    }
                }
            }
        }
    }
    obs
}

fn rand_leaky(
    rng: &mut ChaCha8Rng,
    dim_obs: usize,
    dim_latent: usize,
    hidden_dim: usize,
) -> LeakyParams {
    let weight_dist = Normal::new(0.0f32, 0.5).expect("std=0.5 is a positive literal");
    let mut alphas = Array2::<f32>::zeros((dim_obs, hidden_dim));
    let mut omegas = Array2::<f32>::zeros((hidden_dim, dim_latent));
    let mut betas = Array1::<f32>::zeros(hidden_dim);
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
    pi: &[f32],
) -> (Array3<f32>, Array3<f32>, Array2<i32>) {
    let seq_len = cfg.seq_length;
    let k = cfg.num_states;
    let dim_latent = cfg.dim_latent;
    let hidden_dim = cfg.emission_hidden_dim;
    let q = get_trans_mat(k);
    let leaky = rand_leaky(rng, cfg.dim_obs, cfg.dim_latent, hidden_dim);

    let sim = match cfg.kind {
        SimulatorKind::Cosine => {
            let mut state_params = Vec::with_capacity(k);
            for _ in 0..k {
                state_params.push(rand_cosine_state(
                    rng,
                    cfg.dim_latent,
                    cfg.sparsity_prob,
                    hidden_dim,
                ));
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

    let init_noise = Normal::new(0.0f32, cfg.init_noise_std)
        .expect("init_noise_std validated by validate_simulator_hparams");
    let init_means: Array2<f32> = {
        let normal = Normal::new(0.0f32, cfg.init_mean_std)
            .expect("init_mean_std validated by validate_simulator_hparams");
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
        let init_state = sample_from_probs(rng, pi);
        states[[ni, 0]] = init_state as i32;
        for j in 0..dim_latent {
            latents[[ni, 0, j]] = init_means[[init_state, j]] + init_noise.sample(rng);
        }
    }
    {
        let obs_t0 = func_leaky_relu_batch(latents.slice(s![.., 0, ..]), &leaky);
        obs.slice_mut(s![.., 0, ..]).assign(&obs_t0);
    }

    let scale_var = cfg.transition_step_var;
    let scale_std = scale_var.sqrt();
    let step_noise = Normal::new(0.0f32, scale_std)
        .expect("transition_step_var validated by validate_simulator_hparams (>= 0)");

    for ti in 1..seq_len {
        for ni in 0..n {
            let prev_state = states[[ni, ti - 1]] as usize;
            let trans_probs = q.row(prev_state);
            let next_state = sample_from_probs(
                rng,
                trans_probs
                    .as_slice()
                    .expect("q built by Array2::zeros is row-major; row view has unit stride"),
            );
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
    hidden_dim: usize,
) -> CosineStateParams {
    let weight_dist = Normal::new(0.0f32, 0.5).expect("std=0.5 is a positive literal");
    let mut alphas = Array3::<f32>::zeros((1, dim_latent, hidden_dim));
    let mut omegas = Array3::<f32>::zeros((hidden_dim, dim_latent, dim_latent));
    let mut betas = Array2::<f32>::zeros((dim_latent, hidden_dim));
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

fn sample_from_probs<R: Rng + ?Sized>(rng: &mut R, probs: &[f32]) -> usize {
    let uniform_draw = rng.random::<f32>();
    let mut cum_prob = 0f32;
    for (idx, prob) in probs.iter().enumerate() {
        cum_prob += *prob;
        if uniform_draw < cum_prob {
            return idx;
        }
    }
    probs.len().saturating_sub(1)
}
