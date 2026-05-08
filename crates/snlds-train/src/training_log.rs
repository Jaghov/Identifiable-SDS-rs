//! Shared minibatch logging cadence for training loops.

use std::io::{self, Write};
use std::path::Path;

use burn::prelude::Backend;
use burn::tensor::activation::log_softmax;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;

/// Whether to print diagnostics for this minibatch (`batch_idx` is 0-based).
///
/// `log_every_batch == 0` disables per-batch lines (epoch summaries still print).
/// Otherwise logs every `log_every_batch` batches (including batch 0) and always
/// logs the final batch of the epoch.
#[inline]
pub(crate) fn should_log_minibatch(
    log_every_batch: usize,
    batch_idx: usize,
    total_batches: usize,
) -> bool {
    log_every_batch > 0 && (batch_idx % log_every_batch == 0 || batch_idx + 1 == total_batches)
}

/// After optimizer step for minibatch `batch_idx` (0-based), whether to print learned `Q`.
#[inline]
pub(crate) fn should_log_transition_every_n_batches(every: usize, batch_idx_0based: usize) -> bool {
    every > 0 && (batch_idx_0based + 1) % every == 0
}

#[cfg(test)]
mod test_hooks {
    use std::sync::atomic::{AtomicUsize, Ordering};

    static MID_EPOCH_Q_LOGS: AtomicUsize = AtomicUsize::new(0);
    static EPOCH_END_Q_LOGS: AtomicUsize = AtomicUsize::new(0);

    pub fn reset() {
        MID_EPOCH_Q_LOGS.store(0, Ordering::Relaxed);
        EPOCH_END_Q_LOGS.store(0, Ordering::Relaxed);
    }

    pub fn bump_mid_epoch() {
        MID_EPOCH_Q_LOGS.fetch_add(1, Ordering::Relaxed);
    }

    pub fn bump_epoch_end() {
        // `log_true_transition_matrix_from_data` also prints "true Q" — not counted here.
        EPOCH_END_Q_LOGS.fetch_add(1, Ordering::Relaxed);
    }

    pub fn mid_epoch_count() -> usize {
        MID_EPOCH_Q_LOGS.load(Ordering::Relaxed)
    }

    pub fn epoch_end_count() -> usize {
        EPOCH_END_Q_LOGS.load(Ordering::Relaxed)
    }
}

/// Reset counters used by [`q_log_mid_epoch_count_for_test`] (integration tests only).
#[cfg(test)]
pub fn reset_q_log_counters_for_test() {
    test_hooks::reset();
}

#[cfg(test)]
pub fn q_log_mid_epoch_count_for_test() -> usize {
    test_hooks::mid_epoch_count()
}

#[cfg(test)]
pub fn q_log_epoch_end_count_for_test() -> usize {
    test_hooks::epoch_end_count()
}

/// Print ground-truth `q_true` from M1 `sequences.safetensors` when present (schema v3+).
pub(crate) fn log_true_transition_matrix_from_data(data_dir: &Path, num_states: usize) {
    println!("--- true Markov Q (from `q_true` in sequences.safetensors) ---");

    let st_path = data_dir.join("sequences.safetensors");
    let st_path = if st_path.is_file() {
        st_path
    } else {
        let shard0 = data_dir.join("shard_000").join("sequences.safetensors");
        if shard0.is_file() {
            shard0
        } else {
            println!(
                "true transition Q: missing file {:?} (expected next to metadata.json)",
                st_path
            );
            let _ = io::stdout().flush();
            return;
        }
    };
    let flat = match snlds_data::load_tensor_f32(&st_path, "q_true") {
        Ok(v) => v,
        Err(e) => {
            println!(
                "true transition Q: could not read tensor `q_true` from {:?}: {:#}",
                st_path, e
            );
            println!(
                "  (Regenerate data with a current `snlds-gen` if you need q_true; older exports may omit it.)"
            );
            let _ = io::stdout().flush();
            return;
        }
    };
    if flat.len() != num_states * num_states {
        println!(
            "true transition Q: `q_true` has {} entries, expected K² with K={} from manifest; skipping print",
            flat.len(),
            num_states
        );
        let _ = io::stdout().flush();
        return;
    }
    println!("true transition Q from data (rows=from-state, cols=to-state):");
    let k = num_states;
    for i in 0..k {
        let row: Vec<String> = (0..k).map(|j| format!("{:.4}", flat[i * k + j])).collect();
        println!("  [{}] {}", i, row.join("  "));
    }
    let _ = io::stdout().flush();
}

/// Row-stochastic `Q[i,j] = P(s_{t+1}=j | s_t=i)` matching `log_softmax(q_logits/temp, 1)` in the models.
///
/// `batch_in_epoch`: when `Some((batch_1based, n_batches))`, label as a mid-epoch minibatch; otherwise epoch summary.
pub(crate) fn log_learned_transition_matrix<B: AutodiffBackend + Backend<FloatElem = f32>>(
    line_prefix: &str,
    epoch: usize,
    q_logits: Tensor<B, 2>,
    temperature: f32,
    batch_in_epoch: Option<(usize, usize)>,
) {
    let [k, k2] = q_logits.dims();
    if k == 0 || k2 != k {
        return;
    }
    let log_q = log_softmax(q_logits / temperature, 1);
    let flat = log_q
        .exp()
        .detach()
        .into_data()
        .to_vec::<f32>()
        .unwrap_or_default();
    if flat.len() != k * k {
        return;
    }
    match batch_in_epoch {
        None => println!(
            "{}epoch {:04} learned transition Q (rows=from-state, cols=to-state):",
            line_prefix, epoch
        ),
        Some((b, n_batches)) => println!(
            "{}epoch {:04} batch {:04}/{} learned transition Q (rows=from-state, cols=to-state):",
            line_prefix, epoch, b, n_batches
        ),
    }
    for i in 0..k {
        let row: Vec<String> = (0..k).map(|j| format!("{:.4}", flat[i * k + j])).collect();
        println!("  [{}] {}", i, row.join("  "));
    }
    #[cfg(test)]
    match batch_in_epoch {
        Some(_) => test_hooks::bump_mid_epoch(),
        None => test_hooks::bump_epoch_end(),
    }
    let _ = io::stdout().flush();
}

#[cfg(test)]
mod tests {
    use super::should_log_transition_every_n_batches;

    #[test]
    fn transition_log_every_n_batches_cadence() {
        assert!(!should_log_transition_every_n_batches(0, 0));
        assert!(!should_log_transition_every_n_batches(0, 99));

        assert!(should_log_transition_every_n_batches(1, 0));
        assert!(should_log_transition_every_n_batches(1, 1));

        assert!(!should_log_transition_every_n_batches(2, 0));
        assert!(should_log_transition_every_n_batches(2, 1));
        assert!(!should_log_transition_every_n_batches(2, 2));
        assert!(should_log_transition_every_n_batches(2, 3));

        assert!(should_log_transition_every_n_batches(50, 49));
        assert!(!should_log_transition_every_n_batches(50, 48));
    }
}
