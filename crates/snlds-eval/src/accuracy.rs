//! Hungarian-aligned discrete-state prediction accuracy.
//!
//! The SNLDS model has no a-priori correspondence between its discrete latent
//! indices and the ground-truth state labels, so a raw `inferred == truth`
//! comparison is meaningless. [`align_with_hungarian`] finds the relabelling
//! `π` (a permutation over `[0..K)`) that maximises matched accuracy, then
//! returns the matched accuracy, the chosen permutation, and the post-match
//! confusion matrix.
//!
//! Brute-force permutation search via Heap's algorithm: `K!` for `K = 10` is
//! ~3.6 M permutations which runs in well under a second. Beyond that the
//! function returns `Err` and the caller should switch to a proper `O(K^3)`
//! Hungarian implementation.

use anyhow::{bail, ensure, Result};
use ndarray::{Array2, ArrayView2};

/// Upper bound on `num_states` for the brute-force permutation search used
/// by [`align_with_hungarian`]. `10! = 3,628,800` permutations × `O(K)`
/// scoring runs well under a second on a modern CPU. Values above this cap
/// return `Err` rather than silently spinning — when SNLDS use cases
/// genuinely need `K > 10`, replace [`align_with_hungarian`]'s body with a
/// proper `O(K^3)` Kuhn–Munkres implementation; the public signature is
/// chosen so that's a pure-body change.
pub const MAX_BRUTE_FORCE_STATES: usize = 10;

/// Outcome of label-matched accuracy scoring.
#[derive(Clone, Debug)]
pub struct AccuracyReport {
    /// `permutation[inferred_label] = matched_true_label`. Applied to a stream
    /// of inferred labels, this yields predictions in the ground-truth label
    /// space.
    pub permutation: Vec<usize>,
    /// Matched accuracy in `[0, 1]`.
    pub accuracy: f32,
    /// Confusion matrix in the permuted frame: rows = ground truth, columns =
    /// inferred label after permutation (so the diagonal is correct).
    pub confusion: Array2<u32>,
    /// Number of correctly classified timesteps after permutation.
    pub correct: u64,
    /// Total number of timesteps scored.
    pub total: u64,
}

/// Build the raw confusion matrix `raw[true, inferred]`.
fn build_confusion(
    true_states: ArrayView2<i32>,
    inferred_states: ArrayView2<i32>,
    num_states: usize,
) -> Result<Array2<u32>> {
    ensure!(
        true_states.shape() == inferred_states.shape(),
        "true_states shape {:?} != inferred_states shape {:?}",
        true_states.shape(),
        inferred_states.shape()
    );
    let mut raw = Array2::<u32>::zeros((num_states, num_states));
    let kmax = num_states as i32;
    for (true_label, inferred_label) in true_states.iter().zip(inferred_states.iter()) {
        ensure!(
            *true_label >= 0 && *true_label < kmax,
            "true_state {} out of range [0, {})",
            true_label,
            num_states
        );
        ensure!(
            *inferred_label >= 0 && *inferred_label < kmax,
            "inferred_state {} out of range [0, {})",
            inferred_label,
            num_states
        );
        raw[[*true_label as usize, *inferred_label as usize]] += 1;
    }
    Ok(raw)
}

/// Score the permutation `perm[inferred] = matched_true_label` against
/// `raw[true, inferred]`. Higher is better.
fn score_permutation(raw: &Array2<u32>, perm: &[usize]) -> u64 {
    let mut score = 0u64;
    for inferred_label in 0..perm.len() {
        let true_label = perm[inferred_label];
        score += raw[[true_label, inferred_label]] as u64;
    }
    score
}

/// Find the relabelling that maximises matched accuracy and return the full
/// report. `num_states` must match both `true_states` and `inferred_states`
/// value ranges and must be `<= MAX_BRUTE_FORCE_STATES`.
pub fn align_with_hungarian(
    true_states: ArrayView2<i32>,
    inferred_states: ArrayView2<i32>,
    num_states: usize,
) -> Result<AccuracyReport> {
    ensure!(
        num_states >= 1,
        "num_states must be >= 1 (got {num_states})"
    );
    if num_states > MAX_BRUTE_FORCE_STATES {
        bail!(
            "num_states {} exceeds brute-force cap {} — implement a proper Hungarian solver",
            num_states,
            MAX_BRUTE_FORCE_STATES
        );
    }
    let raw = build_confusion(true_states, inferred_states, num_states)?;
    let total: u64 = raw.iter().map(|count| *count as u64).sum();
    if total == 0 {
        bail!("no scored timesteps (true_states and inferred_states were both empty)");
    }

    let mut perm: Vec<usize> = (0..num_states).collect();
    let mut best_score = score_permutation(&raw, &perm);
    let mut best_perm = perm.clone();

    // Heap's algorithm: enumerate every permutation of `perm` in place.
    let mut counters = vec![0usize; num_states];
    let mut pivot = 0usize;
    while pivot < num_states {
        if counters[pivot] < pivot {
            if pivot.is_multiple_of(2) {
                perm.swap(0, pivot);
            } else {
                perm.swap(counters[pivot], pivot);
            }
            let score = score_permutation(&raw, &perm);
            if score > best_score {
                best_score = score;
                best_perm = perm.clone();
            }
            counters[pivot] += 1;
            pivot = 0;
        } else {
            counters[pivot] = 0;
            pivot += 1;
        }
    }

    // Permuted confusion: rows = ground truth, columns = inferred *after*
    // applying `best_perm`. So a perfectly matched diagonal means perfect
    // accuracy.
    let mut confusion = Array2::<u32>::zeros((num_states, num_states));
    for inferred_label in 0..num_states {
        let mapped = best_perm[inferred_label];
        for true_label in 0..num_states {
            confusion[[true_label, mapped]] += raw[[true_label, inferred_label]];
        }
    }

    let accuracy = best_score as f32 / total as f32;
    Ok(AccuracyReport {
        permutation: best_perm,
        accuracy,
        confusion,
        correct: best_score,
        total,
    })
}

/// Pretty-print the report to stdout in a human-readable form.
pub fn print_report(report: &AccuracyReport) {
    println!(
        "Matched accuracy: {:.4} ({}/{} timesteps)",
        report.accuracy, report.correct, report.total,
    );
    let perm_pairs: Vec<String> = report
        .permutation
        .iter()
        .enumerate()
        .map(|(inferred_label, true_label)| format!("{inferred_label}->{true_label}"))
        .collect();
    println!(
        "Label mapping (inferred -> true): {}",
        perm_pairs.join(", ")
    );
    println!("Permuted confusion matrix (rows=true, cols=inferred-after-permutation):");
    let num_states = report.confusion.shape()[0];
    for true_label in 0..num_states {
        let row: Vec<String> = (0..num_states)
            .map(|inferred_label| format!("{:>6}", report.confusion[[true_label, inferred_label]]))
            .collect();
        println!("  truth={true_label}: [{}]", row.join(", "));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn identity_match_perfect_accuracy() {
        let truth = array![[0, 1, 2, 0, 1, 2]];
        let inferred = array![[0, 1, 2, 0, 1, 2]];
        let report = align_with_hungarian(truth.view(), inferred.view(), 3).unwrap();
        assert_eq!(report.accuracy, 1.0);
        assert_eq!(report.correct, 6);
        assert_eq!(report.total, 6);
        assert_eq!(report.permutation, vec![0, 1, 2]);
    }

    #[test]
    fn swapped_labels_match_perfectly_after_permutation() {
        let truth = array![[0, 0, 1, 1, 2, 2]];
        let inferred = array![[2, 2, 0, 0, 1, 1]];
        let report = align_with_hungarian(truth.view(), inferred.view(), 3).unwrap();
        assert_eq!(report.accuracy, 1.0);
        assert_eq!(report.correct, 6);
        // permutation maps inferred 2->true 0, inferred 0->true 1, inferred 1->true 2.
        assert_eq!(report.permutation[2], 0);
        assert_eq!(report.permutation[0], 1);
        assert_eq!(report.permutation[1], 2);
    }

    #[test]
    fn partial_accuracy_counts_correctly() {
        // 4/6 correct under the best permutation (identity here).
        let truth = array![[0, 0, 1, 1, 2, 2]];
        let inferred = array![[0, 0, 1, 2, 2, 1]];
        let report = align_with_hungarian(truth.view(), inferred.view(), 3).unwrap();
        assert_eq!(report.correct, 4);
        assert_eq!(report.total, 6);
        assert!((report.accuracy - 4.0 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn out_of_range_label_rejected() {
        let truth = array![[0, 1, 3]];
        let inferred = array![[0, 1, 2]];
        let err = align_with_hungarian(truth.view(), inferred.view(), 3).expect_err("label 3 out");
        assert!(format!("{err:#}").contains("out of range"));
    }

    #[test]
    fn shape_mismatch_rejected() {
        let truth = array![[0, 1, 2]];
        let inferred = array![[0, 1]];
        let err = align_with_hungarian(truth.view(), inferred.view(), 3).expect_err("shape");
        assert!(format!("{err:#}").contains("shape"));
    }

    #[test]
    fn confusion_diagonal_is_correct_count() {
        let truth = array![[0, 0, 1, 1, 2, 2]];
        let inferred = array![[0, 0, 1, 2, 2, 1]];
        let report = align_with_hungarian(truth.view(), inferred.view(), 3).unwrap();
        let diag: u32 = (0..3).map(|i| report.confusion[[i, i]]).sum();
        assert_eq!(diag as u64, report.correct);
    }

    #[test]
    fn too_many_states_rejected() {
        let truth = Array2::<i32>::zeros((1, 1));
        let inferred = Array2::<i32>::zeros((1, 1));
        let err = align_with_hungarian(truth.view(), inferred.view(), MAX_BRUTE_FORCE_STATES + 1)
            .expect_err("over cap");
        assert!(format!("{err:#}").contains("Hungarian"));
    }

    #[test]
    fn single_state_yields_trivial_perfect_accuracy() {
        // K = 1 — there's only one permutation (identity), every timestep is
        // by definition correctly classified.
        let truth = array![[0, 0, 0, 0]];
        let inferred = array![[0, 0, 0, 0]];
        let report = align_with_hungarian(truth.view(), inferred.view(), 1).unwrap();
        assert_eq!(report.accuracy, 1.0);
        assert_eq!(report.permutation, vec![0]);
        assert_eq!(report.correct, 4);
        assert_eq!(report.total, 4);
    }

    #[test]
    fn empty_input_rejected() {
        // Empty input has nothing to score against; bail with a clear error
        // rather than reporting accuracy = 0/0 = NaN.
        let truth = Array2::<i32>::zeros((0, 4));
        let inferred = Array2::<i32>::zeros((0, 4));
        let err = align_with_hungarian(truth.view(), inferred.view(), 3)
            .expect_err("empty input must fail");
        assert!(format!("{err:#}").contains("no scored timesteps"));
    }
}
