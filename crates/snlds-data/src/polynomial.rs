//! Polynomial feature expansion matching scikit-learn `PolynomialFeatures`
//! (`interaction_only=False`, `include_bias=True`, single `degree` int).
//!
//! Order follows [`PolynomialFeatures._combinations`] with
//! `combinations_with_replacement` (see sklearn `preprocessing/_polynomial.py`).

use itertools::Itertools;

/// Binomial coefficient C(n, k) (sklearn / scipy `comb` with `exact=True`).
pub fn comb(n: usize, k: usize) -> u64 {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut num = 1u64;
    let mut den = 1u64;
    for i in 0..k {
        num *= (n - i) as u64;
        den *= (i + 1) as u64;
    }
    num / den
}

/// Number of output features =
/// `C(n_features + max_degree, max_degree)` for sklearn defaults
/// (`min_degree=0`, `interaction_only=False`, `include_bias=True`).
#[inline]
pub fn sklearn_poly_output_count(dim_latent: usize, degree: usize) -> usize {
    comb(dim_latent + degree, degree) as usize
}

/// Exponent vectors `powers_[i]` (length `n_features`), same as `PolynomialFeatures.powers_`.
pub fn sklearn_powers(n_features: usize, max_degree: usize) -> Vec<Vec<u32>> {
    let mut out = Vec::with_capacity(sklearn_poly_output_count(n_features, max_degree));
    // bias: combinations_with_replacement(range(n_features), 0) -> ()
    out.push(vec![0u32; n_features]);

    for k in 1..=max_degree {
        for tup in (0..n_features).combinations_with_replacement(k) {
            out.push(bincount_indices(&tup, n_features));
        }
    }
    out
}

fn bincount_indices(indices: &[usize], len: usize) -> Vec<u32> {
    let mut v = vec![0u32; len];
    for &ix in indices {
        v[ix] += 1;
    }
    v
}

/// One row `transform` for dense `PolynomialFeatures`-equivalent expansion.
#[inline]
pub fn expand_polynomial_row(x: &[f32], powers: &[Vec<u32>]) -> Vec<f32> {
    let n = powers.len();
    let mut row = Vec::with_capacity(n);
    for p in powers {
        let mut prod = 1.0f32;
        for (xi, pi) in x.iter().zip(p.iter()) {
            prod *= xi.powi(*pi as i32);
        }
        row.push(prod);
    }
    row
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comb_matches_scipy_math() {
        assert_eq!(comb(5, 3), 10);
        assert_eq!(comb(6, 6), 1);
        assert_eq!(comb(10, 2), 45);
        assert_eq!(sklearn_poly_output_count(2, 3), 10);
    }

    /// Degree-2 polynomial on `[a,b]` yields `[1, a, b, a², ab, b²]` (example in sklearn docs).
    #[test]
    fn sklearn_example_degree2_two_features() {
        let powers = sklearn_powers(2, 2);
        assert_eq!(powers.len(), 6);

        let a = 3.0f32;
        let b = 4.0f32;
        let x = vec![a, b];
        let row = expand_polynomial_row(&x, &powers);

        assert!((row[0] - 1.0).abs() < 1e-6);
        assert!((row[1] - a).abs() < 1e-6);
        assert!((row[2] - b).abs() < 1e-6);
        assert!((row[3] - a * a).abs() < 1e-6);
        assert!((row[4] - a * b).abs() < 1e-6);
        assert!((row[5] - b * b).abs() < 1e-6);
    }
}
