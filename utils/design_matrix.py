"""
design_matrix.py
=================

Builds the basis-projected stimulus regressors from Section 5:

    s0_j(t) = sum_tau  phi_j(tau) * s(t - tau)
    s1_j(t) = sum_tau  phi_j(tau) * surp[i(t - tau)] * s(t - tau)

This is the module that encodes the fix to the original draft. The
original (wrong) version multiplied the whole convolved regressor
s_tilde_j(t) by surp[i(t)] -- the surprisal of the word most recently
heard at the *response* time t. That's wrong because s_tilde_j(t) is
already a convolution mixing stimulus energy from several preceding
words (any tau up to tau_max=800ms), so surprisal has to be looked up
at the LAG-SHIFTED time (t - tau), i.e. the word that was actually
driving the stimulus at that lag, not the most recent word overall.

s1_j(t) is exactly the same convolution as s0_j(t), except each sample
of the stimulus is pre-weighted by the surprisal of whichever word was
active at that particular lag before being convolved with phi_j.

Depends on: basis_functions.py
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from basis.basis_functions import RaisedCosineBasis


@dataclass(frozen=True)
class DesignMatrix:
    """
    Container for a built design matrix.

    Attributes
    ----------
    S0 : np.ndarray, shape (n_times, n_basis)
        Plain basis-projected stimulus, s0_j(t).
    S1 : np.ndarray, shape (n_times, n_basis)
        Surprisal-weighted basis-projected stimulus, s1_j(t).
    X : np.ndarray, shape (n_times, 2 * n_basis)
        [S0 | S1] concatenated -- what you actually hand to a
        regression / PyMC model. Columns 0..J-1 are S0, columns
        J..2J-1 are S1, matching mu_j and beta_j respectively.
    """

    S0: np.ndarray
    S1: np.ndarray
    X: np.ndarray


def build_design_matrix(
    stimulus: np.ndarray,
    surprisal_at_t: np.ndarray,
    basis: RaisedCosineBasis,
    dt: float,
) -> DesignMatrix:
    """
    build [S0 | S1] via explicit lag-by-lag convolution.
    """
    stimulus = np.asarray(stimulus, dtype=float)
    surprisal_at_t = np.asarray(surprisal_at_t, dtype=float)
    n_times = stimulus.shape[0]
    if surprisal_at_t.shape[0] != n_times:
        raise ValueError("stimulus and surprisal_at_t must have the same length")

    n_lags = int(np.floor(basis.tau_max / dt)) + 1
    tau_grid = np.arange(n_lags) * dt  # 0, dt, 2dt, ..., <= tau_max
    Phi = basis.eval(tau_grid)  #(n_lags, n_basis)
    n_basis = basis.n_basis

    S0 = np.zeros((n_times, n_basis))
    S1 = np.zeros((n_times, n_basis))

    #ausal convolution
    for k in range(n_lags):
        if k == 0:
            s_shifted = stimulus
            surp_shifted = surprisal_at_t
        else:
            s_shifted = np.concatenate([np.zeros(k), stimulus[:-k]])
            surp_shifted = np.concatenate([np.zeros(k), surprisal_at_t[:-k]])

        S0 += np.outer(s_shifted, Phi[k, :])
        S1 += np.outer(s_shifted * surp_shifted, Phi[k, :])

    X = np.concatenate([S0, S1], axis=1)
    return DesignMatrix(S0=S0, S1=S1, X=X)


def expand_word_level_to_samples(
    n_times: int,
    dt: float,
    word_onset_times: np.ndarray,
    word_surprisal: np.ndarray,
) -> np.ndarray:
    """
    expand word-level surprisal
    """
    word_onset_times = np.asarray(word_onset_times, dtype=float)
    word_surprisal = np.asarray(word_surprisal, dtype=float)
    if word_onset_times.shape != word_surprisal.shape:
        raise ValueError("word_onset_times and word_surprisal must match in shape")
    if np.any(np.diff(word_onset_times) <= 0):
        raise ValueError("word_onset_times must be strictly increasing")

    sample_times = np.arange(n_times) * dt
    
    word_idx = np.searchsorted(word_onset_times, sample_times, side="right") - 1

    surp_at_t = np.zeros(n_times)
    valid = word_idx >= 0
    surp_at_t[valid] = word_surprisal[word_idx[valid]]
    return surp_at_t



# Self-tests
def _self_test() -> None:
    from basis.basis_functions import make_raised_cosine_basis

    dt = 10.0  # ms
    n_times = 500
    basis = make_raised_cosine_basis(n_basis=6, tau_max=200.0, c=5.0)

    #Test 1: word-level expansion is correct step function
    word_onsets = np.array([0.0, 100.0, 250.0, 400.0])
    word_surp = np.array([1.0, 5.0, 2.0, 8.0])
    surp_t = expand_word_level_to_samples(n_times, dt, word_onsets, word_surp)
    # sample at t=50ms (index 5) should belong to word 0 (surp=1.0)
    assert surp_t[5] == 1.0
    # sample at t=150ms (index 15) should belong to word 1 (surp=5.0)
    assert surp_t[15] == 5.0
    assert surp_t[45] == 8.0
    print("design_matrix.py: word-level expansion test passed.")

    #Test 2: S0 with a single impulse recovers the basis shape

    stimulus = np.zeros(n_times)
    stimulus[0] = 1.0
    zero_surp = np.zeros(n_times)
    dm = build_design_matrix(stimulus, zero_surp, basis, dt)

    n_lags = int(np.floor(basis.tau_max / dt)) + 1
    tau_grid = np.arange(n_lags) * dt
    Phi_expected = basis.eval(tau_grid)
    np.testing.assert_allclose(dm.S0[:n_lags, :], Phi_expected, atol=1e-10)
    print("design_matrix.py: impulse-response (S0 == basis) test passed.")

    #Test 3: S1 with constant surprisal equals surp * S0
    const_surp = np.full(n_times, 3.0)
    dm_const = build_design_matrix(stimulus, const_surp, basis, dt)
    np.testing.assert_allclose(dm_const.S1, 3.0 * dm_const.S0, atol=1e-10)
    print("design_matrix.py: constant-surprisal (S1 == c*S0) test passed.")

    #test 4: THE key regression test for the Section-5 fix 
    stimulus2 = np.zeros(n_times)
    stimulus2[0] = 1.0
    stimulus2[20] = 1.0
    word_onsets2 = np.array([0.0, 150.0])
    word_surp2 = np.array([10.0, 1.0])
    surp_t2 = expand_word_level_to_samples(n_times, dt, word_onsets2, word_surp2)

    dm2 = build_design_matrix(stimulus2, surp_t2, basis, dt)

    t_idx = 20
    expected_s1_at_t = (
        Phi_expected[0, :] * 1.0 * 1.0  # tau=0: surp[i(200)]=1 (word B), s=1
        + Phi_expected[20, :] * 10.0 * 1.0  # tau=200: surp[i(0)]=10 (word A), s=1
    )
    np.testing.assert_allclose(dm2.S1[t_idx, :], expected_s1_at_t, atol=1e-10)

    
    wrong_s1_at_t = surp_t2[t_idx] * dm2.S0[t_idx, :]  # old (buggy) formula
    assert not np.allclose(wrong_s1_at_t, expected_s1_at_t), (
        "test case doesn't actually distinguish correct vs buggy indexing "
        "-- strengthen the test"
    )
    print("design_matrix.py: lag-shifted surprisal indexing (Section-5 fix) test passed.")
    print(f"  correct S1 at t=200ms: {np.round(expected_s1_at_t, 4)}")
    print(f"  buggy (response-time-indexed) S1 would have been: {np.round(wrong_s1_at_t, 4)}")

    # test 5: shape/dimension sanity
    assert dm2.X.shape == (n_times, 2 * basis.n_basis)
    np.testing.assert_allclose(dm2.X[:, : basis.n_basis], dm2.S0)
    np.testing.assert_allclose(dm2.X[:, basis.n_basis :], dm2.S1)
    print("design_matrix.py: all self-tests passed.")


if __name__ == "__main__":
    _self_test()