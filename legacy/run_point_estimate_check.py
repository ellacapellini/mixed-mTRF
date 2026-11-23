"""
run_point_estimate_check.py
============================

Section 7.1's "fat point estimate sanity check, before any of the
expensive fits" -- for Model B specifically. This is the thing to run
BEFORE re-running the full 4-hour NUTS fits after the jitter fix.

WHY THIS EXISTS: the seven-model comparison (compare.py) showed every
Model B NUTS fit failing to converge (divergences, Rhat > 1.01, ESS <
100), with SE-B1 landing BELOW the no-surprisal baseline. Two possible
explanations:

  (a) the sampler is fighting bad geometry (near-singular covariance
      at short lengthscales on a dense 91-lag grid) -- a numerical/
      jitter problem, fixable without touching the model or data, OR
  (b) something upstream is actually wrong (design matrix, ground
      truth, kernel implementation) and NUTS is just the first place
      it became visible.

This script re-fits the SAME shared synthetic dataset with GPmTRF's
non-hierarchical MLE point estimate (gp_mtrf.py) -- no NUTS, no
hierarchy, just per-unit empirical-Bayes optimisation of (output_scale,
length_scale) via L-BFGS-B on the log marginal likelihood. If this
recovers sensible kernel hyperparameters and decent held-out r for
BOTH kernels, that isolates the fault to (a): the sampler/jitter, not
the model or data. If it ALSO fails for SE, that points to (b) instead
-- and no amount of jitter tuning in bayesian_gp_trf.py will fix that.

Uses compare.py's build_shared_dataset / split_train_test directly, so
this is fit to the exact same data the seven-model comparison uses --
not a fresh synthetic draw that could mask or introduce a different
issue.
"""

from __future__ import annotations

import numpy as np

from compare import build_shared_dataset, split_train_test
from models.gp_mtrf import GPmTRF
from utils.kernels import Matern52, SquaredExponential
from metrics import pearson_r


def run_point_estimate_check(
    n_patients: int = 4,
    n_electrodes: int = 3,
    n_times: int = 1000,
    n_basis: int = 6,
    seed: int = 0,
):
    print("Building shared dataset (same as compare.py)...")
    data = build_shared_dataset(
        n_patients=n_patients, n_electrodes=n_electrodes, n_times=n_times,
        n_basis=n_basis, seed=seed,
    )
    train, test = split_train_test(data)
    taus = data["taus"]
    n_units = data["n_units"]

    for kernel_type, kernel_pair in [
        ("matern52", (Matern52(1.0, 50.0), Matern52(1.0, 50.0))),
        ("squared_exponential", (SquaredExponential(1.0, 50.0), SquaredExponential(1.0, 50.0))),
    ]:
        print(f"\n=== {kernel_type} ===")
        r_list = []
        ell0_list, ell1_list = [], []
        for i in range(n_units):
            X_train = np.concatenate([train["S0_raw"][i], train["S1_raw"][i]], axis=1)
            y_train = train["r"][i]
            X_test = np.concatenate([test["S0_raw"][i], test["S1_raw"][i]], axis=1)
            y_test = test["r"][i]

            k0, k1 = kernel_pair[0].__class__(1.0, 50.0), kernel_pair[1].__class__(1.0, 50.0)
            gp = GPmTRF(lags=taus, kernels=[k0, k1], noise_var=1.0)
            gp.optimise_hyperparams(X_train, y_train, verbose=False)

            y_pred = gp.predict(X_test)
            r = pearson_r(y_test, y_pred)
            r_list.append(r)
            ell0_list.append(k0.length_scale)
            ell1_list.append(k1.length_scale)

            print(f"  unit {i:2d}: r={r:+.3f}  ell_h0={k0.length_scale:7.2f}  "
                  f"ell_h1={k1.length_scale:7.2f}  sigma_h0={k0.output_scale:.3f}  "
                  f"sigma_h1={k1.output_scale:.3f}")

        r_arr = np.array(r_list)
        print(f"  --- {kernel_type}: mean r={r_arr.mean():.4f}  "
              f"min r={r_arr.min():.4f}  max r={r_arr.max():.4f}")
        print(f"  --- recovered length_scale: h0 mean={np.mean(ell0_list):.2f}, "
              f"h1 mean={np.mean(ell1_list):.2f} (grid spans "
              f"{taus[0]:.0f}-{taus[-1]:.0f} at dt={taus[1]-taus[0]:.0f})")

    print("\nHow to read this:")
    print("  - If BOTH kernels recover r well above the no-surprisal baseline (~0.65)")
    print("    and reasonable, non-degenerate length_scales, the fault in compare.py's")
    print("    NUTS fits is very likely the sampler/jitter geometry (fixable) -- proceed")
    print("    to re-run bayesian_gp_trf.py with the relative-jitter fix.")
    print("  - If SE specifically comes back with near-zero or wildly unstable")
    print("    length_scale (e.g. collapsing toward the grid spacing dt), that's the")
    print("    same near-singular-covariance signature showing up even without NUTS --")
    print("    still consistent with (a), just visible in the L-BFGS-B optimisation too.")
    print("  - If accuracy is poor for BOTH kernels even at the point-estimate level,")
    print("    stop and check design_matrix.py / ground_truth.py before re-running NUTS.")


if __name__ == "__main__":
    run_point_estimate_check()