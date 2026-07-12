from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

from basis.basis_functions import make_raised_cosine_basis
from ground_truth import (
    make_stage1_static,
    make_stage2_amplitude_only,
    make_stage3_full_modulation,
)
from data.simulate import simulate_recording
from utils.design_matrix import DesignMatrix
from models.mixed_trf import fit_ridge_mixed_trf
from metrics import train_test_split_contiguous, tf_parameter_recovery_error
from derived_quantities import trf_at_surprisal, surprisal_sweep


def run_stage(name, gt, basis, dt, n_times, rng, alpha=1.0, snr_target=6.0, noise_type="white"):
    rec = simulate_recording(basis, gt, n_times, dt, rng, snr_target=snr_target, noise_type=noise_type)
    train_idx, test_idx = train_test_split_contiguous(n_times, test_fraction=0.2)

    d_train = DesignMatrix(
        S0=rec.design.S0[train_idx], S1=rec.design.S1[train_idx],
        X=rec.design.X[train_idx],
    )
    d_test = DesignMatrix(
        S0=rec.design.S0[test_idx], S1=rec.design.S1[test_idx],
        X=rec.design.X[test_idx],
    )
    r_train, r_test = rec.r[train_idx], rec.r[test_idx]

    fit = fit_ridge_mixed_trf(d_train, r_train, d_test, r_test, n_basis=basis.n_basis, alpha=alpha)

    mu_err = tf_parameter_recovery_error(gt.mu, fit.mu_hat)
    beta_err = tf_parameter_recovery_error(gt.beta, fit.beta_hat)

    print(f"\n[{name}]")
    print(f"  held-out Pearson r  : {fit.r_test:.3f}")
    print(f"  mu  recovery  nRMSE : {mu_err['normalised_rmse']:.3f}  (corr={mu_err['correlation']:.3f})")
    if np.abs(gt.beta).sum() > 1e-9:
        print(f"  beta recovery nRMSE : {beta_err['normalised_rmse']:.3f}  (corr={beta_err['correlation']:.3f})")
    else:
        print(f"  beta recovery raw RMSE : {beta_err['rmse']:.4f}  (true beta = 0, nRMSE undefined)")

    return {
        "name": name, "gt": gt, "fit": fit, "rec": rec,
        "mu_err": mu_err, "beta_err": beta_err,
    }


def main():
    rng = np.random.default_rng(123)
    basis = make_raised_cosine_basis(n_basis=10, tau_max=600.0, c=5.0)
    dt = 5.0
    n_times = 24000  # 120 s @ 200 Hz-equivalent (dt=5ms)
    tau = np.linspace(0, 600, 1201)
    surp_levels = np.linspace(0, 6, 25)

    gt1 = make_stage1_static(basis, rng)
    gt2 = make_stage2_amplitude_only(basis, rng, gain=-0.3)
    gt3 = make_stage3_full_modulation(basis, rng)

    results = [
        run_stage("Stage 1: static TRF recovery", gt1, basis, dt, n_times, rng),
        run_stage("Stage 2: amplitude-only modulation (negative control)", gt2, basis, dt, n_times, rng),
        run_stage("Stage 3: full amplitude-latency-scale modulation (positive control)", gt3, basis, dt, n_times, rng),
    ]

    #fig 1: true vs recovered TRF shape (mu, i.e. surp=0 baseline)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, res in zip(axes, results):
        h_true = trf_at_surprisal(res["gt"].mu, res["gt"].beta, basis, 0.0, tau)
        h_hat = trf_at_surprisal(res["fit"].mu_hat, res["fit"].beta_hat, basis, 0.0, tau)
        ax.plot(tau, h_true, 'k-', lw=2, label="true h(tau | surp=0)")
        ax.plot(tau, h_hat, 'r--', lw=1.8, label="ridge-recovered")
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_title(res["name"].split(":")[0], fontsize=10)
        ax.set_xlabel("lag (ms)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("h(tau)")
    plt.tight_layout()
    plt.savefig("figures/02_true_vs_recovered_trf.png", dpi=130)
    plt.close(fig)

    #fig 2: amplitude / latency / scale vs surprisal, true vs recovered
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharex=True)
    row_labels = ["amplitude", "latency", "scale"]
    for col, res in zip([0, 1], [results[1], results[2]]):  # stage 2, stage 3
        sweep_true = surprisal_sweep(res["gt"].mu, res["gt"].beta, basis, surp_levels, tau)
        sweep_hat = surprisal_sweep(res["fit"].mu_hat, res["fit"].beta_hat, basis, surp_levels, tau)
        for row, key in enumerate(row_labels):
            ax = axes[row, col]
            ax.plot(surp_levels, sweep_true[key], 'k-', lw=2, label="true")
            ax.plot(surp_levels, sweep_hat[key], 'r--', lw=1.8, label="recovered")
            if row == 0:
                ax.set_title(res["name"].split(":")[0] + "\n" + res["name"].split(":")[1].strip(), fontsize=9)
            if col == 0:
                ax.set_ylabel(key)
            if row == 2:
                ax.set_xlabel("surprisal (bits)")
            ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("figures/03_amplitude_latency_scale_vs_surprisal.png", dpi=130)
    plt.close(fig)

    print("\nSaved: figures/02_true_vs_recovered_trf.png")
    print("Saved: figures/03_amplitude_latency_scale_vs_surprisal.png")

    #fnal assertions
    for res in results:
        assert res["fit"].r_test > 0.85, f"{res['name']}: held-out r too low ({res['fit'].r_test})"
        assert res["mu_err"]["normalised_rmse"] < 0.2, f"{res['name']}: mu recovery too poor"

    # Stage-2-specific: latency/scale should be near-flat in BOTH true and recovered sweeps (- control should stay -).
    sweep2_hat = surprisal_sweep(results[1]["fit"].mu_hat, results[1]["fit"].beta_hat, basis, surp_levels, tau)
    lat_range2_hat = np.nanmax(sweep2_hat["latency"]) - np.nanmin(sweep2_hat["latency"])
    assert lat_range2_hat < 15.0, (
        f"stage 2 recovered latency should stay ~flat, got range {lat_range2_hat:.1f}ms"
    )

    # Stage-3-specific: latency should shift meaningfully in the recovered fit too, not just in the ground truth (+ control should stay + after fitting).
    sweep3_hat = surprisal_sweep(results[2]["fit"].mu_hat, results[2]["fit"].beta_hat, basis, surp_levels, tau)
    lat_range3_hat = np.nanmax(sweep3_hat["latency"]) - np.nanmin(sweep3_hat["latency"])
    assert lat_range3_hat > 30.0, (
        f"stage 3 recovered latency shift too small: {lat_range3_hat:.1f}ms"
    )

    print(f"\nStage 2 recovered latency range: {lat_range2_hat:.1f} ms (expect ~flat)")
    print(f"Stage 3 recovered latency range: {lat_range3_hat:.1f} ms (expect a real shift)")
    print("\nALL VALIDATION CHECKS PASSED (white noise).")
    print("Ridge baseline correctly: (a) recovers static TRF shape, (b) recovers")
    print("amplitude-only modulation without hallucinating a latency shift, and")
    print("(c) recovers a genuine latency shift when the ground truth has one.")

    #robustness check: does recovery hold up under realistic 1/f (pink)observation noise, not just idealised white noise? 
    print("\n--- Robustness check: white noise vs. realistic pink (1/f) noise ---")
    rng_pink = np.random.default_rng(123)
    pink_results = []
    for name, gt in [("Stage 1", gt1), ("Stage 2", gt2), ("Stage 3", gt3)]:
        r = run_stage(f"{name} (pink noise)", gt, basis, dt, n_times, rng_pink, noise_type="pink")
        pink_results.append(r)

    for white_res, pink_res, name in zip(results, pink_results, ["Stage 1", "Stage 2", "Stage 3"]):
        white_r, pink_r = white_res["fit"].r_test, pink_res["fit"].r_test
        print(f"{name}: held-out r = {white_r:.3f} (white) vs {pink_r:.3f} (pink)")
        assert pink_r > 0.85, f"{name} under pink noise: held-out r too low ({pink_r:.3f})"

    print("\nRecovery holds under realistic 1/f noise, not just idealised white noise.")


if __name__ == "__main__":
    main()