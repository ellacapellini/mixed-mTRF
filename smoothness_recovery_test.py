"""
smoothness_recovery_test.py
===========================
Can the hyperparameter hierarchy (B-pop-hyp) recover a KNOWN patient-level difference in smoothness?

The testbed gives each patient a different response-width scale (--patient_width_sd). If the hierarchy works,
patients with broader responses should get longer posterior deviation length scales (positive correlation).
Run the amplitude-pooled version too (B-pop-amp): it has ONE length scale, so it cannot show this -- that
contrast is what the extra machinery has to earn.

Put in the project root (next to erp_testbed.py). Examples (Euler):
  python smoothness_recovery_test.py --hierarchy hyperparameter --draws 1000 --tune 1500 --chains 4
  python smoothness_recovery_test.py --hierarchy amplitude      --draws 1000 --tune 1500 --chains 4
Measured in a small sandbox (8 units, 81 lags, 1 core): ~0.4 s/iteration (amplitude) vs ~2.8 s/iteration
(hyperparameter), tree depth at its maximum early on. Expect hours per chain at 32 units: request >= 24 h,
4 CPUs, 16 GB, and calibrate with one task first.
"""
import argparse, json, time
import numpy as np
import arviz as az
import erp_testbed as T
from models.bayesian_pop_trf import prepare_suffstats, fit_pop_trf

ap = argparse.ArgumentParser()
ap.add_argument("--hierarchy", choices=["amplitude", "hyperparameter"], default="hyperparameter")
ap.add_argument("--kind", default="matern52", choices=["matern52", "squared_exponential"])
ap.add_argument("--n_patients", type=int, default=6)
ap.add_argument("--n_electrodes", type=int, default=2)
ap.add_argument("--n_times", type=int, default=24000)
ap.add_argument("--snr", type=float, default=0.6)
ap.add_argument("--patient_width_sd", type=float, default=0.45)
ap.add_argument("--draws", type=int, default=1000)
ap.add_argument("--tune", type=int, default=1500)
ap.add_argument("--chains", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=None)
a = ap.parse_args()

L = 81
taus = np.arange(L) * T.DT
rng = np.random.default_rng(a.seed)
units = T.make_dataset(rng, n_patients=a.n_patients, n_electrodes=a.n_electrodes, n_times=a.n_times, taus=taus,
                       mode="null", effect_frac=1.0, effect_on="surprisal", shared=False, snr=a.snr, noise="white",
                       common_frac=0.0, rho=0.0, surp_mean=7.0, surp_sd=3.0, patient_width_sd=a.patient_width_sd)
ss = prepare_suffstats([T.unit_design(u, a.n_times, L, ["surp"]) for u in units], [u["r"] for u in units])
pat = np.array([u["patient"] for u in units])
wtrue = np.array([u["wscale"] for u in units])

t0 = time.time()
res = fit_pop_trf(ss, pat, taus, kind=a.kind, hierarchy=a.hierarchy, draws=a.draws, tune=a.tune, chains=a.chains, seed=a.seed)
tr = res.trace
out = {"hierarchy": a.hierarchy, "kind": a.kind, "seed": a.seed, "elapsed_s": time.time() - t0,
       "n_divergences": int(tr.sample_stats["diverging"].sum()),
       "max_rhat_h_pop": float(az.rhat(tr, var_names=["h_pop"])["h_pop"].max())}
if a.hierarchy == "hyperparameter":
    ell = np.asarray(tr.posterior["ell_unit_0"].values)               # (chain, draw, unit)
    ell_mean = ell.mean(axis=(0, 1))
    out["corr_log_ell_vs_log_true_width"] = float(np.corrcoef(np.log(ell_mean), np.log(wtrue))[0, 1])
    out["max_rhat_ell_unit_0"] = float(az.rhat(tr, var_names=["ell_unit_0"])["ell_unit_0"].max())
    out["per_patient_ell_vs_true_width"] = [[float(wtrue[pat == p][0]), float(ell_mean[pat == p].mean())]
                                            for p in range(a.n_patients)]
print(json.dumps(out, indent=1))
if a.out:
    with open(a.out, "a") as f:
        f.write(json.dumps(out) + "\n")