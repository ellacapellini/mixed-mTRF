"""
erp_testbed.py
==============
A harder, more realistic synthetic testbed, plus the analyses that answer
"can I tell gain from latency from width, and is it surprisal at all?".

What is different from the earlier synthetic data
-------------------------------------------------
* 800 ms window (the N400 lives at ~300-500 ms; a 200 ms window cannot see it).
* ERP-shaped ground truth: onset-anchored N1 / P2 / N400 (sums of Gaussians),
  not a stationary GP draw. Surprisal modulates ONLY the N400 (like the
  literature), through one of three KNOWN mechanisms, per z-scored word:
      gain   : N400 amplitude *= (1 + g z)
      latency: N400 peak latency += shift_ms * z
      width  : N400 width *= exp(omega z)
  This truth is NONLINEAR in z for shift/width, so the linear-in-surprisal
  models are mildly misspecified, as they will be on real data.
* Word-level: the stimulus is word-onset impulses (the ERP/rERP framing),
  every word has its own surprisal, responses overlap (words every ~330 ms,
  window 800 ms).
* Realistic-ish noise: white or 1/f, optional component shared across
  electrodes within a patient (volume conduction / common reference),
  and a REALISTIC SNR (default 0.2 => max attainable r ~ 0.4; the earlier
  SNR=5 corresponds to r ~ 0.9+, which real data never reach).
* Optional shared stimulus (all units hear the same words).
* Optional CONFOUND (e.g. word frequency) correlated with surprisal, with
  the true effect assigned to either surprisal or the confound.
* Optional heterogeneity: only a fraction of electrodes carry the effect.

Analyses
--------
* per-unit ridge (lambda by contiguous CV): h0, h1 for each electrode;
* gain / shift / width decomposition (first-order, no argmax):
      h1 ~ g*h0 + shift*(-h0') + omega*(-(tau - tc) h0')   within the N400 window
  with a PATIENT-cluster bootstrap (the real sample size is patients);
* Section-3-style argmax latency, for comparison;
* leave-one-patient-out prediction: M0 (no surprisal) vs M1 (+surprisal)
  vs M2 (+confound).

Run:  python erp_testbed.py --modes null gain latency width --reps 3
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter1d

DT = 10.0  # ms per sample
COMPONENTS = {              # name: (latency ms, sd ms, amplitude)
    "N1": (100.0, 25.0, -1.0),
    "P2": (200.0, 40.0, +0.8),
    "N400": (400.0, 80.0, -1.2),
}
STRENGTH = {"gain": 0.30, "shift": 25.0, "width": 0.25}   # per 1 SD of the driver


# ------------------------------------------------------------------ truth
def effect_for(mode, scale=1.0):
    on = {"null": [], "gain": ["gain"], "latency": ["shift"], "width": ["width"],
          "all": ["gain", "shift", "width"]}[mode]
    return {k: (STRENGTH[k] * scale if k in on else 0.0) for k in STRENGTH}


def erp(taus, z, eff, amp_scale=1.0, shift=0.0, wscale=1.0):
    """h(tau | z) for ONE word. Onset-anchored; z is the (z-scored) driver."""
    out = np.zeros_like(taus, dtype=float)
    for name, (t, sd, a) in COMPONENTS.items():
        aa, tt, ss = a, t + shift, sd * wscale
        if name == "N400":
            aa = a * max(1.0 + eff["gain"] * z, 0.05)      # never flips sign
            tt = t + shift + eff["shift"] * z
            ss = sd * wscale * np.exp(eff["width"] * z)
        out += aa * np.exp(-0.5 * ((taus - tt) / ss) ** 2)
    return amp_scale * out


# ------------------------------------------------------------- simulation
def make_words(n_times, rng, rate=3.0, surp_mean=7.0, surp_sd=3.0, rho=0.0):
    """Word onsets (samples) with gamma-distributed surprisal (bits).
    surp_mean/sd are ASSUMPTIONS -- replace with your GPT-2 values."""
    mean_isi = 1000.0 / rate
    isi = 150.0 + rng.gamma(4.0, (mean_isi - 150.0) / 4.0, size=int(n_times * DT / mean_isi * 2) + 10)
    onsets = np.round(np.cumsum(isi) / DT).astype(int)
    onsets = onsets[onsets < n_times - 1]
    k, scale = (surp_mean / surp_sd) ** 2, surp_sd ** 2 / surp_mean
    surp = rng.gamma(k, scale, size=len(onsets))
    z = (surp - surp_mean) / surp_sd                         # centred by a GLOBAL constant
    conf = rho * z + np.sqrt(max(1 - rho ** 2, 0.0)) * rng.normal(size=len(z))
    return {"onsets": onsets, "surp": surp, "z": z, "conf": conf}


def make_noise(n_times, rng, kind):
    x = rng.normal(size=n_times)
    if kind == "pink":
        f = np.fft.rfftfreq(n_times)
        f[0] = f[1]
        x = np.fft.irfft(np.fft.rfft(x) / np.sqrt(f), n=n_times)
    return (x - x.mean()) / x.std()


def make_dataset(rng, *, n_patients, n_electrodes, n_times, taus, mode, effect_frac, effect_on,
                 shared, snr, noise, common_frac, rho, surp_mean, surp_sd, patient_width_sd=0.0):
    n_lags = len(taus)
    shared_words = make_words(n_times, rng, rho=rho, surp_mean=surp_mean, surp_sd=surp_sd) if shared else None
    units = []
    for p in range(n_patients):
        p_scale, p_shift = np.exp(rng.normal(0, 0.15)), rng.normal(0, 8.0)
        p_w = float(np.exp(rng.normal(0, patient_width_sd))) if patient_width_sd > 0 else 1.0
        common = make_noise(n_times, rng, noise)
        for e in range(n_electrodes):
            words = shared_words or make_words(n_times, rng, rho=rho, surp_mean=surp_mean, surp_sd=surp_sd)
            has = rng.random() < effect_frac
            eff = effect_for(mode if has else "null")
            drive = words["z"] if effect_on == "surprisal" else words["conf"]
            e_scale = np.exp(rng.normal(0, 0.25))
            sig = np.zeros(n_times)
            for o, zz in zip(words["onsets"], drive):
                h = erp(taus, zz, eff, p_scale * e_scale, p_shift, p_w)
                end = min(n_times, o + n_lags)
                sig[o:end] += h[: end - o]
            nz = np.sqrt(1 - common_frac) * make_noise(n_times, rng, noise) + np.sqrt(common_frac) * common
            nz = nz / nz.std() * np.sqrt(sig.var() / snr)
            units.append({"patient": p, "wscale": p_w, "words": words, "r": sig + nz, "signal": sig, "has_effect": has})
    return units


# ------------------------------------------------------------------ fitting
def lag_design(onsets, values, n_times, n_lags):
    """Lagged word-onset design as a SPARSE matrix: S[t, j] = value of the word whose onset is j samples before t."""
    rows, cols, vals = [], [], []
    for j in range(n_lags):
        idx = onsets + j
        ok = idx < n_times
        rows.append(idx[ok]); cols.append(np.full(int(ok.sum()), j)); vals.append(values[ok])
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(n_times, n_lags))


def unit_design(u, n_times, n_lags, mods):
    w = u["words"]
    cols = [lag_design(w["onsets"], np.ones(len(w["onsets"])), n_times, n_lags)]
    for m in mods:
        cols.append(lag_design(w["onsets"], w[{"surp": "z", "conf": "conf"}[m]], n_times, n_lags))
    return sp.hstack(cols, format="csr")


def _blocks(n, k):
    return np.array_split(np.arange(n), k)


def ridge_cv_c(X, y, cs=np.logspace(-3, 1, 9), k=5):
    """Pick relative ridge strength c (lambda = c * tr(X'X)/p) by contiguous k-fold CV."""
    n, p = X.shape
    XtX, Xty = (X.T @ X).toarray(), X.T @ y
    scale = np.trace(XtX) / p
    folds = _blocks(n, k)
    blk = [((X[b[0]:b[-1] + 1].T @ X[b[0]:b[-1] + 1]).toarray(), X[b[0]:b[-1] + 1].T @ y[b], b) for b in folds]
    best, best_err = cs[0], np.inf
    for c in cs:
        err = 0.0
        for xx, xy, b in blk:
            w = np.linalg.solve(XtX - xx + c * scale * np.eye(p), Xty - xy)
            err += np.sum((y[b] - X[b[0]:b[-1] + 1] @ w) ** 2)
        if err < best_err:
            best, best_err = c, err
    return best


def ridge_fit(XtX, Xty, c):
    p = XtX.shape[0]
    return np.linalg.solve(XtX + c * np.trace(XtX) / p * np.eye(p), Xty)


# ------------------------------------------------------------- decomposition
def decompose(h0, h1, taus, window=(250.0, 600.0), center=None, smooth=1.0):
    """First-order gain / shift / width of h1 relative to h0, inside `window`.
    h(tau|z) ~ h0 + z*h1,  h1 ~ g*h0 + shift*(-h0') + omega*(-(tau-tc)*h0').
    Returns (g per z, shift in ms per z, omega = log-width change per z)."""
    h0s, h1s = gaussian_filter1d(h0, smooth), gaussian_filter1d(h1, smooth)
    d0 = np.gradient(h0s, taus)
    sel = (taus >= window[0]) & (taus <= window[1])
    tc = center if center is not None else taus[sel][np.argmax(np.abs(h0s[sel]))]
    B = np.stack([h0s, -d0, -(taus - tc) * d0], axis=1)[sel]
    coef, *_ = np.linalg.lstsq(B, h1s[sel], rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2]), float(tc)


def argmax_latency_shift(h0, h1, taus, window=None):
    """ms between the |h| peak at +1 SD and at -1 SD. window=None: global argmax over all lags
    (what Section 3 does). window=(lo, hi): argmax restricted to a pre-specified component window."""
    sel = np.ones_like(taus, bool) if window is None else (taus >= window[0]) & (taus <= window[1])
    lat = lambda s: taus[sel][np.argmax(np.abs((h0 + s * h1)[sel]))]
    return float(lat(+1.0) - lat(-1.0))


# --------------------------------------------------------------------- run
def run_scenario(args, mode, seed):
    rng = np.random.default_rng(seed)
    n_lags = int(args.tau_max / DT) + 1
    taus = np.arange(n_lags) * DT
    units = make_dataset(
        rng, n_patients=args.n_patients, n_electrodes=args.n_electrodes, n_times=args.n_times,
        taus=taus, mode=mode, effect_frac=args.effect_frac, effect_on=args.effect_on,
        shared=args.shared_stimulus, snr=args.snr, noise=args.noise, common_frac=args.common_noise,
        rho=args.confound_rho, surp_mean=args.surp_mean, surp_sd=args.surp_sd,
        patient_width_sd=getattr(args, "patient_width_sd", 0.0))

    mods_sets = {"M0": [], "M1": ["surp"], "M2": ["surp", "conf"]}
    cache = {}
    for m, mods in mods_sets.items():
        for i, u in enumerate(units):
            X = unit_design(u, args.n_times, n_lags, mods)
            cache[(m, i)] = ((X.T @ X).toarray(), X.T @ u["r"], X, u["r"])

    # per-unit M1 fits (lambda by CV), population = mean over units
    cs, H0, H1 = [], [], []
    for i, u in enumerate(units):
        XtX, Xty, X, y = cache[("M1", i)]
        c = ridge_cv_c(X, y)
        w = ridge_fit(XtX, Xty, c)
        cs.append(c)
        H0.append(w[:n_lags]); H1.append(w[n_lags: 2 * n_lags])
    H0, H1 = np.array(H0), np.array(H1)
    pat = np.array([u["patient"] for u in units])
    g, sh, om, tc = decompose(H0.mean(0), H1.mean(0), taus, center=args.center_ms)

    # patient-cluster bootstrap of the population decomposition
    brng = np.random.default_rng(seed + 1000)
    boots = []
    for _ in range(args.boot):
        ps = brng.choice(args.n_patients, args.n_patients, replace=True)
        idx = np.concatenate([np.where(pat == p)[0] for p in ps])
        bg, bs, bo, _ = decompose(H0[idx].mean(0), H1[idx].mean(0), taus, center=args.center_ms)
        boots.append((bg, bs, bo))
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [5, 95], axis=0)

    # Alternative interval: summary-statistics approach. One decomposition PER PATIENT,
    # then a t-interval over patients (df = P-1). Honest about n = patients.
    from scipy import stats as _st
    pg = np.array([decompose(H0[pat == q].mean(0), H1[pat == q].mean(0), taus, center=args.center_ms)[:3]
                   for q in range(args.n_patients)])
    t_est = pg.mean(0)
    t_se = pg.std(0, ddof=1) / np.sqrt(args.n_patients)
    tq = _st.t.ppf(0.95, args.n_patients - 1)
    t_lo, t_hi = t_est - tq * t_se, t_est + tq * t_se

    # M2: surprisal's OWN modulation after the confound is also in the model
    H0b, Hs = [], []
    for i in range(len(units)):
        XtX, Xty, X, y = cache[("M2", i)]
        w = ridge_fit(XtX, Xty, ridge_cv_c(X, y))
        H0b.append(w[:n_lags]); Hs.append(w[n_lags: 2 * n_lags])
    m2 = decompose(np.mean(H0b, 0), np.mean(Hs, 0), taus, center=args.center_ms)[:3]

    # leave-one-patient-out prediction, same ridge strength for every model
    c_pop = float(np.median(cs))
    r_by = {m: [] for m in mods_sets}
    for p in range(args.n_patients):
        tr = [i for i in range(len(units)) if pat[i] != p]
        te = [i for i in range(len(units)) if pat[i] == p]
        for m in mods_sets:
            XtX = sum(cache[(m, i)][0] for i in tr)
            Xty = sum(cache[(m, i)][1] for i in tr)
            w = ridge_fit(XtX, Xty, c_pop)
            r_by[m].append(np.mean([np.corrcoef(cache[(m, i)][2] @ w, units[i]["r"])[0, 1] for i in te]))
    r_by = {m: np.array(v) for m, v in r_by.items()}

    eff_true = effect_for(mode, args.effect_frac if args.effect_on == "surprisal" else 0.0)
    truth = (eff_true["gain"], eff_true["shift"], eff_true["width"])
    return {
        "mode": mode, "seed": seed, "tc_est": tc,
        "truth_g_shift_omega": truth,
        "est_g_shift_omega": [g, sh, om], "ci90_lo": lo.tolist(), "ci90_hi": hi.tolist(),
        "covered": [bool(lo[k] <= truth[k] <= hi[k]) for k in range(3)],
        "t_est": t_est.tolist(), "t_lo": t_lo.tolist(), "t_hi": t_hi.tolist(),
        "t_covered": [bool(t_lo[k] <= truth[k] <= t_hi[k]) for k in range(3)],
        "boot_halfwidth": ((hi - lo) / 2).tolist(),
        "m2_surprisal_g_shift_omega_given_confound": list(m2),
        "argmax_latency_shift_ms": argmax_latency_shift(H0.mean(0), H1.mean(0), taus),
        "argmax_window_shift_ms": argmax_latency_shift(H0.mean(0), H1.mean(0), taus, window=(250.0, 600.0)),
        "lopo_r": {m: float(v.mean()) for m, v in r_by.items()},
        "lopo_gain_M1_minus_M0": float((r_by["M1"] - r_by["M0"]).mean()),
        "lopo_frac_patients_M1_better": float(((r_by["M1"] - r_by["M0"]) > 0).mean()),
        "lopo_gain_M2_minus_M1": float((r_by["M2"] - r_by["M1"]).mean()),
        "n_units": len(units), "n_words_unit0": int(len(units[0]["words"]["onsets"])),
        "mean_signal_r_ceiling": float(np.sqrt(args.snr / (1 + args.snr))),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", nargs="+", default=["null", "gain", "latency", "width"],
                    choices=["null", "gain", "latency", "width", "all"])
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--n_patients", type=int, default=6)
    ap.add_argument("--n_electrodes", type=int, default=4)
    ap.add_argument("--n_times", type=int, default=6000)           # 60 s per unit
    ap.add_argument("--tau_max", type=float, default=800.0)
    ap.add_argument("--snr", type=float, default=0.2)
    ap.add_argument("--noise", choices=["white", "pink"], default="white")
    ap.add_argument("--common_noise", type=float, default=0.0, help="fraction of noise shared within patient")
    ap.add_argument("--effect_frac", type=float, default=1.0, help="fraction of electrodes carrying the effect")
    ap.add_argument("--effect_on", choices=["surprisal", "confound"], default="surprisal")
    ap.add_argument("--confound_rho", type=float, default=0.0, help="corr(confound, surprisal)")
    ap.add_argument("--patient_width_sd", type=float, default=0.0,
                    help="sd of log width-scale across PATIENTS (patients differ in smoothness); 0 = off")
    ap.add_argument("--shared_stimulus", action="store_true")
    ap.add_argument("--surp_mean", type=float, default=7.0)
    ap.add_argument("--surp_sd", type=float, default=3.0)
    ap.add_argument("--center_ms", type=float, default=None, help="fix N400 centre; default = estimated")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"truth per 1 SD of driver: gain {STRENGTH['gain']:+.2f}, shift {STRENGTH['shift']:+.0f} ms, "
          f"width {STRENGTH['width']:+.2f} (log); effect_frac={args.effect_frac}, effect_on={args.effect_on}, "
          f"rho={args.confound_rho}, noise={args.noise}, common={args.common_noise}, snr={args.snr}, "
          f"shared={args.shared_stimulus}")
    hdr = f"{'mode':8} {'':4}{'gain':>8}{'shift ms':>10}{'width':>8}   {'argmax':>7}{'argmaxW':>8} {'LOPO dr M1-M0':>14} {'M2-M1':>7}"
    print(hdr)
    rows = []
    for mode in args.modes:
        res = [run_scenario(args, mode, args.seed + 17 * k) for k in range(args.reps)]
        rows += res
        t = np.mean([r["truth_g_shift_omega"] for r in res], 0)
        e = np.mean([r["est_g_shift_omega"] for r in res], 0)
        sd = np.std([r["est_g_shift_omega"] for r in res], 0)
        cov = np.mean([r["covered"] for r in res], 0)
        print(f"{mode:8} truth {t[0]:8.2f}{t[1]:10.1f}{t[2]:8.2f}")
        print(f"{'':8} est   {e[0]:8.2f}{e[1]:10.1f}{e[2]:8.2f}   "
              f"{np.mean([r['argmax_latency_shift_ms'] for r in res]):7.1f}"
              f"{np.mean([r['argmax_window_shift_ms'] for r in res]):8.1f} "
              f"{np.mean([r['lopo_gain_M1_minus_M0'] for r in res]):14.4f} "
              f"{np.mean([r['lopo_gain_M2_minus_M1'] for r in res]):7.4f}")
        m2e = np.mean([r["m2_surprisal_g_shift_omega_given_confound"] for r in res], 0)
        print(f"{'':8} est(surprisal | confound in model, M2) {m2e[0]:6.2f}{m2e[1]:8.1f}{m2e[2]:8.2f}")
        print(f"{'':8} sd    {sd[0]:8.2f}{sd[1]:10.1f}{sd[2]:8.2f}   90%-CI coverage of truth: "
              f"gain {cov[0]:.2f}, shift {cov[1]:.2f}, width {cov[2]:.2f}   "
              f"(LOPO: M1>M0 in {np.mean([r['lopo_frac_patients_M1_better'] for r in res]):.0%} of patients)")
        tc_ = np.mean([r["t_covered"] for r in res], 0)
        bh = np.mean([r["boot_halfwidth"] for r in res], 0)
        th = np.mean([(np.array(r["t_hi"]) - np.array(r["t_lo"])) / 2 for r in res], 0)
        te = np.mean([r["t_est"] for r in res], 0)
        print(f"{'':8} CALIBRATION  bias(est-truth): {e[0]-t[0]:+.3f} {e[1]-t[1]:+.2f} {e[2]-t[2]:+.3f} | "
              f"empirical sd: {sd[0]:.3f} {sd[1]:.2f} {sd[2]:.3f} | boot half-width: {bh[0]:.3f} {bh[1]:.2f} {bh[2]:.3f} "
              f"| t half-width: {th[0]:.3f} {th[1]:.2f} {th[2]:.3f}")
        print(f"{'':8} coverage90   bootstrap: {cov[0]:.2f} {cov[1]:.2f} {cov[2]:.2f}   patient-t: {tc_[0]:.2f} {tc_[1]:.2f} {tc_[2]:.2f}")
    if args.out:
        with open(args.out, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"[{time.time() - t0:.0f}s]")