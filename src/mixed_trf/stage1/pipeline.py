"""Stage-1 pipeline at iEEG scale: simulate -> time split -> electrode selection on development data -> confirmatory estimates (first-order or binned) -> patient-level inference -> leave-one-patient-out. One JSON row per mechanism."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from mixed_trf.estimators.binned import binned_patient_decomp
from mixed_trf.estimators.decomposition import argmax_latency_shift
from mixed_trf.estimators.firstorder import patient_decomp
from mixed_trf.estimators.ridge_cv import CS, conf_fit, cv_ridge
from mixed_trf.evaluation.inference import lopo_gain, t_interval
from mixed_trf.simulation.ieeg import generate
from mixed_trf.simulation.templates import DT, effect_for


def analyze(args, d, taus, L):
    XtXb, Xty, yty, patient = d["XtXb"], d["Xty"], d["yty"], d["patient"]
    U = len(patient)
    best0, r2_0 = cv_ridge(XtXb, Xty, yty, np.arange(L))              # M0 CV R^2 -> selection score
    best1, _ = cv_ridge(XtXb, Xty, yty, np.arange(2 * L))             # M1 ridge strengths
    sel = np.zeros(U, bool)
    for p in np.unique(patient):
        ids = np.where(patient == p)[0]
        sel[ids[np.argsort(-r2_0[ids])[:args.select_k]]] = True
    mc = []
    for p in np.unique(patient):
        ids = np.where(sel & (patient == p))[0] - np.where(patient == p)[0][0]
        if len(ids) > 1:
            S_ = d["Sigma"][p][np.ix_(ids, ids)]
            mc.append(S_[np.triu_indices(len(ids), 1)].mean())
    XtXc, Xtyc, ytyc = XtXb[5], Xty[:, 5, :], yty[:, 5]
    # c_mult < 1 relaxes the ridge ONLY for the decomposition fits (prediction/LOPO keep the CV-chosen strength):
    # prediction-optimal shrinkage over-shrinks the surprisal term (width/shift estimates came out ~45% too small).
    W = conf_fit(XtXc, Xtyc, best1, np.arange(2 * L), c_mult=args.ridge_c_mult)
    binned = args.estimator == "binned"
    if binned:
        K = len(d["bins"]["centers"])
        bin_cols = np.arange(2 * L, 2 * L + K * L)
        best_b, _ = cv_ridge(XtXb, Xty, yty, bin_cols)
        Wb = conf_fit(XtXc, Xtyc, best_b, bin_cols)
        W0 = conf_fit(XtXc, Xtyc, best0, np.arange(L))
    strength = effect_for(args.mode, args.effect_scale)
    out = {"n_units": int(U), "sel_precision": float(d["responsive"][sel].mean()),
           "sel_recall": float(sel[d["responsive"]].mean()) if d["responsive"].any() else None,
           "n_selected": int(sel.sum()), "sel_mean_noise_corr": float(np.mean(mc)) if mc else None}
    for tag, mask in (("selected", sel), ("all", np.ones(U, bool))):
        pd_fo = patient_decomp(W, patient, mask, taus, L, window=tuple(args.decomp_window))
        if binned:
            pd_, fit_expl = binned_patient_decomp(Wb, W0, patient, mask, taus, L, d["bins"]["centers"], d["bins"]["counts"],
                                                  tuple(args.param_window))
        else:
            pd_ = pd_fo
        m, lo, hi = t_interval(pd_)
        # Estimand = modulation among RESPONSIVE electrodes: non-responsive ones add noise to the pooled TRF
        # but nothing to the surprisal term, so they must not dilute the truth.
        rm = mask & d["responsive"]
        frac = float(d["has_eff"][rm].mean()) if rm.any() else 0.0
        truth = [strength["gain"] * frac, strength["shift"] * frac, strength["width"] * frac]
        if binned:                                   # keep the first-order result alongside for paired comparison
            m1, lo1, hi1 = t_interval(pd_fo)
            out.setdefault("first_order", {})[tag] = {"est": m1.tolist(), "lo": lo1.tolist(), "hi": hi1.tolist()}
        out[tag] = {"est": m.tolist(), "lo": lo.tolist(), "hi": hi.tolist(), "truth": truth,
                    "fit_explained": (fit_expl if binned else None),
                    "covered": [bool(lo[k] <= truth[k] <= hi[k]) for k in range(3)],
                    "patient_sd": pd_.std(0, ddof=1).tolist(), "n_patients": int(len(pd_))}
        gM = float(np.median(CS[best1[mask]]))
        dg, r0 = lopo_gain(XtXc, Xtyc, ytyc, patient, mask, gM, L)
        out[tag]["lopo_dR2"] = float(dg.mean()); out[tag]["lopo_frac_patients_pos"] = float((dg > 0).mean())
        out[tag]["lopo_R2_M0"] = float(r0.mean())
    h0 = W[sel, :L].mean(0); h1 = W[sel, L:].mean(0)
    out["argmax_global_ms"] = argmax_latency_shift(h0, h1, taus)
    out["argmax_window_ms"] = argmax_latency_shift(h0, h1, taus, window=(250.0, 600.0))
    return out


def run(args):
    t0 = time.time()
    L = int(args.tau_max / DT) + 1
    taus = np.arange(L) * DT
    rng = np.random.default_rng(args.seed)
    d = generate(args, rng, taus, L)
    t_gen = time.time() - t0
    res = analyze(args, d, taus, L)
    res.update(mode=args.mode, snr=args.snr, seed=args.seed, n_patients=args.n_patients, electrodes=args.electrodes,
               responsive_frac=args.responsive_frac, effect_frac=args.effect_frac, select_k=args.select_k,
               n_times=args.n_times, noise=args.noise, common_noise=args.common_noise,
               rhythm=args.rhythm, violation_p=args.violation_p, effect_scale=args.effect_scale, ridge_c_mult=args.ridge_c_mult, decomp_window=list(args.decomp_window), estimator=args.estimator,
               n_bins=args.n_bins, param_window=list(args.param_window), band=args.band, geometry=args.geometry, spacing_mm=args.spacing_mm, resp_len_mm=args.resp_len_mm,
               t_generate_s=t_gen, t_total_s=time.time() - t0)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", nargs="+", default=["null", "gain", "latency", "width"],
                    choices=["null", "gain", "latency", "width", "all", "late", "decay"])
    ap.add_argument("--n_patients", type=int, default=8)
    ap.add_argument("--electrodes", type=int, default=128, help="electrodes per patient (iEEG ~100-256)")
    ap.add_argument("--n_times", type=int, default=720000, help="samples at 100 Hz (720000 = 2 h)")
    ap.add_argument("--tau_max", type=float, default=800.0)
    ap.add_argument("--snr", type=float, default=0.03)
    ap.add_argument("--noise", choices=["white", "pink"], default="pink")
    ap.add_argument("--common_noise", type=float, default=0.3)
    ap.add_argument("--band", choices=["lfp", "hg"], default="lfp",
                    help="lfp: ERP/low-frequency-like (noise length ~15 mm); hg: high-gamma (~1.5 mm)")
    ap.add_argument("--noise_len_mm", type=float, default=0.0, help="override the band's noise correlation length")
    ap.add_argument("--geometry", choices=["seeg", "grid"], default="seeg")
    ap.add_argument("--spacing_mm", type=float, default=3.5, help="contact spacing (sEEG ~3.5, ECoG ~4-10)")
    ap.add_argument("--resp_len_mm", type=float, default=12.0, help="spatial scale of responsiveness/effect clusters")
    ap.add_argument("--responsive_frac", type=float, default=0.25, help="ASSUMPTION: share of electrodes responding to speech")
    ap.add_argument("--effect_frac", type=float, default=0.5, help="ASSUMPTION: share of RESPONSIVE electrodes modulated by surprisal")
    ap.add_argument("--dev_frac", type=float, default=0.30)
    ap.add_argument("--select_k", type=int, default=16, help="electrodes kept per patient, chosen on development data")
    ap.add_argument("--estimator", choices=["firstorder", "binned"], default="firstorder",
                    help="firstorder: linear-in-surprisal decomposition; binned: direct gain/shift/dilation fit on surprisal bins")
    ap.add_argument("--n_bins", type=int, default=5, help="surprisal bins for the binned estimator")
    ap.add_argument("--param_window", type=float, nargs=2, default=[300.0, 600.0],
                    help="lag window (ms) for the binned estimator's transform fit (start >= 300 avoids the unmodulated P2 tail)")
    ap.add_argument("--decomp_window", type=float, nargs=2, default=[250.0, 600.0],
                    help="lag window (ms) for the gain/shift/width regression; the plan fixes 250-600")
    ap.add_argument("--ridge_c_mult", type=float, default=1.0,
                    help="multiply the CV-chosen ridge strength in the DECOMPOSITION fits (1.0 = unchanged; try 0.1, 0.01)")
    ap.add_argument("--effect_scale", type=float, default=1.0,
                    help="multiply the true per-SD effect sizes (gain 0.30, shift 25 ms, width 0.25) by this factor: 1.0 = unchanged; "
                         "use 0.5 / 2 / 3 to map power against effect size")
    ap.add_argument("--rhythm", choices=["natural", "isochronous"], default="natural",
                    help="natural: word-like timing; isochronous: strict grid with occasional timing violations (oddball/rhythm paradigm)")
    ap.add_argument("--period_ms", type=float, default=300.0)
    ap.add_argument("--violation_p", type=float, default=0.15, help="probability an isochronous event is a timing violation")
    ap.add_argument("--jitter_ms", type=float, default=90.0)
    ap.add_argument("--surp_mean", type=float, default=7.0)
    ap.add_argument("--surp_sd", type=float, default=3.0)
    ap.add_argument("--patient_width_sd", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rows = []
    for mode in a.modes:
        a.mode = mode
        r = run(a)
        rows.append(r)
        s, al = r["selected"], r["all"]
        if "first_order" in r:
            f1 = r["first_order"]["selected"]
            print(f"           (first-order on the same data: est g {f1['est'][0]:+.3f} sh {f1['est'][1]:+.1f} w {f1['est'][2]:+.3f})", flush=True)
        print(f"[{mode:8s}] truth(sel) g {s['truth'][0]:.2f} sh {s['truth'][1]:.1f} w {s['truth'][2]:.2f} | "
              f"est g {s['est'][0]:+.3f} sh {s['est'][1]:+.1f} w {s['est'][2]:+.3f}  CI-cover {s['covered']} | "
              f"sel precision {r['sel_precision']:.2f} noise-corr(sel) {r['sel_mean_noise_corr'] if r['sel_mean_noise_corr'] is not None else float('nan'):.2f} | LOPO dR2 sel {s['lopo_dR2']:+.5f} all {al['lopo_dR2']:+.5f} | "
              f"argmax glob/win {r['argmax_global_ms']:.0f}/{r['argmax_window_ms']:.0f} | {r['t_total_s']:.0f}s", flush=True)
    if a.out:
        with open(a.out, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
