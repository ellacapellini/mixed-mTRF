#!/usr/bin/env python
"""
summarize_ieeg1.py -- merge result files from the Mac AND Euler and evaluate the plan's gates.

  python -m mixed_trf summarize                                   # main run (all seeds found in results/stage1/)
  python -m mixed_trf summarize --glob 'results/stage1/ieeg1_e256_*_seed*.jsonl'   # a sensitivity variant (its own tag)

Per (band, SNR, mechanism), on the SELECTED electrodes (and ALL electrodes for contrast):
  bias, empirical SD across seeds, mean t-interval half-width, coverage of the truth, power / false-positive rate
  (90% interval excludes 0), leave-one-patient-out R^2 gain, selection precision, noise correlation among selected.
GATES (analysis_plan_v1.md, 5a), evaluated per (band, SNR) on the selected electrodes:
  (i)  null: 90% interval covers 0 in 85-95% of seeds for gain/shift/width, and false-positive rate <= 10%
  (ii) effect scenarios: |bias| < 0.3 x SESOI for EVERY parameter, including the off-target ones (cross-talk)
       SESOI = gain 0.10, shift 10 ms, width 0.10  ->  tolerance 0.03 / 3 ms / 0.03
  (iii) SPECIFICITY: in an effect scenario, how often the 90% interval of an OFF-target parameter excludes 0
       (a false mechanism claim). Must be <= 15%. Added after the 20-seed run: bias can be inside tolerance while the
       interval is narrow enough that the off-target parameter is still 'significant' (latency scenario: gain flagged in 60% of seeds).
  (info) power = how often the TRUE mechanism's interval excludes 0; recovery = estimate / truth for it.
Rows from different machines are merged automatically; duplicate (band, snr, mode, seed) rows are dropped.
"""
import argparse, glob, json, collections
import numpy as np

SESOI = np.array([0.10, 10.0, 0.10]); TOL = 0.3 * SESOI


def wilson(k, n, z=1.645):
    """90% Wilson interval for a proportion k/n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def verdict_range(rate, n, lo, hi):
    """PASS if the whole 90% interval of the observed rate lies in [lo, hi]; FAIL if it lies entirely outside;
    otherwise INCONCLUSIVE (more seeds needed). rate*n must be a count."""
    a, b = wilson(round(rate * n), n)
    if a >= lo and b <= hi:
        return "PASS"
    if b < lo or a > hi:
        return "FAIL"
    return "INCONCLUSIVE"


def verdict_cov(rate, n, nominal=0.90):
    """Calibration of a nominal-90% interval. PASS = consistent with nominal (90% interval of the observed rate contains it) AND not
    clearly low (lower bound >= 0.80). FAIL = clearly off (interval entirely below 0.85 or above 0.95). Else INCONCLUSIVE.
    (The earlier rule 'whole interval inside [0.85, 0.95]' could not PASS even for a perfectly calibrated procedure at 100 seeds.)"""
    a, b = wilson(round(rate * n), n)
    if b < 0.85 or a > 0.95:
        return "FAIL"
    if a <= nominal <= b and a >= 0.80:
        return "PASS"
    return "INCONCLUSIVE"


def verdict_fp(rate, n, nominal=0.10):
    """False-positive rate with nominal 10%. PASS = consistent with nominal (lower bound <= 0.10) and not too uncertain (upper bound <= 0.20).
    FAIL = lower bound > 0.10 (significantly inflated). Else INCONCLUSIVE (too few seeds). A rate of exactly 0.10 has an upper bound of
    ~0.16 at 100 seeds, so any cap below that would make a perfectly calibrated procedure unable to pass."""
    a, b = wilson(round(rate * n), n)
    if a > nominal:
        return "FAIL"
    return "PASS" if b <= 0.20 else "INCONCLUSIVE"


def verdict_bias(bias, se, tol):
    if abs(bias) + 1.645 * se < tol:
        return "PASS"
    if abs(bias) - 1.645 * se > tol:
        return "FAIL"
    return "INCONCLUSIVE"
NAMES = ["gain", "shift", "width"]
ap = argparse.ArgumentParser()
ap.add_argument("--glob", nargs="+", default=["results/stage1/ieeg1_lfp_snr*_seed*.jsonl", "results/stage1/ieeg1_hg_snr*_seed*.jsonl"],
                help="file patterns; default = MAIN run only (tagged sensitivity runs are excluded, pass their own pattern)")
ap.add_argument("--min-seeds", type=int, default=1)
ap.add_argument("--table", action="store_true", help="also print a compact one-line-per-cell table, with a paired first-order comparison if the rows contain it")
a = ap.parse_args()

rows, seen = [], set()
for f in sorted(set(sum((glob.glob(g) for g in a.glob), []))):
    for line in open(f):
        r = json.loads(line)
        key = (r.get("band"), r["snr"], r["mode"], r["seed"])
        if key in seen:
            continue
        seen.add(key); rows.append(r)
if not rows:
    raise SystemExit(f"no result rows found for {a.glob}")
groups = collections.defaultdict(list)
for r in rows:
    groups[(r.get("band"), r["snr"], r["mode"])].append(r)
print(f"{len(rows)} rows from {len(set(r['seed'] for r in rows))} seeds, {len(groups)} cells\n")

def stat(rs, tag):
    est = np.array([r[tag]["est"] for r in rs]); tru = np.array([r[tag]["truth"] for r in rs])
    lo = np.array([r[tag]["lo"] for r in rs]); hi = np.array([r[tag]["hi"] for r in rs])
    cov = np.array([r[tag]["covered"] for r in rs]).mean(0)
    claim = ((lo > SESOI) | (hi < -SESOI)).mean(0)          # plan 5c: whole 90% interval beyond the smallest effect of interest
    absent = ((lo > -SESOI) & (hi < SESOI)).mean(0)         # plan 5c: whole interval inside the band = practical absence
    excl0 = ((lo > 0) | (hi < 0)).mean(0)
    return dict(n=len(rs), bias=(est - tru).mean(0), bias_se=((est - tru).std(0, ddof=1) / np.sqrt(len(rs))) if len(rs) > 1 else np.full(3, np.nan), sd=est.std(0, ddof=1) if len(rs) > 1 else np.full(3, np.nan),
                hw=((hi - lo) / 2).mean(0), cov=cov, excl0=excl0, claim=claim, absent=absent, truth=tru.mean(0), est=est.mean(0),
                lopo=np.mean([r[tag]["lopo_dR2"] for r in rs]), lopo_pos=np.mean([r[tag]["lopo_dR2"] > 0 for r in rs]))

order = {"null": 0, "gain": 1, "latency": 2, "width": 3}
summ = {}
for (band, snr, mode), rs in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1], order.get(kv[0][2], 9))):
    if len(rs) < a.min_seeds:
        continue
    S, A = stat(rs, "selected"), stat(rs, "all")
    summ[(band, snr, mode)] = S
    print(f"[{band} snr={snr} {mode:8s}] seeds={S['n']}  sel.precision={np.mean([r['sel_precision'] for r in rs]):.2f}  "
          f"noise-corr(sel)={np.nanmean([r['sel_mean_noise_corr'] if r['sel_mean_noise_corr'] is not None else np.nan for r in rs]):.2f}")
    print(f"    SELECTED  truth {np.round(S['truth'], 3)} est {np.round(S['est'], 3)} bias {np.round(S['bias'], 3)} sdSeeds {np.round(S['sd'], 3)} "
          f"CIhalf {np.round(S['hw'], 3)}")
    print(f"              coverage90 {np.round(S['cov'], 2)}  P(CI excludes 0) {np.round(S['excl0'], 2)}  "
          f"LOPO dR2 {S['lopo']:+.5f} (>0 in {S['lopo_pos']:.0%} of seeds)")
    print(f"    ALL       bias {np.round(A['bias'], 3)}  P(CI excludes 0) {np.round(A['excl0'], 2)}  LOPO dR2 {A['lopo']:+.5f}")
    fe = [r["selected"].get("fit_explained") for r in rs if r["selected"].get("fit_explained") is not None]
    if fe:
        print(f"    family adequacy (binned estimator): share of surprisal-dependent change explained by gain/shift/dilation = {np.mean(fe):.2f} "
              f"(min {np.min(fe):.2f}); testbed: in-family 0.93-0.99, out-of-family 0.71-0.91 at SNR 0.2 (falls with noise: calibrate per SNR)")
    print(f"    argmax shift (global / windowed): {np.mean([r['argmax_global_ms'] for r in rs]):.0f} / {np.mean([r['argmax_window_ms'] for r in rs]):.0f} ms")

print("\n================ GATES (selected electrodes) ================")
print("Verdicts: PASS / FAIL only when the 90% interval of the observed rate (or bias) is entirely on one side of the threshold;")
print("          INCONCLUSIVE = within Monte-Carlo error, more seeds needed.\n")
for (band, snr) in sorted({(b, s) for (b, s, m) in summ}, key=lambda x: (str(x[0]), x[1])):
    print(f"\n[{band} snr={snr}]")
    nul = summ.get((band, snr, "null"))
    if nul is not None:
        n = nul["n"]
        vc = [verdict_cov(c, n) for c in nul["cov"]]
        vf = [verdict_fp(e, n) for e in nul["excl0"]]
        print(f"  (i) null coverage of 0: {np.round(nul['cov'], 2)} -> {vc};  false-positive rate {np.round(nul['excl0'], 2)} -> {vf}   (seeds={n})")
    for mode in ("gain", "latency", "width"):
        S = summ.get((band, snr, mode))
        if S is None:
            continue
        vb = [verdict_bias(S["bias"][i], S["bias_se"][i], TOL[i]) for i in range(3)]
        print(f"  (ii) {mode:8s}: bias {np.round(S['bias'], 3)} (+/-{np.round(1.645 * S['bias_se'], 3)}) vs tolerance {TOL} -> "
              + ", ".join(f"{n_}:{v}" for n_, v in zip(NAMES, vb)))
        on = {"gain": 0, "latency": 1, "width": 2}[mode]
        off = [i for i in range(3) if i != on]
        fm = S["excl0"][off]
        fmd = {NAMES[i]: round(float(v), 2) for i, v in zip(off, fm)}
        vs = [verdict_range(v, S["n"], 0.0, 0.15) for v in fm]
        print(f"       (iii) specificity: false-mechanism rate {fmd} -> {vs};  power({NAMES[on]}) {S['excl0'][on]:.2f};  "
              f"recovery est/truth {S['est'][on] / S['truth'][on]:.2f}")
print("\n---- (iv) the PLAN's claim rule (5c): 'effect present' only if the WHOLE 90% interval lies beyond the smallest effect of interest ----")
print("     This is the rule the plan actually uses; (iii) above tests 'excludes 0', which is stricter.\n")
for (band, snr) in sorted({(b, s) for (b, s, m) in summ}, key=lambda x: (str(x[0]), x[1])):
    for mode in ("gain", "latency", "width"):
        S = summ.get((band, snr, mode))
        if S is None:
            continue
        on = {"gain": 0, "latency": 1, "width": 2}[mode]; off = [i for i in range(3) if i != on]
        fc = S["claim"][off]
        vs = [verdict_range(v, S["n"], 0.0, 0.15) for v in fc]
        print(f"[{band} snr={snr} {mode:8s}] true mechanism claimed present in {S['claim'][on]:.2f} of seeds | "
              f"FALSE mechanism claimed: { {NAMES[i]: round(float(v), 2) for i, v in zip(off, fc)} } -> {vs}")
print("\nGATE STATUS: (ii) bias and (iv) the plan's SESOI claim rule are the gates. (iii) 'excludes 0' is INFORMATIONAL only (tiny structural leakage is always")
print("'significant' at high SNR; it only matters if it crosses the smallest effect of interest). Until (ii) and (iv) pass for all mechanisms -- AND the family-adequacy check")
print("holds -- only 'surprisal modulates the response' (M1 vs M0) may be claimed, not which mechanism.")


if a.table:
    print("\n================ COMPACT TABLE (selected electrodes) ================")
    print("rec = estimate/truth for the TRUE mechanism; power = true mechanism beyond SESOI (plan rule 5c); false = worst OFF-target mechanism beyond SESOI.")
    print("bin = binned estimator, fo = first-order estimator on the SAME data (paired). bias: P=pass F=fail ?=inconclusive on (gain,shift,width).\n")
    print(f"{'band':4s} {'snr':>6s} {'mech':8s} {'n':>4s} | {'rec bin':>7s} {'rec fo':>7s} | {'power bin':>9s} {'power fo':>8s} | {'false bin':>9s} {'false fo':>8s} | bias | adequacy")
    sym = {"PASS": "P", "FAIL": "F", "INCONCLUSIVE": "?"}
    for (band, snr, mode) in sorted(summ, key=lambda k: (str(k[0]), k[1], order.get(k[2], 9))):
        if mode == "null":
            continue
        rs = groups[(band, snr, mode)]; S = summ[(band, snr, mode)]
        on = {"gain": 0, "latency": 1, "width": 2}[mode]; off = [i for i in range(3) if i != on]
        rec = S["est"][on] / S["truth"][on] if S["truth"][on] else float("nan")
        pw, fb = S["claim"][on], S["claim"][off].max()
        if all("first_order" in r for r in rs):
            est = np.array([r["first_order"]["selected"]["est"] for r in rs]); tru = np.array([r["selected"]["truth"] for r in rs])
            lo = np.array([r["first_order"]["selected"]["lo"] for r in rs]); hi = np.array([r["first_order"]["selected"]["hi"] for r in rs])
            cl = ((lo > SESOI) | (hi < -SESOI)).mean(0)
            rec_fo = est[:, on].mean() / tru[:, on].mean() if tru[:, on].mean() else float("nan")
            pw_fo, fb_fo = cl[on], cl[off].max()
            fo = (f"{rec_fo:7.2f}", f"{pw_fo:8.2f}", f"{fb_fo:8.2f}")
        else:
            fo = ("   n/a ", "    n/a ", "    n/a ")
        vb = "".join(sym[verdict_bias(S["bias"][i], S["bias_se"][i], TOL[i])] for i in range(3))
        fe = [r["selected"].get("fit_explained") for r in rs if r["selected"].get("fit_explained") is not None]
        print(f"{band:4s} {snr:>6} {mode:8s} {S['n']:>4d} | {rec:7.2f} {fo[0]} | {pw:9.2f} {fo[1]} | {fb:9.2f} {fo[2]} | {vb:4s} | {np.mean(fe) if fe else float('nan'):.2f}")
    print("\nNULL rows (false-positive rate of 'interval excludes 0' / 'beyond SESOI'):")
    for (band, snr, mode) in sorted(summ, key=lambda k: (str(k[0]), k[1])):
        if mode != "null":
            continue
        S = summ[(band, snr, mode)]
        print(f"  {band:4s} snr={snr:>6}  n={S['n']:>3d}  excl0 {np.round(S['excl0'], 2)}  beyondSESOI {np.round(S['claim'], 2)}  coverage {np.round(S['cov'], 2)}")
