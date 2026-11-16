#!/usr/bin/env python
"""
summarize_ieeg1.py -- merge result files from the Mac AND Euler and evaluate the plan's gates.

  python summarize_ieeg1.py                                   # main run (all seeds found in figures/)
  python summarize_ieeg1.py --glob 'figures/ieeg1_e256_*_seed*.jsonl'   # a sensitivity variant (its own tag)

Per (band, SNR, mechanism), on the SELECTED electrodes (and ALL electrodes for contrast):
  bias, empirical SD across seeds, mean t-interval half-width, coverage of the truth, power / false-positive rate
  (90% interval excludes 0), leave-one-patient-out R^2 gain, selection precision, noise correlation among selected.
GATES (analysis_plan_v1.md, 5a), evaluated per (band, SNR) on the selected electrodes:
  (i)  null: 90% interval covers 0 in 85-95% of seeds for gain/shift/width, and false-positive rate <= 10%
  (ii) effect scenarios: |bias| < 0.3 x SESOI for EVERY parameter, including the off-target ones (cross-talk)
       SESOI = gain 0.10, shift 10 ms, width 0.10  ->  tolerance 0.03 / 3 ms / 0.03
Rows from different machines are merged automatically; duplicate (band, snr, mode, seed) rows are dropped.
"""
import argparse, glob, json, collections
import numpy as np

SESOI = np.array([0.10, 10.0, 0.10]); TOL = 0.3 * SESOI
NAMES = ["gain", "shift", "width"]
ap = argparse.ArgumentParser()
ap.add_argument("--glob", nargs="+", default=["figures/ieeg1_lfp_snr*_seed*.jsonl", "figures/ieeg1_hg_snr*_seed*.jsonl"],
                help="file patterns; default = MAIN run only (tagged sensitivity runs are excluded, pass their own pattern)")
ap.add_argument("--min-seeds", type=int, default=1)
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
    excl0 = ((lo > 0) | (hi < 0)).mean(0)
    return dict(n=len(rs), bias=(est - tru).mean(0), sd=est.std(0, ddof=1) if len(rs) > 1 else np.full(3, np.nan),
                hw=((hi - lo) / 2).mean(0), cov=cov, excl0=excl0, truth=tru.mean(0), est=est.mean(0),
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
    print(f"    argmax shift (global / windowed): {np.mean([r['argmax_global_ms'] for r in rs]):.0f} / {np.mean([r['argmax_window_ms'] for r in rs]):.0f} ms")

print("\n================ GATES (selected electrodes) ================")
for (band, snr) in sorted({(b, s) for (b, s, m) in summ}, key=lambda x: (str(x[0]), x[1])):
    print(f"\n[{band} snr={snr}]")
    nul = summ.get((band, snr, "null"))
    if nul is not None:
        okc = [0.85 <= c <= 0.95 for c in nul["cov"]]; okf = [e <= 0.10 for e in nul["excl0"]]
        print(f"  (i) null coverage of 0: {np.round(nul['cov'], 2)}  -> {'PASS' if all(okc) else 'FAIL'};   false-positive rate {np.round(nul['excl0'], 2)} -> {'PASS' if all(okf) else 'FAIL'}"
              f"   (seeds={nul['n']}; MC error ~ +/-{1.645 * np.sqrt(0.09 / max(nul['n'], 1)):.2f})")
    for mode in ("gain", "latency", "width"):
        S = summ.get((band, snr, mode))
        if S is None:
            continue
        ok = np.abs(S["bias"]) < TOL
        print(f"  (ii) {mode:8s}: |bias| {np.round(np.abs(S['bias']), 3)} vs tolerance {TOL} -> {'PASS' if ok.all() else 'FAIL on ' + ', '.join(n for n, o in zip(NAMES, ok) if not o)}")
print("\nUntil (ii) passes for all mechanisms, only 'surprisal modulates the response' (M1 vs M0) may be claimed -- not which mechanism.")