"""
ieeg_stage1.py -- Stage 1 at iEEG scale: 8 patients x ~128 electrodes x 2 h.

Why a new file (what breaks at ~1000 units)
-------------------------------------------
erp_testbed.py keeps a sparse design per unit (~140 MB at 2 h) -> ~140 GB for 1000 units. But every patient hears
the SAME story, so every unit shares ONE design matrix X. Then per unit all that is needed is
    X'y (per time block), y'y (per time block)
and X'X is computed once. Memory per unit is a few KB; time per unit is dominated by simulating the signal.
Everything below (ridge CV, electrode selection, fits, held-out scores) runs from these sufficient statistics.

Analysis follows the plan (analysis_plan_v1.md):
  * time split inside every patient: first --dev_frac (30%) = development (5 contiguous CV blocks),
    the rest = confirmatory (one block). Nothing in the confirmatory block is used to choose anything.
  * electrode selection on DEVELOPMENT data only: cross-validated R^2 of the surprisal-free model M0
    (so selection cannot be driven by the surprisal effect); top-k per patient (--select_k).
  * confirmatory estimates: per-unit ridge (strength chosen on development), gain/shift/width decomposition
    per patient, then a t-interval ACROSS PATIENTS (df = P-1); the patient is the unit of inference.
  * leave-one-patient-out (LOPO) on the confirmatory block: R^2 gain of M1 (with surprisal) over M0.
  * reported twice: on the selected electrodes, and on ALL electrodes (to show the dilution).

Realism knobs (all ASSUMPTIONS -- change them): --responsive_frac (share of electrodes with any speech response),
--effect_frac (share of RESPONSIVE electrodes whose response depends on surprisal), noise 1/f plus a component
shared within a patient (--common_noise; common reference / volume conduction).
Spatial structure (added): electrodes have 3-D positions (sEEG shafts or an ECoG grid). Noise covariance is
    Sigma = c * 11' + (1 - c) * exp(-d / L_band)
c = global common-reference share (--common_noise); L_band = correlation length of the BAND: high-gamma ('hg') ~1.5 mm
(uncorrelated by 2-4 mm), low-frequency/ERP-like ('lfp') ~15 mm (spreads over centimetres). Responsiveness, surprisal
sensitivity and amplitude are thresholded / scaled spatial Gaussian fields (--resp_len_mm), so responsive electrodes CLUSTER
(as in superior temporal cortex) and selection tends to pick neighbouring contacts. The patient-level t-interval absorbs the
resulting loss of effective independence. More electrodes still do NOT raise the number of independent patients (8).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import scipy.sparse as sp
import scipy.fft as sfft
from scipy import stats
from scipy.spatial.distance import cdist

import erp_testbed as T

DT = T.DT
N_DEV_BLOCKS = 5
CS = np.logspace(-3, 1, 9)            # relative ridge strengths (lambda = c * trace(X'X)/p)


# ------------------------------------------------------------ fast simulation
def erp_words(taus, z, eff, amp, shift, wscale):
    """Vectorised twin of erp_testbed.erp for ALL words at once -> (n_words, L)."""
    tt = taus[None, :]
    out = 0.0
    for name, (t, sd, a) in T.COMPONENTS.items():
        if name == "N400":
            aa = a * np.maximum(1.0 + eff["gain"] * z, 0.05)[:, None]
            tc = (t + shift + eff["shift"] * z)[:, None]
            ss = (sd * wscale * np.exp(eff["width"] * z))[:, None]
        else:
            aa, tc, ss = a, t + shift, sd * wscale
        out = out + aa * np.exp(-0.5 * ((tt - tc) / ss) ** 2)
    return amp * out


def place(onsets, H, n_times):
    idx = onsets[:, None] + np.arange(H.shape[1])[None, :]
    ok = idx < n_times
    return np.bincount(idx[ok], weights=H[ok], minlength=n_times)[:n_times]


def fast_pink(n, rng):
    x = rng.normal(size=n)
    f = np.fft.rfftfreq(n); f[0] = f[1]
    x = np.fft.irfft(np.fft.rfft(x) / np.sqrt(f), n=n)
    return (x - x.mean()) / x.std()


# ------------------------------------------------------------ spatial structure
BAND_LEN_MM = {"hg": 1.5, "lfp": 15.0}


def make_positions(E, geometry, spacing_mm, rng):
    """(E, 3) contact positions in mm. sEEG: shafts of 12 contacts at random places/directions in an ~80x80x60 mm
    volume; grid: 16-column lattice in a plane."""
    if geometry == "grid":
        cols = 16
        i = np.arange(E)
        return np.stack([(i // cols) * spacing_mm, (i % cols) * spacing_mm, np.zeros(E)], axis=1).astype(float)
    per = 12
    pos = []
    for _ in range(int(np.ceil(E / per))):
        o = rng.uniform([0, 0, 0], [80, 80, 60])
        v = rng.normal(size=3); v /= np.linalg.norm(v)
        pos += [o + k * spacing_mm * v for k in range(per)]
    return np.array(pos[:E])


def pink_rows(E, n, rng, chunk=32):
    """E independent unit-variance 1/f series, float32, generated in chunks to cap memory."""
    out = np.empty((E, n), dtype=np.float32)
    f = np.fft.rfftfreq(n); f[0] = f[1]
    shape = (1.0 / np.sqrt(f)).astype(np.float32)
    for a in range(0, E, chunk):
        b = min(E, a + chunk)
        x = rng.standard_normal((b - a, n)).astype(np.float32)
        x = sfft.irfft(sfft.rfft(x, axis=1, workers=1) * shape[None, :], n=n, axis=1, workers=1)
        x -= x.mean(axis=1, keepdims=True); x /= x.std(axis=1, keepdims=True)
        out[a:b] = x
    return out


def spatial_noise(pos, n_times, rng, length_mm, common, noise_kind):
    """Noise matrix (E, n_times), unit variance per row, covariance c*11' + (1-c)*exp(-d/L)."""
    E = len(pos)
    D = cdist(pos, pos)
    Sigma = common * np.ones((E, E)) + (1 - common) * np.exp(-D / length_mm) + 1e-6 * np.eye(E)
    Lc = np.linalg.cholesky(Sigma).astype(np.float32)
    if noise_kind == "pink":
        Z = pink_rows(E, n_times, rng)
    else:
        Z = rng.standard_normal((E, n_times)).astype(np.float32)
    return Lc @ Z, Sigma


def thresholded_field(D, length_mm, frac, rng):
    """Spatially smooth Gaussian field -> binary mask with expected share `frac` (clusters, not salt-and-pepper)."""
    R = np.exp(-D / length_mm) + 1e-6 * np.eye(len(D))
    g = np.linalg.cholesky(R) @ rng.standard_normal(len(D))
    return g > stats.norm.ppf(1 - frac), g


# ------------------------------------------------------------ blocks / design
def block_edges(n_times, dev_frac):
    n_dev = int(n_times * dev_frac)
    return list(np.linspace(0, n_dev, N_DEV_BLOCKS + 1).astype(int)) + [n_times]     # 6 blocks: 5 dev + 1 conf


def make_blocks(words, n_times, L, dev_frac):
    on = words["onsets"]
    X = sp.hstack([T.lag_design(on, np.ones(len(on)), n_times, L),
                   T.lag_design(on, words["z"], n_times, L)], format="csr")
    edges = block_edges(n_times, dev_frac)
    XtXb, XbT = [], []
    for b in range(6):
        Xb = X[edges[b]:edges[b + 1]]
        XtXb.append((Xb.T @ Xb).toarray())
        XbT.append(Xb.T.tocsr())
    return edges, np.array(XtXb), XbT


def unit_stats(y, edges, XbT):
    Xty = np.array([XbT[b] @ y[edges[b]:edges[b + 1]] for b in range(6)])
    yty = np.array([y[edges[b]:edges[b + 1]] @ y[edges[b]:edges[b + 1]] for b in range(6)])
    return Xty, yty


# ------------------------------------------------------------ generation
def generate(args, rng, taus, L):
    n_times = args.n_times
    words = T.make_words(n_times, rng, surp_mean=args.surp_mean, surp_sd=args.surp_sd)
    edges, XtXb, XbT = make_blocks(words, n_times, L, args.dev_frac)
    # nominal noise level: SNR is defined against a unit-amplitude, effect-free responsive electrode
    null = T.effect_for("null")
    ref_var = place(words["onsets"], erp_words(taus, words["z"], null, 1.0, 0.0, 1.0), n_times).var()
    noise_sd = np.sqrt(ref_var / args.snr)
    eff_on = T.effect_for(args.mode)
    P, E = args.n_patients, args.electrodes
    L_noise = args.noise_len_mm if args.noise_len_mm > 0 else BAND_LEN_MM[args.band]
    patient, responsive, has_eff, Xty_all, yty_all, mean_corr = [], [], [], [], [], []
    for p in range(P):
        p_scale, p_shift = np.exp(rng.normal(0, 0.15)), rng.normal(0, 8.0)
        p_w = float(np.exp(rng.normal(0, args.patient_width_sd))) if args.patient_width_sd > 0 else 1.0
        pos = make_positions(E, args.geometry, args.spacing_mm, rng)
        D = cdist(pos, pos)
        resp_mask, _ = thresholded_field(D, args.resp_len_mm, args.responsive_frac, rng)
        eff_mask, _ = thresholded_field(D, args.resp_len_mm, args.effect_frac, rng)
        amp_field = np.linalg.cholesky(np.exp(-D / args.resp_len_mm) + 1e-6 * np.eye(E)) @ rng.standard_normal(E)
        N, Sigma = spatial_noise(pos, n_times, rng, L_noise, args.common_noise, args.noise)
        for e in range(E):
            resp = bool(resp_mask[e]); has = bool(resp and eff_mask[e])
            sig = 0.0
            if resp:
                eff = eff_on if has else null
                sig = place(words["onsets"], erp_words(taus, words["z"], eff, p_scale * np.exp(0.25 * amp_field[e]), p_shift, p_w), n_times)
            y = sig + noise_sd * N[e].astype(np.float64)
            Xty, yty = unit_stats(y, edges, XbT)
            patient.append(p); responsive.append(resp); has_eff.append(has)
            Xty_all.append(Xty); yty_all.append(yty)
        mean_corr.append(Sigma)
        del N
    corr_mats = mean_corr
    return {"patient": np.array(patient), "responsive": np.array(responsive), "has_eff": np.array(has_eff),
            "Xty": np.array(Xty_all), "yty": np.array(yty_all), "XtXb": XtXb, "Sigma": corr_mats}


# ------------------------------------------------------------ analysis from sufficient stats
def cv_ridge(XtXb, Xty, yty, cols, cs=CS):
    """Per-unit CV over the dev blocks. Returns best c index per unit and CV R^2 per unit."""
    idx = np.array(cols); p = len(idx); U = Xty.shape[0]
    A = [XtXb[b][np.ix_(idx, idx)] for b in range(N_DEV_BLOCKS)]
    tot = sum(A); scale = np.trace(tot) / p
    xty_tot = Xty[:, :N_DEV_BLOCKS, :][:, :, idx].sum(1)
    sse = np.zeros((len(cs), U))
    for ci, c in enumerate(cs):
        for b in range(N_DEV_BLOCKS):
            xb = Xty[:, b, idx]
            W = np.linalg.solve(tot - A[b] + c * scale * np.eye(p), (xty_tot - xb).T).T
            sse[ci] += yty[:, b] - 2 * np.einsum("up,up->u", W, xb) + np.einsum("up,pq,uq->u", W, A[b], W)
    best = sse.argmin(0)
    r2 = 1.0 - sse[best, np.arange(U)] / yty[:, :N_DEV_BLOCKS].sum(1)
    return best, r2


def conf_fit(XtXc, Xtyc, best, cols, cs=CS):
    idx = np.array(cols); p = len(idx)
    A = XtXc[np.ix_(idx, idx)]; scale = np.trace(A) / p
    W = np.zeros((Xtyc.shape[0], p))
    for ci in np.unique(best):
        m = best == ci
        W[m] = np.linalg.solve(A + cs[ci] * scale * np.eye(p), Xtyc[m][:, idx].T).T
    return W


def patient_decomp(W, patient, sel, taus, L):
    out = []
    for p in np.unique(patient):
        m = sel & (patient == p)
        if m.sum() == 0:
            continue
        g, sh, om, _ = T.decompose(W[m, :L].mean(0), W[m, L:].mean(0), taus)
        out.append((g, sh, om))
    return np.array(out)


def t_interval(pd_, level=0.90):
    n = len(pd_)
    m = pd_.mean(0); se = pd_.std(0, ddof=1) / np.sqrt(n)
    q = stats.t.ppf(0.5 + level / 2, n - 1)
    return m, m - q * se, m + q * se


def lopo_gain(XtXc, Xtyc, yty_c, patient, sel, c_pop, L):
    """Leave-one-patient-out R^2 gain of M1 over M0 on the confirmatory block (selected units only)."""
    res = []
    for p in np.unique(patient):
        tr = sel & (patient != p); te = sel & (patient == p)
        if tr.sum() == 0 or te.sum() == 0:
            continue
        r2 = {}
        for name, cols in (("M0", np.arange(L)), ("M1", np.arange(2 * L))):
            A = XtXc[np.ix_(cols, cols)] * tr.sum()
            sc = np.trace(A) / len(cols)
            w = np.linalg.solve(A + c_pop * sc * np.eye(len(cols)), Xtyc[tr][:, cols].sum(0))
            sse = yty_c[te] - 2 * Xtyc[te][:, cols] @ w + w @ XtXc[np.ix_(cols, cols)] @ w
            r2[name] = np.mean(1 - sse / yty_c[te])
        res.append((r2["M0"], r2["M1"]))
    res = np.array(res)
    return res[:, 1] - res[:, 0], res[:, 0]


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
    W = conf_fit(XtXc, Xtyc, best1, np.arange(2 * L))
    strength = T.effect_for(args.mode)
    out = {"n_units": int(U), "sel_precision": float(d["responsive"][sel].mean()),
           "sel_recall": float(sel[d["responsive"]].mean()) if d["responsive"].any() else None,
           "n_selected": int(sel.sum()), "sel_mean_noise_corr": float(np.mean(mc)) if mc else None}
    for tag, mask in (("selected", sel), ("all", np.ones(U, bool))):
        pd_ = patient_decomp(W, patient, mask, taus, L)
        m, lo, hi = t_interval(pd_)
        # Estimand = modulation among RESPONSIVE electrodes: non-responsive ones add noise to the pooled TRF
        # but nothing to the surprisal term, so they must not dilute the truth.
        rm = mask & d["responsive"]
        frac = float(d["has_eff"][rm].mean()) if rm.any() else 0.0
        truth = [strength["gain"] * frac, strength["shift"] * frac, strength["width"] * frac]
        out[tag] = {"est": m.tolist(), "lo": lo.tolist(), "hi": hi.tolist(), "truth": truth,
                    "covered": [bool(lo[k] <= truth[k] <= hi[k]) for k in range(3)],
                    "patient_sd": pd_.std(0, ddof=1).tolist(), "n_patients": int(len(pd_))}
        gM = float(np.median(CS[best1[mask]]))
        dg, r0 = lopo_gain(XtXc, Xtyc, ytyc, patient, mask, gM, L)
        out[tag]["lopo_dR2"] = float(dg.mean()); out[tag]["lopo_frac_patients_pos"] = float((dg > 0).mean())
        out[tag]["lopo_R2_M0"] = float(r0.mean())
    h0 = W[sel, :L].mean(0); h1 = W[sel, L:].mean(0)
    out["argmax_global_ms"] = T.argmax_latency_shift(h0, h1, taus)
    out["argmax_window_ms"] = T.argmax_latency_shift(h0, h1, taus, window=(250.0, 600.0))
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
               band=args.band, geometry=args.geometry, spacing_mm=args.spacing_mm, resp_len_mm=args.resp_len_mm,
               t_generate_s=t_gen, t_total_s=time.time() - t0)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", nargs="+", default=["null", "gain", "latency", "width"],
                    choices=["null", "gain", "latency", "width", "all"])
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
    ap.add_argument("--surp_mean", type=float, default=7.0)
    ap.add_argument("--surp_sd", type=float, default=3.0)
    ap.add_argument("--patient_width_sd", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    for mode in a.modes:
        a.mode = mode
        r = run(a)
        rows.append(r)
        s, al = r["selected"], r["all"]
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