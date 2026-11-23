"""ERP-shaped ground truth: onset-anchored N1/P2/N400 templates and the surprisal-dependent mechanisms (gain, latency shift, width, plus two MISSPECIFIED truths: a late component and an asymmetric decay)."""
from __future__ import annotations

import numpy as np



DT = 10.0  # ms per sample


COMPONENTS = {              # name: (latency ms, sd ms, amplitude)
    "N1": (100.0, 25.0, -1.0),
    "P2": (200.0, 40.0, +0.8),
    "N400": (400.0, 80.0, -1.2),
}


STRENGTH = {"gain": 0.30, "shift": 25.0, "width": 0.25, "late": 0.35, "decay": 0.25}   # per 1 SD of the driver


def effect_for(mode, scale=1.0):
    # "late" and "decay" are MISSPECIFIED truths (not a gain/shift/dilation of the N400): a new late component scaled by
    # surprisal, and an asymmetric N400 whose decay (right flank) width changes -- i.e. "recovery speed".
    on = {"null": [], "gain": ["gain"], "latency": ["shift"], "width": ["width"],
          "all": ["gain", "shift", "width"], "late": ["late"], "decay": ["decay"]}[mode]
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
            if eff.get("decay", 0.0) != 0.0:                       # asymmetric: only the right (decay) flank changes width
                ss = np.where(taus < tt, ss, ss * np.exp(eff["decay"] * z))
        out += aa * np.exp(-0.5 * ((taus - tt) / ss) ** 2)
    if eff.get("late", 0.0) != 0.0:                                # additive late component, linear in z
        out += eff["late"] * z * np.exp(-0.5 * ((taus - (550.0 + shift)) / 60.0) ** 2)
    return amp_scale * out
