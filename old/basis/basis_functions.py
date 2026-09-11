"""
raised-cosine temporal basis functions (Pillow-style), log-spaced over
the lag window. This is the phi_j(tau) from Section 2 of the equations
note
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class RaisedCosineBasis:
    # fitted raised-cosine basis over a lag window.
    n_basis: int
    tau_max: float
    centres: np.ndarray
    spacing: float
    c: float
    def eval(self, tau: np.ndarray) -> np.ndarray:
        # evaluate all J basis functions at the given lags.
        tau = np.asarray(tau, dtype=float)
        if np.any(tau < 0):
            raise ValueError("tau must be non-negative (causal lags only).")
        log_tau = np.log(tau + self.c)  # shape (n_lags,)
        #broadcast: (n_lags, 1) - (1, n_basis) -> (n_lags, n_basis)
        diff = log_tau[:, None] - self.centres[None, :]
        arg = (np.pi / self.spacing) * diff
        Phi = np.where(
            np.abs(diff) < self.spacing,
            0.5 * np.cos(arg) + 0.5,
            0.0,
        )
        return Phi


def make_raised_cosine_basis(
    n_basis: int,
    tau_max: float,
    c: float = 5.0,
    n_lags: int | None = None,
) -> RaisedCosineBasis:
    #construct a raised-cosine basis spanning [0, tau_max].
    if n_basis < 2:
        raise ValueError("n_basis must be >= 2 to form overlapping bumps.")
    if tau_max <= 0:
        raise ValueError("tau_max must be positive.")
    if c <= 0:
        raise ValueError("c must be positive (avoids log(0) at tau=0).")
    log_min = np.log(0.0 + c)
    log_max = np.log(tau_max + c)
    centres = np.linspace(log_min, log_max, n_basis)
    spacing = centres[1] - centres[0]
    return RaisedCosineBasis(
        n_basis=n_basis,
        tau_max=tau_max,
        centres=centres,
        spacing=spacing,
        c=c,
    )


# Self-tests -> python basis_functions.py
def _self_test() -> None:
    basis = make_raised_cosine_basis(n_basis=10, tau_max=800.0, c=5.0)
    tau = np.linspace(0, 800, 2001)
    Phi = basis.eval(tau)

    #1. Shape sanity.
    assert Phi.shape == (2001, 10), f"unexpected shape {Phi.shape}"

    #2. values must be within [0, 1] (raised cosine bump range).
    assert Phi.min() >= -1e-12, f"negative basis value: {Phi.min()}"
    assert Phi.max() <= 1.0 + 1e-12, f"basis value > 1: {Phi.max()}"

    # 3.each basis function should be nonzero somewhere (no dead basis functions from a degenerate centre/spacing configuration).
    nonzero_mass = (Phi > 1e-9).sum(axis=0)
    assert np.all(nonzero_mass > 0), "found a basis function that is zero everywhere"

    #4. boundary continuity: at |diff| = spacing exactly, cos(pi) = -1, so the bump value should be ~0 there (continuous match with the "outside support" branch), not a jump discontinuity.
    edge_tau = np.exp(basis.centres[0] + basis.spacing) - basis.c
    val_at_edge = basis.eval(np.array([edge_tau]))[0, 0]
    assert abs(val_at_edge) < 1e-6, f"discontinuity at basis boundary: {val_at_edge}"

    #5. Resolution check: basis function *peaks* (in tau-space) should be much more tightly packed near tau=0 than near tau_max, since centres are evenly spaced in LOG time, not linear time.
    peak_taus = np.exp(basis.centres) - basis.c
    early_gaps = np.diff(peak_taus)[:3]
    late_gaps = np.diff(peak_taus)[-3:]
    assert early_gaps.mean() < late_gaps.mean(), (
        "expected finer resolution near tau=0 than near tau_max "
        f"(early gaps {early_gaps}, late gaps {late_gaps})"
    )

    # 6. Sum of all basis functions should be reasonably flat in the interior (classic raised-cosine "50% overlap sums to ~constant" property), away from the very edges where boundary effects bite.
    total = Phi.sum(axis=1)
    interior = total[200:-200]
    assert interior.std() / interior.mean() < 0.15, (
        f"basis sum not flat in interior: mean={interior.mean():.3f}, "
        f"std={interior.std():.3f}"
    )

    print("basis_functions.py: all self-tests passed.")
    print(f"  peak lags (ms): {np.round(peak_taus, 1)}")
    print(f"  early adjacent-peak gaps (ms): {np.round(early_gaps, 1)}")
    print(f"  late adjacent-peak gaps (ms):  {np.round(late_gaps, 1)}")

if __name__ == "__main__":
    _self_test()