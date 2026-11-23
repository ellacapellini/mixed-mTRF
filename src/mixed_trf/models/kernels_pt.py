"""Symbolic (PyTensor) GP kernels, vendored verbatim from the legacy bayesian_gp_trf.py.
sigma is the output scale (variance multiplier) directly."""
import pytensor.tensor as pt

def _matern52_pt(taus, ell, sigma):
    """sigma is output_scale (variance multiplier) DIRECTLY, matching
    kernels.py's Matern52/SquaredExponential convention and
    hierarchical_kernel.py's own documented semantics (log_sigma_0 ->
    exp -> used as output_scale, no further squaring). NOT a standard
    deviation to be squared -- see module docstring "Design decision"
    note on the sigma/output_scale units bug this replaces."""
    diff = taus[:, None] - taus[None, :]
    r = pt.abs(diff) / ell
    sqrt5_r = pt.sqrt(5.0) * r
    return sigma * (1 + sqrt5_r + (5.0 / 3.0) * r**2) * pt.exp(-sqrt5_r)


def _squared_exponential_pt(taus, ell, sigma):
    """See _matern52_pt's docstring: sigma is output_scale directly."""
    diff = taus[:, None] - taus[None, :]
    return sigma * pt.exp(-(diff**2) / (2 * ell**2))


def _build_kernel_pt(kernel_type, taus, ell, sigma):
    if kernel_type == "matern52":
        return _matern52_pt(taus, ell, sigma)
    elif kernel_type == "squared_exponential":
        return _squared_exponential_pt(taus, ell, sigma)
    else:
        raise ValueError(f"unknown kernel_type {kernel_type!r}")
