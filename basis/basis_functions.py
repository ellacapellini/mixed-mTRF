"""
Raised cosine basis functions for the mixedTRF model
what this needs to do:
  1. build raised cosine basis matrix Φ of shape (n_lags, n_basis)
  2. project stim feature through that basis to get compressed design matrix (n_times, n_basis)
  3. self-contained!
"""
import numpy as np 

def make_raised_cosine_basis(
    lags: np.ndarray,
    n_basis: int = 10,
    log_offset: float = 1.0, 
    overlap_factor: float = 1.5,
) -> np.ndarray:
    # 1. map lags to logtime axis
    #nl(tau) = log(tau+c)   
    nl = np.log(lags+log_offset)
    nl_min, nl_max = nl[0], nl[-1]
    # 2. = spaced centers on log time axis
    centers = nl.linspace(nl_min, nl_max, n_basis)
    spacing = (nl_max - nl_min) / (n_basis - 1) if n_basis > 1 else 1.0
    # 3. half width with overlap 
    #delta > spacing ensures adjancent bumps overlap
    Delta = overlap_factor * spacing
    # 4. evalaute each basis function
    # phi_j peaks at 1.0 when lag is exactly at centre u_j, smoothly decays to 0 at distance Delta on log axis
    Phi = np.zeros((len(lags), n_basis))
    for j, u_j in enumerate(centers):
        dist = nl - u_j
        inside = np.abs(dist)<Delta
        Phi[inside, j] = 0.5 * np.cos(np.pi*dist[inside]/Delta) + 0.5
        return Phi

def project_stimulus(
    stimulus:np.ndarray,
    Phi:np.ndarray,
) -> np.ndarray:
    n_times = len(stimulus)
    n_basis = Phi.shape[1]
    S_tilde = np.zeros((n_times, n_basis))
    for j in range(n_basis):
        conv = np.convolve(stimulus, Phi[:, j], mode='full')
        S_tilde[:, j] = conv[:n_times]
    return S_tilde

def reconstruct_trf(
    weights: np.darray,
    Phi:np.darray,
) -> np.darray:
    return Phi@weights