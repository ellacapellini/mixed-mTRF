import numpy as np 
class SquaredExponential:
    """SE kernel: k(τ, τ') = σ² · exp(−(τ−τ')² / 2ℓ²)"""

def __init__(self, output_scale=1.0, length_scale=50.0):
    self.output_scale = output_scale
    self.length_scale = length_scale
def __call__(self, taus1, taus2):
    """ Evaluate kernel matric between 2 sets of lags.
    Param
    taus1 : np.ndarray, shape (n,)
    taus2 : np.ndarray, shape (m,)
    return
    K : np.ndarray, shape (n,m)"""
    diff = taus1[:, None] - taus2[None, :]
    return self.output_scale * np.exp(-diff**2/(2*self.length_scale**2))


    