import numpy as np 
class SquaredExponential:
    def __init__(self, output_scale=1.0, length_scale=50.0):
        self.output_scale = output_scale
        self.length_scale = length_scale
    def __call__(self, taus1, taus2):
        diff = taus1[:, None] - taus2[None, :]
        return self.output_scale * np.exp(-diff**2/(2*self.length_scale**2))

class Matern52:
    def __init__(self, output_scale=1.0, length_scale=50.0):
        self.output_scale = output_scale
        self.length_scale = length_scale
    def __call__(self, taus1, tau2):
        diff = taus1[:, None] - taus2[None, :]
        r = np.abs(diff)/self.length_scale
        sqrt5_r = np.sqrt(5)*r
        return self.output_scale * (1+sqrt5_r + (5/3) * r**2) * np.exp(-sqrt5_r)

class WhiteNoise:
    def __init__(self, output_scale=1.0):
        self.output_scale = output_scale
    def __call__(self, taus1, taus2):
        diff = taus1[:, None] - taus2[None, :]
        return self.output_scale * (diff==0).astype(float)

class AdditiveKernel:
    def __init__(self, kernel1, kernel2):
        self.kernel1 = kernel1
        self.kernel2 = kernel2
    def __call__(self, taus1, taus2):
        return self.kernel1(taus1, taus2) + self.kernel2(taus1, taus2)