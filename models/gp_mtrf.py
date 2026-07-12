# this is gp_mtrf.py

import numpy as np
from scipy.optimize import minimize

class GPmTRF:
    """mTRF with GP prior"""
    def __init__(self, lags, kernel, noise_var=1.0):
        self.lags      = lags
        self.kernel    = kernel
        self.noise_var = noise_var
        self.mean_ = None   # post mean, shape (n_weights,)
        self.cov_  = None   # post covariance, shape (n_weights, n_weights)
        self.std_  = None   # Marginal std x weight (sqrt of cov diagonal)

    def _build_prior_K(self, n_features):
        """block-diagonal prior covariance over all TRF weights"""
        K_single = self.kernel(self.lags, self.lags)  # (n_lags, n_lags)
        K = np.kron(np.eye(n_features), K_single)
        return K + 1e-6 * np.eye(K.shape[0])

    def fit(self, X, y):
        """compute exact GP posterior (weight-space formulation)"""
        n_times, n_weights = X.shape
        n_features = n_weights // len(self.lags)
        K     = self._build_prior_K(n_features)
        K_inv = np.linalg.inv(K)
        # post precision and covariance
        precision  = K_inv + (X.T @ X) / self.noise_var  # (n_w, n_w)
        self.cov_  = np.linalg.inv(precision)
        #post mean
        self.mean_ = (self.cov_ @ X.T @ y) / self.noise_var
        #marginal std: sqrt of diagonal of Σ
        # clip ensures no - values from floating-point noise
        self.std_  = np.sqrt(np.clip(np.diag(self.cov_), 0, None))
        return self

    def predict(self, X):
        """Predict EEG signal using post mean TRF."""
        return X @ self.mean_

    def score(self, X, y):
        """Pearson r between predicted and actual EEG"""
        return np.corrcoef(y, self.predict(X))[0, 1]

    def get_trf(self, feature_name, feature_slices):
        """Extract post mean and uncertainty for one feature"""
        sl = feature_slices[feature_name]
        return self.mean_[sl], self.std_[sl]

    def credible_interval(self, feature_name, feature_slices, z=2.0):
        """lower + upper bounds of the credible interval"""
        mean, std = self.get_trf(feature_name, feature_slices)
        return mean - z * std, mean + z * std

    def log_marginal_likelihood(self, X, y):
        """Log marginal likelihood in weight space: log p(y | X, θ) """
        if self.cov_ is None:
            self.fit(X, y)

        n_times, n_weights = X.shape
        n_features = n_weights // len(self.lags)

        K     = self._build_prior_K(n_features)
        K_inv = np.linalg.inv(K)
        precision = K_inv + X.T @ X / self.noise_var

        _, log_det_precision = np.linalg.slogdet(precision)
        _, log_det_K         = np.linalg.slogdet(K)
        log_det_S = (n_times * np.log(self.noise_var)
                     + log_det_K
                     + log_det_precision)

        Sigma_Xty = self.cov_ @ (X.T @ y)
        quad = (y @ y - (X.T @ y) @ Sigma_Xty / self.noise_var) / self.noise_var

        return -0.5 * (quad + log_det_S + n_times * np.log(2 * np.pi))

    def optimise_hyperparams(self, X, y, verbose=True):
        """Learn kernel hyperparameters from data via empirical Bayes """
        def negative_lml(log_params):
            log_os, log_ls, log_nv = log_params
            self.kernel.output_scale = np.exp(log_os)
            self.kernel.length_scale = np.exp(log_ls)
            self.noise_var           = np.exp(log_nv)
            self.fit(X, y)
            return -self.log_marginal_likelihood(X, y)

        x0 = np.array([
            np.log(self.kernel.output_scale),
            np.log(self.kernel.length_scale),
            np.log(self.noise_var)
        ])

        result = minimize(
            negative_lml, x0,
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-6}
        )

        log_os, log_ls, log_nv = result.x
        self.kernel.output_scale = np.exp(log_os)
        self.kernel.length_scale = np.exp(log_ls)
        self.noise_var           = np.exp(log_nv)
        self.fit(X, y)

        if verbose:
            print("Optimised hyperparameters:")
            print(f"  output_scale = {self.kernel.output_scale:.4f}")
            print(f"  length_scale = {self.kernel.length_scale:.4f} samples "
                  f"({self.kernel.length_scale * 10:.1f}ms at 100Hz)")
            print(f"  noise_var    = {self.noise_var:.4f}")
            print(f"  LML          = {self.log_marginal_likelihood(X, y):.2f}")
            print(f"  Converged    : {result.success}")

        return self