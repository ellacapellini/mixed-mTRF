"""
gp_mtrf.py -- GP-prior mTRF, point estimate (empirical Bayes / MAP)

WHY this is the non-hierarchical sanity-check layer for Model B,
mirroring exactly the role mixed_trf.py (ridge) plays for Model A:
NUTS is slow, and the hierarchical Bayesian GP model
(bayesian_gp_trf.py, not yet built) has a lot that can go wrong --
wrong design matrix, wrong kernel, wrong hyperparameter composition.
If this fast, exact, non-hierarchical GP fit can't recover a known
kernel on synthetic data, there's no point debugging NUTS yet.

WHAT changed from the first version: originally used ONE kernel
shared across every feature block via np.kron(eye(n_features),
K_single) -- meaning h^(0) and h^(1) (equations doc Section 7.3)
were forced to share identical hyperparameters. That's a real
mismatch with the doc's formulation, which gives them independent
theta^(0), theta^(1). Generalised to accept a LIST of kernels, one
per feature block, block-diagonal rather than Kronecker -- while
still accepting a single kernel object for backward compatibility
(broadcast to every block), so the simple single-kernel case (e.g.
the Ridge = GP(WhiteNoise) equivalence check below) still reads the
same as before.

HOW blocks are inferred: n_features = n_weights // len(self.lags),
i.e. this assumes every feature block is the same size (n_lags
columns each) -- true for our S0/S1 design matrix (equations doc
Section 4/7.3), where both the plain and surprisal-weighted
regressors have exactly n_lags columns. Documented explicitly here
since it's an assumption baked into the block-size inference, not
something the code checks for you.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import block_diag


class GPmTRF:
    """mTRF with GP prior(s), one kernel per feature block."""

    def __init__(self, lags, kernels, noise_var=1.0):
        """
        kernels : a single kernel object (broadcast to every feature
            block -- the old single-kernel behaviour, unchanged) OR a
            list of kernel objects, one per feature block, allowing
            h^(0) and h^(1) to have independent hyperparameters.
        """
        self.lags = lags
        # normalise to a list internally; remember whether the caller
        # passed a single kernel so optimise_hyperparams can decide
        # whether "the same kernel for every block" was actually intended
        # (relevant for the Ridge=GP(WhiteNoise) check below, where a
        # single shared kernel is the whole point).
        self._single_kernel_input = not isinstance(kernels, (list, tuple))
        self.kernels = [kernels] if self._single_kernel_input else list(kernels)
        self.noise_var = noise_var
        self.mean_ = None   # post mean, shape (n_weights,)
        self.cov_ = None    # post covariance, shape (n_weights, n_weights)
        self.std_ = None    # marginal std x weight (sqrt of cov diagonal)

    def _n_features(self, n_weights: int) -> int:
        n_lags = len(self.lags)
        assert n_weights % n_lags == 0, (
            f"n_weights ({n_weights}) is not a multiple of n_lags ({n_lags}) -- "
            "GPmTRF assumes equal-sized feature blocks (e.g. S0/S1 from the design matrix)"
        )
        return n_weights // n_lags

    def _kernels_for(self, n_features: int) -> list:
        """
        Resolve self.kernels (possibly length 1, from a single-kernel
        constructor call) against the actual number of feature blocks
        in the data. A length-1 list broadcasts to every block --
        this is what makes the single-kernel call path behave exactly
        as the original shared-kernel implementation did.
        """
        if len(self.kernels) == n_features:
            return self.kernels
        if len(self.kernels) == 1:
            return self.kernels * n_features
        raise ValueError(
            f"got {len(self.kernels)} kernels but data has {n_features} feature blocks -- "
            "pass either one kernel (broadcast to all blocks) or exactly n_features kernels"
        )

    def _build_prior_K(self, n_features, jitter_rel: float = 1e-6):
        """Block-diagonal prior covariance over all TRF weights, one
        block per feature, each block from its own kernel.

        Jitter is added PER BLOCK, scaled to that block's own mean
        diagonal (= output_scale, both kernels are stationary), not
        as one fixed constant added to the whole matrix -- h^(0) and
        h^(1) can have very different output_scale after optimisation,
        and a fixed nugget appropriate for one block can be negligible
        or disproportionate for the other."""
        kernels = self._kernels_for(n_features)
        blocks = []
        for k in kernels:
            Kb = k(self.lags, self.lags)
            jitter = jitter_rel * np.mean(np.diag(Kb))
            blocks.append(Kb + jitter * np.eye(Kb.shape[0]))
        return block_diag(*blocks)

    def fit(self, X, y):
        """Compute exact GP posterior (weight-space formulation)."""
        n_times, n_weights = X.shape
        n_features = self._n_features(n_weights)
        K = self._build_prior_K(n_features)
        K_inv = np.linalg.inv(K)
        # post precision and covariance
        precision = K_inv + (X.T @ X) / self.noise_var  # (n_w, n_w)
        self.cov_ = np.linalg.inv(precision)
        # post mean
        self.mean_ = (self.cov_ @ X.T @ y) / self.noise_var
        # marginal std: sqrt of diagonal of Sigma
        # clip ensures no negative values from floating-point noise
        self.std_ = np.sqrt(np.clip(np.diag(self.cov_), 0, None))
        return self

    def predict(self, X):
        """Predict EEG signal using post mean TRF."""
        return X @ self.mean_

    def score(self, X, y):
        """Pearson r between predicted and actual EEG."""
        return np.corrcoef(y, self.predict(X))[0, 1]

    def get_trf(self, feature_name, feature_slices):
        """Extract post mean and uncertainty for one feature."""
        sl = feature_slices[feature_name]
        return self.mean_[sl], self.std_[sl]

    def credible_interval(self, feature_name, feature_slices, z=2.0):
        """Lower + upper bounds of the credible interval."""
        mean, std = self.get_trf(feature_name, feature_slices)
        return mean - z * std, mean + z * std

    def log_marginal_likelihood(self, X, y):
        """Log marginal likelihood in weight space: log p(y | X, theta)."""
        if self.cov_ is None:
            self.fit(X, y)

        n_times, n_weights = X.shape
        n_features = self._n_features(n_weights)

        K = self._build_prior_K(n_features)
        K_inv = np.linalg.inv(K)
        precision = K_inv + X.T @ X / self.noise_var

        _, log_det_precision = np.linalg.slogdet(precision)
        _, log_det_K = np.linalg.slogdet(K)
        log_det_S = (n_times * np.log(self.noise_var)
                     + log_det_K
                     + log_det_precision)

        Sigma_Xty = self.cov_ @ (X.T @ y)
        quad = (y @ y - (X.T @ y) @ Sigma_Xty / self.noise_var) / self.noise_var

        return -0.5 * (quad + log_det_S + n_times * np.log(2 * np.pi))

    def optimise_hyperparams(self, X, y, verbose=True):
        """
        Learn kernel hyperparameters from data via empirical Bayes.
        Optimises EACH feature block's (output_scale, length_scale)
        independently, plus one shared noise_var -- 2*n_features + 1
        parameters total, up from 3 in the single-shared-kernel version.
        """
        n_features = self._n_features(X.shape[1])
        kernels = self._kernels_for(n_features)
        # from here on, self.kernels is definitely the resolved
        # per-feature list (not the possibly-length-1 constructor input),
        # since we're about to mutate each kernel's hyperparameters
        # independently during optimisation.
        self.kernels = kernels

        def negative_lml(log_params):
            for i, k in enumerate(self.kernels):
                k.output_scale = np.exp(log_params[2 * i])
                k.length_scale = np.exp(log_params[2 * i + 1])
            self.noise_var = np.exp(log_params[-1])
            self.fit(X, y)
            return -self.log_marginal_likelihood(X, y)

        x0 = np.concatenate([
            [np.log(k.output_scale), np.log(k.length_scale)] for k in self.kernels
        ] + [[np.log(self.noise_var)]])

        result = minimize(
            negative_lml, x0,
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-6}
        )

        for i, k in enumerate(self.kernels):
            k.output_scale = np.exp(result.x[2 * i])
            k.length_scale = np.exp(result.x[2 * i + 1])
        self.noise_var = np.exp(result.x[-1])
        self.fit(X, y)

        if verbose:
            print("Optimised hyperparameters:")
            for i, k in enumerate(self.kernels):
                print(f"  block {i}: output_scale={k.output_scale:.4f}, "
                      f"length_scale={k.length_scale:.4f}")
            print(f"  noise_var    = {self.noise_var:.4f}")
            print(f"  LML          = {self.log_marginal_likelihood(X, y):.2f}")
            print(f"  Converged    : {result.success}")

        return self


# --------------------------------------------------------------------------
# Self-tests. Run this file directly: `python -m models.gp_mtrf`
# --------------------------------------------------------------------------
def _self_test() -> None:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from utils.kernels import WhiteNoise, Matern52

    rng = np.random.default_rng(0)
    lags = np.arange(15, dtype=float)
    n_times = 500

    # --- Test 1: single-kernel call broadcasts identically to an explicit
    # list of two identical kernels (backward-compatibility guarantee) ------
    X = rng.normal(0, 1, size=(n_times, 2 * len(lags)))  # two feature blocks
    true_w = np.zeros(2 * len(lags))
    true_w[3] = 2.0
    true_w[len(lags) + 7] = -1.5
    y = X @ true_w + rng.normal(0, 0.5, n_times)

    single = GPmTRF(lags=lags, kernels=WhiteNoise(output_scale=1.0), noise_var=1.0).fit(X, y)
    explicit_list = GPmTRF(
        lags=lags, kernels=[WhiteNoise(output_scale=1.0), WhiteNoise(output_scale=1.0)], noise_var=1.0
    ).fit(X, y)
    assert np.allclose(single.mean_, explicit_list.mean_, atol=1e-10), (
        "single-kernel broadcast should give identical results to an explicit "
        "list of the same kernel repeated per block"
    )
    print("gp_mtrf.py: single-kernel broadcast matches explicit per-block list.")

    # --- Test 2: Ridge = GP(WhiteNoise) equivalence, now a permanent test --
    # (previously only checked once by hand; making it a real regression
    # test so this claim, cited in the equations doc Section 6, can't
    # silently stop being true after a future edit).
    from sklearn.linear_model import Ridge
    alpha = 2.0
    ridge = Ridge(alpha=alpha, fit_intercept=False).fit(X, y)
    noise_var = 1.0
    gp_wn = GPmTRF(
        lags=lags, kernels=WhiteNoise(output_scale=noise_var / alpha), noise_var=noise_var
    ).fit(X, y)
    max_diff = np.max(np.abs(ridge.coef_ - gp_wn.mean_))
    assert max_diff < 1e-6, (
        f"Ridge vs GP(WhiteNoise) should match to numerical precision, got max diff {max_diff:.2e}"
    )
    print(f"gp_mtrf.py: Ridge = GP(WhiteNoise) equivalence still holds "
          f"(max weight difference: {max_diff:.2e}).")

    # --- Test 3: different kernels per block actually produce a block- -----
    # diagonal K with the RIGHT sub-blocks, not silently reusing one kernel.
    k0 = Matern52(output_scale=1.0, length_scale=10.0)
    k1 = Matern52(output_scale=3.0, length_scale=40.0)
    gp_diff = GPmTRF(lags=lags, kernels=[k0, k1], noise_var=1.0)
    K = gp_diff._build_prior_K(n_features=2)
    n_lags = len(lags)
    # jitter is now RELATIVE to each block's own output_scale (k0=1.0, k1=3.0),
    # so the two blocks get different absolute nugget amounts -- strip each
    # block's own jitter, not one shared constant.
    K_block0 = K[:n_lags, :n_lags] - (1e-6 * k0.output_scale) * np.eye(n_lags)
    K_block1 = K[n_lags:, n_lags:] - (1e-6 * k1.output_scale) * np.eye(n_lags)
    expected_block0 = k0(lags, lags)
    expected_block1 = k1(lags, lags)
    assert np.allclose(K_block0, expected_block0, atol=1e-9), "block 0 of K doesn't match kernel 0"
    assert np.allclose(K_block1, expected_block1, atol=1e-9), "block 1 of K doesn't match kernel 1"
    assert not np.allclose(K_block0, K_block1), (
        "the two blocks should differ, since k0 and k1 have different hyperparameters -- "
        "if this fails, the block-diagonal construction may be silently reusing one kernel"
    )
    # off-diagonal cross-blocks must be exactly zero (block-diagonal, not full/dense)
    K_cross = K[:n_lags, n_lags:]
    assert np.allclose(K_cross, 0.0), "cross-block entries should be exactly zero (block-diagonal prior)"
    print("gp_mtrf.py: independent per-block kernels produce the correct block-diagonal "
          "prior, with zero cross-block correlation.")

    # --- Test 4: optimise_hyperparams recovers DIFFERENT lengthscales per --
    # block when the data genuinely needs different smoothness per block --
    # confirms optimisation isn't accidentally tying the two blocks together.
    n_times_opt = 3000
    lags_opt = np.arange(20, dtype=float)
    true_k_short = Matern52(output_scale=1.0, length_scale=3.0)
    true_k_long = Matern52(output_scale=1.0, length_scale=25.0)
    w_short = rng.multivariate_normal(np.zeros(len(lags_opt)), true_k_short(lags_opt, lags_opt))
    w_long = rng.multivariate_normal(np.zeros(len(lags_opt)), true_k_long(lags_opt, lags_opt))
    true_w_opt = np.concatenate([w_short, w_long])
    X_opt = rng.normal(0, 1, size=(n_times_opt, 2 * len(lags_opt)))
    y_opt = X_opt @ true_w_opt + rng.normal(0, 0.3, n_times_opt)

    gp_opt = GPmTRF(
        lags=lags_opt,
        kernels=[Matern52(output_scale=1.0, length_scale=10.0), Matern52(output_scale=1.0, length_scale=10.0)],
        noise_var=1.0,
    )
    gp_opt.optimise_hyperparams(X_opt, y_opt, verbose=False)
    ls_recovered_short = gp_opt.kernels[0].length_scale
    ls_recovered_long = gp_opt.kernels[1].length_scale
    assert ls_recovered_long > ls_recovered_short, (
        f"expected block 0 (true ell=3) to recover a shorter lengthscale than "
        f"block 1 (true ell=25), got block0={ls_recovered_short:.2f}, block1={ls_recovered_long:.2f}"
    )
    print(f"gp_mtrf.py: optimise_hyperparams recovers genuinely different lengthscales "
          f"per block (block0={ls_recovered_short:.2f} vs true 3.0, "
          f"block1={ls_recovered_long:.2f} vs true 25.0) -- not tied together.")

    print("gp_mtrf.py: all self-tests passed.")


if __name__ == "__main__":
    _self_test()