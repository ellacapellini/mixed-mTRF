import numpy as np

class RidgemTRF:
    """ Standard  mTRF estimated via ridge regression
    baseline model -> equivalent to a GP with white noise kernel (identity prior — all lagstreated as independent)."""
    def __init__(self, lags, alpha=1.0):
        self.lags = lags
        self.alpha = alpha
        self.coef_ = None       #TRF weights after fitting, shape (n_lags * n_features,)
        self.feature_slices_ = None

    def fit(self, X, y):
        """Fit TRF via ridge regression"""
        n_features = X.shape[1]
        A = X.T @ X + self.alpha * np.eye(n_features)
        b = X.T @ y
        self.coef_ = np.linalg.solve(A, b)
        return self

    def predict(self, X):
        """predic EEG from design matrix"""
        return X @ self.coef_

    def score(self, X, y):
        """pearson correlat between predicted and actual EEG"""
        y_pred = self.predict(X)
        if y.ndim == 1:
            return np.corrcoef(y, y_pred)[0, 1]
        return np.array([
            np.corrcoef(y[:, i], y_pred[:, i])[0, 1]
            for i in range(y.shape[1])
        ])

    def get_trf(self, feature_name, feature_slices):
        """extract TRF for a spec feature."""
        return self.coef_[feature_slices[feature_name]]

    def fit_cv(self, X, y, alphas, n_folds=5):
        n_times = len(y)
        fold_size = n_times // n_folds
        cv_scores = np.zeros(len(alphas))

        for i, alpha in enumerate(alphas):
            fold_corrs = []
            for fold in range(n_folds):
                val_start = fold * fold_size
                val_end = val_start + fold_size
                mask = np.ones(n_times, dtype=bool)
                mask[val_start:val_end] = False
                X_train, y_train = X[mask], y[mask]
                X_val, y_val = X[~mask], y[~mask]
                model = RidgemTRF(self.lags, alpha=alpha)
                model.fit(X_train, y_train)
                fold_corrs.append(model.score(X_val, y_val))
            cv_scores[i] = np.mean(fold_corrs)
        best_alpha = alphas[np.argmax(cv_scores)]
        self.alpha = best_alpha
        self.fit(X, y)
        return best_alpha, cv_scores