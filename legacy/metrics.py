"""
shared scoring functions
"""

from __future__ import annotations

import numpy as np


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    pearson corr between true and predicted signal. 
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.std() < 1e-12 or y_pred.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def train_test_split_contiguous(
    n_times: int, test_fraction: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """
    Contiguous (not shuffled) train/test split, held out from the END
    of the recording
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    n_test = int(round(n_times * test_fraction))
    n_train = n_times - n_test
    train_idx = np.arange(0, n_train)
    test_idx = np.arange(n_train, n_times)
    return train_idx, test_idx


def tf_parameter_recovery_error(
    true_params: np.ndarray, estimated_params: np.ndarray
) -> dict:
    """
    Simple recovery-error summary for synthetic validation
    """
    true_params = np.asarray(true_params, dtype=float)
    estimated_params = np.asarray(estimated_params, dtype=float)
    rmse = float(np.sqrt(np.mean((true_params - estimated_params) ** 2)))
    scale = float(np.sqrt(np.mean(true_params ** 2)))

    if scale < 1e-6:
        normalised_rmse = float("nan")
    else:
        normalised_rmse = rmse / scale

    return {
        "rmse": rmse,
        "normalised_rmse": normalised_rmse,
        "correlation": pearson_r(true_params, estimated_params) if len(true_params) > 1 else float("nan"),
    }


# Self-tests.
def _self_test() -> None:
    rng = np.random.default_rng(3)

    # pearson_r: perfect correlation, zero correlation, anti-correlation.
    x = rng.normal(0, 1, 500)
    assert abs(pearson_r(x, x) - 1.0) < 1e-10
    assert abs(pearson_r(x, -x) - (-1.0)) < 1e-10
    assert abs(pearson_r(x, np.ones_like(x))) < 1e-10  # constant -> 0, not NaN

    # train_test_split_contiguous: correct sizes, no overlap, no gaps.
    train_idx, test_idx = train_test_split_contiguous(1000, test_fraction=0.2)
    assert len(train_idx) == 800 and len(test_idx) == 200
    assert set(train_idx).isdisjoint(set(test_idx))
    assert train_idx.max() + 1 == test_idx.min()  # contiguous, no gap

    # tf_parameter_recovery_error: perfect recovery -> zero error.
    true_p = np.array([1.0, -2.0, 0.5, 3.0])
    result = tf_parameter_recovery_error(true_p, true_p.copy())
    assert result["rmse"] < 1e-10
    assert result["normalised_rmse"] < 1e-10

    print("metrics.py: all self-tests passed.")


if __name__ == "__main__":
    _self_test()