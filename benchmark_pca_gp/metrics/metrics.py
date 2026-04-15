"""Evaluation metrics for the PCA-GP benchmark."""
import numpy as np
from typing import List, Optional, Dict, Any
from scipy import stats


# ---------------------------------------------------------------------------
# Scalar metrics
# ---------------------------------------------------------------------------

def compute_q2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Q² coefficient of determination.

    Q² = 1 - SS_res / SS_tot
    where SS_tot uses the mean of y_true.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-15))


def compute_rrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Relative Root Mean Square Error.

    RRMSE = RMSE / std(y_true)  (or / RMS(y_true) if mean ≈ 0)
    """
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    scale = np.sqrt(np.mean(y_true ** 2))
    return float(rmse / (scale + 1e-15))


def compute_interval_width(
    y_var: np.ndarray, confidence: float = 0.95
) -> np.ndarray:
    """Mean 95% predictive interval width.

    Width = 2 * z_{α/2} * sqrt(var)

    Parameters
    ----------
    y_var      : (N, S) or (N,) variance array
    confidence : coverage level

    Returns
    -------
    mean_width : mean over all entries (scalar)
    """
    z = float(stats.norm.ppf(0.5 + confidence / 2))
    widths = 2.0 * z * np.sqrt(np.maximum(y_var, 0.0))
    return float(widths.mean())


def compute_constraint_error(
    fields_pred: List[np.ndarray], u: np.ndarray
) -> Dict[str, float]:
    """Measure violation of the sum constraint u.T @ f = 0.

    Parameters
    ----------
    fields_pred : List[Q of (N, S)] predicted fields (in original units)
    u           : constraint vector of length Q

    Returns
    -------
    dict with 'max_violation' and 'mean_violation'
    """
    total = sum(u[i] * fields_pred[i] for i in range(len(fields_pred)))
    abs_viol = np.abs(total)
    return {
        "max_violation":  float(abs_viol.max()),
        "mean_violation": float(abs_viol.mean()),
    }


def compute_constraint_error_samples(
    samples: np.ndarray, u: np.ndarray
) -> Dict[str, float]:
    """Constraint violation on posterior samples.

    Parameters
    ----------
    samples : (N_test, Q, n_samples) array of posterior field samples
    u       : constraint vector of length Q

    Returns
    -------
    dict with 'max_violation_samples' and 'mean_violation_samples'
    """
    # weighted sum over Q outputs: (N_test, n_samples)
    total = sum(u[i] * samples[:, i, :] for i in range(samples.shape[1]))
    abs_viol = np.abs(total)
    return {
        "max_violation_samples":  float(abs_viol.max()),
        "mean_violation_samples": float(abs_viol.mean()),
    }


# ---------------------------------------------------------------------------
# Per-field metrics
# ---------------------------------------------------------------------------

def compute_field_metrics(
    f_true: np.ndarray,
    f_mean: np.ndarray,
    f_var: np.ndarray,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """Q², RRMSE, and interval width for a single field.

    Parameters
    ----------
    f_true  : (N_test, S) true field
    f_mean  : (N_test, S) predicted mean
    f_var   : (N_test, S) predicted variance

    Returns
    -------
    dict with 'q2', 'rrmse', 'interval_width'
    """
    return {
        "q2":             compute_q2(f_true.ravel(), f_mean.ravel()),
        "rrmse":          compute_rrmse(f_true.ravel(), f_mean.ravel()),
        "interval_width": compute_interval_width(f_var),
    }


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def compute_all_metrics(
    predictions: Dict[str, List[np.ndarray]],
    fields_test_orig: List[np.ndarray],
    means_train: List[np.ndarray],
    u: np.ndarray,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """Compute all benchmark metrics for all output fields.

    Parameters
    ----------
    predictions     : {'fields_mean': List[Q arrays (N,S)], 'fields_var': ...}
    fields_test_orig: List[Q of (N_test, S)] – true original-scale test fields
    means_train     : List[Q of (S,)] – training means (already added back in preds)
    u               : constraint vector
    confidence      : coverage for interval width

    Returns
    -------
    dict with:
        'q2'              : (Q,) array
        'rrmse'           : (Q,) array
        'interval_width'  : (Q,) array
        'constraint_mean' : scalar  (on mean predictions)
        'constraint_max'  : scalar
    """
    Q = len(fields_test_orig)
    fields_mean = predictions["fields_mean"]
    fields_var  = predictions["fields_var"]

    q2_arr    = np.zeros(Q)
    rrmse_arr = np.zeros(Q)
    iw_arr    = np.zeros(Q)

    for i in range(Q):
        f_true = fields_test_orig[i]
        m = compute_field_metrics(f_true, fields_mean[i], fields_var[i], confidence)
        q2_arr[i]    = m["q2"]
        rrmse_arr[i] = m["rrmse"]
        iw_arr[i]    = m["interval_width"]

    constr = compute_constraint_error(fields_mean, u)

    return {
        "q2":             q2_arr,
        "rrmse":          rrmse_arr,
        "interval_width": iw_arr,
        "constraint_mean": constr["mean_violation"],
        "constraint_max":  constr["max_violation"],
    }
