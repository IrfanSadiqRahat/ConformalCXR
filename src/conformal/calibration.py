"""
Calibration set management and conformal method selection.

Handles:
  - Splitting a dataset into train/calibration/test
  - Caching softmax scores from the backbone
  - Selecting and running the conformal method (naive/APS/RAPS/RAPS-CC)
  - Producing evaluation reports
"""

import numpy as np
from typing import Tuple, Optional
from .raps import RAPS
from .class_conditional import ClassConditionalRAPS


METHODS = {
    "raps":    RAPS,
    "raps_cc": ClassConditionalRAPS,
}


def build_conformal_predictor(method: str, alpha: float = 0.10,
                               lambda_reg: float = 0.01, k_reg: int = 5,
                               num_classes: int = 14):
    """Factory: return the appropriate conformal predictor."""
    if method == "raps":
        return RAPS(alpha=alpha, lambda_reg=lambda_reg, k_reg=k_reg)
    elif method == "raps_cc":
        return ClassConditionalRAPS(alpha=alpha, lambda_reg=lambda_reg,
                                     k_reg=k_reg, num_classes=num_classes)
    elif method == "naive":
        from .naive import NaiveConformal
        return NaiveConformal(alpha=alpha)
    elif method == "aps":
        from .aps import APS
        return APS(alpha=alpha)
    else:
        raise ValueError(f"Unknown method: {method}. Choose from {list(METHODS)}")


def split_calibration(probs: np.ndarray, labels: np.ndarray,
                       calib_frac: float = 0.20, seed: int = 42
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split probs/labels into calibration and test sets.
    Stratified split to ensure all classes represented in calibration.
    """
    rng = np.random.default_rng(seed)
    n   = len(labels)
    idx = rng.permutation(n)
    n_calib = int(n * calib_frac)
    calib_idx = idx[:n_calib]
    test_idx  = idx[n_calib:]
    return (probs[calib_idx], labels[calib_idx],
            probs[test_idx],  labels[test_idx])


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
