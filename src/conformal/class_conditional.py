"""
Class-Conditional Conformal Prediction (RAPS-CC).

Standard RAPS provides MARGINAL coverage:
    P(Y in S(X)) >= 1-alpha  [averaged over all examples]

But for clinical use, we need CONDITIONAL coverage per pathology:
    P(Y=c in S(X) | Y=c) >= 1-alpha  for each class c

This matters because rare pathologies (e.g. Pneumothorax) can have
marginal coverage that looks fine while being missed on the specific
cases where it matters most.

RAPS-CC calibrates a separate threshold tau_c for each class c
using only the calibration examples where Y=c.

Warning: requires sufficient calibration examples per class.
Classes with < min_calib_per_class examples fall back to marginal tau.

Reference:
  Angelopoulos et al. (2021) — class-conditional variant discussion
  Venn predictors and their extensions for conditional coverage
"""

import numpy as np
from typing import Dict, List, Optional
from .raps import RAPS


class ClassConditionalRAPS:
    """
    Per-class calibrated RAPS.

    For each class c:
        tau_c = RAPS threshold calibrated on {(x_i, y_i) : y_i = c}

    Prediction set:
        S(x) = {c : score_c(x) <= tau_c}

    Args:
        alpha:           target miscoverage per class
        lambda_reg:      RAPS regularization
        k_reg:           RAPS regularization rank
        min_calib_per_class: minimum examples per class for class-specific tau
                             (fallback to marginal tau if insufficient)
        num_classes:     number of classes
    """

    def __init__(self, alpha=0.10, lambda_reg=0.01, k_reg=5,
                 min_calib_per_class=50, num_classes=14):
        self.alpha                = alpha
        self.lambda_reg           = lambda_reg
        self.k_reg                = k_reg
        self.min_calib_per_class  = min_calib_per_class
        self.num_classes          = num_classes

        self.tau_per_class   = {}    # tau_c for each class
        self.marginal_tau    = None  # fallback
        self.n_calib_per_class = {}

    def calibrate(self, probs: np.ndarray, labels: np.ndarray) -> Dict[int, float]:
        """
        Calibrate per-class thresholds.

        Args:
            probs:  (n, num_classes) softmax probabilities
            labels: (n,) integer primary class labels
        Returns:
            tau_per_class dict
        """
        # First calibrate marginal RAPS as fallback
        base_raps = RAPS(alpha=self.alpha, lambda_reg=self.lambda_reg,
                         k_reg=self.k_reg)
        self.marginal_tau = base_raps.calibrate(probs, labels)

        # Then calibrate per class
        for c in range(self.num_classes):
            mask = (labels == c)
            n_c  = mask.sum()
            self.n_calib_per_class[c] = int(n_c)

            if n_c >= self.min_calib_per_class:
                raps_c = RAPS(alpha=self.alpha, lambda_reg=self.lambda_reg,
                              k_reg=self.k_reg)
                raps_c.calibrate(probs[mask], labels[mask])
                self.tau_per_class[c] = raps_c.tau
            else:
                # Insufficient calibration data — use marginal tau
                self.tau_per_class[c] = self.marginal_tau

        fitted = sum(1 for c in self.tau_per_class
                     if self.n_calib_per_class[c] >= self.min_calib_per_class)
        print(f"[RAPS-CC] Calibrated {fitted}/{self.num_classes} class-specific thresholds")
        return self.tau_per_class

    def _raps_score_for_class(self, probs: np.ndarray, c: int) -> float:
        """Non-conformity score of class c given softmax probs."""
        sorted_idx = np.argsort(probs)[::-1]
        rank = np.where(sorted_idx == c)[0][0]
        cumsum = np.cumsum(probs[sorted_idx])
        score  = cumsum[rank]
        score += self.lambda_reg * max(rank + 1 - self.k_reg, 0)
        score -= np.random.uniform() * probs[c]
        return float(score)

    def predict_single(self, probs: np.ndarray) -> List[int]:
        """Prediction set using per-class thresholds."""
        assert self.tau_per_class, "Call calibrate() first"
        pred_set = []
        for c in range(self.num_classes):
            tau_c = self.tau_per_class.get(c, self.marginal_tau)
            score = self._raps_score_for_class(probs, c)
            if score <= tau_c:
                pred_set.append(c)
        return pred_set

    def predict(self, probs: np.ndarray) -> List[List[int]]:
        return [self.predict_single(probs[i]) for i in range(len(probs))]

    def evaluate(self, probs: np.ndarray, labels: np.ndarray) -> dict:
        sets   = self.predict(probs)
        n      = len(labels)
        covered = sum(labels[i] in sets[i] for i in range(n))
        sizes   = [len(s) for s in sets]

        # Per-class coverage
        per_class_cov = {}
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                cov_c = sum(c in sets[i] for i in np.where(mask)[0]) / mask.sum()
                per_class_cov[c] = float(cov_c)

        return {
            "coverage":         covered / n,
            "avg_set_size":     float(np.mean(sizes)),
            "singleton_rate":   sum(1 for s in sets if len(s)==1) / n,
            "per_class_coverage": per_class_cov,
            "min_class_coverage": min(per_class_cov.values()) if per_class_cov else 0.0,
            "n": n,
        }
