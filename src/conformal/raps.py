"""
RAPS — Regularized Adaptive Prediction Sets.

The core conformal algorithm in this paper.

Standard softmax thresholding gives:
    S(x) = {y : f(x)_y >= tau}
But tau has no formal coverage guarantee and sets are badly calibrated.

APS (Angelopoulos et al.) gives exact coverage but large sets (size ~19
on ImageNet) because it includes all classes needed to reach cumulative
probability 1-alpha, and low-probability classes have noisy scores.

RAPS adds a regularization term that penalizes large sets:
    score(x, y) = sum_{i: f(x)_i >= f(x)_y} f(x)_i
                  + lambda * max(rank(y) - k_reg, 0) * U

where:
  rank(y) = rank of y when classes sorted by descending score
  k_reg   = regularization threshold (default 5)
  lambda  = regularization strength (default 0.01)
  U       ~ Uniform[0,1] (randomization for exact coverage)

Calibration:
  1. Compute score(x_i, y_i) for all calibration examples i
  2. Set tau = (1-alpha) quantile of these scores

Prediction:
  S(x) = {y : score(x, y) <= tau}

This guarantees: P(Y in S(X)) >= 1-alpha  [finite sample guarantee]

Reference:
  Angelopoulos, Bates et al. (ICLR 2021)
  "Uncertainty Sets for Image Classifiers using Conformal Prediction"
  https://arxiv.org/abs/2009.14193
"""

import numpy as np
import torch
from typing import List, Optional


class RAPS:
    """
    RAPS conformal predictor.

    Usage:
        raps = RAPS(alpha=0.1, lambda_reg=0.01, k_reg=5)

        # Calibrate on held-out calibration set
        raps.calibrate(softmax_scores_calib, labels_calib)

        # Get prediction sets at test time
        prediction_sets = raps.predict(softmax_scores_test)

        # Evaluate coverage
        metrics = raps.evaluate(softmax_scores_test, labels_test)

    Args:
        alpha:      target miscoverage rate (e.g. 0.1 for 90% coverage)
        lambda_reg: RAPS regularization weight (penalizes large rank)
        k_reg:      rank threshold for regularization
        randomized: use randomization for exact finite-sample coverage
    """

    def __init__(self, alpha: float = 0.10, lambda_reg: float = 0.01,
                 k_reg: int = 5, randomized: bool = True):
        self.alpha      = alpha
        self.lambda_reg = lambda_reg
        self.k_reg      = k_reg
        self.randomized = randomized
        self.tau        = None   # calibrated threshold

    def _raps_score(self, probs: np.ndarray, label: int,
                    u: Optional[float] = None) -> float:
        """
        Compute RAPS non-conformity score for a single (x, y) pair.

        Args:
            probs: (num_classes,) softmax probabilities
            label: true class index
            u:     uniform random variable (None = sample new)
        Returns:
            score: scalar RAPS score
        """
        # Sort classes by descending probability
        sorted_idx = np.argsort(probs)[::-1]
        rank = np.where(sorted_idx == label)[0][0]  # 0-indexed rank of true label

        # Cumulative sum up to and including the true label
        cumsum = np.cumsum(probs[sorted_idx])
        score  = cumsum[rank]

        # Regularization: penalize if rank >= k_reg
        reg = self.lambda_reg * max(rank + 1 - self.k_reg, 0)
        score += reg

        # Randomization: subtract u * p(y) for exact coverage
        if self.randomized:
            u_val = u if u is not None else np.random.uniform()
            score -= u_val * probs[label]

        return float(score)

    def _raps_scores_batch(self, probs: np.ndarray,
                           labels: np.ndarray) -> np.ndarray:
        """Vectorized RAPS scores for a calibration batch."""
        n = len(labels)
        scores = np.zeros(n)
        for i in range(n):
            scores[i] = self._raps_score(probs[i], labels[i])
        return scores

    def calibrate(self, probs: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute the calibration threshold tau.

        Args:
            probs:  (n_calib, num_classes) softmax probabilities
            labels: (n_calib,) integer class labels (argmax for multi-label)
        Returns:
            tau: calibrated threshold
        """
        scores = self._raps_scores_batch(probs, labels)
        n      = len(scores)
        # Use the (ceil((n+1)*(1-alpha))/n) quantile for exact coverage
        level  = np.ceil((n + 1) * (1 - self.alpha)) / n
        level  = min(level, 1.0)
        self.tau = float(np.quantile(scores, level))
        print(f"[RAPS] Calibrated | n={n} | alpha={self.alpha} | tau={self.tau:.4f}")
        return self.tau

    def predict_single(self, probs: np.ndarray) -> List[int]:
        """
        Prediction set for a single test example.

        Returns: list of class indices included in the prediction set
        """
        assert self.tau is not None, "Call calibrate() first"
        sorted_idx = np.argsort(probs)[::-1]
        cumsum = 0.0
        pred_set = []
        for rank, idx in enumerate(sorted_idx):
            cumsum += probs[idx]
            reg     = self.lambda_reg * max(rank + 1 - self.k_reg, 0)
            score   = cumsum + reg
            if self.randomized:
                score -= np.random.uniform() * probs[idx]
            pred_set.append(int(idx))
            if score > self.tau:
                break
        return pred_set

    def predict(self, probs: np.ndarray) -> List[List[int]]:
        """
        Prediction sets for a batch.

        Args:
            probs: (n, num_classes) softmax probabilities
        Returns:
            List of prediction sets, one per example
        """
        return [self.predict_single(probs[i]) for i in range(len(probs))]

    def predict_as_binary(self, probs: np.ndarray,
                          num_classes: int) -> np.ndarray:
        """
        Returns (n, num_classes) binary matrix where entry [i,j]=1
        iff class j is in the prediction set for example i.
        """
        n     = len(probs)
        sets  = self.predict(probs)
        out   = np.zeros((n, num_classes), dtype=np.int32)
        for i, s in enumerate(sets):
            for j in s:
                if j < num_classes:
                    out[i, j] = 1
        return out

    def evaluate(self, probs: np.ndarray, labels: np.ndarray) -> dict:
        """
        Compute coverage, average set size, singleton rate.

        For multi-label problems, labels should be argmax or primary label.
        """
        assert self.tau is not None, "Call calibrate() first"
        sets     = self.predict(probs)
        n        = len(labels)
        covered  = sum(labels[i] in sets[i] for i in range(n))
        sizes    = [len(s) for s in sets]
        singletons = sum(1 for s in sets if len(s) == 1)
        return {
            "coverage":      covered / n,
            "avg_set_size":  float(np.mean(sizes)),
            "median_set_size": float(np.median(sizes)),
            "singleton_rate": singletons / n,
            "tau":           self.tau,
            "n":             n,
        }
