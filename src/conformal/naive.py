"""Naive softmax threshold baseline (no coverage guarantee)."""
import numpy as np
from typing import List


class NaiveConformal:
    """
    Include all classes with softmax score >= tau.
    tau chosen so empirical coverage on calibration set >= 1-alpha.
    No formal finite-sample guarantee (unlike RAPS).
    """

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.tau   = None

    def calibrate(self, probs: np.ndarray, labels: np.ndarray) -> float:
        """Find tau s.t. coverage on calib set >= 1-alpha."""
        scores = probs[np.arange(len(labels)), labels]
        n      = len(scores)
        level  = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.tau = float(np.quantile(scores, 1 - level))
        print(f"[Naive] tau={self.tau:.4f}")
        return self.tau

    def predict(self, probs: np.ndarray) -> List[List[int]]:
        assert self.tau is not None
        return [[j for j in range(probs.shape[1]) if probs[i,j] >= self.tau]
                for i in range(len(probs))]

    def evaluate(self, probs, labels):
        sets = self.predict(probs)
        n    = len(labels)
        return {
            "coverage":      sum(labels[i] in sets[i] for i in range(n)) / n,
            "avg_set_size":  float(np.mean([len(s) for s in sets])),
            "singleton_rate": sum(1 for s in sets if len(s)==1) / n,
            "tau": self.tau, "n": n,
        }
