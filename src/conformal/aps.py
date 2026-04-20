"""APS (Adaptive Prediction Sets) baseline — RAPS without regularization."""
import numpy as np
from .raps import RAPS


class APS(RAPS):
    """APS = RAPS with lambda_reg=0."""
    def __init__(self, alpha: float = 0.10):
        super().__init__(alpha=alpha, lambda_reg=0.0, k_reg=0)
