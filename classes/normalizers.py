import numpy as np
from typing import Optional


class Normalization:
    _norm_constant: Optional[np.ndarray]
    _shift: Optional[np.ndarray]

    def __init__(self):
        self._norm_constant = None

    def none(self, x: np.ndarray):
        return x

    def fixed_norm(self, x: np.ndarray):
        if self._norm_constant is None or self._shift is None:
            self._norm_constant = (np.max(x, axis=0) - np.min(x, axis=0))
            self._norm_constant[self._norm_constant == 0] = 1
            self._shift = np.mean(x / self._norm_constant, axis=0)
        return x / self._norm_constant - self._shift


if __name__ == '__main__':
    pass
