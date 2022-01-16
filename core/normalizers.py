import numpy as np
import abc
from typing import Optional

from sklearn.preprocessing import StandardScaler


class Normalizer(abc.ABC):
    is_fitted: bool = False

    @abc.abstractmethod
    def fit(self, x: np.ndarray) -> None:
        pass

    @abc.abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray:
        pass


class NoNormalizer(Normalizer):
    def fit(self, x: np.ndarray) -> None:
        self.is_fitted = True
        pass

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x


class StandardNormalizer(Normalizer):
    _standard_scaler: StandardScaler

    def __init__(self):
        self._standard_scaler = StandardScaler()

    def fit(self, x: np.ndarray) -> None:
        self.is_fitted = True
        self._standard_scaler.fit(x)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self._standard_scaler.transform(x)


if __name__ == '__main__':
    pass
