import numpy as np
import abc
from functions import estimators


'''
    ABSTRACT CLASS
'''


class Estimator(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def to_loc(values: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abc.abstractmethod
    def to_scale(values: np.ndarray) -> np.ndarray:
        pass


'''
    SUBCLASSES
'''


class MLEUniEstimator(Estimator):
    @staticmethod
    def to_loc(values: np.ndarray) -> np.ndarray:
        return estimators.mean(values)

    @staticmethod
    def to_scale(values: np.ndarray) -> np.ndarray:
        return estimators.std(values)


class RobustUniEstimator(Estimator):
    @staticmethod
    def to_loc(values: np.ndarray) -> np.ndarray:
        return estimators.median(values)

    @staticmethod
    def to_scale(values: np.ndarray) -> np.ndarray:
        return estimators.std(values, robust=True)


class MLEMultiEstimator(Estimator):
    @staticmethod
    def to_loc(values: np.ndarray) -> np.ndarray:
        return estimators.mean(values)

    @staticmethod
    def to_scale(values: np.ndarray) -> np.ndarray:
        return estimators.cov(values)


if __name__ == '__main__':
    pass
