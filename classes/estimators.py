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


class MeanUniEstimator(Estimator):
    @staticmethod
    def to_loc(values: np.ndarray) -> np.ndarray:
        return estimators.mean(values)

    @staticmethod
    def to_scale(values: np.ndarray) -> np.ndarray:
        return estimators.std(values)


class MedianUniEstimator(Estimator):
    @staticmethod
    def to_loc(values: np.ndarray) -> np.ndarray:
        return estimators.median(values)

    @staticmethod
    def to_scale(values: np.ndarray) -> np.ndarray:
        return estimators.std(values, loc=MedianUniEstimator.to_loc(values))


class HLUniEstimator(Estimator):
    @staticmethod
    def to_loc(values: np.ndarray) -> np.ndarray:
        return estimators.hodges_lehmann(values)

    @staticmethod
    def to_scale(values: np.ndarray) -> np.ndarray:
        return estimators.std(values, loc=HLUniEstimator.to_loc(values))


class MeanMultiEstimator(Estimator):
    @staticmethod
    def to_loc(values: np.ndarray) -> np.ndarray:
        return estimators.mean(values)[:, None]

    @staticmethod
    def to_scale(values: np.ndarray) -> np.ndarray:
        return estimators.cov(values)


if __name__ == '__main__':
    pass
