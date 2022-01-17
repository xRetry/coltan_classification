import numpy as np
import abc
from core.functions import summary


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
        return summary.mean(values)

    @staticmethod
    def to_scale(values: np.ndarray) -> np.ndarray:
        return summary.std(values)


class MedianUniEstimator(Estimator):
    @staticmethod
    def to_loc(values: np.ndarray) -> np.ndarray:
        return summary.median(values)

    @staticmethod
    def to_scale(values: np.ndarray) -> np.ndarray:
        return summary.std(values, loc=MedianUniEstimator.to_loc(values))


class HLUniEstimator(Estimator):
    @staticmethod
    def to_loc(values: np.ndarray) -> np.ndarray:
        return summary.hodges_lehmann(values)

    @staticmethod
    def to_scale(values: np.ndarray) -> np.ndarray:
        return summary.std(values, loc=HLUniEstimator.to_loc(values))


class MeanMultiEstimator(Estimator):
    @staticmethod
    def to_loc(values: np.ndarray) -> np.ndarray:
        return summary.mean(values)[:, None]

    @staticmethod
    def to_scale(values: np.ndarray) -> np.ndarray:
        return summary.cov(values)


if __name__ == '__main__':
    pass
