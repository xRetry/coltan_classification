import numpy as np
import abc

from functions.mathematical import normal_uni_mean, normal_uni_std, normal_uni_median, normal_multi_cov
from classes.dataset import Sample
from classes.distributions import MultiNormal, NonParametric
from classes.parameters import Parameters
from typing import List, Optional, Callable, Iterable


'''
    ++++++++++++++++++
    +++ SUPERCLASS +++
'''


class Mine(abc.ABC):
    _coordinates: np.ndarray
    _status: int
    _func_transform: Callable
    _func_eval: Callable

    def __init__(self, coordinates: Iterable, status: int, parameters: Parameters):
        self._coordinates = np.array(coordinates)
        self._status = status
        self._func_transform = parameters.func_transform
        self._func_eval = parameters.func_eval
        self._distribution = None  # Dummy distribution (gets overwritten)

    def add_sample(self, sample: Sample) -> None:
        self._add_sample(self._func_transform(sample.attributes))

    @abc.abstractmethod
    def _add_sample(self, values) -> None:
        pass

    def eval_sample(self, sample: Sample) -> float:
        attr_values = self._func_transform(sample.attributes)
        return self._func_eval(self, attr_values)

    @property
    def coordinates(self) -> np.ndarray:
        return self._coordinates

    @property
    def status(self) -> int:
        return self._status

    @property
    def distribution(self):
        return self._distribution


''' 
    ++++++++++++++++++
    +++ SUBCLASSES +++
'''


class OrigMine(Mine):
    _distribution: NonParametric

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_sample(self, values) -> None:
        mean_new = normal_uni_mean(values)
        if self._distribution is None:
            means_stacked = np.array([mean_new])
        else:
            means_stacked = np.row_stack([self._distribution.values, mean_new])
        self._distribution = NonParametric(means_stacked)

    @staticmethod
    def _to_normal(sample: np.ndarray) -> MultiNormal:
        mean = normal_uni_mean(sample)
        std = normal_uni_std(sample, corrected=False)
        return MultiNormal(mean, std=std)

    def eval_pdf(self, sample: np.ndarray) -> float:
        sample_means = sample.mean(axis=0)
        return self._to_normal(self._distribution.values).pdf(sample_means)


class AggregationUniMine(Mine):
    _distribution: NonParametric

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_sample(self, values: np.ndarray) -> None:
        if self._distribution is None:
            stacked_samples = np.array(values)
        else:
            stacked_samples = np.row_stack([self._distribution.values, values])
        self._distribution = NonParametric(stacked_samples)

    @staticmethod
    def _to_normal(sample: np.ndarray) -> MultiNormal:
        mean = normal_uni_mean(sample)
        std = normal_uni_std(sample, corrected=True)
        return MultiNormal(mean, std=std)

    def eval_dot(self, sample: np.ndarray) -> float:
        return self._distribution.test_dot(sample.mean(axis=0))

    def eval_frobenius(self, sample: np.ndarray) -> float:
        return self._distribution.test_norm_frobenius(sample)

    def eval_norm1(self, sample: np.ndarray) -> float:
        return self._distribution.test_norm_1(sample)

    def eval_norm2(self, sample: np.ndarray) -> float:
        return self._distribution.test_norm_2(sample)

    def eval_ttest(self, sample: np.ndarray) -> float:
        return self._to_normal(self._distribution.values).test_1sample(sample)

    def eval_pdf(self, sample: np.ndarray) -> float:
        sample_means = sample.mean(axis=0)
        return self._to_normal(self._distribution.values).pdf(sample_means)

    def eval_kldivergence(self, sample: np.ndarray) -> float:
        sample_distribution = self._to_normal(sample)
        return -self._to_normal(self._distribution.values).kl_divergence(sample_distribution.mean, sample_distribution.cov)

    def eval_ranksums(self, sample: np.ndarray) -> float:
        return self._distribution.test_ranksums(sample)

    def eval_mannwhitneyu(self, sample: np.ndarray) -> float:
        return self._distribution.test_mannwhitneyu(sample)


class AggregationUniMineRobust(AggregationUniMine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _to_normal(sample: np.ndarray) -> MultiNormal:
        median = normal_uni_median(sample)
        std = normal_uni_std(sample, mean=median, corrected=True)
        return MultiNormal(median, std=std)


class AggregationMultiMine(AggregationUniMine):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _to_normal(sample: np.ndarray) -> MultiNormal:
        mean = normal_uni_mean(sample)
        cov = normal_multi_cov(sample, corrected=True)
        return MultiNormal(mean, cov)


class BayesianUniMine(Mine):
    _distribution: MultiNormal

    def __init__(self, mean: np.ndarray, cov: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self._distribution = MultiNormal(mean, cov)

    def _add_sample(self, values) -> None:
        cov_known = np.diag(np.ones(len(self._distribution)))
        self._distribution = self._distribution.posterior(values, cov_known)

    @staticmethod
    def _to_normal(sample: np.ndarray) -> MultiNormal:
        mean = normal_uni_mean(sample)
        std = normal_uni_std(sample, corrected=True)
        return MultiNormal(mean, std=std)

    def eval_pdf(self, sample: np.ndarray) -> float:
        sample_means = sample.mean(axis=0)
        return self._distribution.pdf(sample_means)

    def eval_ttest(self, sample: np.ndarray) -> float:
        return self._distribution.test_1sample(sample)

    def eval_kldivergence(self, sample: np.ndarray) -> float:
        sample_distribution = self._to_normal(sample)
        return -self._distribution.kl_divergence(sample_distribution.mean, sample_distribution.cov)


if __name__ == '__main__':
    pass
