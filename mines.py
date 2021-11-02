import numpy as np
import abc

import data
import mathematical
from samples import Samples
from typing import List, Optional, Callable
import scipy.stats


'''
    +++ SUPERCLASS +++
'''


class Mine(abc.ABC):
    _x: float
    _y: float
    _z: float
    _status: int
    _transform: Callable

    def __init__(self, x: float, y: float, z: float, status: int, samples: Optional[List[np.ndarray]] = None, transform: Callable=data.no_transform):
        self._x = x
        self._y = y
        self._z = z
        self._status = status
        if samples is not None:
            [self.add_sample(sample) for sample in samples]
        self._transform = transform

    @abc.abstractmethod
    def add_sample(self, values) -> None:
        pass

    @abc.abstractmethod
    def pdf(self, x):
        pass

    def eval_samples(self, samples: Samples) -> np.ndarray:
        result = np.zeros(len(samples))
        for i, sample in enumerate(samples):
            result[i] = self._eval_sample(self._transform(sample))
        return result

    @abc.abstractmethod
    def _eval_sample(self, sample: np.ndarray) -> np.ndarray:
        pass

    @property
    def coordinates(self) -> np.ndarray:
        return np.array([self._x, self._y, self._z])

    @property
    def status(self):
        return self._status


''' 
     +++ SUBCLASSES +++
'''


class OrigMine(Mine):
    _mu_samples: List[np.ndarray]

    def __init__(self, x: float, y: float, z: float, status: int, samples: Optional[List[np.ndarray]] = None,
                 transform: Callable = lambda x: x):
        self._mu_samples = []
        super(OrigMine, self).__init__(x, y, z, status, samples, transform)

    def add_sample(self, values) -> None:
        mu_new = mathematical.normal_mu(self._transform(np.array(values)))
        self._mu_samples.append(mu_new)

    def pdf(self, x) -> np.ndarray:
        return mathematical.normal_pdf(self.mean, self.std, x)

    def _eval_sample(self, sample: np.ndarray):
        sample_means = sample.mean(axis=0)
        attr_probabilities = mathematical.normal_pdf(self.mean, self.std, sample_means)
        return np.product(attr_probabilities)

    @property
    def mean(self) -> np.ndarray:
        return mathematical.normal_mu(self._mu_samples)

    @property
    def std(self) -> np.ndarray:
        return mathematical.normal_sigma_orig(self._mu_samples)


class BaselineMine(Mine):
    _mu: np.ndarray
    _sigma: np.ndarray
    _samples: List[np.ndarray]

    def __init__(self, x: float, y: float, z: float, status: int, samples: Optional[List[np.ndarray]] = None,
                 transform: Callable = lambda x: x):
        self._samples = []
        super(BaselineMine, self).__init__(x, y, z, status, samples, transform)

    def add_sample(self, values) -> None:
        self._samples.append(self._transform(np.array(values)))
        stacked_samples = np.row_stack(self._samples)
        self._mu = mathematical.normal_mu(stacked_samples)
        self._sigma = mathematical.normal_sigma(stacked_samples)

    def pdf(self, x):
        return mathematical.normal_pdf(self.mean, self.std, x)

    def _eval_sample(self, sample: np.ndarray) -> np.ndarray:
        attr_probabilities = scipy.stats.ttest_1samp(sample, self.mean)[1]
        return np.product(attr_probabilities)

    @property
    def mean(self):
        return self._mu

    @property
    def std(self):
        return self._sigma


class BayesianSimpleMine(Mine):
    _mu: np.ndarray
    _sigma: np.ndarray

    def __init__(self, x: float, y: float, z: float, status: int, mu_prior: np.ndarray, sigma_prior: np.ndarray,
                 samples: Optional[List[np.ndarray]] = None, transform: Callable = lambda x: x):
        self._mu = np.ones(37) * mu_prior  # TODO: remove hardcoded prior-dim
        self._sigma = np.ones(37) * sigma_prior
        super(BayesianSimpleMine, self).__init__(x, y, z, status, samples, transform)

    def add_sample(self, values) -> None:
        if len(values[0]) != len(self._mu):
            raise ValueError(f'Attribute dims do not agree with prior dims: {len(self._mu)} vs {len(values[0])}')

        values = self._transform(values)
        #mu_post, sigma_post = mathematical.norm_posterior_skwn(self.mean, self.std, np.std(values, axis=0), values)
        mu_post, sigma_post = mathematical.norm_posterior_skwn(self.mean, self.std, 1, values)
        self._mu = mu_post
        self._sigma = sigma_post

    def pdf(self, x):
        return mathematical.normal_pdf(self.mean, self.std, x)

    def _eval_sample(self, sample: np.ndarray) -> np.ndarray:
        attr_probabilities = scipy.stats.ttest_1samp(sample, self.mean)[1]
        return np.product(attr_probabilities)

    @property
    def mean(self) -> np.ndarray:
        return self._mu

    @property
    def std(self) -> np.ndarray:
        return self._sigma


if __name__ == '__main__':
    pass
