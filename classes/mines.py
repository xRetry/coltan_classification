import numpy as np
import abc

from functions.mathematical import normal_uni_mu, normal_uni_sigma_orig, normal_uni_sigma
from classes.dataset import Sample
from classes.distributions import Distribution, MultiNormal
from classes.parameters import Parameters
from typing import List, Optional, Callable, Iterable


'''
    +++ SUPERCLASS +++
'''


class Mine(abc.ABC):
    _coordinates: np.ndarray
    _status: int
    _func_transform: Callable
    _func_eval: Callable
    _distribution: Distribution

    def __init__(self, coordinates: Iterable, status: int, parameters: Parameters,
                 samples: Optional[List[Sample]] = None):
        self._coordinates = np.array(coordinates)
        self._status = status
        self._func_transform = parameters.func_transform
        self._func_eval = parameters.func_eval
        if samples is not None:
            [self.add_sample(sample) for sample in samples]

    def add_sample(self, sample: Sample) -> None:
        self._add_sample(self._func_transform(sample.attributes))

    @abc.abstractmethod
    def _add_sample(self, values) -> None:
        pass

    def eval_sample(self, sample: Sample) -> float:
        attr_values = self._func_transform(sample.attributes)
        return self._func_eval(self._distribution, attr_values)

    @property
    def coordinates(self) -> np.ndarray:
        return self._coordinates

    @property
    def status(self):
        return self._status

    @property
    def distribution(self):
        return self._distribution


''' 
     +++ SUBCLASSES +++
'''


class OrigMine(Mine):
    _mu_samples: List[np.ndarray]

    def __init__(self, **args):
        self._mu_samples = []
        super(OrigMine, self).__init__(**args)

    def _add_sample(self, values) -> None:
        mu_new = normal_uni_mu(values)
        self._mu_samples.append(mu_new)
        mean = normal_uni_mu(self._mu_samples)
        std = normal_uni_sigma_orig(self._mu_samples)
        self._distribution = MultiNormal(mean, np.diag(np.power(std, 2)))


class BaselineMine(Mine):
    _samples: List[np.ndarray]

    def __init__(self, **args):
        self._samples = []
        super(BaselineMine, self).__init__(**args)

    def _add_sample(self, values) -> None:
        self._samples.append(values)
        stacked_samples = np.row_stack(self._samples)
        mu = normal_uni_mu(stacked_samples)
        sigma = normal_uni_sigma(stacked_samples)
        self._distribution = MultiNormal(mu, np.diag(np.power(sigma, 2)))


class BayesianSimpleMine(Mine):
    def __init__(self, mean: np.ndarray, std: np.ndarray, **kwargs):
        super(BayesianSimpleMine, self).__init__(**kwargs)
        mean = np.ones(37) * mean  # TODO: remove hardcoded prior-dim
        cov = np.diag(np.ones(37) * np.power(std, 2))
        self._distribution = MultiNormal(mean, cov)

    def _add_sample(self, values) -> None:
        cov_known = np.diag(np.ones(37))  # TODO: remove hardcoded prior-dim
        self._distribution = self._distribution.posterior(self._func_transform(values), cov_known)


if __name__ == '__main__':
    pass
