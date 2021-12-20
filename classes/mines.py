import numpy as np
import abc

from classes.dataset import Sample
from functions.distributions import uni_normal, multi_normal, normal_inverse_wishart, non_parametric, normal_inverse_chisquared
from classes.parameters import Parameters
from classes.estimators import Estimator
from classes.normalizers import Normalization
from typing import List, Optional, Callable, Iterable


'''
    ABSTRACT CLASS
'''


class Mine(abc.ABC):
    _coordinates: np.ndarray
    _status: int
    _func_normalize: Callable
    _func_transform: Callable
    _func_eval: Callable
    _eval_args: dict
    _estimator: Estimator

    def __init__(self, coordinates: Iterable, status: int, parameters: Parameters):
        self._coordinates = np.array(coordinates)
        self._status = status
        name_func_normalize = parameters.func_normalize.__name__
        self._func_normalize = Normalization().__getattribute__(name_func_normalize)
        self._func_transform = parameters.func_transform
        self._func_eval = parameters.func_eval
        self._eval_args = parameters.eval_args
        self._estimator = parameters.estimator

    def add_sample(self, sample: Sample) -> None:
        self._add_sample(self._func_normalize(self._func_transform(sample.attributes)))

    @abc.abstractmethod
    def _add_sample(self, values) -> None:
        pass

    def eval_sample(self, sample: Sample) -> float:
        attr_values = self._func_normalize(self._func_transform(sample.attributes))
        return self._func_eval(self, attr_values, **self._eval_args)

    @property
    def coordinates(self) -> np.ndarray:
        return self._coordinates

    @property
    def status(self) -> int:
        return self._status

    @property
    @abc.abstractmethod
    def parameters(self) -> dict:
        pass


''' 
    SUBCLASSES
'''


class OrigMine(Mine):
    _parameters: Optional[Sample]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parameters = None

    def _add_sample(self, attr_values: np.ndarray) -> None:
        loc = self._estimator.to_loc(attr_values)
        if self._parameters is None:
            self._parameters = Sample(attributes=np.expand_dims(loc, axis=0))
        else:
            self._parameters.append(loc)

    def eval_pdf(self, x: np.ndarray) -> float:
        loc = self._estimator.to_loc(self._parameters.attributes)
        scale = self._estimator.to_scale(self._parameters.attributes)
        return uni_normal.pdf(loc, scale, self._estimator.to_loc(x))

    @property
    def parameters(self) -> dict:
        return {
            'Location': self._estimator.to_loc(self._parameters.attributes),
            'Scale': self._estimator.to_scale(self._parameters.attributes)
        }


class AggregationMine(Mine):
    _parameters: Optional[Sample]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parameters = None

    def _add_sample(self, attr_values: np.ndarray) -> None:
        if self._parameters is None:
            self._parameters = Sample(attributes=attr_values)
        else:
            self._parameters.append(attr_values)

    def eval_frobenius(self, x: np.ndarray) -> float:
        loc = self._estimator.to_loc(self._parameters.attributes)
        return non_parametric.test_norm_frobenius(loc, self._estimator.to_loc(x))

    def eval_norm1(self, x: np.ndarray) -> float:
        loc = self._estimator.to_loc(self._parameters.attributes)
        return non_parametric.test_norm1(loc, self._estimator.to_loc(x))

    def eval_norm2(self, x: np.ndarray) -> float:
        loc = self._estimator.to_loc(self._parameters.attributes)
        return non_parametric.test_norm2(loc, self._estimator.to_loc(x))

    def eval_exponential(self, x: np.ndarray, exponent: float=2, scale: float=1) -> float:
        loc = self._estimator.to_loc(self._parameters.attributes)
        return non_parametric.test_exponential(loc, self._estimator.to_loc(x), exponent, scale)

    def eval_ttest(self, x: np.ndarray) -> float:
        loc = self._estimator.to_loc(self._parameters.attributes)
        return uni_normal.test_1sample(loc, x)

    def eval_pdf(self, x: np.ndarray) -> float:
        loc = self._estimator.to_loc(self._parameters.attributes)
        scale = self._estimator.to_scale(self._parameters.attributes)
        return uni_normal.pdf(loc, scale, self._estimator.to_loc(x))

    def eval_ranksums(self, x: np.ndarray) -> float:
        return non_parametric.test_ranksums(self._parameters.attributes, x)

    def eval_mannwhitneyu(self, x: np.ndarray) -> float:
        return non_parametric.test_mannwhitneyu(self._parameters.attributes, x)

    @property
    def parameters(self) -> dict:
        return {
            'Location': self._estimator.to_loc(self._parameters.attributes),
            'Scale': self._estimator.to_scale(self._parameters.attributes)
        }


class AggregationMultiMine(Mine):
    _parameters: Optional[Sample]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._parameters = None

    def _add_sample(self, attr_values: np.ndarray) -> None:
        if self._parameters is None:
            self._parameters = Sample(attributes=attr_values)
        else:
            self._parameters.append(attr_values)

    def eval_pdf(self, x: np.ndarray) -> float:
        loc = self._estimator.to_loc(self._parameters.attributes)
        scale = self._estimator.to_scale(self._parameters.attributes)
        return multi_normal.pdf(loc, scale, self._estimator.to_loc(x))

    @property
    def parameters(self) -> dict:
        return {
            'Location': self._estimator.to_loc(self._parameters.attributes),
            'Scale': self._estimator.to_scale(self._parameters.attributes)
        }


class BayesianSimpleMine(Mine):
    _mean: np.ndarray
    _std: np.ndarray

    def __init__(self, mean: np.ndarray, std: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self._mean = mean
        self._std = std

    def _add_sample(self, values: np.ndarray) -> None:
        std_known = np.ones(len(self._mean))
        self._mean, self._std = uni_normal.posterior(self._mean, self._std, std_known, values)

    def eval_pdf(self, x: np.ndarray) -> float:
        return uni_normal.pdf(self._mean, self._std, self._estimator.to_loc(x))

    def eval_ttest(self, x: np.ndarray) -> float:
        return uni_normal.test_1sample(self._mean, x)

    @property
    def parameters(self) -> dict:
        return {
            'Location': self._mean,
            'Scale': self._std
        }


class BayesianUniMine(Mine):
    _loc: np.ndarray
    _scale: np.ndarray
    _kappa: int
    _nu: int

    def __init__(self, loc, scale, kappa, nu, **kwargs):
        super().__init__(**kwargs)
        self._loc = loc
        self._scale = scale
        self._kappa = kappa
        self._nu = nu

    def _add_sample(self, values: np.ndarray) -> None:
        self._loc, self._scale, self._kappa, self._nu = normal_inverse_chisquared.posterior(
            self._loc,
            self._scale,
            self._kappa,
            self._nu,
            values
        )

    def eval_pdf(self, x: np.ndarray) -> float:
        x_mean = self._estimator.to_loc(x)
        x_std = self._estimator.to_scale(x)
        return normal_inverse_chisquared.pdf(self._loc, self._scale, self._kappa, self._nu, x_mean, x_std)

    def eval_pdf_predictive(self, x: np.ndarray) -> float:
        x_loc = self._estimator.to_loc(x)
        return normal_inverse_chisquared.pdf_predictive(self._loc, self._scale, self._kappa, self._nu, x_loc)

    def eval_ttest(self, x: np.ndarray) -> float:
        return uni_normal.test_1sample(self._loc, x)

    @property
    def parameters(self) -> dict:
        return {
            'Location': self._loc,
            'Scale': self._scale,
            'Kappa': self._kappa,
            'Nu': self._nu
        }


class BayesianMultiMine(BayesianUniMine):  # TODO: Deal with values with 0 std
    def _add_sample(self, values: np.ndarray) -> None:
        self._loc, self._scale, self._kappa, self._nu = normal_inverse_wishart.posterior(
            self._loc,
            self._scale,
            self._kappa,
            self._nu,
            values
        )

    def eval_pdf(self, x: np.ndarray) -> float:
        x_mean = self._estimator.to_loc(x)
        x_prec = self._estimator.to_scale(x)
        return normal_inverse_wishart.pdf(self._loc, self._scale, self._kappa, self._nu, x_mean, x_prec)

    def eval_pdf_predictive(self, x: np.ndarray) -> float:
        x_loc = self._estimator.to_loc(x)
        return normal_inverse_wishart.pdf_predictive(self._loc, self._scale, self._kappa, self._nu, x_loc)

    def eval_ttest(self, x: np.ndarray) -> float:
        return multi_normal.ttest_1sample(self._loc, x)


if __name__ == '__main__':
    pass
