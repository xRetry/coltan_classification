from typing import Optional, Callable, Iterable, Sequence
import abc
from dataclasses import dataclass, field

import numpy as np

from core.dataset import Sample
from core.functions.evaluations import uni_normal, multi_normal, normal_inverse_wishart, non_parametric, normal_inverse_chisquared
from core.estimators import Estimator, MeanUniEstimator
from core.normalizers import Normalizer, NoNormalizer
from core.utils import singular_check


###############
# DATACLASSES #
###############


@dataclass
class MineParameters:
    func_transform: Callable
    func_eval: Callable
    NormalizerClass: type(Normalizer) = NoNormalizer
    EstimatorClass: type(Estimator) = MeanUniEstimator
    eval_kwargs: dict = field(default_factory=dict)
    mine_kwargs: dict = field(default_factory=dict)


##################
# ABSTRACT CLASS #
##################


class Mine(abc.ABC):
    _coordinates: np.ndarray
    _label: int
    _mine_params: MineParameters
    _normalizer: Normalizer
    _estimator: Estimator

    def __init__(self, mine_parameters: MineParameters, label: int, coordinates: Optional[np.ndarray]=None):
        if coordinates is None:
            coordinates = np.zeros(3)
        self._coordinates = coordinates
        self._label = label
        self._mine_params = mine_parameters
        self._normalizer = mine_parameters.NormalizerClass()
        self._estimator = mine_parameters.EstimatorClass()

    def add_sample(self, sample: Sample) -> None:
        x_trans = self._mine_params.func_transform(sample.attributes)
        if not self._normalizer.is_fitted:
            self._normalizer.fit(x_trans)
        self._add_sample(self._normalizer.transform(x_trans))

    @abc.abstractmethod
    def _add_sample(self, values) -> None:
        pass

    def eval_sample(self, sample: Sample) -> float:
        attr_values = self._normalizer.transform(self._mine_params.func_transform(sample.attributes))
        return self._mine_params.func_eval(self, attr_values, **self._mine_params.eval_kwargs)

    @property
    def coordinates(self) -> np.ndarray:
        return np.array(self._coordinates)

    @property
    def status(self) -> int:
        return self._label

    @property
    @abc.abstractmethod
    def parameters(self) -> dict:
        pass


##############
# SUBCLASSES #
##############


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

    def eval_norm2(self, x: np.ndarray, **kwargs) -> float:
        loc = self._estimator.to_loc(self._parameters.attributes)
        return non_parametric.test_norm2(loc, self._estimator.to_loc(x))

    def eval_cosine(self, x: np.ndarray) -> float:
        loc = self._estimator.to_loc(self._parameters.attributes)
        return non_parametric.test_cosine(loc, self._estimator.to_loc(x))

    def eval_canberra(self, x: np.ndarray) -> float:
        loc = self._estimator.to_loc(self._parameters.attributes)
        return non_parametric.test_canberra(loc, self._estimator.to_loc(x))

    def eval_correlation(self, x: np.ndarray) -> float:
        loc = self._estimator.to_loc(self._parameters.attributes)
        return non_parametric.test_correlation(loc, self._estimator.to_loc(x))

    def eval_exponential(self, x: np.ndarray, exponent: float=2, scale: float=1, func_aggr: Callable=np.product) -> float:
        loc = self._estimator.to_loc(self._parameters.attributes)
        return non_parametric.test_exponential(loc, self._estimator.to_loc(x), exponent, scale, func_aggr)

    @singular_check('x')
    def eval_ttest(self, x: np.ndarray, func_aggr: Callable=np.product) -> float:
        loc = self._estimator.to_loc(self._parameters.attributes)
        return uni_normal.test_1sample(loc, x, func_aggr)

    def eval_ttest_2sample(self, x: np.ndarray, func_aggr: Callable=np.product) -> float:
        loc = self._estimator.to_loc(self._parameters.attributes)
        scale = self._estimator.to_scale(self._parameters.attributes)
        n_obs = np.ones_like(loc) * len(self._parameters)
        loc_x = self._estimator.to_loc(x)
        scale_x = self._estimator.to_scale(x)
        n_obs_x = np.ones_like(loc) * len(x)
        return uni_normal.test_2sample(loc, scale, n_obs, loc_x, scale_x, n_obs_x, func_aggr)

    def eval_pdf(self, x: np.ndarray, func_aggr: Callable=np.product) -> float:
        loc = self._estimator.to_loc(self._parameters.attributes)
        scale = self._estimator.to_scale(self._parameters.attributes)
        return uni_normal.pdf(loc, scale, self._estimator.to_loc(x), func_aggr)

    def eval_ranksums(self, x: np.ndarray, func_aggr: Callable=np.product) -> float:
        return non_parametric.test_ranksums(self._parameters.attributes, x, func_aggr)

    def eval_mannwhitneyu(self, x: np.ndarray, func_aggr: Callable=np.product) -> float:
        return non_parametric.test_mannwhitneyu(self._parameters.attributes, x, func_aggr)

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

    def __init__(self, mine_parameters: MineParameters, label: int, coordinates: Optional[np.ndarray] = None):
        super().__init__(mine_parameters, label, coordinates)
        self._mean = self._mine_params.mine_kwargs['mean']
        self._std = self._mine_params.mine_kwargs['std']

    def _add_sample(self, values: np.ndarray) -> None:
        std_known = np.ones(len(self._mean))
        self._mean, self._std = uni_normal.posterior(self._mean, self._std, std_known, values)

    def eval_pdf(self, x: np.ndarray, func_aggr: Callable=np.product) -> float:
        return uni_normal.pdf(self._mean, self._std, self._estimator.to_loc(x), func_aggr)

    def eval_ttest(self, x: np.ndarray, func_aggr: Callable=np.product) -> float:
        return uni_normal.test_1sample(self._mean, x, func_aggr)

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

    def __init__(self, mine_parameters: MineParameters, label: int, coordinates: Optional[np.ndarray] = None):
        super().__init__(mine_parameters, label, coordinates)
        self._loc = self._mine_params.mine_kwargs['loc']
        self._scale = self._mine_params.mine_kwargs['scale']
        self._kappa = self._mine_params.mine_kwargs['kappa']
        self._nu = self._mine_params.mine_kwargs['nu']

    def _add_sample(self, values: np.ndarray) -> None:
        self._loc, self._scale, self._kappa, self._nu = normal_inverse_chisquared.posterior(
            self._loc,
            self._scale,
            self._kappa,
            self._nu,
            values
        )

    def eval_pdf(self, x: np.ndarray, func_aggr: Callable=np.product) -> float:
        x_mean = self._estimator.to_loc(x)
        x_std = self._estimator.to_scale(x)
        return normal_inverse_chisquared.pdf(self._loc, self._scale, self._kappa, self._nu, x_mean, x_std, func_aggr)

    def eval_pdf_predictive(self, x: np.ndarray, func_aggr: Callable=np.product) -> float:
        x_loc = self._estimator.to_loc(x)
        return normal_inverse_chisquared.pdf_predictive(self._loc, self._scale, self._kappa, self._nu, x_loc, func_aggr)

    def eval_ttest_2sample(self, x: np.ndarray, func_aggr: Callable=np.product) -> float:
        loc_x = self._estimator.to_loc(x)
        scale_x = self._estimator.to_scale(x)
        n_obs_x = np.ones_like(self._loc) * len(x)
        n_obs = np.ones_like(self._loc) * self._nu
        return uni_normal.test_2sample(self._loc, self._scale, n_obs, loc_x, scale_x, n_obs_x, func_aggr)

    def eval_ztest(self, x: np.ndarray, func_aggr: Callable=np.product):
        loc_x = self._estimator.to_loc(x)
        n_obs_x = np.ones_like(self._loc) * len(x)
        return uni_normal.ztest_1sample(loc_x, n_obs_x, self._loc, self._scale, func_aggr)

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

    def eval_pdf(self, x: np.ndarray, func_aggr: Callable=np.product) -> float:
        x_mean = self._estimator.to_loc(x)
        x_prec = self._estimator.to_scale(x)
        return normal_inverse_wishart.pdf(self._loc, self._scale, self._kappa, self._nu, x_mean, x_prec)

    def eval_pdf_predictive(self, x: np.ndarray, func_aggr: Callable=np.product) -> float:
        x_loc = self._estimator.to_loc(x)
        return normal_inverse_wishart.pdf_predictive(self._loc, self._scale, self._kappa, self._nu, x_loc)

    def eval_ttest(self, x: np.ndarray, func_aggr: Callable=np.product) -> float:
        return multi_normal.ttest_1sample(self._loc, x, func_aggr)


if __name__ == '__main__':
    pass
