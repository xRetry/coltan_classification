from typing import Sequence, List, Dict, Tuple, Callable
from dataclasses import dataclass, field

import numpy as np

from core.dataset import Sample
from core.functions import transformation
from core.normalizers import Normalizer, NoNormalizer
from core.estimators import Estimator
from analysis.logging import Log


@dataclass
class Parameters:
    MineClass: object.__class__
    func_eval: Callable
    ModelClass: object.__class__ = None
    func_transform: Callable = transformation.none
    func_selection: Callable = None
    func_loss: Callable = None
    NormalizerClass: type(Normalizer) = NoNormalizer
    EstimatorClass: type(Estimator) = None
    eval_kwargs: dict = field(default_factory=dict)
    mine_kwargs: dict = field(default_factory=dict)


@dataclass
class ModelEvaluationParameters:
    eval_idx: int
    parameters: Parameters
    samples_train: Sequence[Sample]
    samples_test: Sequence[Sample]


@dataclass
class ModelEvaluationResult(Log):
    eval_idx: int
    samples_train: Sequence[Sample]
    samples_test: Sequence[Sample]
    predictions: np.ndarray
    warnings: list
    elapsed_time: float

    def __init__(self, model_eval_params: ModelEvaluationParameters):
        self.eval_idx = model_eval_params.eval_idx
        self.samples_train = model_eval_params.samples_train
        self.samples_test = model_eval_params.samples_test
        self.predictions = np.zeros(len(self.samples_test))
        self.warnings = []

    @property
    def labels(self):
        return np.array([sample.label for sample in self.samples_test])


@dataclass
class CrossValResult:
    parameters: List[Parameters]
    eval_results: List[List[ModelEvaluationResult]]
    n_warnings: int
    losses: List[List[np.ndarray]]
    conf_ints: List[List[np.ndarray]]

    def __init__(self, parameters: List[Parameters]):
        self.parameters = parameters
        self.eval_results = [[] for p in parameters]
        self.losses = []
        self.conf_ints = []
        self.n_warnings = 0

    def add_result(self, eval_result: ModelEvaluationResult):
        self.n_warnings += len(eval_result.warnings)
        self.eval_results[eval_result.eval_idx].append(eval_result)

    @property
    def predictions(self) -> List[np.ndarray]:
        preds = []
        for res_mdl in self.eval_results:
            preds.append(np.array([res_fld.predictions for res_fld in res_mdl]))
        return preds

    @property
    def labels(self) -> List[np.ndarray]:
        lbls = []
        for res_mdl in self.eval_results:
            lbls.append(np.array([res_fld.labels for res_fld in res_mdl]))
        return lbls


@dataclass
class ProgressBarData:
    bars: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # {name: (idx_current, n_iter)}
    post_fix: str = ''

    def add_bar(self, name, idx_current, n_iter):
        self.bars[name] = (idx_current, n_iter)


if __name__ == '__main__':
    pass
