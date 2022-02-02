from typing import Callable, Dict, Iterable, NamedTuple
import abc
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression

from core.mines import Mine, MineParameters
from core.dataset import Dataset, Sample
from core.normalizers import Normalizer


@dataclass
class ModelParameters:
    MineClass: type(Mine)
    mine_params: MineParameters
    func_classification: Callable
    threshold: int = 0.5


@dataclass
class ModelResult(NamedTuple):
    label: int
    scores: np.ndarray


class Model(abc.ABC):
    @abc.abstractmethod
    def __init__(self, parameters: ModelParameters, samples: Iterable[Sample]):
        pass

    @abc.abstractmethod
    def classify(self, sample: Sample, return_summary: bool=False) -> int or ModelResult:
        """
        Classifies a sample.
        """
        pass


class MineModel(Model):
    _mines: Dict[str, Mine]
    _model_params: ModelParameters

    def __init__(self, model_parameters: ModelParameters, samples: Iterable[Sample]):
        super().__init__(model_parameters, samples)
        self._mines = {}
        self._model_params = model_parameters
        self._create_mines(samples)

    def _create_mines(self, samples: Iterable[Sample]) -> None:
        """
        Adds all samples to the model.
        """
        for sample in samples:
            self._add_sample(sample)

    def _add_sample(self, sample: Sample) -> None:
        """
        Creates mines if necessary and adds new sample to it.
        """
        mine_id = sample.mine_id
        mine = self._mines.get(mine_id)
        if mine is None:
            self._mines[mine_id] = self._model_params.MineClass(
                coordinates=sample.coordinates,
                label=sample.label,
                mine_parameters=self._model_params.mine_params,
            )
        self._mines[mine_id].add_sample(sample)

    def classify(self, sample: Sample, return_summary: bool = False) -> int or ModelResult:
        """
        Classifies a sample.
        """
        # Getting mines from dictionary
        mines = list(self._mines.values())
        # Evaluation sample for all mines
        mine_scores, labels = np.zeros(len(mines)), np.zeros(len(mines))
        for i, mine in enumerate(mines):
            mine_scores[i] = mine.eval_sample(sample)
            labels[i] = mine.status
        # Getting class prediction from mine scores
        prediction_label = self._model_params.func_classification(mine_scores, labels, self._model_params.threshold)
        if return_summary:
            return ModelResult(prediction_label, mine_scores)
        return prediction_label


class LabelModel(MineModel):
    def _add_sample(self, sample: Sample) -> None:
        """
        Creates mines if necessary and adds new sample to it.
        """
        mine = self._mines.get(str(sample.label))
        if mine is None:
            self._mines[str(sample.label)] = self._model_params.MineClass(
                coordinates=sample.coordinates,
                label=sample.label,
                parameters=self._model_params,
            )
        self._mines[str(sample.label)].add_sample(sample)


class LogisticRegressionModel(Model):
    _logistic_model: LogisticRegression
    _normalizer: Normalizer
    _func_transform: Callable

    def __init__(self, model_parameters: ModelParameters, samples: Iterable[Sample]):
        super().__init__(model_parameters, samples)
        self._func_transform = model_parameters.mine_params.func_transform
        self._normalizer = model_parameters.mine_params.NormalizerClass()

        data_train = Dataset(samples=samples)
        data_trans = self._func_transform(np.concatenate(data_train.attributes))
        self._normalizer.fit(data_trans)
        attributes = self._normalizer.transform(data_trans)
        labels = data_train.labels_analysis
        self._logistic_model = LogisticRegression(random_state=0, solver='newton-cg').fit(attributes, labels)

    def classify(self, sample: Sample, return_summary: bool = False) -> int or ModelResult:
        """
        Classifies a sample.
        """
        x = self._normalizer.transform(self._func_transform(sample.attributes))
        predictions = self._logistic_model.predict(x)
        prediction_label = 1 if (predictions == 1).sum() / len(predictions) > 0.5 else -1
        if return_summary:
            return ModelResult(prediction_label, predictions)
        return prediction_label


if __name__ == '__main__':
    pass
