import numpy as np
import pandas as pd
import abc
from classes.mines import Mine
from classes.parameters import Parameters
from classes.dataset import Dataset, Sample
from classes.normalizers import Normalizer
from typing import List, Optional, Callable, Dict, Iterable
from sklearn.linear_model import LogisticRegression


class Model(abc.ABC):
    @abc.abstractmethod
    def __init__(self, parameters: Parameters, samples: Iterable[Sample]):
        pass

    @abc.abstractmethod
    def classify(self, sample: Sample, return_scores: bool=False) -> int or (int, np.ndarray):
        """
        Classifies a sample.
        """
        pass


class MineModel(Model):
    _mines: Dict[str, Mine]
    _parameters: Parameters

    def __init__(self, parameters: Parameters, samples: Iterable[Sample]):
        super().__init__(parameters, samples)
        self._mines = {}
        self._parameters = parameters
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
            self._mines[mine_id] = self._parameters.MineClass(
                coordinates=sample.coordinates,
                label=sample.label,
                parameters=self._parameters,
                **self._parameters.mine_kwargs
            )
        self._mines[mine_id].add_sample(sample)

    def classify(self, sample: Sample, return_scores: bool=False) -> int or (int, np.ndarray):
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
        prediction_label = self._parameters.func_selection(mine_scores, labels)
        if return_scores:
            return prediction_label, mine_scores
        return prediction_label


class LabelModel(MineModel):
    def _add_sample(self, sample: Sample) -> None:
        """
        Creates mines if necessary and adds new sample to it.
        """
        mine = self._mines.get(str(sample.label))
        if mine is None:
            self._mines[str(sample.label)] = self._parameters.MineClass(
                coordinates=sample.coordinates,
                label=sample.label,
                parameters=self._parameters,
                **self._parameters.mine_kwargs
            )
        self._mines[str(sample.label)].add_sample(sample)


class LogisticRegressionModel(Model):
    _logistic_model: LogisticRegression
    _normalizer: Normalizer
    _func_transform: Callable

    def __init__(self, parameters: Parameters, samples: Iterable[Sample]):
        super().__init__(parameters, samples)
        self._func_transform = parameters.func_transform
        self._normalizer = parameters.NormalizerClass()

        data_train = Dataset(samples=samples)
        data_trans = self._func_transform(np.concatenate(data_train.attributes))
        self._normalizer.fit(data_trans)
        attributes = self._normalizer.transform(data_trans)
        labels = data_train.labels
        self._logistic_model = LogisticRegression(random_state=0, solver='newton-cg').fit(attributes, labels)

    def classify(self, sample: Sample, return_scores: bool=False) -> int or (int, np.ndarray):
        """
        Classifies a sample.
        """
        x = self._normalizer.transform(self._func_transform(sample.attributes))
        predictions = self._logistic_model.predict(x)
        prediction_label = 1 if (predictions == 1).sum() / len(predictions) > 0.5 else -1
        if return_scores:
            return prediction_label, predictions
        return prediction_label


if __name__ == '__main__':
    pass
