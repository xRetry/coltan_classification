import numpy as np
import pandas as pd
import abc
from classes.mines import Mine
from classes.parameters import Parameters
from classes.dataset import Dataset, Sample
from typing import List, Optional, Callable, Dict, Iterable
from sklearn.linear_model import LogisticRegression


class Model(abc.ABC):
    @staticmethod
    def __init__(self, parameters: Parameters, samples: Iterable[Sample]):
        pass

    @staticmethod
    def classify(self, sample: Sample) -> int:
        pass


class KernelModel(Model):
    _mines: Dict[str, Mine]
    _parameters: Parameters

    def __init__(self, parameters: Parameters, samples: Iterable[Sample]):
        super().__init__(self, parameters, samples)
        self._mines = {}
        self._parameters = parameters
        self._create_mines(samples)

    def _create_mines(self, samples: Iterable[Sample]) -> None:
        for sample in samples:
            self._add_sample(sample)

    def _add_sample(self, sample: Sample):
        mine_id = sample.mine_id
        mine = self._mines.get(mine_id)
        if mine is None:
            self._mines[mine_id] = self._parameters.MineClass(
                coordinates=sample.coordinates,
                status=sample.label,
                parameters=self._parameters,
                **self._parameters.mine_kwargs
            )
        self._mines[mine_id].add_sample(sample)

    def classify(self, sample: Sample) -> int:
        mines = list(self._mines.values())
        eval_results, labels = np.zeros(len(mines)), np.zeros(len(mines))
        for i, mine in enumerate(mines):
            eval_results[i] = mine.eval_sample(sample)
            labels[i] = mine.status
        return self._parameters.func_selection(eval_results, labels)


class LogisticRegressionModel(Model):
    _logistic_model: LogisticRegression

    def __init__(self, parameters: Parameters, samples: Iterable[Sample]):
        super().__init__(self, parameters, samples)
        data_train = Dataset(samples=samples)
        labels = data_train.labels
        attributes = np.concatenate(data_train.attributes)
        self._logistic_model = LogisticRegression(random_state=0).fit(attributes, labels)

    def classify(self, sample: Sample) -> int:
        predictions = self._logistic_model.predict(sample.attributes)
        return 1 if (predictions == 1).sum() / len(predictions) > 0.5 else -1


if __name__ == '__main__':
    pass
