import numpy as np
import pandas as pd
from classes.mines import Mine
from classes.parameters import Parameters
from classes.dataset import Dataset, Sample
from typing import List, Optional, Callable, Dict


class Model:
    _mines: Dict[str, Mine]
    _parameters: Parameters

    def __init__(self,parameters: Parameters, dataset: Dataset):
        self._mines = {}
        self._parameters = parameters
        self._create_mines(dataset)

    def _create_mines(self, dataset: Dataset) -> None:
        for sample in dataset:
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


if __name__ == '__main__':
    pass
