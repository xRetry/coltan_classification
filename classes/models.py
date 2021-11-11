import numpy as np
import pandas as pd
from classes.samples import Samples
from classes.mines import Mine
from classes.parameters import Parameters
from typing import List, Optional, Callable


class Model:
    _mines: List[Mine]
    _func_selection: Callable

    def __init__(self,parameters: Parameters, data: pd.DataFrame):
        self._mines = []
        self._func_selection = parameters.func_selection
        self._create_mines(data, parameters)

    def _create_mines(self, data, parameters: Parameters) -> None:
        mines = {}
        sample_ids = pd.unique(data['smp'])
        for i in range(len(sample_ids)):
            sample_data = data[data['smp'] == sample_ids[i]]

            mine_id = sample_data['mineID'].iloc[0]
            mine = mines.get(mine_id)
            if mine is None:
                mines[mine_id] = parameters.MineClass(
                    x=sample_data['x'].iloc[0],
                    y=sample_data['y'].iloc[0],
                    z=sample_data['z'].iloc[0],
                    status=sample_data['FP'].iloc[0],
                    parameters=parameters,
                    **parameters.mine_kwargs
                )
            mines[mine_id].add_sample(sample_data.filter(regex='Att*').values)
        self._mines = list(mines.values())

    def classify(self, samples: Samples) -> np.ndarray:
        eval_results = np.zeros((len(samples), len(self._mines)))
        for i, mine in enumerate(self._mines):
            eval_results[:, i] = mine.eval_samples(samples)

        return self._func_selection(eval_results, self._mines)

    def __getitem__(self, item) -> Mine:
        return self._mines[item]


if __name__ == '__main__':
    pass
