import numpy as np
import pandas as pd
from functions import evaluation
from classes.samples import Samples
from classes.mines import Mine
from typing import List, Optional, Callable


class Model:
    _mines: List[Mine]
    _eval_func: Callable

    def __init__(self, data: pd.DataFrame, mine_class: Mine.__class__, eval_func: Callable = evaluation.best_mine, mine_kwargs: Optional[dict] = None):
        self._mines = []
        self._eval_func = eval_func
        self._create_mines(data, mine_class, mine_kwargs)

    def _create_mines(self, data, mine_class, mine_kwargs: Optional[dict]=None) -> None:
        if mine_kwargs is None:
            mine_kwargs = {}

        mines = {}
        sample_ids = pd.unique(data['smp'])
        for i in range(len(sample_ids)):
            sample_data = data[data['smp'] == sample_ids[i]]

            mine_id = sample_data['mineID'].iloc[0]
            mine = mines.get(mine_id)
            if mine is None:
                mines[mine_id] = mine_class(
                    x=sample_data['x'].iloc[0],
                    y=sample_data['y'].iloc[0],
                    z=sample_data['z'].iloc[0],
                    status=sample_data['FP'].iloc[0],
                    **mine_kwargs
                )
            mines[mine_id].add_sample(sample_data.filter(regex='Att*').values)
        self._mines = list(mines.values())

    def classify(self, samples: Samples) -> np.ndarray:
        eval_results = np.zeros((len(samples), len(self._mines)))
        for i, mine in enumerate(self._mines):
            eval_results[:, i] = mine.eval_samples(samples)

        return self._eval_func(eval_results, self._mines)
        # return evaluation.best_label(eval_results, self._mines)

    def __getitem__(self, item) -> Mine:
        return self._mines[item]


if __name__ == '__main__':
    pass
