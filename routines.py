import pandas as pd
import numpy as np

import evaluation
import plotting
from mines import Mine
from models import Model
from samples import TestSamples
from typing import Callable, Optional


def cross_validate(data: pd.DataFrame, mine_class: Mine.__class__, n_fold: int, loss_func: Callable[[TestSamples, np.ndarray], float]=evaluation.error, model_kwargs: Optional[dict]=None,  mine_kwargs: Optional[dict] = None) -> float:
    if model_kwargs is None:
        model_kwargs = {}
    data_shuffled = data.iloc[np.random.permutation(len(data))]

    samples = pd.unique(data['smp'])
    n_samples = len(samples)
    n_folds = (n_samples // n_fold) + 1
    loss = np.zeros(n_folds)

    # Iterate through folds
    for i in range(n_folds):
        # Setting up mask indices
        idx_begin = i * n_fold
        idx_end = idx_begin + n_fold
        if idx_end > n_samples:
            idx_end = n_samples
        # Creating sample mask
        mask = np.ones(n_samples).astype(np.bool)
        mask[idx_begin:idx_end] = False
        # Splitting data according to mask
        data_model = data_shuffled[np.isin(data_shuffled['smp'], samples[mask])]
        data_test = data_shuffled[np.isin(data_shuffled['smp'], samples[np.invert(mask)])]
        # Creating model and test samples
        model = Model(data_model, mine_class, mine_kwargs=mine_kwargs, **model_kwargs)
        samples_test = TestSamples(data_test)
        # Evaluate test samples
        labels = model.classify(samples_test)
        loss[i] = loss_func(samples_test, labels)
    return loss.mean()


def show_training(data: pd.DataFrame, Mine_Class: Mine.__class__, n_samples: int, mine_kwargs: Optional[dict]=None):
    if mine_kwargs is None:
        mine_kwargs = {}
    sample_ids = pd.unique(data['smp'])

    mine: Mine = Mine_Class(
        x=0,
        y=0,
        z=0,
        status=0,
        **mine_kwargs
    )

    samples = []
    for i in range(n_samples):
        samples.append(data[data['smp'] == sample_ids[i]].filter(regex='Att*').values)

    stacked = np.row_stack(samples)
    min_value = np.min(stacked, axis=0)
    max_value = np.max(stacked, axis=0)
    delta = max_value - min_value
    x = np.linspace(min_value - 0.1*delta, max_value + 0.1*delta, 500)
    y = [mine.pdf(x)]
    for i in range(n_samples):
        mine.add_sample(samples[i])
        y.append(mine.pdf(x))

    plotting.plot_training(x, y, samples, attr_idx=0)


if __name__ == '__main__':
    pass
