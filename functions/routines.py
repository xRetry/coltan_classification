import pandas as pd
import numpy as np

from functions.plotting import plot_training
from classes.mines import Mine
from classes.models import Model, Parameters
from classes.samples import TestSamples
from typing import Callable, Optional


def cross_validate(parameters: Parameters, data: pd.DataFrame, n_fold: int) -> float:
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
        model = Model(parameters, data_model)
        samples_test = TestSamples(data_test)
        # Evaluate test samples
        predictions = model.classify(samples_test)
        loss[i] = parameters.func_loss(samples_test.labels, predictions)
    return loss.mean()


def show_training(parameters: Parameters, data: pd.DataFrame, n_samples: int):
    sample_ids = pd.unique(data['smp'])

    mine: Mine = parameters.MineClass(
        x=0,
        y=0,
        z=0,
        status=0,
        parameters=parameters,
        **parameters.mine_kwargs
    )

    samples = []
    for i in range(n_samples):
        samples.append(data[data['smp'] == sample_ids[i]].filter(regex='Att*').values)

    stacked = np.row_stack(samples)
    min_value = np.min(stacked, axis=0)
    max_value = np.max(stacked, axis=0)
    delta = max_value - min_value
    x = np.linspace(min_value - 0.1*delta, max_value + 0.1*delta, 500)
    y = [mine.distribution.pdf(x)]
    for i in range(n_samples):
        mine.add_sample(samples[i])
        y.append(mine.distribution.pdf(x))

    plot_training(x, y, samples, attr_idx=0)


if __name__ == '__main__':
    pass
