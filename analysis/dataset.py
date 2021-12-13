import numpy as np
from classes.dataset import Dataset
from functions import plotting
from functions import transformation
from typing import Callable, Iterable
import statsmodels.api as sm


class DatasetAnalyser:
    _dataset: Dataset

    def __init__(self, dataset: Dataset):
        self._dataset = dataset

    def plot_correlation(self):
        values = np.row_stack(self._dataset.attributes)
        plotting.plot_correlation_matrix(values.T, self._dataset.attribute_labels)

    def plot_samples(self, attr_idx: int or Iterable[int], sample_idx: int or Iterable[int]):
        attr_values = self._dataset.attributes[sample_idx]
        attr_labels = self._dataset.attribute_labels
        plotting.plot_samples(attr_values, attr_idx, attr_labels)

    def plot_qq(self, attr_idx:int):
        plotting.plot_qq(self._dataset.attributes[0], attr_idx=attr_idx)

    def test_normality(self, func_trans: Callable=transformation.none):
        p_vals_all = np.zeros((len(self._dataset), self._dataset.n_attributes))
        for i, sample in enumerate(self._dataset):
            statistic, p_vals = sm.stats.diagnostic.normal_ad(func_trans(sample.attributes))
            p_vals_all[i, :] = p_vals
        plotting.plot_norm_test(p_vals_all)


if __name__ == '__main__':
    pass
