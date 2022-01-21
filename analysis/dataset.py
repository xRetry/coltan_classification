import numpy as np
from core.dataset import Dataset
import analysis.plotting.dataset as plot
from core.functions import transformation
from typing import Callable, Iterable, Optional
import statsmodels.api as sm
from sklearn.decomposition import PCA


class DatasetAnalyser:
    _dataset: Dataset

    def __init__(self, dataset: Dataset):
        self._dataset = dataset

    def plot_correlation(self):
        values = np.row_stack(self._dataset.attributes)
        plot.plot_correlation_matrix(values.T, self._dataset.attribute_labels)

    def plot_samples(self, attr_idx: int or Iterable[int], sample_idx: int or Iterable[int]):
        attr_values = self._dataset.attributes[sample_idx]
        attr_labels = self._dataset.attribute_labels
        plot.plot_samples(attr_values, attr_idx, attr_labels)

    def plot_qq(self, attr_idx:int):
        plot.plot_qq(self._dataset.attributes[0], attr_idx=attr_idx)

    def test_normality(self, func_trans: Callable= transformation.none):
        p_vals_all = np.zeros((len(self._dataset), self._dataset.n_attributes))
        for i, sample in enumerate(self._dataset):
            statistic, p_vals = sm.stats.diagnostic.normal_ad(func_trans(sample.attributes))
            p_vals_all[i, :] = p_vals
        plot.plot_norm_test(p_vals_all)

    def pca_ratio(self, func_trans: Callable= transformation.none, n_components: Optional[int]=None):
        """
        Plots the explained variance ratio for a PCA.
        """
        # Stack values of all samples on top of each other
        data = func_trans(np.row_stack(self._dataset.attributes))
        # Create PCA model
        pca = PCA(n_components=n_components)
        pca.fit(data)
        # Plotting variance ratio
        plot.plot_pca_ratio(pca.explained_variance_ratio_)

    def pca(self, func_trans: Callable= transformation.none):
        """
        Plots the first two principal components.
        """
        # Stack values of all samples on top of each other
        data = func_trans(np.row_stack(self._dataset.attributes))
        # Compute transformed data
        data_trans = PCA(n_components=2).fit_transform(data)
        # Plot transformed data
        plot.plot_pca(data_trans[:, 0], data_trans[:, 1], self._dataset.labels_analysis)


if __name__ == '__main__':
    pass
