from typing import Callable, Iterable, Optional

import numpy as np
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from core.dataset import Dataset
import analysis.plotting.dataset as plot
from core.functions import transformation


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

    def test_normality(self, func_trans: Callable= transformation.none, title: str=''):
        p_vals_all = np.zeros((len(self._dataset), self._dataset.n_attributes))
        for i, sample in enumerate(self._dataset):
            statistic, p_vals = sm.stats.diagnostic.normal_ad(func_trans(sample.attributes))
            p_vals_all[i, :] = p_vals
        plot.plot_norm_test(p_vals_all, title)

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

    def lda(self, func_trans: Callable= transformation.none):
        """
        Performs a linear discriminant analysis for all analysis in the dataset and plot the result.
        """
        # Stack values of all samples on top of each other
        data = func_trans(np.row_stack(self._dataset.attributes))
        # Getting labels for each individual analysis
        labels = self._dataset.labels_analysis
        # Perform LDA of the data
        data_trans = LinearDiscriminantAnalysis().fit_transform(data, labels)
        # Plot result
        plot.plot_lda(data_trans, labels)

    @staticmethod
    def lda_mine(func_trans: Callable= transformation.none):
        """
        Performs a linear discriminant analysis for all mines in the dataset and plot the result.
        """
        # Get data grouped by mine
        dataset = Dataset(group_by_mine=True)
        # Stack transformed attributes
        data = np.row_stack([np.mean(func_trans(a), axis=0) for a in dataset.attributes])
        # Getting labels for all mines
        labels = np.array([m.label for m in dataset])
        # Perform linear discriminate analysis
        data_trans = LinearDiscriminantAnalysis().fit_transform(data, labels)
        # Plot the result
        plot.plot_lda(data_trans, labels)

    @staticmethod
    def lda_mine_2(func_trans: Callable= transformation.none):
        dataset = Dataset()
        # Stack values of all samples on top of each other
        data = func_trans(np.row_stack(dataset.attributes))
        labels = dataset.labels_mine
        data_trans = LinearDiscriminantAnalysis().fit_transform(data, labels)
        plot.plot_lda(data_trans, labels)


if __name__ == '__main__':
    pass
