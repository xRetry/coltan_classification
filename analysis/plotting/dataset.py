from typing import Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats


def plot_correlation_matrix(values: np.ndarray, labels: Optional[Iterable[str]] = None):
    # Compute values of correlation matrix
    corr_matrix = np.corrcoef(values)
    # Plotting correlation matrix
    f, ax = plt.subplots(figsize=(30, 30))
    heatmap = sns.heatmap(
        corr_matrix,
        square=True,
        linewidths=.5,
        cmap='coolwarm',
        cbar=False,
        fmt=".2f",
        vmin=-1,
        vmax=1,
        annot=True,
        annot_kws={"size": 12}
    )
    # Add labels
    if labels is not None:
        ax.set_yticklabels(labels, rotation=0)
        ax.set_xticklabels(labels)
    # Modify layout
    sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
    plt.tight_layout()
    plt.show()


def plot_qq(attribute_values: np.ndarray, attr_idx:int=0):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # sm.graphics.qqplot(attribute_values[:, attr_idx])  # Proper Q-Q Plot
    scipy.stats.probplot(attribute_values[:, attr_idx], plot=ax)  # Technically not a Q-Q Plot?
    ax.set_title(f'Attribute {attr_idx}')
    plt.show()


def plot_norm_test(p_vals: np.ndarray):
    # Create attribute labels
    attr_labels = [f'Attr {i+1}' for i in range(len(p_vals[0, :]))]
    # Plot p values
    plt.figure(figsize=(10, 10))
    sns.boxplot(data=p_vals, orient='h')
    # Modifying layout
    plt.yticks(list(range(len(p_vals[0, :]))), attr_labels)
    plt.title('Anderson-Darling Normality Test')
    plt.ylabel('Attribute')
    plt.xlabel('P-Value')
    plt.show()


def plot_samples(attr_values: np.ndarray, attr_idx: int or Iterable[int], attr_labels: Optional[np.ndarray]=None):
    """
    Plots histograms of attribute values in a grid.
    """
    # Convert attribute index to list if is a single index.
    if isinstance(attr_idx, int):
        attr_idx = [attr_idx]
    # Determine sizes and grid dimensions
    n_samples = len(attr_values)
    n_attr = len(attr_idx)
    n_cols = 10
    n_rows = int(np.ceil(n_samples / n_cols))
    # Defining colormap
    colors = plt.get_cmap('tab10')(range(n_attr))
    # Creating plot
    fig, axs = plt.subplots(nrows=n_rows * n_attr, ncols=n_cols, figsize=(n_cols * 2, n_rows * n_attr * 2))
    # Iterating through attributes
    for k, a in enumerate(attr_idx):
        # Iterating through samples
        n = 0
        for i in range(n_rows):
            for idx_col in range(n_cols):
                idx_row = k * n_rows + i
                # Removing x and y ticks
                axs[idx_row, idx_col].set_xticks([])
                axs[idx_row, idx_col].set_yticks([])
                # Skip if no more samples left
                if n >= n_samples:
                    continue
                # Creating histogram
                axs[idx_row, idx_col].hist(attr_values[n][:, a], 20, density=True, facecolor=colors[k])
                n += 1
                # Adding y labels if provided
                if attr_values is not None and idx_col == 0:
                    axs[idx_row, idx_col].set_ylabel(attr_labels[k])

    plt.tight_layout()
    plt.show()


def plot_pca_ratio(variance_ratio: np.ndarray):
    plt.plot(range(len(variance_ratio)), variance_ratio)
    plt.show()


def plot_pca(pc1: np.ndarray, pc2: np.ndarray, labels):
    """
    Plots the first two principal components, color-coded by labels.
    """
    # Creating mask for positive label
    is_pos = labels == 1
    # Plotting points with positive label
    plt.scatter(pc1[is_pos], pc2[is_pos], c='g')
    # Plotting points with negative label
    plt.scatter(pc1[~is_pos], pc2[~is_pos], c='r')
    # Adding labels
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    # Showing plot
    plt.tight_layout()
    plt.show()


def plot_lda(data, labels):
    """
    Plots the result of a Linear Discriminant Analysis.
    """
    plt.scatter(data[labels == 1].flatten(), np.zeros(np.sum(labels == 1)), c='g')
    plt.scatter(data[labels == -1].flatten(), np.ones(np.sum(labels == -1)), c='r')
    plt.show()


if __name__ == '__main__':
    pass
