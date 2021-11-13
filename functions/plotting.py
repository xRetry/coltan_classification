import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Iterable


def plot_samples(attribute_values: List[np.ndarray], attr_idx: int, ax: plt.Subplot = None):
    has_parent = ax is not None
    if not has_parent:
        ax = plt.subplot()

    for sample_attributes in attribute_values:
        ax.hist(sample_attributes[:, attr_idx], 20, density=True, histtype='step', facecolor='g', alpha=0.75)

    if not has_parent:
        plt.show()


def plot_locations(mines: dict):
    xyz = np.array([m.coordinates for m in mines.values()])

    plt.scatter(xyz[:, 0], xyz[:, 1])
    plt.show()


def plot_training(x: np.ndarray, y: list, samples: List[np.ndarray], attr_idx: int=0):
    n = len(samples)
    len_diff = len(y) - n
    plt.figure()
    for i in range(len_diff):
        plt.subplot(n + len_diff, 1, i + 1)
        plt.plot(x[:, attr_idx], y[i][:, attr_idx])

    for i in range(n):
        plt.subplot(n+len_diff, 1, len_diff+i+1)
        plt.hist(samples[i][:, attr_idx], bins=40, density=True)
        plt.plot(x[:, attr_idx], y[len_diff+i][:, attr_idx])
    plt.show()
    pass


def plot_correlation_matrix(values: np.ndarray, labels: Optional[Iterable[str]] = None):
    corr_matrix = np.corrcoef(values)

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

    sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    pass
