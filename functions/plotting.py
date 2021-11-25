import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import statsmodels.api as sm
from typing import List, Optional, Iterable


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


def plot_progression(mine_parameters: List[dict], attr_idx: int=0):
    """
    Plots the mine parameter values over the amount of samples added.
    """
    # Determine amount of parameters
    n_plots = len(mine_parameters[0])
    # Getting axis labels
    labels = list(mine_parameters[0].keys())
    # Plotting figure
    fig, axs = plt.subplots(nrows=n_plots, sharex='col')
    for i in range(n_plots):
        # Collecting parameter values
        y = [list(p.values())[i] for p in mine_parameters]
        # Select certain index if parameter is an array
        if isinstance(y[0], np.ndarray):
            y = [v[attr_idx] for v in y]
        # Creating plot
        axs[i].plot(np.arange(1, len(mine_parameters)+1), y)
        # Adding label on y axis
        axs[i].set_ylabel(labels[i])
    # Modifying presentation
    axs[-1].set_xlabel("Amount of Samples added to Mine")
    plt.tight_layout()
    plt.show()


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


def plot_eval_results(x: np.ndarray, y: List[np.ndarray] or np.ndarray, labels=None):
    # Wrap y value if only one eval result
    if not isinstance(y, list):
        y = [y]
    # Normalize y values if multiple eval results are provided
    if len(y) > 1:
        max_val = np.max(y)
        y = [(y_cur + np.abs(y_cur.min()) if y_cur.min() < 0 else y_cur)/max_val for y_cur in y]
    # Plot all eval results
    plt.figure()
    for y_current in y:
        plt.plot(x.T, y_current.T)
    # Setting labels
    plt.xlabel('x Offsets')
    if len(y) > 1:
        plt.yticks([])
    else:
        plt.ylabel('Evaluation Value')
    # Add legend if labels are provided
    if labels is not None:
        plt.legend(labels)
    plt.show()


def plot_eval_results_2d(x, y, z):
    plt.figure()
    plt.contourf(x, y, z)
    plt.show()


def plot_pdf(x, y):
    plt.figure()
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    pass
