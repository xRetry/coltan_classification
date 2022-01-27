from typing import Iterable, List
import itertools
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns


def plot_cv(splits: np.ndarray, conf_ints: np.ndarray, model_names: Iterable) -> None:
    """
    Plots the result of stepwise cross-validation.
    """
    # Iterate through all models and plotting result
    for i in range(len(conf_ints[0, :, 0])):
        # Determine center of confident intervals
        mean = conf_ints[:, i, :].mean(axis=1)
        # Plot confident intervals
        plt.errorbar(splits, mean, yerr=np.abs(mean[:, None]-conf_ints[:, i, :]).T, capsize=5)
    # Setting up layout
    plt.xlabel('Test Proportion for Cross-Validation')
    plt.ylabel('Accuracy')
    plt.legend(model_names)
    plt.show()


def _calc_pval_grid(predictions: List[np.ndarray]) -> np.ndarray:
    """
    Computes the model-wise p-values from tests for equality.
    """
    # Setting up parameters
    n_models = len(predictions)
    p_vals = np.zeros((n_models, n_models))
    # Iterate through all combinations
    for idxs in itertools.product(range(n_models), range(n_models)):
        # Combine predictions from all folds
        preds_model1 = np.concatenate(predictions[idxs[0]])
        preds_model2 = np.concatenate(predictions[idxs[1]])
        # Create table for fisher test
        table = np.array([
            [(preds_model1 == 1).sum(), (preds_model1 != 1).sum()],
            [(preds_model2 == 1).sum(), (preds_model2 != 1).sum()]
        ]).T
        # Perform fisher test for equality
        _, p_val = scipy.stats.fisher_exact(table)
        p_vals[idxs] = p_val
    return p_vals


def plot_cv_grid(cv_result, model_names: Iterable, title:str = "") -> None:
    """
    Computes the model-wise p-values of equality tests and plots the results.
    """
    # Compute p-value matrix
    p_vals = _calc_pval_grid(cv_result.predictions)
    # Create mask for the upper triangle
    mask = np.zeros_like(p_vals)
    mask[np.triu_indices_from(mask)] = True
    # Plot heatmap
    f, ax = plt.subplots(figsize=(2*len(p_vals), 2*len(p_vals)))
    sns.heatmap(
        p_vals,
        square=True,
        linewidths=.5,
        cmap='coolwarm_r',
        cbar_kws={'shrink': .8, 'ticks': [0, 0.5, 1]},
        vmin=0,
        vmax=1,
        annot=True,
        #annot_kws={"size": 12},
        mask=mask
    )
    # Add the model names as labels
    ax.set_yticklabels(model_names, rotation=0)
    ax.set_xticklabels(model_names)
    # Set title
    ax.set_title(title)
    # Set tick positions
    sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
    # Show plot
    plt.show()


if __name__ == '__main__':
    pass
