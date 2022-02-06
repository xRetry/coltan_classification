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


def plot_cv_grid(cv_result, model_names: Iterable, title: str = "") -> None:
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


def plot_crossval_distributions(accs: List[np.ndarray], conf_ints: np.ndarray, model_labels: List[str], use_beta: bool=False) -> None:
    plt.figure()
    n_losses = len(accs[0])
    x_vals = np.linspace(0, 1, 500) if use_beta else np.arange(n_losses + 1)
    y_vals = np.zeros((len(accs), len(x_vals)))
    for i in range(len(accs)):
        if use_beta:  # TODO: Find out why shape of beta is different from binomial
            y_vals[i] = scipy.stats.beta.pdf(x_vals, accs[i].sum(), len(accs[i]) - accs[i].sum())
        else:
            y_vals[i] = scipy.stats.binom.pmf(x_vals, n=len(accs[i]), p=accs[i].sum() / len(accs[i]))

    y_offset = np.max(y_vals) * 1.2
    _, idx_above_threshold = np.where(y_vals > 1e-4)
    x_min, x_max = x_vals[idx_above_threshold.min()], x_vals[idx_above_threshold.max()]
    for i, y in enumerate(y_vals):
        if use_beta:
            plt.plot(x_vals, y+(i*y_offset))
            x_ci = conf_ints[i]
        else:
            plt.bar(x_vals, y, bottom=i * y_offset)
            plt.plot([0, n_losses + 1], [i * y_offset] * 2, linewidth=1)
            x_ci = np.array(conf_ints[i]) * (n_losses + 1)

        plt.plot(x_ci, [i * y_offset] * 2, linewidth=4, color='k')

    plt.yticks(np.arange(len(y_vals)) * y_offset + y_offset / 3, model_labels)
    plt.xticks(
        np.linspace(x_min, x_max, 5),
        (np.round(np.linspace(x_min, x_max, 5) / (n_losses-1), 2) * 100).astype(int)
    )
    plt.xlim(x_min, x_max)
    plt.ylim(-y_offset * 0.1, len(y_vals) * y_offset)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.xlabel('Model Accuracy [%]')
    plt.show()


if __name__ == '__main__':
    pass
