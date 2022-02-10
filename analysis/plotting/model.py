from typing import Iterable, List, Optional
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


def plot_crossval_distributions_single(accs: List[np.ndarray], conf_ints: np.ndarray,
                                model_labels: List[str], use_beta: bool = False) -> None:
    """
    Plots distributions from cross-validation results.
    """
    plt.figure(figsize=(7, 5))
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
    colors = plt.get_cmap('tab10')(range(len(y_vals)))
    for i, y in enumerate(y_vals):
        if use_beta:
            plt.plot(x_vals, y_vals[i] - i*y_offset, color=colors[i])
            x_ci = [conf_ints[i], conf_ints[i]]
        else:
            plt.bar(x_vals, y_vals[i], bottom=-i * y_offset, color=colors[i])
            plt.plot([0, n_losses + 1], [-i * y_offset] * 2, linewidth=1, color=colors[i])
            x_ci = [np.array(conf_ints[i]) * (n_losses + 1), np.array(conf_ints[i]) * (n_losses + 1)]

        plt.plot(x_ci[0], [-i * y_offset] * 2, linewidth=4, color='k')

    plt.yticks(-np.arange(len(y_vals)) * y_offset + y_offset/3, model_labels)
    plt.xticks(
        np.linspace(x_min, x_max, 5),
        (np.round(np.linspace(x_min, x_max, 5) / (n_losses-1), 2) * 100).astype(int)
    )
    plt.gca().tick_params(axis='y', which='both', length=0)
    plt.xlim(x_min, x_max)
    plt.ylim(-(len(y_vals)-1) * y_offset * 1.05, y_offset * 1.1)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.xlabel('Model Accuracy [%]')
    plt.tight_layout()
    plt.show()

def plot_crossval_distributions_double(accs: List[np.ndarray] or List[List[np.ndarray]], conf_ints: List[np.ndarray],
                                model_labels: List[str], comparison_labels: Optional[List[str]] = None,
                                use_beta: bool = False) -> None:
    """
    Plots distributions from cross-validation results while comparing two categories.
    """
    if not isinstance(conf_ints, list):
        conf_ints = [conf_ints]
        accs = [accs]

    plt.figure(figsize=(8, 6))
    n_losses = len(accs[0][0])
    x_vals = np.linspace(0, 1, 500) if use_beta else np.arange(n_losses + 1)
    y_vals = np.zeros((len(accs), len(accs[0]), len(x_vals)))
    for j in range(len(accs)):
        for i in range(len(accs[0])):
            if use_beta:  # TODO: Find out why shape of beta is different from binomial
                y_vals[j][i] = scipy.stats.beta.pdf(x_vals, accs[j][i].sum(), len(accs[j][i]) - accs[j][i].sum())
            else:
                y_vals[j][i] = scipy.stats.binom.pmf(x_vals, n=len(accs[j][i]), p=accs[j][i].sum() / len(accs[j][i]))

    y_centers = np.max(np.sum(np.max(y_vals, axis=2), axis=0)) * 1.2
    delta = 0.05 * y_centers
    y_offset = y_centers + 2*delta
    _, _, idx_above_threshold = np.where(y_vals > 1e-4)
    x_min, x_max = x_vals[idx_above_threshold.min()], x_vals[idx_above_threshold.max()]
    colors = plt.get_cmap('tab10')(range(len(y_vals[0])))
    for i, y in enumerate(y_vals[0]):
        if use_beta:
            plt.plot(x_vals, y_vals[0][i] - i*y_offset + delta, color=colors[i])
            plt.plot(x_vals, -y_vals[1][i] - (i * y_offset - delta), color=colors[i])
            x_ci = [conf_ints[0][i], conf_ints[1][i]]
        else:
            plt.bar(x_vals, y_vals[0][i], bottom=-i * y_offset + delta, color=colors[i])
            plt.bar(x_vals, -y_vals[1][i], bottom=-i * y_offset - delta, color=colors[i])
            plt.plot([0, n_losses + 1], [-i * y_offset + delta] * 2, linewidth=1, color=colors[i])
            plt.plot([0, n_losses + 1], [-i * y_offset - delta] * 2, linewidth=1, color=colors[i])
            x_ci = [np.array(conf_ints[0][i]) * (n_losses + 1), np.array(conf_ints[1][i]) * (n_losses + 1)]

        plt.plot(x_ci[0], [-i * y_offset + delta] * 2, linewidth=4, color='k')
        plt.plot(x_ci[1], [-i * y_offset - delta] * 2, linewidth=4, color='k')
        plt.text(x_min + 0.05, -i * y_offset + y_offset / 8, comparison_labels[0], fontstyle='italic', color='grey')
        plt.text(x_min + 0.05, -i * y_offset - y_offset / 8, comparison_labels[1], fontstyle='italic', color='grey', verticalalignment='top')

    plt.yticks(-np.arange(len(y_vals[0])) * y_offset, model_labels)
    plt.xticks(
        np.linspace(x_min, x_max, 5),
        (np.round(np.linspace(x_min, x_max, 5) / (n_losses-1), 2) * 100).astype(int)
    )
    plt.gca().tick_params(axis='y', which='both', length=0)
    plt.xlim(x_min, x_max)
    #plt.ylim(-y_offset * 0.1, len(y_vals[0]) * y_offset)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.xlabel('Model Accuracy [%]')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    pass
