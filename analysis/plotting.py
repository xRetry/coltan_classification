import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import statsmodels.api as sm
from typing import List, Optional, Iterable, Dict


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


def plot_mine_evaluation(eval_values: np.ndarray, labels: np.ndarray, is_added: np.ndarray):
    """
    Plots the mine evaluation values for samples.
    """
    plt.figure()
    # Plot unused samples
    plt.scatter(eval_values[np.invert(is_added)], labels[np.invert(is_added)], c='b', alpha=0.5)
    # Plot used samples
    plt.scatter(eval_values[is_added], labels[is_added], c='r', alpha=0.5)
    # Modify appearance and layout
    plt.yticks([1, -1])
    plt.ylabel('Mine Labels')
    plt.xlabel('Sample Evaluation Value')
    plt.legend(['unused', 'used for training'])
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


def plot_eval_result(x1, x2, results: Dict[str, np.ndarray], sample_test: Optional[np.ndarray]=None):
    """
    Plots the result of an evaluation function analysis.
    """
    # Getting function names
    names = list(results.keys())
    # Determine type of plot
    use_surface_plot = False
    if len(names) == 1:
        use_surface_plot = True
    # Creating plot
    if use_surface_plot:
        # Get y values
        y = results[names[0]]
        # Defining figure
        fig, axs = plt.subplots(
            2, 2,
            sharex='col',
            sharey='row',
            gridspec_kw={'width_ratios': [4, 2], 'height_ratios': [2, 4]}
        )
        # Defining sample axis and orientation
        sample_axis = [axs[0, 0], axs[1, 1]]
        sample_orient = ['vertical', 'horizontal']
        # Plotting
        axs[1, 0].contourf(x1, x2, y)
        axs[0, 0].plot(x1[0], np.max(y, axis=0))
        axs[1, 1].plot(np.max(y, axis=1), x2[:, 0])
        # Disable axis of empty subplot
        axs[0, 1].axis('off')
    else:
        # Defining figure
        fig, axs = plt.subplots(2, 1)
        # Defining sample axis and orientation
        sample_axis = axs
        sample_orient = ['vertical', 'vertical']
        # Combining x values to list
        x = [x1[0, :], x2[:, 0]]
        # Plotting all functions
        for func_name in names:
            y = results[func_name]
            for i in range(2):
                y_vals = np.max(y, axis=i)
                # Shift to zero
                y_vals -= min(y_vals)
                # Scale to one
                y_vals /= max(y_vals)
                # Plot function values
                axs[i].plot(x[i], y_vals)
        # Adding legend
        plt.legend(names)
    # Plotting histogram of sample as reference
    if sample_test is not None:
        for i in range(len(sample_axis)):
            sample_axis[i].hist(sample_test[i, :], orientation=sample_orient[i], density=True)  # TODO: Replace density with accurate normalization
    # Showing figure
    plt.tight_layout()
    plt.show()


def plot_cv_stepwise(splits: np.ndarray, conf_ints: np.ndarray, model_names: Iterable) -> None:
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


def plot_kwargs_accs(x: Dict[str, np.ndarray], y: np.ndarray) -> None:
    """
    Plots result of kwargs function analysis.
    """
    # Get keys and values as from x dictionary
    values = list(x.values())
    labels = list(x.keys())
    # 1D Plot
    if len(values) == 1:
        mean = np.mean(y, axis=1)
        plt.errorbar(values[0], mean, yerr=np.abs(mean[:, None]-y).T, capsize=5)
        plt.xlabel(labels[0])
        plt.ylabel('Accuracy')
    # 2D plot
    else:
        plt.contourf(values[0], values[1], y)
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Accuracy')
    plt.show()


def plot_pdf(x, y):
    plt.figure()
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    pass
