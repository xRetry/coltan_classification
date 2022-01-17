from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    pass
