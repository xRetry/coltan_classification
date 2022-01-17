from typing import List
import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    pass
