from typing import Iterable
import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    pass
