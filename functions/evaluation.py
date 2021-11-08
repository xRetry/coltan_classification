import numpy as np
from typing import Callable
import scipy.stats


def eval_pdf(pdf_func: Callable, sample_means, *args):
    attr_probabilities = pdf_func(*args, sample_means)
    return np.product(attr_probabilities)


def eval_ttest(sample, *args):
    attr_probabilities = scipy.stats.ttest_1samp(sample, *args)[1]
    return np.product(attr_probabilities)


'''
    +++ LOSS FUNCTIONS +++
'''


def accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    n_correct = (labels == predictions).sum()
    return n_correct / len(predictions)


def error(labels: np.ndarray, predictions: np.ndarray) -> float:
    acc = accuracy(labels, predictions)
    return 1-acc


'''
    +++ VALUE-TO-LABEL CONVERSIONS +++
'''


def best_mine(eval_results, mines):
    eval_results = np.array(eval_results)

    if len(eval_results.shape) == 1:
        eval_results = eval_results.reshape((1, -1))

    selection = np.ones(len(eval_results)) * np.nan
    is_all_nan = np.all(np.isnan(eval_results),axis=1)
    idx = np.nanargmax(eval_results[np.invert(is_all_nan), :], axis=1)
    selection[np.invert(is_all_nan)] = np.array([mines[i].status for i in idx])
    return selection


def best_label(eval_results, mines):
    eval_results = np.array(eval_results)

    if len(eval_results.shape) == 1:
        eval_results = eval_results.reshape((1, -1))

    labels = np.array([m.status for m in mines])
    sum_full = np.nansum(eval_results, axis=1)
    sum_pos = np.nansum(eval_results[:, labels == 1], axis=1)
    p_pos = np.divide(sum_pos, sum_full, out=np.ones_like(sum_pos)*np.nan, where=sum_full != 0)
    selection = np.ones_like(p_pos)*np.nan
    is_valid = np.invert(np.isnan(p_pos))
    selection[is_valid] = -np.ones(len(p_pos[is_valid]), dtype=int) + 2 * (p_pos[is_valid] > 0.5).astype(int)
    return selection


if __name__ == '__main__':
    pass
