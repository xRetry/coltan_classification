import numpy as np
from classes.distributions import Distribution, MultiNormal


'''
    +++ DATA TRANSFORMATIONS +++
'''


def transform_none(x):
    """
    Applies no transformation to the input.

    :param x: Any object or value
    :return: Returns input without modifications
    """
    return x


def transform_log(x: np.ndarray or list) -> np.ndarray or list:
    is_list = True
    if not isinstance(x, list):
        is_list = False
        x = [x]
    for i in range(len(x)):
        x[i] = np.array(x[i])
        x[i] = np.log(x[i], out=np.ones_like(x[i])*-20, where=x[i] > 0)  # TODO: Find appropriate value for negative inputs
    if is_list:
        return x
    return x[0]


'''
    +++ SAMPLE EVALUATION +++
'''


def eval_pdf(distribution: Distribution, sample: np.ndarray) -> float:
    sample_means = sample.mean(axis=0)
    return distribution.pdf(sample_means)


def eval_ttest(distribution: MultiNormal, sample: np.ndarray) -> float:
    return distribution.ttest_1sample(sample)


'''
    +++ LOSS FUNCTIONS +++
'''


def loss_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    n_correct = (labels == predictions).sum()
    return n_correct / len(predictions)


def loss_error(labels: np.ndarray, predictions: np.ndarray) -> float:
    acc = loss_accuracy(labels, predictions)
    return 1-acc


'''
    +++ VALUE-TO-LABEL CONVERSIONS +++
'''


def select_mine(eval_results: np.ndarray, labels: np.ndarray) -> int:
    if np.all(np.isnan(eval_results)):
        return np.nan
    idx = np.nanargmax(eval_results)
    return labels[idx]


def select_label(eval_results: np.ndarray, labels: np.ndarray) -> int:
    sum_full = np.nansum(eval_results)
    sum_pos = np.nansum(eval_results[labels == 1])
    if sum_full == 0:
        return np.nan
    p_pos = sum_pos/sum_full
    return (p_pos > 0.5) * 2 - 1


if __name__ == '__main__':
    pass
