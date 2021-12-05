import numpy as np
from functions.decorators import verification


@verification('a', 'a')
def mine(eval_results: np.ndarray, labels: np.ndarray) -> int:
    if np.all(np.isnan(eval_results)):
        return np.nan
    idx = np.nanargmax(eval_results)
    return labels[idx]


@verification('a', 'a')
def label(eval_results: np.ndarray, labels: np.ndarray) -> int:
    sum_full = np.nansum(eval_results)
    sum_pos = np.nansum(eval_results[labels == 1])
    if sum_full == 0:
        return np.nan
    p_pos = sum_pos / sum_full
    return (p_pos > 0.5) * 2 - 1


if __name__ == '__main__':
    pass
