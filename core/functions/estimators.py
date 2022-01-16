import numpy as np
from core.utils import verification
from typing import Optional


@verification('ba')
def mean(attr_values: np.ndarray) -> np.ndarray:
    return np.mean(attr_values, axis=0)


@verification('ba')
def median(attr_values: np.ndarray) -> np.ndarray:
    return np.median(attr_values, axis=0)


@verification('ba')
def hodges_lehmann(attr_values: np.ndarray) -> np.ndarray:
    n, n_attr = attr_values.shape
    tri_mask = np.tril_indices(n)
    loc = np.zeros(n_attr)
    for a in range(n_attr):
        means = (attr_values[:, a] + attr_values[:, a][:, None]) / 2
        loc[a] = np.median(means[tri_mask])
    return loc


@verification('ba', None, None)
def std(attr_values: np.ndarray, corrected=True, loc: Optional[np.ndarray]=None) -> np.ndarray:
    if loc is None:
        loc = mean(attr_values)

    n = len(attr_values)
    if corrected and n > 1:
        n -= 1
    return np.sqrt(1 / n * np.sum(np.power((attr_values - loc), 2), axis=0))


@verification('ba')
def cov(attr_values: np.ndarray):
    return np.cov(attr_values.T)


if __name__ == '__main__':
    pass
