import numpy as np
from functions.decorators import verification


@verification('ba')
def mean(attr_values: np.ndarray) -> np.ndarray:
    return np.mean(attr_values, axis=0)


@verification('ba')
def median(attr_values: np.ndarray) -> np.ndarray:
    return np.median(attr_values, axis=0)


@verification('ba', None, None)
def std(attr_values: np.ndarray, corrected=True, robust=False) -> np.ndarray:
    if robust:
        loc = median(attr_values)
    else:
        loc = median(attr_values)

    n = len(attr_values)
    if corrected and n > 1:
        n -= 1
    return np.sqrt(1 / n * np.sum(np.power((attr_values - loc), 2), axis=0))


@verification('ba')
def cov(attr_values: np.ndarray):
    return np.cov(attr_values.T)


if __name__ == '__main__':
    pass
