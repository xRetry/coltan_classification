import numpy as np
import scipy.stats
from functions.decorators import verification


@verification('a', 'a')
def test_norm_frobenius(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Computes the difference between two vectors using the frobenius norm.
    """
    return -float(np.linalg.norm(x1 - x2))


@verification('a', 'a')
def test_norm1(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Computes the difference between two vectors using the L1 norm.
    """
    return -float(np.linalg.norm(x1 - x2, 1))


@verification('a', 'a')
def test_norm2(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Computes the difference between two vectors using the L2 norm.
    """
    return -float(np.linalg.norm(x1 - x2, 2))


@verification('ba', 'ca')
def test_ranksums(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Performs a ranksums test between two non-parametric distributions.
    """
    test_results = np.zeros(x1[0])
    for i in range(len(x1[0])):
        test_results[i] = scipy.stats.ranksums(x1[:, i], x2[:, i])[1]
    return np.product(test_results)


@verification('ba', 'ca')
def test_mannwhitneyu(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Performs a mann-whitney-u test between two non-parametric distributions.
    """
    test_results = np.zeros(x1[0])
    for i in range(len(x1[0])):
        test_results[i] = scipy.stats.mannwhitneyu(x1[:, i], x2[:, i])[1]
    return np.product(test_results)


@verification('a', 'a', None, None)
def test_exponential(x1: np.ndarray, x2: np.ndarray, exponent: float=2, scale: float=1) -> float:
    """
    Generic exponential kernel with adjustable parameters for shape and scale.
    """
    evals = np.exp(-np.power(np.abs(x1 - x2), exponent) * scale)
    return np.product(evals)


if __name__ == '__main__':
    pass
