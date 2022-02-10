import numpy as np
import scipy.stats
from scipy import spatial
from core.utils import verification
from typing import Callable


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


@verification('a', 'a')
def test_cosine(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Computes the difference between two vectors using the cosine distance.
    """
    return 1 - spatial.distance.cosine(x1, x2)


@verification('a', 'a')
def test_canberra(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Computes the difference between two vectors using the canberra distance.
    """
    return 1 - spatial.distance.canberra(x1, x2)


@verification('a', 'a')
def test_correlation(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Computes the difference between two vectors using the correlation distance.
    """
    return 1 - spatial.distance.correlation(x1, x2)


@verification('ba', 'ca', None)
def test_ranksums(x1: np.ndarray, x2: np.ndarray, func_aggr: Callable=np.product) -> float:
    """
    Performs a ranksums test between two non-parametric distributions.
    """
    test_results = np.zeros(x1[0])
    for i in range(len(x1[0])):
        test_results[i] = scipy.stats.ranksums(x1[:, i], x2[:, i])[1]
    return func_aggr(test_results)


@verification('ba', 'ca', None)
def test_mannwhitneyu(x1: np.ndarray, x2: np.ndarray, func_aggr: Callable=np.product) -> float:
    """
    Performs a mann-whitney-u test between two non-parametric distributions.
    """
    test_results = np.zeros(x1[0])
    for i in range(len(x1[0])):
        test_results[i] = scipy.stats.mannwhitneyu(x1[:, i], x2[:, i])[1]
    return func_aggr(test_results)


@verification('a', 'a', None, None, None)
def test_exponential(x1: np.ndarray, x2: np.ndarray, exponent: float=2, scale: float=1, func_aggr: Callable=np.product) -> float:
    """
    Generic exponential kernel with adjustable parameters for shape and scale.
    """
    evals = np.exp(-np.power(np.abs(x1 - x2), exponent) * scale)
    return func_aggr(evals)


if __name__ == '__main__':
    pass
