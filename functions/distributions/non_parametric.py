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


@verification('a', 'a')
def test_ranksums(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Performs a ranksums test between two non-parametric distributions.
    """
    test_result = scipy.stats.ranksums(x1, x2)
    return test_result[1]


@verification('a', 'a')
def test_mannwhitneyu(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Performs a mann-whitney-u test between two non-parametric distributions.
    """
    test_result = scipy.stats.mannwhitneyu(x1, x2)
    return test_result[1]


if __name__ == '__main__':
    pass
