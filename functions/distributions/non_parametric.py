import numpy as np
import scipy.stats


def test_norm_frobenius(x1: np.ndarray, x2: np.ndarray) -> float:
    return -float(np.linalg.norm(x1 - x2))


def test_norm1(x1: np.ndarray, x2: np.ndarray) -> float:
    return -float(np.linalg.norm(x1 - x2, 1))


def test_norm2(x1: np.ndarray, x2: np.ndarray) -> float:
    return -float(np.linalg.norm(x1 - x2, 2))


def test_ranksums(x1: np.ndarray, x2: np.ndarray) -> float:
    test_result = scipy.stats.ranksums(x1, x2)
    return test_result[1]


def test_mannwhitneyu(x1: np.ndarray, x2: np.ndarray) -> float:
    test_result = scipy.stats.mannwhitneyu(x1, x2)
    return test_result[1]


if __name__ == '__main__':
    pass
