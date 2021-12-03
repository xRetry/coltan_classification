import numpy as np
import scipy.stats
import functools


def eval_verification(func):
    """
    Verifies the inputs for non-parametric evaluation functions.
    """
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        # Put all inputs into kwargs
        arg_names = ('x1', 'x2')
        kwargs.update({arg_names[i]: a for i, a in enumerate(args)})
        # Check for correct dims
        if len(kwargs['x1'].shape) != 1:
            raise ValueError('Invalid input dimensions! (dims={})'.format(kwargs['x1'].shape))
        # Check for equal dims
        if kwargs['x1'].shape != kwargs['x2'].shape:
            raise ValueError('Input dimensions must be equal! {} != {}'.format(kwargs['x1'].shape, kwargs['x2'].shape))

        value = func(**kwargs)
        return value
    return wrapper_decorator


@eval_verification
def test_norm_frobenius(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Computes the difference between two vectors using the frobenius norm.
    """
    return -float(np.linalg.norm(x1 - x2))


@eval_verification
def test_norm1(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Computes the difference between two vectors using the L1 norm.
    """
    return -float(np.linalg.norm(x1 - x2, 1))


@eval_verification
def test_norm2(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Computes the difference between two vectors using the L2 norm.
    """
    return -float(np.linalg.norm(x1 - x2, 2))


@eval_verification
def test_ranksums(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Performs a ranksums test between two non-parametric distributions.
    """
    test_result = scipy.stats.ranksums(x1, x2)
    return test_result[1]


@eval_verification
def test_mannwhitneyu(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Performs a mann-whitney-u test between two non-parametric distributions.
    """
    test_result = scipy.stats.mannwhitneyu(x1, x2)
    return test_result[1]


if __name__ == '__main__':
    pass
