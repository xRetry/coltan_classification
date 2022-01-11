import numpy as np
import scipy.stats
from functions.utils import verification
from typing import Callable


@verification('a', 'a', 'a', 'ba')
def posterior(mu_prior, sigma_prior, sigma_known, x):
    """
    Calculates the posterior for known sigma.
    """
    n_new = len(x)
    mu = (mu_prior / np.power(sigma_prior, 2) + np.sum(x, axis=0) / np.power(sigma_known, 2)) / (
                1 / np.power(sigma_prior, 2) + n_new / np.power(sigma_known, 2))
    sigma = np.power(1 / np.power(sigma_prior, 2) + n_new / np.power(sigma_known, 2), -1 / 2)
    return mu, sigma


@verification('a', 'a', 'a', None)
def pdf(mean: np.ndarray, std: np.ndarray, x: np.ndarray, func_aggr: Callable=np.product) -> float:
    """
    Calculates the value of the normal pdf function at given locations.
    """
    norm = np.divide(1, (std * np.sqrt(2 * np.pi)), out=np.zeros_like(std) * np.nan, where=std != 0)
    exponent = np.divide(-np.power(x - mean, 2), (2 * np.power(std, 2)), out=np.zeros_like(x) * np.nan,
                         where=std != 0)
    return func_aggr(norm * np.exp(exponent))


@verification('a', 'ba', None)
def test_1sample(mean: np.ndarray, x: np.ndarray, func_aggr: Callable=np.product) -> float:
    """
    Performs a 1 sample t-test.
    """
    attr_probabilities = scipy.stats.ttest_1samp(x, mean.T)
    return func_aggr(attr_probabilities[1])


@verification('a', 'a', 'a', 'a', 'a', 'a', None)
def test_2sample(mean1: np.ndarray, std1: np.ndarray, nobs1: np.ndarray,
                 mean2: np.ndarray, std2: np.ndarray, nobs2: np.ndarray, func_aggr: Callable=np.product) -> float:
    """
    Performs a 2 sample t-test.
    """
    attr_scores = scipy.stats.ttest_ind_from_stats(mean1, std1, nobs1, mean2, std2, nobs2)
    return func_aggr(attr_scores[1])


if __name__ == '__main__':
    pass
