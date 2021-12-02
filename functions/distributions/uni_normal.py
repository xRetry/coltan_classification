import numpy as np
import scipy.stats


def pdf(mean: np.ndarray, std: np.ndarray, x: np.ndarray):
    norm = np.divide(1, (std * np.sqrt(2 * np.pi)), out=np.zeros_like(std) * np.nan, where=std != 0)
    exponent = np.divide(-np.power(x - mean, 2), (2 * np.power(std, 2)), out=np.zeros_like(x) * np.nan,
                         where=std != 0)
    return norm * np.exp(exponent)


def posterior(mu_prior, sigma_prior, sigma_known, x):
    """
    Calculates the posterior for known sigma.
    """
    n_new = len(x)
    mu = (mu_prior / np.power(sigma_prior, 2) + np.sum(x, axis=0) / np.power(sigma_known, 2)) / (
                1 / np.power(sigma_prior, 2) + n_new / np.power(sigma_known, 2))
    sigma = np.power(1 / np.power(sigma_prior, 2) + n_new / np.power(sigma_known, 2), -1 / 2)
    return mu, sigma


def test_1sample(mean, x: np.ndarray):
    attr_probabilities = scipy.stats.ttest_1samp(x, mean.T)
    return np.product(attr_probabilities[1])


if __name__ == '__main__':
    pass
