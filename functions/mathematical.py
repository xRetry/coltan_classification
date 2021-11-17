import numpy as np


'''
    ++++++++++++++++++
    +++ UNIVARIATE +++
'''


def normal_uni_pdf(mean: np.ndarray, std: np.ndarray, x: np.ndarray):
    norm = np.divide(1, (std * np.sqrt(2 * np.pi)), out=np.zeros_like(std) * np.nan, where=std != 0)
    exponent = np.divide(-np.power(x - mean, 2), (2 * np.power(std, 2)), out=np.zeros_like(x) * np.nan, where=std != 0)
    return norm * np.exp(exponent)


def normal_uni_mean(values) -> np.ndarray:
    values = np.array(values)
    if len(values.shape) > 2:
        raise ValueError(f'Dimension of input array too big ({len(values.shape)} > 2)')
    return np.mean(values, axis=0)


def normal_uni_std(values, corrected=True, mean=None) -> np.ndarray:
    if mean is None:
        mean = normal_uni_mean(values)
    n = len(values)
    if corrected:
        n -= 1
    std = np.sqrt(1 / n * np.sum(np.power((values - mean), 2), axis=0))
    return std


def normal_uni_median(values) -> np.ndarray:
    values = np.array(values)
    if len(values.shape) > 2:
        raise ValueError(f'Dimension of input array too big ({len(values.shape)} > 2)')
    return np.median(values, axis=0)


def normal_uni_posterior_sigmaknown(mu_prior, sigma_prior, sigma_known, x):
    """
    Calculates the posterior for known sigma.
    """
    n_new = len(x)
    mu = (mu_prior/np.power(sigma_prior, 2) + np.sum(x, axis=0)/np.power(sigma_known, 2)) / (1/np.power(sigma_prior, 2) + n_new/np.power(sigma_known, 2))
    sigma = np.power(1/np.power(sigma_prior, 2) + n_new/np.power(sigma_known, 2), -1/2)
    return mu, sigma


'''
    ++++++++++++++++++++
    +++ MULTIVARIATE +++
'''


def normal_multi_cov(values: np.ndarray, corrected=True) -> np.ndarray:
    return np.cov(values, bias=not corrected)


def normal_multi_posterior_sigmaknown(mu_prior: np.ndarray, cov_prior: np.ndarray, cov_known: np.ndarray, x: np.ndarray):
    if len(mu_prior.shape) == 1:
        mu_prior = mu_prior[:, None]

    n = len(x)
    inverse = np.linalg.inv(cov_prior + (1 / n) * cov_known)
    mu_post = cov_prior @ inverse * x.mean() + (1 / n) * cov_known @ inverse @ mu_prior
    cov_post = cov_prior @ inverse @ cov_known * (1 / n)
    return np.diag(mu_post), cov_post


if __name__ == '__main__':
    pass
