import numpy as np


def normal_pdf(mean, std, x: np.ndarray):
    norm = np.divide(1, (std * np.sqrt(2 * np.pi)), out=np.zeros_like(std) * np.nan, where=std != 0)
    exponent = np.divide(-np.power(x - mean, 2), (2 * np.power(std, 2)), out=np.zeros_like(x) * np.nan, where=std != 0)
    return norm * np.exp(exponent)


def normal_mu(values) -> np.ndarray:
    return np.mean(values, axis=0)


def normal_sigma(values) -> np.ndarray:
    mu = normal_mu(values)
    n = len(values)
    sigma = np.sqrt(1 / (n - 1) * np.sum(np.power((values - mu), 2), axis=0))
    return sigma


def normal_sigma_orig(values) -> np.ndarray:
    # Calculation is strictly following the paper (n instead of (n-1))
    mu = normal_mu(values)
    n = len(values)
    sigma = np.sqrt(1 / n * np.sum(np.power((values - mu), 2), axis=0))

    # Not working: negative sign in sqrt
    # sigma = np.sqrt(np.sum(np.power(values, 2) / n - np.power(mu, 2), axis=0))
    return sigma


def norm_posterior_skwn(mu_prior, sigma_prior, sigma_known, x):
    """
    Calculates the posterior for known sigma.
    For simplification sigma can be assumed to be equal to the sample sigma.
    """
    n_new = len(x)
    mu = (mu_prior/np.power(sigma_prior, 2) + np.sum(x, axis=0)/np.power(sigma_known, 2)) / (1/np.power(sigma_prior, 2) + n_new/np.power(sigma_known, 2))
    sigma = np.power(1/np.power(sigma_prior, 2) + n_new/np.power(sigma_known, 2), -1/2)
    return mu, sigma


if __name__ == '__main__':
    pass
