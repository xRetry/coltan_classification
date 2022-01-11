import numpy as np
import scipy.stats
import statsmodels.api as sm

from functions.utils import verification


@verification('a1', 'aa', 'a1')
def pdf(mean: np.ndarray, cov: np.ndarray, x: np.ndarray) -> float:
    """
    Computes the pdf of a multivariate normal at the specified location.
    """
    p = scipy.stats.multivariate_normal.pdf(x.flatten(), mean.flatten(), cov, allow_singular=True)
    return p


@verification('a1', 'aa', 'ba', 'aa')
def posterior(mean: np.ndarray, cov:np.ndarray, x: np.ndarray, cov_known: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Computes the parameters of the posterior distribution in case the true covariance matrix is known.
    """
    n = len(x)
    mean_x = x.mean(axis=0)[:, None]

    inverse = np.linalg.inv(cov + (1 / n) * cov_known)
    mu_post = cov @ inverse * mean_x + (1 / n) * cov_known @ inverse @ mean
    cov_post = cov @ inverse @ cov_known * (1 / n)
    return np.diag(mu_post)[:, None], cov_post


@verification('a1', 'aa', 'a1', 'aa')
def kl_divergence(mean1: np.ndarray, cov1: np.ndarray, mean2: np.ndarray, cov2: np.ndarray) -> float:
    """
    Computes the KL-Divergence between two multivariate normal distributions.
    """
    # Removing zeros in diagonal of covariance matrix
    is_not_zero = (np.diag(cov1) != 0) & (np.diag(cov2) != 0)
    mean1 = mean1[is_not_zero, :]
    mean2 = mean2[is_not_zero, :]
    cov1 = cov1[is_not_zero, :][:, is_not_zero]
    cov2 = cov2[is_not_zero, :][:, is_not_zero]
    # Computing KL-divergence
    n = len(mean1)
    return 1 / 2 * (np.log(np.linalg.det(cov2) / np.linalg.det(cov1)) - n +
                    np.trace(np.linalg.inv(cov2) @ cov1) +
                    (mean2 - mean1).T @ np.linalg.inv(cov2) @ (mean2 - mean1))[0][0]


@verification('a1', 'ba')
def ttest_1sample(mean: np.ndarray, x: np.ndarray) -> float:  # TODO: Check computation
    x = np.asarray(x)
    nobs, k_vars = x.shape
    mean_x = x.mean(0)
    cov = np.cov(x, rowvar=False, ddof=1)
    diff = mean_x - mean.flatten()
    t2 = nobs * diff.dot(np.linalg.solve(cov, diff))
    factor = (nobs - 1) * k_vars / (nobs - k_vars)
    statistic = t2 / factor
    df = (k_vars, nobs - k_vars)
    pvalue = scipy.stats.f.sf(statistic, df[0], df[1])
    return pvalue


if __name__ == '__main__':
    pass
