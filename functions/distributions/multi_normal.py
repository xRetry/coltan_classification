import numpy as np
from functions.decorators import verification


@verification('a1', 'aa', 'a1')
def pdf(mean: np.ndarray, cov: np.ndarray, x: np.ndarray) -> float:
    """
    Computes the pdf of a multivariate normal at the specified location.
    """
    if not np.any(np.diag(cov) == 0):
        n = len(mean)
        det = np.linalg.det(cov)
        norm_const = 1 / (np.power(2 * np.pi, n / 2) * np.power(det, 1 / 2))
        p = norm_const * np.exp(-1 / 2 * (x - mean).T @ np.linalg.inv(cov) @ (x - mean))  # TODO: Figure out why underflow occurs
        return p[0][0]
    return np.nan


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


if __name__ == '__main__':
    pass
