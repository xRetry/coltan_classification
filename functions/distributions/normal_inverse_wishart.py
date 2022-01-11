import numpy as np
import scipy.special
import scipy.stats
from functions.utils import verification


@verification('a1', 'aa', '', '', 'ba')
def posterior(mean: np.ndarray, prec: np.ndarray, kappa: int, nu: int, x: np.ndarray) \
        -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Calculates the parameters of the posterior given the new data.
    """
    # Precompute necessary values
    n = len(x)
    x_mean = x.mean(axis=0)[:, None]
    mean_diff = (x_mean - mean)
    S = np.cov(x.T) * (len(x) - 1)
    # Compute posterior parameters
    mean_p = (kappa / (kappa + n) * mean + n / (kappa + n) * x_mean)
    kappa_p = kappa + n
    nu_p = nu + n
    prec_p = prec + S + kappa * n / (kappa + n) * mean_diff @ mean_diff.T
    # Return new parameters
    return mean_p, prec_p, kappa_p, nu_p


@verification('a1', 'aa', '', '', 'a1', 'aa')
def pdf(mean: np.ndarray, prec: np.ndarray, kappa: int, nu: int, x_mean: np.ndarray,
        x_cov: np.ndarray) -> float:
    """
    Computes the pdf value for provided mean and cov.
    """
    d = len(x_mean)
    lam_det = np.linalg.det(prec)
    cov_det = np.linalg.det(x_cov)
    cov_inv = np.linalg.inv(x_cov)
    mean_diff = (x_mean - mean)
    Z = (np.power(2, nu * d / 2) * scipy.special.gamma(nu / 2) * np.power(2 * np.pi / kappa, d / 2)) /\
        np.power(lam_det, nu / 2)  # TODO: Fix underflow
    p = 1 / Z * np.power(cov_det, -((nu + d) / 2 + 1)) * np.exp(
        -1 / 2 * np.trace(prec @ cov_inv) - kappa / 2 * mean_diff.T @ cov_inv @ mean_diff
    )
    return p


@verification('a1', 'aa', '', '', 'a1')
def pdf_predictive(mean: np.ndarray, prec: np.ndarray, kappa: int, nu: int, x: np.ndarray) -> float:  # TODO: Check correct output
    """
    Computes the pdf value for a new sample.
    """
    d = len(mean)
    return scipy.stats.t.pdf(
        x=x,
        df=nu - d + 1,
        loc=mean,
        scale=prec * (kappa + 1) / (kappa * (nu - d + 1))
    )


if __name__ == '__main__':
    pass
