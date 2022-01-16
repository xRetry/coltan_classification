import numpy as np
import scipy.special
import scipy.stats
from core.utils import verification
from typing import Callable


@verification('a', 'a', '', '', 'ba')
def posterior(mean: np.ndarray, std: np.ndarray, kappa: int, nu: int, x: np.ndarray) \
        -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Computes the posterior parameters of a normal inverse chi-squared distribution given new samples.
    """
    # Pre-compute necessary values
    n = len(x)
    x_mean = x.mean(axis=0)
    var = np.power(std, 2)
    # Computing the posterior parameters
    kappa_p = kappa + n
    mean_p = (kappa*mean + n*x_mean) / kappa_p
    nu_p = nu + n
    std_p = np.sqrt(1/nu_p * (nu*var + np.sum(np.power(x - x_mean, 2), axis=0) + (n*kappa)/(kappa+n) * np.power(mean - x_mean, 2)))
    return mean_p, std_p, kappa_p, nu_p


@verification('a', 'a', '', '', 'a', 'a')
def pdf(mean, std, kappa: int, nu: int, x_mean: np.ndarray, x_std: np.ndarray, func_aggr: Callable=np.product) -> float:
    """
        Evaluates the pdf of a NIX distribution for a given mean and std.
    """
    var = np.power(std, 2)
    x_var = np.power(x_std, 2)
    Z = np.sqrt(2*np.pi)/np.sqrt(kappa) * scipy.special.gamma(nu/2) * np.power(2 / (nu * var), nu / 2)
    p = 1 / Z * np.power(x_std, -1) * np.power(x_var, -(nu / 2 + 1)) * np.exp(-1 / (2 * x_var) * (nu * var + kappa * np.power(mean - x_mean, 2)))  # TODO: Fix underflow
    return func_aggr(p)


@verification('a', 'a', '', '', 'a')
def pdf_predictive(mean: np.ndarray, std: np.ndarray, kappa: int, nu: int, x: np.ndarray, func_aggr: Callable=np.product) -> float:
    """
        Evaluates the predictive pdf of a NIX distribution for new data.
    """
    p = scipy.stats.t.pdf(  # TODO: Fix underflow
        x=x,
        df=nu,
        loc=mean,
        scale=(1+kappa)*np.power(std, 2) / kappa
    )
    return func_aggr(p)


if __name__ == '__main__':
    pass
