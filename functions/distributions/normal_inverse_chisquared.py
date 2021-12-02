import numpy as np
import scipy.special
import scipy.stats


def posterior(mean: np.ndarray, std: np.ndarray, kappa: int, nu: int, x: np.ndarray) \
        -> (np.ndarray, np.ndarray, np.ndarray):
    n = len(x)
    x_mean = x.mean(axis=0)[:, None]
    _var = np.power(std, 2)

    kappa_p = kappa + n
    mean_p = (kappa*mean + n*x_mean) / kappa_p
    nu_p = nu + n
    std_p = np.sqrt(1/nu_p * (nu*_var + np.sum(np.power(x.T-x_mean, 2), axis=1)[:, None] + (n*kappa)/(kappa+n) * np.power(mean - x_mean, 2)))
    return mean_p, std_p, kappa_p, nu_p


def pdf(mean, std, kappa: int, nu: int, x_mean: np.ndarray, x_std: np.ndarray) -> float:
    _var = np.power(std, 2)
    var = np.power(x_std, 2)
    Z = np.sqrt(2*np.pi)/np.sqrt(kappa) * scipy.special.gamma(nu/2) * np.power(2/(nu * _var), nu/2)
    return 1/Z * np.power(x_std, -1) * np.power(var, -(nu/2+1)) * np.exp(-1/(2*var) * (nu*_var + kappa*np.power(mean - x_mean, 2)))


def pdf_predictive(mean: np.ndarray, std: np.ndarray, kappa: int, nu: int, x) -> float:
    p = scipy.stats.t.pdf(  # TODO: Fix output dimensions
        x=x,
        df=nu,
        loc=mean,
        scale=(1+kappa)*np.power(std, 2) / kappa
    )
    return np.product(p)


if __name__ == '__main__':
    pass
