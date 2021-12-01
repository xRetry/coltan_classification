from abc import ABC

import numpy as np
import scipy.stats
import scipy.special
import abc
from typing import Optional


class Distribution(abc.ABC):
    @property
    @abc.abstractmethod
    def parameters(self) -> dict:
        pass


class MultiNormal(Distribution):
    _mean: np.ndarray
    _cov: np.ndarray

    def __init__(self, mean: np.ndarray, cov: Optional[np.ndarray]=None, std: Optional[np.ndarray]=None):
        if len(mean.shape) == 1:
            mean = mean[:, None]
        if mean.shape[0] < mean.shape[1]:
            mean = mean.T
        self._mean = mean

        if cov is not None:
            if cov.shape[0] != cov.shape[1]:
                raise AttributeError('Covariance matrix is not square!')
            self._cov = cov
        if cov is None and std is not None:
            if len(std.shape) != 1:
                raise AttributeError('Invalid std dimension (dim={})!'.format(std.shape))
            if len(std) != len(mean):
                raise AttributeError('Means and Stds need to be the same size (len(mean)={}, len(std)={})'.format(len(mean), len(std)))
            self._cov = np.diag(np.power(std, 2))
        if cov is None and std is None:
            raise AttributeError('Either covariance matrix or standard deviations have to be provided!')

    def pdf(self, x: np.ndarray) -> float:
        if not np.any(np.diag(self._cov) == 0):
            if len(x.shape) == 1:
                x = x[:, None]
            n = len(self._mean)
            det = np.linalg.det(self._cov)
            norm_const = 1 / (np.power(2 * np.pi, n / 2) * np.power(det, 1 / 2))
            z = norm_const * np.exp(-1 / 2 * (x - self._mean).T @ np.linalg.inv(self._cov) @ (x - self._mean))
            return z[0][0]
        return np.nan

    def posterior(self, x: np.ndarray, cov_known: np.ndarray):
        mean_prior = self._mean
        cov_prior = self._cov
        if len(x.shape) == 1:
            n = 1
            x = x[:, None]
            mean_x = x
        else:
            n = len(x)
            mean_x = x.mean(axis=0)[:, None]

        inverse = np.linalg.inv(cov_prior + (1 / n) * cov_known)
        mu_post = cov_prior @ inverse * mean_x + (1 / n) * cov_known @ inverse @ mean_prior
        cov_post = cov_prior @ inverse @ cov_known * (1 / n)
        return MultiNormal(np.diag(mu_post)[:, None], cov_post)

    def test_1sample(self, x: np.ndarray):
        attr_probabilities = scipy.stats.ttest_1samp(x, self._mean.T)
        return np.product(attr_probabilities[1])

    def kl_divergence(self, mean, cov):
        # Removing zeros in diagonal of covariance matrix
        is_not_zero = (np.diag(self._cov) != 0) & (np.diag(cov) != 0)
        mean1 = self._mean[is_not_zero, :]
        mean2 = mean[is_not_zero, :]
        cov1 = self._cov[is_not_zero, :][:, is_not_zero]
        cov2 = cov[is_not_zero, :][:, is_not_zero]
        # Computing KL-divergence
        n = len(mean1)
        return 1 / 2 * (np.log(np.linalg.det(cov2) / np.linalg.det(cov1)) - n +
                        np.trace(np.linalg.inv(cov2) @ cov1) +
                        (mean2 - mean1).T @ np.linalg.inv(cov2) @ (mean2 - mean1))[0][0]

    @property
    def mean(self):
        return self._mean

    @property
    def cov(self):
        return self._cov

    @property
    def parameters(self) -> dict:
        return {'mean':self.mean, 'covariance':self.cov}

    def __len__(self):
        return len(self._mean)


class NonParametric(Distribution):
    _samples: np.ndarray

    def __init__(self, samples: np.ndarray=np.array([])):
        self._samples = samples

    def test_norm_frobenius(self, sample: np.ndarray) -> float:
        return -float(np.linalg.norm(self._samples.mean(axis=0) - sample.mean(axis=0)))

    def test_norm_1(self, sample: np.ndarray) -> float:
        return -float(np.linalg.norm(self._samples.mean(axis=0) - sample.mean(axis=0), 1))

    def test_norm_2(self, sample: np.ndarray) -> float:
        return -float(np.linalg.norm(self._samples.mean(axis=0) - sample.mean(axis=0), 2))

    def test_ranksums(self, sample: np.ndarray) -> float:
        n_attr = self._samples.shape[1]
        p_vals = np.zeros(n_attr)
        for i in range(n_attr):
            test_result = scipy.stats.ranksums(self._samples[:, i], sample[:, i])
            p_vals[i] = test_result[1]
        return np.product(p_vals)  # TODO: Try other method of aggregation

    def test_mannwhitneyu(self, sample: np.ndarray) -> float:
        n_attr = self._samples.shape[1]
        p_vals = np.zeros(n_attr)
        for i in range(n_attr):
            test_result = scipy.stats.mannwhitneyu(self._samples[:, i], sample[:, i])
            p_vals[i] = test_result[1]
        return np.product(p_vals)

    @property
    def values(self):
        return self._samples

    @property
    def parameters(self) -> dict:
        return {'values': self.values}

    def __len__(self):
        raise NotImplementedError()


class NormalInverseChiSquared(Distribution):  # TODO: Test distribution
    _mean: float
    _std: float
    _kappa: float
    _nu: float

    def __init__(self, mean, std, kappa, nu):
        if len(mean.shape) == 1:
            mean = mean[:, None]
        if len(std.shape) == 1:
            std = std[:, None]

        self._mean = mean
        self._std = std
        self._kappa = kappa
        self._nu = nu

    def posterior(self, x:np.ndarray) -> Distribution:
        if len(x.shape) != 2:
            raise ValueError('New data needs to be two dimensional!')

        n = len(x)
        x_mean = x.mean(axis=0)[:, None]
        _var = np.power(self._std, 2)

        kappa = self._kappa + n
        mean = (self._kappa*self._mean + n*x_mean) / kappa
        nu = self._nu + n
        std = np.sqrt(1/nu * (self._nu*_var + np.sum(np.power(x.T-x_mean, 2), axis=1)[:, None] + (n*self._kappa)/(self._kappa+n) * np.power(self._mean - x_mean, 2)))
        return NormalInverseChiSquared(mean, std, kappa, nu)

    def pdf(self, mean, std) -> float:
        _var = np.power(self._std, 2)
        var = np.power(std, 2)
        Z = np.sqrt(2*np.pi)/np.sqrt(self._kappa) * scipy.special.gamma(self._nu/2) * np.power(2/(self._nu * _var), self._nu/2)
        return 1/Z * np.power(std, -1) * np.power(var, -(self._nu/2+1)) * np.exp(-1/(2*var) * (self._nu*_var + self._kappa*np.power(self._mean - mean, 2)))

    def pdf_predictive(self, x) -> float:
        p = scipy.stats.t.pdf(  # TODO: Fix output dimensions
            x=x,
            df=self._nu,
            loc=self._mean,
            scale=(1+self._kappa)*np.power(self._std, 2) / self._kappa
        )
        return np.product(p)

    @property
    def parameters(self):
        return {'mean': self._mean, 'std': self._std, 'nu': self._nu, 'kappa': self._kappa}


class NormalInverseWishart(Distribution):  # TODO: Test distribution
    _mean: np.ndarray
    _prec: np.ndarray
    _kappa: np.ndarray
    _nu: np.ndarray

    def __init__(self, mean, prec, kappa, nu):
        self._mean = mean
        self._prec = prec
        self._kappa = kappa
        self._nu = nu

    def posterior(self, x: np.ndarray) -> Distribution:
        """
        Creates new NIW distribution with parameters of the posterior given the new data.
        """
        # Check sample dimensions
        if len(x.shape) != 2:
            raise ValueError('New data needs to be two dimensional!')
        # Precompute necessary values
        n = len(x)
        x_mean = x.mean(axis=0)[:, None]
        mean_diff = (x_mean - self._mean)
        #S_mean = n * x_mean[:, None] @ x_mean[:, None].T
        #S2 = np.zeros((len(x[0]), len(x[0])))
        #for i in range(n):
        #    S2 += x[i, None].T @ x[i, None]
        #S = S2 - S_mean
        S = np.cov(x.T) * (len(x)-1)
        # Compute posterior parameters
        mean = (self._kappa / (self._kappa + n) * self._mean + n / (self._kappa + n) * x_mean)
        kappa = self._kappa + n
        nu = self._nu + n
        prec = self._prec + S + self._kappa * n / (self._kappa + n) * mean_diff @ mean_diff.T
        # Return distribution with new parameters
        return NormalInverseWishart(mean, prec, kappa, nu)

    def pdf(self, mean, cov) -> float:
        """
        Computes the pdf value for provided mean and cov.
        """
        d = len(mean)
        lam_det = np.linalg.det(self._prec)
        cov_det = np.linalg.det(cov)
        cov_inv = np.linalg.inv(cov)
        mean_diff = (mean - self._mean)
        Z = (np.power(2, self._nu * d / 2) * scipy.special.gamma(self._nu/2) * np.power(2*np.pi/self._kappa, d/2)) / np.power(lam_det, self._nu/2)
        return 1/Z * np.power(cov_det, -((self._nu+d)/2+1)) * np.exp(-1 / 2 * np.trace(self._prec @ cov_inv) - self._kappa / 2 * mean_diff.T @ cov_inv @ mean_diff)

    def pdf_predictive(self, x: np.ndarray) -> float:
        """
        Computes the pdf value for a new sample.
        """
        d = len(self._mean)
        return scipy.stats.t.pdf(
            x=x,
            df=self._nu-d+1,
            loc=self._mean,
            scale=self._prec * (self._kappa+1) / (self._kappa * (self._nu-d+1))
        )

    @property
    def parameters(self) -> dict:
        return {'mean': self._mean, 'precision': self._prec, 'kappa': self._kappa, 'nu': self._nu}


if __name__ == '__main__':
    pass
