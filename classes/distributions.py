from abc import ABC

import numpy as np
import scipy.stats
import scipy.special
import abc
from typing import Optional


class UniNormal:
    @staticmethod
    def pdf(mean: np.ndarray, std: np.ndarray, x: np.ndarray):
        norm = np.divide(1, (std * np.sqrt(2 * np.pi)), out=np.zeros_like(std) * np.nan, where=std != 0)
        exponent = np.divide(-np.power(x - mean, 2), (2 * np.power(std, 2)), out=np.zeros_like(x) * np.nan,
                             where=std != 0)
        return norm * np.exp(exponent)

    @staticmethod
    def posterior(mu_prior, sigma_prior, sigma_known, x):
        """
        Calculates the posterior for known sigma.
        """
        n_new = len(x)
        mu = (mu_prior / np.power(sigma_prior, 2) + np.sum(x, axis=0) / np.power(sigma_known, 2)) / (
                    1 / np.power(sigma_prior, 2) + n_new / np.power(sigma_known, 2))
        sigma = np.power(1 / np.power(sigma_prior, 2) + n_new / np.power(sigma_known, 2), -1 / 2)
        return mu, sigma

    @staticmethod
    def test_1sample(mean, x: np.ndarray):
        attr_probabilities = scipy.stats.ttest_1samp(x, mean.T)
        return np.product(attr_probabilities[1])


class MultiNormal:
    @staticmethod
    def pdf(mean: np.ndarray, cov: np.ndarray, x: np.ndarray) -> float:
        if not np.any(np.diag(cov) == 0):
            n = len(mean)
            det = np.linalg.det(cov)
            norm_const = 1 / (np.power(2 * np.pi, n / 2) * np.power(det, 1 / 2))
            z = norm_const * np.exp(-1 / 2 * (x - mean).T @ np.linalg.inv(cov) @ (x - mean))
            return z[0][0]
        return np.nan

    @staticmethod
    def posterior(mean: np.ndarray, cov:np.ndarray, x: np.ndarray, cov_known: np.ndarray) -> (np.ndarray, np.ndarray):
        mean_prior = mean
        cov_prior = cov
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
        return np.diag(mu_post)[:, None], cov_post

    @staticmethod
    def kl_divergence(mean1: np.ndarray, cov1: np.ndarray, mean2: np.ndarray, cov2: np.ndarray) -> float:
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


class NonParametric:
    @staticmethod
    def test_norm_frobenius(x1: np.ndarray, x2: np.ndarray) -> float:
        return -float(np.linalg.norm(x1 - x2))

    @staticmethod
    def test_norm1(x1: np.ndarray, x2: np.ndarray) -> float:
        return -float(np.linalg.norm(x1 - x2, 1))

    @staticmethod
    def test_norm2(x1: np.ndarray, x2: np.ndarray) -> float:
        return -float(np.linalg.norm(x1 - x2, 2))

    @staticmethod
    def test_ranksums(x1: np.ndarray, x2: np.ndarray) -> float:
        test_result = scipy.stats.ranksums(x1, x2)
        return test_result[1]

    @staticmethod
    def test_mannwhitneyu(x1: np.ndarray, x2: np.ndarray) -> float:
        test_result = scipy.stats.mannwhitneyu(x1, x2)
        return test_result[1]


class NormalInverseChiSquared:  # TODO: Test distribution
    @staticmethod
    def posterior(mean: np.ndarray, std: np.ndarray, kappa: int, nu: int, x: np.ndarray):
        n = len(x)
        x_mean = x.mean(axis=0)[:, None]
        _var = np.power(std, 2)

        kappa_p = kappa + n
        mean_p = (kappa*mean + n*x_mean) / kappa_p
        nu_p = nu + n
        std_p = np.sqrt(1/nu_p * (nu*_var + np.sum(np.power(x.T-x_mean, 2), axis=1)[:, None] + (n*kappa)/(kappa+n) * np.power(mean - x_mean, 2)))
        return mean_p, std_p, kappa_p, nu_p

    @staticmethod
    def pdf(mean, std, kappa: int, nu: int, x_mean: np.ndarray, x_std: np.ndarray) -> float:
        _var = np.power(std, 2)
        var = np.power(x_std, 2)
        Z = np.sqrt(2*np.pi)/np.sqrt(kappa) * scipy.special.gamma(nu/2) * np.power(2/(nu * _var), nu/2)
        return 1/Z * np.power(x_std, -1) * np.power(var, -(nu/2+1)) * np.exp(-1/(2*var) * (nu*_var + kappa*np.power(mean - x_mean, 2)))

    @staticmethod
    def pdf_predictive(mean: np.ndarray, std: np.ndarray, kappa: int, nu: int, x) -> float:
        p = scipy.stats.t.pdf(  # TODO: Fix output dimensions
            x=x,
            df=nu,
            loc=mean,
            scale=(1+kappa)*np.power(std, 2) / kappa
        )
        return np.product(p)


class NormalInverseWishart:  # TODO: Test distribution
    @staticmethod
    def posterior(mean: np.ndarray, prec: np.ndarray, kappa: np.ndarray, nu: np.ndarray, x: np.ndarray) \
            -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        Calculates the parameters of the posterior given the new data.
        """
        # Precompute necessary values
        n = len(x)
        x_mean = x.mean(axis=0)[:, None]
        mean_diff = (x_mean - mean)
        S = np.cov(x.T) * (len(x)-1)
        # Compute posterior parameters
        mean_p = (kappa / (kappa + n) * mean + n / (kappa + n) * x_mean)
        kappa_p = kappa + n
        nu_p = nu + n
        prec_p = prec + S + kappa * n / (kappa + n) * mean_diff @ mean_diff.T
        # Return new parameters
        return mean_p, prec_p, kappa_p, nu_p

    @staticmethod
    def pdf(mean: np.ndarray, prec: np.ndarray, kappa: np.ndarray, nu: np.ndarray, x_mean: np.ndarray, x_cov: np.ndarray) -> float:
        """
        Computes the pdf value for provided mean and cov.
        """
        d = len(x_mean)
        lam_det = np.linalg.det(prec)
        cov_det = np.linalg.det(x_cov)
        cov_inv = np.linalg.inv(x_cov)
        mean_diff = (x_mean - mean)
        Z = (np.power(2, nu * d / 2) * scipy.special.gamma(nu/2) * np.power(2*np.pi/kappa, d/2)) / np.power(lam_det, nu/2)
        return 1/Z * np.power(cov_det, -((nu+d)/2+1)) * np.exp(-1 / 2 * np.trace(prec @ cov_inv) - kappa / 2 * mean_diff.T @ cov_inv @ mean_diff)

    @staticmethod
    def pdf_predictive(mean:np.ndarray, prec: np.ndarray, kappa: np.ndarray, nu: np.ndarray, x: np.ndarray) -> float:
        """
        Computes the pdf value for a new sample.
        """
        d = len(mean)
        return scipy.stats.t.pdf(
            x=x,
            df=nu-d+1,
            loc=mean,
            scale=prec * (kappa+1) / (kappa * (nu-d+1))
        )


if __name__ == '__main__':
    pass
