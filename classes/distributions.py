import numpy as np
import scipy.stats
import abc


class Distribution(abc.ABC):

    @abc.abstractmethod
    def pdf(self, x: np.ndarray) -> float:
        pass

    def posterior(self, x: np.ndarray, cov_known: np.ndarray):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass


class MultiNormal(Distribution):
    _mean: np.ndarray
    _cov: np.ndarray

    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        if len(mean.shape) == 1:
            mean = mean[:, None]
        if mean.shape[0] < mean.shape[1]:
            mean = mean.T
        if cov.shape[0] != cov.shape[1]:
            raise TypeError('Covariance matrix is not square!')
        self._mean = mean
        self._cov = cov

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

    def ttest_1sample(self, x: np.ndarray):
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

    def __len__(self):
        return len(self._mean)


if __name__ == '__main__':
    pass
