import unittest
import numpy as np
import scipy.stats
from classes.distributions import MultiNormal
from functions.mathematical import normal_uni_posterior_sigmaknown


class MultiNormalTestCase(unittest.TestCase):
    def test_pdf(self):
        means = np.array([2, 3, 4])
        stds = np.array([0.5, 1, 1.5])
        cov = np.diag(stds**2)
        x = np.array([1, 2, 3])
        p_true = scipy.stats.multivariate_normal.pdf(x, means, cov)
        p = MultiNormal(means, cov).pdf(x)
        self.assertAlmostEqual(p_true, p, delta=1e-7)

    def test_posterior(self):
        mu_prior = np.array([0, 3])
        sigma_prior = np.array([2, 2])
        sigma_known = np.array([1.5, 1])
        x = np.array([[1, 2], [1.2, 2.1]])
        mu_uni_post, sigma_uni_post = normal_uni_posterior_sigmaknown(mu_prior, sigma_prior, sigma_known, x)

        mu_p = mu_prior[:, None]
        cov_p = np.diag(np.power(sigma_prior, 2))
        cov_kwn = np.diag(np.power(sigma_known, 2))
        posterior = MultiNormal(mu_p, cov_p).posterior(x, cov_kwn)

        self.assertTrue(np.allclose(mu_uni_post, posterior._mean.flatten(), rtol=1e-05, atol=1e-08))
        self.assertTrue(np.allclose(sigma_uni_post, np.sqrt(np.diag(posterior._cov)), rtol=1e-05, atol=1e-08))


if __name__ == '__main__':
    unittest.main()
