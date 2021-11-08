import unittest
import numpy as np
import scipy.stats

from functions import mathematical


class NormalUniMu_TestCase(unittest.TestCase):
    def test_single(self):
        values = np.array([2, 3, 4])
        result_true = np.mean(values)
        result = mathematical.normal_uni_mu(values)
        self.assertEqual(result, result_true)

    def test_1d_column(self):
        values = np.array([2, 3, 4])[:, None]
        result_true = np.mean(values)
        result = mathematical.normal_uni_mu(values)
        self.assertEqual(result[0], result_true)

    def test_1d_row(self):
        values = np.array([[2, 3, 4]])
        result = mathematical.normal_uni_mu(values)
        self.assertTrue(np.all(result == values[0]))

    def test_2d(self):
        values = np.array([
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6]
        ])
        result_true = np.mean(values, axis=0)
        result = mathematical.normal_uni_mu(values)
        self.assertTrue(np.all(result == result_true))

    def test_3d(self):
        values = np.ones((3, 3, 3))
        raised_error = False
        try:
            result = mathematical.normal_uni_mu(values)
        except ValueError:
            raised_error = True
        self.assertTrue(raised_error)


class NormalUniSigma_TestCase(unittest.TestCase):
    def test_single(self):
        values = np.array([2, 3, 4])
        result_true = np.std(values)
        result = mathematical.normal_uni_sigma(values)
        self.assertEqual(result, result_true)

    def test_1d_column(self):
        values = np.array([2, 3, 4])[:, None]
        result_true = np.std(values)
        result = mathematical.normal_uni_sigma(values)
        self.assertEqual(result[0], result_true)

    def test_1d_row(self):
        values = np.array([[2, 3, 4]])
        result = mathematical.normal_uni_sigma(values)
        self.assertTrue(np.all(result == [0, 0, 0]))

    def test_2d(self):
        values = np.array([
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6]
        ])
        result_true = np.std(values, axis=0)
        result = mathematical.normal_uni_sigma(values)
        self.assertTrue(np.all(result == result_true))

    def test_3d(self):
        values = np.ones((3, 3, 3))
        raised_error = False
        try:
            result = mathematical.normal_uni_sigma(values)
        except ValueError:
            raised_error = True
        self.assertTrue(raised_error)


class NormalUniPdf_TestCase(unittest.TestCase):
    def test_1d(self):
        mean = np.array([-1, 2, 3])
        std = np.array([1, 1, 1])
        x = np.array([2, 2, 2])

        result = mathematical.normal_uni_pdf(mean, std, x)
        result_true = scipy.stats.norm.pdf(x, loc=mean, scale=std)
        self.assertTrue(np.all(result == result_true))

    def test_2d(self):
        mean = np.array([-1, 2, 3])
        std = np.array([1, 1, 1])
        x = np.array([[2, 2, 2], [3, 3, 3]])

        result = mathematical.normal_uni_pdf(mean, std, x)
        result_true = scipy.stats.norm.pdf(x, loc=mean, scale=std)
        self.assertTrue(np.all(result == result_true))

    def test_zero_std(self):
        mean = np.array([-1, 2, 3])
        std = np.array([1, 0, 1])
        x = np.array([[2, 2, 2], [3, 3, 3]])

        result = mathematical.normal_uni_pdf(mean, std, x)
        std[1] = 1
        result_true = scipy.stats.norm.pdf(x, loc=mean, scale=std)
        self.assertTrue(np.all(np.isnan(result[:, 1])))
        self.assertTrue(np.all(result[:, 0] == result_true[:, 0]))
        self.assertTrue(np.all(result[:, 2] == result_true[:, 2]))


class NormalPosteriorSigmaKnown_TestCase(unittest.TestCase):  # TODO: Add tests for invalid inputs
    def test_posterior(self):
        mu_prior = np.array([0, 3])
        sigma_prior = np.array([2, 2])
        sigma_known = np.array([1.5, 1])
        x = np.array([1, 2])
        mu_uni_post, sigma_uni_post = mathematical.normal_uni_posterior_sigmaknown(mu_prior, sigma_prior, sigma_known, x)

        mu_p = mu_prior[:, None]
        cov_p = np.diag(np.power(sigma_prior, 2))
        cov_kwn = np.diag(np.power(sigma_known, 2))
        mu_multi_post, cov_multi_post = mathematical.normal_multi_posterior_sigmaknown(mu_p, cov_p, cov_kwn, x)

        self.assertTrue(np.allclose(mu_uni_post, mu_multi_post, rtol=1e-05, atol=1e-08))
        self.assertTrue(np.allclose(sigma_uni_post, np.sqrt(np.diag(cov_multi_post)), rtol=1e-05, atol=1e-08))


if __name__ == '__main__':
    unittest.main()
