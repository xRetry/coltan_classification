import unittest

import numpy as np
from functions.evaluation import transform_none, eval_pdf, eval_ttest
from functions.mathematical import normal_uni_sigma_orig
from classes.mines import BaselineMine, OrigMine
from classes.parameters import Parameters
from classes.dataset import Sample
import scipy.stats


class BaselineMineTestCase(unittest.TestCase):
    def test_eval(self):
        sample1 = Sample(
            sample_id=1, mine_id='1', coordinates=np.zeros(3), label=1,
            attributes=np.array([
                [0, 2, 2, 3, 0, 4],
                [10, 20, 20, 30, 10, 40]
            ]).T
        )
        sample2 = Sample(
            sample_id=1, mine_id='1', coordinates=np.zeros(3), label=1,
            attributes=np.array([
                [3, 6, 3, 2, 2, 3],
                [30, 60, 30, 20, 20, 30]
            ]).T
        )
        test_sample = Sample(
            sample_id=1, mine_id='1', coordinates=np.zeros(3), label=1,
            attributes=np.array([
                [4, 3, 3, 5, 6, 5],
                [40, 30, 30, 50, 60, 50]
            ]).T
        )

        params = Parameters(
                MineClass=None,
                func_transform=transform_none,
                func_eval=eval_pdf,
                func_selection=transform_none,
                func_loss=transform_none,
            )
        mine = BaselineMine(
            coordinates=np.zeros(3),
            status=1,
            samples=[sample1],
            parameters=params
        )
        self.assertTrue(np.all(mine.distribution._mean.T == sample1.attributes.mean(axis=0)))
        self.assertTrue(np.all(np.round(np.diag(mine.distribution._cov), 5) == np.round(np.power(sample1.attributes.std(axis=0), 2), 5)))

        mine.add_sample(sample2)
        mean2_true = np.row_stack([sample1.attributes, sample2.attributes]).mean(axis=0)
        std2_true = np.row_stack([sample1.attributes, sample2.attributes]).std(axis=0)
        self.assertTrue(np.all(mine.distribution._mean.T == mean2_true))
        self.assertTrue(np.all(np.round(np.diag(mine.distribution._cov), 5) == np.round(np.power(std2_true, 2), 5)))

        params.func_eval = eval_ttest
        mine = BaselineMine(
            coordinates=np.zeros(3),
            status=1,
            samples=[sample1, sample2],
            parameters=params
        )
        attr_probs_true = scipy.stats.ttest_1samp(test_sample.attributes, mean2_true)
        mine_eval_true = np.product(attr_probs_true[1])
        mine_eval = mine.eval_sample(test_sample)
        self.assertEqual(mine_eval_true, mine_eval)


class OrigMineTestCase(unittest.TestCase):
    def test_eval(self):
        sample1 = Sample(
            sample_id=1, mine_id='1', coordinates=np.zeros(3), label=1,
            attributes=np.array([
                [0, 2, 2, 3, 0, 4],
                [10, 20, 20, 30, 10, 40]
            ]).T
        )
        sample2 = Sample(
            sample_id=1, mine_id='1', coordinates=np.zeros(3), label=1,
            attributes=np.array([
                [3, 6, 3, 2, 2, 3],
                [30, 60, 30, 20, 20, 30]
            ]).T
        )
        test_sample = Sample(
            sample_id=1, mine_id='1', coordinates=np.zeros(3), label=1,
            attributes=np.array([
                [4, 3, 3, 5, 6, 5],
                [40, 30, 30, 50, 60, 50]
            ]).T
        )

        params = Parameters(
            MineClass=None,
            func_transform=transform_none,
            func_eval=eval_pdf,
            func_selection=transform_none,
            func_loss=transform_none,
        )
        mine = OrigMine(
            coordinates=np.zeros(3),
            status=1,
            samples=[sample1],
            parameters=params
        )

        self.assertTrue(np.all(mine.distribution._mean.T == sample1.attributes.mean(axis=0)))
        self.assertTrue(np.all(np.round(np.diag(mine.distribution._cov), 5) == np.round(np.zeros_like(mine.distribution._mean.T), 5)))
        self.assertTrue(np.isnan(mine.eval_sample(sample1)))

        mine.add_sample(sample2)
        sample_means_true = np.array([sample1.attributes.mean(axis=0), sample2.attributes.mean(axis=0)])
        mean2_true = np.mean(sample_means_true, axis=0)
        std2_true = normal_uni_sigma_orig(sample_means_true)
        self.assertTrue(np.all(mine.distribution._mean.T == mean2_true))
        self.assertTrue(np.all(np.round(np.diag(mine.distribution._cov), 5) == np.round(np.power(std2_true, 2), 5)))

        attr_probs_true = scipy.stats.norm.pdf(test_sample.attributes.mean(axis=0), mean2_true, std2_true)
        mine_eval_true = np.product(attr_probs_true)
        mine_eval = mine.eval_sample(test_sample)
        self.assertAlmostEqual(mine_eval_true, mine_eval, places=7)


if __name__ == '__main__':
    unittest.main()
