import unittest

import numpy as np
from classes.evaluation import Normalization, Transformation
from functions.mathematical import normal_uni_std
from classes.mines import AggregationUniMine, OrigMine
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
                func_normalize=Normalization.none,
                func_transform=Transformation.none,
                func_eval=AggregationUniMine.eval_pdf,
                func_selection=Transformation.none,
                func_loss=Transformation.none,
            )
        mine = AggregationUniMine(
            coordinates=np.zeros(3),
            status=1,
            parameters=params
        )
        mine.add_sample(sample1)
        self.assertTrue(np.all(mine._to_normal(mine.distribution.parameters['values'])._mean.T == sample1.attributes.mean(axis=0)))
        self.assertTrue(np.all(np.round(np.diag(mine._to_normal(mine.distribution.parameters['values'])._cov), 5) == np.round(np.diag(np.cov(sample1.attributes.T)), 5)))

        mine.add_sample(sample2)
        mean2_true = np.row_stack([sample1.attributes, sample2.attributes]).mean(axis=0)
        var2_true = np.diag(np.cov(np.row_stack([sample1.attributes, sample2.attributes]).T))
        self.assertTrue(np.all(mine._to_normal(mine.distribution.parameters['values'])._mean.T == mean2_true))
        self.assertTrue(np.all(np.round(np.diag(mine._to_normal(mine.distribution.parameters['values'])._cov), 5) == np.round(var2_true, 5)))

        params.func_eval = AggregationUniMine.eval_ttest
        mine = AggregationUniMine(
            coordinates=np.zeros(3),
            status=1,
            parameters=params
        )
        mine.add_sample(sample1)
        mine.add_sample(sample2)

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
            func_normalize=Normalization.none,
            func_transform=Transformation.none,
            func_eval=OrigMine.eval_pdf,
            func_selection=Transformation.none,
            func_loss=Transformation.none,
        )
        mine = OrigMine(
            coordinates=np.zeros(3),
            status=1,
            parameters=params
        )
        mine.add_sample(sample1)

        self.assertTrue(np.all(mine._to_normal(mine.distribution.parameters['values'])._mean.T == sample1.attributes.mean(axis=0)))
        self.assertTrue(np.all(np.round(np.diag(mine._to_normal(mine.distribution.parameters['values'])._cov), 5) == np.round(np.zeros_like(len(mine.distribution._samples[0])), 5)))
        self.assertTrue(np.isnan(mine.eval_sample(sample1)))

        mine.add_sample(sample2)
        sample_means_true = np.array([sample1.attributes.mean(axis=0), sample2.attributes.mean(axis=0)])
        mean2_true = np.mean(sample_means_true, axis=0)
        std2_true = normal_uni_std(sample_means_true, corrected=False)
        self.assertTrue(np.all(mine._to_normal(mine.distribution.parameters['values'])._mean.T == mean2_true))
        self.assertTrue(np.all(np.round(np.diag(mine._to_normal(mine.distribution.parameters['values'])._cov), 5) == np.round(np.power(std2_true, 2), 5)))

        attr_probs_true = scipy.stats.norm.pdf(test_sample.attributes.mean(axis=0), mean2_true, std2_true)
        mine_eval_true = np.product(attr_probs_true)
        mine_eval = mine.eval_sample(test_sample)
        self.assertAlmostEqual(mine_eval_true, mine_eval, places=7)


if __name__ == '__main__':
    unittest.main()
