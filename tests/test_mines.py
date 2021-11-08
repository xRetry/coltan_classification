import unittest
from classes.mines import *


class BaselineMineTestCase(unittest.TestCase):
    def test_eval(self):
        sample1 = np.array([
            [0, 2, 2, 3, 0, 4],
            [10, 20, 20, 30, 10, 40]
        ]).T
        sample2 = np.array([
            [3, 6, 3, 2, 2, 3],
            [30, 60, 30, 20, 20, 30]
        ]).T
        test_sample = np.array([
            [4, 3, 3, 5, 6, 5],
            [40, 30, 30, 50, 60, 50]
        ]).T
        mine = BaselineMine(
            x=0,
            y=0,
            z=0,
            status=1,
            samples=[sample1]
        )
        self.assertTrue(np.all(mine.mean == sample1.mean(axis=0)))
        self.assertTrue(np.all(np.round(mine.std, 5) == np.round(sample1.std(axis=0), 5)))

        mine.add_sample(sample2)
        mean2_true = np.row_stack([sample1, sample2]).mean(axis=0)
        std2_true = np.row_stack([sample1, sample2]).std(axis=0)
        self.assertTrue(np.all(mine.mean == mean2_true))
        self.assertTrue(np.all(np.round(mine.std, 5) == np.round(std2_true, 5)))

        attr_probs_true = scipy.stats.ttest_1samp(test_sample, mean2_true)
        mine_eval_true = np.product(attr_probs_true[1])
        mine_eval = mine.eval_samples(np.reshape(test_sample, (1,-1, 2)))[0]
        self.assertEqual(mine_eval_true, mine_eval)


class OrigMineTestCase(unittest.TestCase):
    def test_eval(self):
        sample1 = np.array([
            [0, 2, 2, 3, 0, 4],
            [10, 20, 20, 30, 10, 40]
        ]).T
        sample2 = np.array([
            [3, 6, 3, 2, 2, 3],
            [30, 60, 30, 20, 20, 30]
        ]).T
        test_sample = np.array([
            [4, 3, 3, 5, 6, 5],
            [40, 30, 30, 50, 60, 50]
        ]).T
        mine = OrigMine(
            x=0,
            y=0,
            z=0,
            status=1,
            samples=[sample1]
        )

        self.assertTrue(np.all(mine.mean == sample1.mean(axis=0)))
        self.assertTrue(np.all(np.round(mine.std, 5) == np.round(np.zeros_like(mine.std), 5)))
        self.assertTrue(np.isnan(mine.eval_samples(np.reshape(test_sample, (1, -1, 2)))[0]))

        mine.add_sample(sample2)
        sample_means_true = np.array([sample1.mean(axis=0), sample2.mean(axis=0)])
        mean2_true = np.mean(sample_means_true, axis=0)
        std2_true = mathematical.normal_uni_sigma_orig(sample_means_true)
        self.assertTrue(np.all(mine.mean == mean2_true))
        self.assertTrue(np.all(np.round(mine.std, 5) == np.round(std2_true, 5)))

        attr_probs_true = mathematical.normal_uni_pdf(mean2_true, std2_true, test_sample.mean(axis=0))
        mine_eval_true = np.product(attr_probs_true)
        mine_eval = mine.eval_samples(np.reshape(test_sample, (1, -1, 2)))[0]
        self.assertEqual(mine_eval_true, mine_eval)


if __name__ == '__main__':
    unittest.main()
