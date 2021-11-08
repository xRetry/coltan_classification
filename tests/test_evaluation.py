import unittest

from functions import evaluation
from functions.evaluation import *
from classes.mines import BaselineMine


class LossFunctionTestCase(unittest.TestCase):
    def test_accuracy(self):
        labels = np.array([1, 1, 1, 1])
        predictions = np.array([1, 1, np.nan, 1])
        result = evaluation.accuracy(labels, predictions)

        self.assertEqual(result, 0.75)

    def test_error(self):
        labels = np.array([1, 1, 1, 1])
        predictions = np.array([1, 1, np.nan, 1])
        result = evaluation.error(labels, predictions)

        self.assertEqual(result, 0.25)


class LabelSelectionTestCase(unittest.TestCase):
    def test_best_mine(self):
        mines = [BaselineMine(0, 0, 0, i) for i in range(3)]
        eval_results = [0, 1, np.nan]
        label_selection = evaluation.best_mine(eval_results, mines)
        self.assertEqual(label_selection[0], 1)

        eval_results = [
            [0, 1, np.nan],
            [1, 2, 3],
        ]
        label_selection = evaluation.best_mine(eval_results, mines)
        self.assertEqual(label_selection[0], 1)


if __name__ == '__main__':
    unittest.main()
