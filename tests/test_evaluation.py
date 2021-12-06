import unittest
import numpy as np
from functions import loss
from functions import selection


class LossFunctionTestCase(unittest.TestCase):
    def test_accuracy(self):
        labels = np.array([1, 1, 1, 1])
        predictions = np.array([1, 1, np.nan, 1])
        result = loss.accuracy(labels, predictions)

        self.assertEqual(result, 0.75)

    def test_error(self):
        labels = np.array([1, 1, 1, 1])
        predictions = np.array([1, 1, np.nan, 1])
        result = loss.error(labels, predictions)

        self.assertEqual(result, 0.25)


class LabelSelectionTestCase(unittest.TestCase):
    def test_best_mine(self):
        # Normal function
        labels = np.array(list(range(3)))
        eval_result = np.array([0, 1, 3])
        label_selection = selection.mine(eval_result, labels)
        self.assertEqual(label_selection, 2)
        # NaN value
        eval_result = np.array([0, 1, np.nan])
        label_selection = selection.mine(eval_result, labels)
        self.assertEqual(label_selection, 1)
        # all NaN values
        eval_result = np.array([np.nan, np.nan, np.nan])
        label_selection = selection.mine(eval_result, labels)
        self.assertTrue(np.isnan(label_selection))

    def test_best_label(self):
        # Normal function
        labels = np.array([-1, -1, 1, 1])
        eval_result = np.array([0.35, 0.05, 0.3, 0.3])
        label_selection = selection.label(eval_result, labels)
        self.assertEqual(label_selection, 1)
        # NaN value
        eval_result = np.array([0.35, 0.05, np.nan, 0.3])
        label_selection = selection.label(eval_result, labels)
        self.assertEqual(label_selection, -1)
        # all NaN values
        eval_result = np.array([np.nan, np.nan, np.nan, np.nan])
        label_selection = selection.label(eval_result, labels)
        self.assertTrue(np.isnan(label_selection))


if __name__ == '__main__':
    unittest.main()
