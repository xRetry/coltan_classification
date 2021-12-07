import numpy as np
from classes.dataset import Dataset
from classes.parameters import Parameters
from classes.models import Model
from typing import List


class ModelAnalyser:
    _parameters: Parameters

    def __init__(self, parameters: Parameters):
        self._parameters = parameters

    def cross_validate(self, dataset: Dataset, n_folds: int) -> float or np.ndarray:
        samples = dataset.crossval_samples(n_folds, shuffle=True)
        predictions, labels = [], []
        # Iterate through folds
        for i, (samples_train, samples_test) in enumerate(samples):
            # Creating model and test values
            model = Model(self._parameters, samples_train)
            # Evaluate test values
            predictions_fold, labels_fold = np.zeros(len(samples_test)), np.zeros(len(samples_test))
            for j, sample_test in enumerate(samples_test):
                predictions_fold[j] = model.classify(sample_test)
                labels_fold[j] = sample_test.label
            predictions.append(predictions_fold)
            labels.append(labels_fold)
        return self._compute_loss(labels, predictions)

    def _compute_loss(self, labels:List[np.ndarray], predictions:List[np.ndarray]) -> float or np.ndarray:
        loss = []
        for i in range(len(labels)):
            loss.append(self._parameters.func_loss(labels[i], predictions[i]))
        if isinstance(loss[0], np.ndarray):
            return np.sum(loss, axis=0) / np.sum(loss)
        return np.mean(loss)


if __name__ == '__main__':
    pass
