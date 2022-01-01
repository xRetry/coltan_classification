import numpy as np
from classes.dataset import Dataset
from classes.parameters import Parameters
from classes.models import KernelModel
from functions import plotting
from typing import List, Callable, Optional
import statsmodels.api as sm


class ModelAnalyser:
    @staticmethod
    def cross_validate(parameters: Parameters or List[Parameters], dataset: Dataset, pct_folds: float, return_confint: bool=False) -> list:
        if not isinstance(parameters, list):
            parameters = [parameters]
        sample_gen = dataset.cv_generator(pct_folds, shuffle=True, verbose=True)
        # Iterate through folds
        predictions, labels = [[] for p in parameters], [[] for p in parameters]
        for i, (samples_train, samples_test) in enumerate(sample_gen):
            # Iterate through model parameters
            for j, params in enumerate(parameters):
                # Creating model and test values
                model = params.ModelClass(params, samples_train)
                # Iterating and evaluating all test values
                predictions_fold, labels_fold = np.zeros(len(samples_test)), np.zeros(len(samples_test))
                for k, sample_test in enumerate(samples_test):
                    predictions_fold[k] = model.classify(sample_test)
                    labels_fold[k] = sample_test.label
                predictions[j].append(predictions_fold)
                labels[j].append(labels_fold)

        result = []
        for i in range(len(predictions)):
            if return_confint:
                # Computing confident interval of accuracy
                labels_cat = np.concatenate(labels[i])
                n_correct = (labels_cat == np.concatenate(predictions[i])).sum()
                confint = sm.stats.proportion_confint(n_correct, len(labels_cat), alpha=0.05, method='beta')
                result.append(confint)
            else:
                # Computing loss from true labels and predictions
                loss = ModelAnalyser._compute_loss(parameters[i].func_loss, labels[i], predictions[i])
                result.append(loss)
        return result

    @staticmethod
    def cross_validate_stepwise(parameters: Parameters or List[Parameters], dataset:Dataset, n_splits: int) -> None:
        pct_folds = np.linspace(0.1, 0.9, n_splits)
        losses, conf_ints = [], []
        for i, n in enumerate(pct_folds):
            conf_int = ModelAnalyser.cross_validate(parameters, dataset, n, return_confint=True)
            conf_ints.append(conf_int)
            print()
        model_names = ['{}-{}-{}'.format(p.ModelClass.__name__, p.MineClass.__name__, p.func_eval.__name__) for p in parameters]  # TODO: dynamically change model names
        plotting.plot_cv_stepwise(pct_folds, np.array(conf_ints), model_names=model_names)

    @staticmethod
    def _compute_loss(func_loss: Callable, labels:List[np.ndarray], predictions:List[np.ndarray]) -> float or np.ndarray:
        loss = []
        for i in range(len(labels)):
            loss.append(func_loss(labels[i], predictions[i]))
        if isinstance(loss[0], np.ndarray):
            return np.sum(loss, axis=0) / np.sum(loss)
        return np.mean(loss)


if __name__ == '__main__':
    pass
