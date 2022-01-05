import numpy as np
import itertools
from classes.dataset import Dataset
from classes.parameters import Parameters
from functions import plotting
from typing import List, Callable, Optional, Tuple
from multiprocessing import Pool
import statsmodels.api as sm


class ModelAnalyser:
    @staticmethod
    def cross_validate(parameters: Parameters or List[Parameters], dataset: Dataset, pct_test: float, return_confint: bool=False) -> list:
        """
        Cross-validates models with provided model parameters.
        """
        # Wrapping parameters in list if not
        if not isinstance(parameters, list):
            parameters = [parameters]
        # Creating sample generator
        sample_gen = dataset.cv_generator(pct_test, shuffle=True, verbose=True)
        # Setting up lists for results
        predictions, labels = [[] for p in parameters], [[] for p in parameters]
        # Evaluating all folds and parameters
        with Pool(processes=4) as pool:
            eval_map = pool.imap_unordered(
                ModelAnalyser._evaluate_model,
                itertools.product(zip(parameters, range(len(parameters))), sample_gen)
            )
            for eval_result in eval_map:
                labels[eval_result[2]].append(eval_result[0])
                predictions[eval_result[2]].append(eval_result[1])
        # Iterating through predictions
        losses = []
        for i in range(len(predictions)):
            if return_confint:
                # Computing confident interval of accuracy
                labels_cat = np.concatenate(labels[i])
                n_correct = (labels_cat == np.concatenate(predictions[i])).sum()
                confint = sm.stats.proportion_confint(n_correct, len(labels_cat), alpha=0.05, method='beta')
                losses.append(confint)
            else:
                # Computing loss from true labels and predictions
                loss = ModelAnalyser._compute_loss(parameters[i].func_loss, labels[i], predictions[i])
                losses.append(loss)
        return losses

    @staticmethod
    def cross_validate_stepwise(parameters: Parameters or List[Parameters], dataset: Dataset, n_splits: int, model_names: Optional[Iterable[str]]=None) -> None:
        """
        Shows model loss for different amount of training samples.
        """
        # Setting up data splits
        pct_test = np.linspace(0.1, 0.9, n_splits)
        # Iterating through data splits
        losses, conf_ints = [], []
        for i, n in enumerate(pct_test):
            conf_int = ModelAnalyser.cross_validate(parameters, dataset, n, return_confint=True)
            conf_ints.append(conf_int)
            print()
        # Creating model names for plotting
        if model_names is None:
            model_names = ['{}-{}-{}'.format(p.ModelClass.__name__, p.MineClass.__name__, p.func_eval.__name__) for p in parameters]  # TODO: dynamically change model names
        # Plotting results
        plotting.plot_cv_stepwise(pct_test, np.array(conf_ints), model_names=model_names)

    @staticmethod
    def _evaluate_model(inputs: Tuple[Tuple[Parameters, int], Tuple[np.ndarray, np.ndarray]]) -> (np.ndarray, np.ndarray, int):
        """
        Wrapper function for model evaluation to be used in maps.
        """
        # Unwrapping inputs
        params = inputs[0][0]
        mine_idx = inputs[0][1]
        samples_train = inputs[1][0]
        samples_test = inputs[1][1]
        # Creating model and test values
        model = params.ModelClass(params, samples_train)
        # Iterating and evaluating all test values
        predictions_fold, labels_fold = np.zeros(len(samples_test)), np.zeros(len(samples_test))
        for k, sample_test in enumerate(samples_test):
            predictions_fold[k] = model.classify(sample_test)
            labels_fold[k] = sample_test.label
        return labels_fold, predictions_fold, mine_idx

    @staticmethod
    def _compute_loss(func_loss: Callable, labels:List[np.ndarray], predictions:List[np.ndarray]) -> float or np.ndarray:
        """
        Helper function to compute the mean loss for all folds.
        """
        # Iterating through all folds
        loss = []
        for i in range(len(labels)):
            loss.append(func_loss(labels[i], predictions[i]))
        # Handle case where losses are matrices
        if isinstance(loss[0], np.ndarray):
            return np.sum(loss, axis=0) / np.sum(loss)
        # Compute and return mean loss otherwise
        return np.mean(loss)


if __name__ == '__main__':
    pass
