import numpy as np
import itertools
from classes.dataset import Dataset
from classes.parameters import Parameters
from functions import plotting, utils
from typing import List, Callable, Optional, Tuple, Iterable, Dict
from multiprocessing import Pool
import statsmodels.api as sm


class ModelAnalyser:
    @staticmethod
    def cross_validate(parameters: Parameters or List[Parameters], dataset: Dataset, pct_test: float,
                       return_confint: bool=False, progress_dict: Optional[Dict[str, Tuple[int, int]]]=None) -> list:
        """
        Cross-validates models with provided model parameters.
        """
        # Wrapping parameters in list if not
        if not isinstance(parameters, list):
            parameters = [parameters]
        # Initializing dict for printing function
        if progress_dict is None:
            progress_dict = {}
        # Creating sample generator
        sample_gen, n_folds = dataset.cv_generator(pct_test, shuffle=True)
        # Calculating the total amount of iterations
        n_iter = n_folds * len(parameters)
        # Setting up lists for results
        predictions, labels = [[] for p in parameters], [[] for p in parameters]
        # Evaluating all folds and parameters
        with Pool(processes=4) as pool:
            eval_map = pool.imap_unordered(
            #eval_map = map(
                ModelAnalyser._evaluate_model,
                itertools.product(zip(parameters, range(len(parameters))), sample_gen)
            )
            for i in range(n_iter):
                # Printing progress bar
                progress_dict['Cross-Validation'] = (i, n_iter)
                utils.print_progressbar(progress_dict, is_end=False)
                # Evaluating and storing result
                eval_result = next(eval_map)
                labels[eval_result[2]].append(eval_result[0])
                predictions[eval_result[2]].append(eval_result[1])
                # Printing progress bar
                progress_dict['Cross-Validation'] = (i, n_iter)
                utils.print_progressbar(progress_dict)
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
        # Setting up printing dictionary
        progress_dict = {}
        # Iterating through data splits
        losses, conf_ints = [], []
        for i, n in enumerate(pct_test):
            progress_dict['Step'] = (i, len(pct_test))
            conf_int = ModelAnalyser.cross_validate(parameters, dataset, n, return_confint=True, progress_dict=progress_dict)
            conf_ints.append(conf_int)
        # Creating model names for plotting
        if model_names is None:
            model_names = ['{}-{}-{}'.format(p.ModelClass.__name__, p.MineClass.__name__, p.func_eval.__name__) for p in parameters]  # TODO: dynamically change model names
        # Plotting results
        plotting.plot_cv_stepwise(pct_test, np.array(conf_ints), model_names=model_names)

    @staticmethod
    def mine_distances(parameters: Parameters):
        dataset = Dataset()
        samples_train, samples_test = dataset.train_test_split(0.00001)
        model = parameters.ModelClass(parameters, samples_train)
        mine_labels = np.array([m._label for m in model._mines.values()])
        prediction, score = model.classify(samples_test[0], return_scores=True)
        score_pos = score[mine_labels == 1]
        score_neg = score[mine_labels == -1]
        print('Average Score\n\t1: {}\n\t-1: {}\n -> Label: {} (true: {})'.format(
            score_pos.mean(),
            score_neg.mean(),
            1 if score_pos.mean() > score_neg.mean() else -1,
            samples_test[0].proportional_score
        ))
        print('Max Score\n\t1: {}\n\t-1: {}\n -> Label: {} (true: {})'.format(
            score_pos.max(),
            score_neg.max(),
            1 if score_pos.max() > score_neg.max() else -1,
            samples_test[0].proportional_score
        ))

    @staticmethod
    def params_generator(all_combinations=False, **kwargs):
        """
        Generates a list of Parameters from multiple parameters in form of a list.
        """
        # Wrap values in list if not already
        for k, v in kwargs.items():
            if not isinstance(v, Iterable):
                kwargs[k] = [v]
        # Fills up values to reach the length of the longest values
        if not all_combinations:
            # Find the longest value list
            len_max = max([len(v) for v in kwargs.values()])
            # Iterating through max length of values
            params = []
            for i in range(len_max):
                # Creating kwargs dict for Parameters creation
                param_dict = {}
                for k, v in kwargs.items():
                    if len(v) > 1:
                        if len(v) != len_max:
                            raise ValueError('Invalid argument length!')
                        param_dict[k] = v[i]
                    else:
                        param_dict[k] = v[0]
                # Create Parameters and add to output
                params.append(Parameters(**param_dict))
        # Creates the Cartesian product of all values
        else:
            # Creating kwargs tuples with of (arg_name, arg_value)
            tuples = []
            for k, val in kwargs.items():
                tuples.append([])
                for v in val:
                    tuples[-1].append((k, v))
            # Cartesian product of kwargs tuples
            combinations = list(itertools.product(*tuples))
            # Creating Parameters from kwargs tuples
            params = [Parameters(**dict(c)) for c in combinations]
        return params

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
