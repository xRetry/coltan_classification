from typing import List, Callable, Optional, Tuple, Iterable, Dict, Sequence
import itertools
from multiprocessing import Pool

import numpy as np
import statsmodels.api as sm

from core.dataset import Dataset, Sample
from analysis.datastructs import Parameters, ProgressBarData, CrossValResult, ModelEvaluationParameters, ModelEvaluationResult
from analysis import logging, plotting


def print_progressbar(progress_data: ProgressBarData, postfix: str='', is_end: bool=True) -> None:
    """
    Prints progress bars from items of dictionary. Format: {name: (current progress, full length)}.
    """
    # Iterating through progress bars and building console output
    output = ''
    for k, v in progress_data.bars.items():
        # Determine amount of current and completed elements
        n_load = int((v[0] + 1) / v[1] * 10)
        n_full = int((v[0]) / v[1] * 10) if not is_end else n_load
        # Creating line string
        line = '=' * n_full + '-' * (n_load - n_full) + ' ' * (10 - n_load)
        # Adding line string to console output
        output += '{}: |{}| {}/{}\t\t'.format(k, line, v[0] + 1 if is_end else v[0], v[1])
    # Printing output to console
    print('\r{}\t\t{}'.format(output, postfix), end='')


class ModelAnalyser:
    @staticmethod
    def cross_validate(parameters: Parameters or List[Parameters], dataset: Dataset, pct_test: float,
                       progress_data: Optional[ProgressBarData]=None) -> CrossValResult:
        """
        Cross-validates models with provided model parameters.
        """
        # Wrapping parameters in list if not
        if not isinstance(parameters, list or tuple):
            parameters = [parameters]
        # Initializing dict for printing function
        if progress_data is None:
            progress_data = ProgressBarData()
        # Creating sample generator
        sample_gen, n_folds = dataset.cv_generator(pct_test, shuffle=True)
        # Calculating the total amount of iterations
        n_iter = n_folds * len(parameters)
        # Setting up lists for results
        cv_result = CrossValResult(parameters)
        # Evaluating all folds and parameters
        with Pool(processes=4) as pool:
            eval_map = pool.imap_unordered(
            #eval_map = map(
                ModelAnalyser._evaluate_model,
                map(ModelAnalyser._map_to_eval_parameters, itertools.product(zip(parameters, range(len(parameters))), sample_gen))
            )
            for i in range(n_iter):
                # Printing progress bar
                progress_data.add_bar('Cross-Validation', i, n_iter)
                print_progressbar(progress_data, postfix=f'Warnings: {cv_result.n_warnings}', is_end=False)
                # Evaluating and storing result
                eval_result = next(eval_map)
                cv_result.add_result(eval_result)
                # Printing progress bar
                progress_data.add_bar('Cross-Validation', i, n_iter)
                print_progressbar(progress_data, postfix=f'Warnings: {cv_result.n_warnings}')
        # Iterating through predictions
        for i, (labels, predictions) in enumerate(zip(cv_result.labels, cv_result.predictions)):
            # Computing confident interval of accuracy
            labels_cat = np.concatenate(labels)
            n_correct = (labels_cat == np.concatenate(predictions)).sum()
            confint = sm.stats.proportion_confint(n_correct, len(labels_cat), alpha=0.05, method='beta')
            cv_result.conf_ints.append(confint)

            # Computing loss from true labels and predictions
            loss = ModelAnalyser._compute_loss(parameters[i].func_loss, labels, predictions)
            cv_result.losses.append(loss)
        return cv_result

    @staticmethod
    def cross_validate_stepwise(parameters: Parameters or List[Parameters], dataset: Dataset, n_splits: int, model_names: Optional[Iterable[str]]=None) -> None:
        """
        Shows model loss for different amount of training samples.
        """
        # Setting up data splits
        pct_test = np.linspace(0.1, 0.9, n_splits)
        # Setting up progress bar data
        progress_data = ProgressBarData()
        # Iterating through data splits
        cv_results = []
        for i, n in enumerate(pct_test):
            progress_data.add_bar('Step', i, len(pct_test))
            cv_results.append(ModelAnalyser.cross_validate(parameters, dataset, n, progress_data=progress_data))
        # Plotting results
        plotting.plot_cv_stepwise(pct_test, np.array([c.conf_ints for c in cv_results]), model_names=model_names)

    @staticmethod
    def mine_distances(parameters: Parameters):
        dataset = Dataset()
        samples_train, samples_test = dataset.train_test_split(0.00001)
        model = parameters.ModelClass(parameters, samples_train)
        mine_labels = np.array([m._label for m in model._mines.values()])
        prediction, score = model.classify(samples_test[0])
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
            if not isinstance(v, list or tuple):
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
    def _map_to_eval_parameters(inputs: Tuple[Tuple[Parameters, int], Tuple[np.ndarray, np.ndarray]]):
        """ Function to map the generator of evaluation inputs to a class """
        return ModelEvaluationParameters(
            parameters=inputs[0][0],
            eval_idx=inputs[0][1],
            samples_train=inputs[1][0],
            samples_test=inputs[1][1]
        )

    @staticmethod
    def _evaluate_model(inputs: ModelEvaluationParameters) -> ModelEvaluationResult:
        """ Wrapper function for model evaluation to be used in maps. """
        with logging.Logger(ModelEvaluationResult(inputs)) as eval_result:
            # Creating model and test values
            model = inputs.parameters.ModelClass(inputs.parameters, inputs.samples_train)
            # Iterating and evaluating all test values
            for k, sample_test in enumerate(inputs.samples_test):
                eval_result.predictions[k] = model.classify(sample_test)
        return eval_result

    @staticmethod
    def _compute_loss(func_loss: Callable, labels: Sequence[np.ndarray], predictions:Sequence[np.ndarray]) -> float or np.ndarray:
        """ Helper function to compute the mean loss for all folds. """
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
