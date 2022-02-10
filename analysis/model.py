from typing import List, Callable, Optional, Tuple, Iterable, Sequence
import itertools
from multiprocessing import Pool
from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm

from core.dataset import Dataset, Sample
from core.models import ModelParameters, Model
from core.mines import MineParameters, BayesianUniMine, AggregationMine
import analysis.plotting.model as plot
from analysis.utils import console, logging


###############
# DATACLASSES #
###############


@dataclass
class ModelEvaluationParameters:
    ModelClass: type(Model)
    eval_idx: int
    model_params: ModelParameters
    samples_train: Sequence[Sample]
    samples_test: Sequence[Sample]


@dataclass
class ModelEvaluationResult(logging.Log):
    eval_idx: int
    samples_train: Sequence[Sample]
    samples_test: Sequence[Sample]
    predictions: np.ndarray
    warnings: list
    elapsed_time: float

    def __init__(self, model_eval_params: ModelEvaluationParameters):
        self.eval_idx = model_eval_params.eval_idx
        self.samples_train = model_eval_params.samples_train
        self.samples_test = model_eval_params.samples_test
        self.predictions = np.zeros(len(self.samples_test))
        self.warnings = []

    @property
    def labels(self):
        return np.array([sample.label for sample in self.samples_test])


@dataclass
class CrossValParameters:
    ModelClass: List[type(Model)]
    model_params: List[ModelParameters]
    dataset: Dataset
    pct_test: float
    func_loss: Callable


@dataclass
class CrossValResult:
    cv_params: CrossValParameters
    eval_params: List[ModelEvaluationParameters]
    eval_results: List[List[ModelEvaluationResult]]
    n_warnings: int

    def __init__(self, cv_parameters: CrossValParameters):
        self.cv_params = cv_parameters
        self.eval_results = [[] for p in cv_parameters.model_params]
        self.n_warnings = 0

    def add_result(self, eval_result: ModelEvaluationResult):
        self.n_warnings += len(eval_result.warnings)
        self.eval_results[eval_result.eval_idx].append(eval_result)

    @property
    def predictions(self) -> List[List[np.ndarray]]:
        preds = []
        for res_mdl in self.eval_results:
            preds.append([res_fld.predictions for res_fld in res_mdl])
        return preds

    @property
    def labels(self) -> List[List[np.ndarray]]:
        lbls = []
        for res_mdl in self.eval_results:
            lbls.append([res_fld.labels for res_fld in res_mdl])
        return lbls

    @property
    def conf_ints(self) -> np.ndarray:
        intervals = []
        for i, (labels, predictions) in enumerate(zip(self.labels, self.predictions)):
            # Computing confident interval of accuracy
            labels_cat = np.concatenate(labels)
            n_correct = (labels_cat == np.concatenate(predictions)).sum()
            intervals.append(sm.stats.proportion_confint(n_correct, len(labels_cat), alpha=0.05, method='beta'))
        return np.array(intervals)

    @property
    def losses(self) -> np.ndarray:
        losses = []
        for i, (labels, predictions) in enumerate(zip(self.labels, self.predictions)):
            # Computing loss from true labels and predictions
            losses.append(ModelAnalyser.compute_loss(self.cv_params.func_loss, labels, predictions))
        return np.array(losses)


######################
# ANALYSIS FUNCTIONS #
######################


class ModelAnalyser:
    @staticmethod
    def cross_validate(cv_params: CrossValParameters, progress_bar: Optional[console.ProgressBar]=None) -> CrossValResult:
        """
        Cross-validates models with provided model parameters.
        """
        # Wrapping parameters in list if not
        if not isinstance(cv_params.model_params, list or tuple):
            cv_params.model_params = [cv_params.model_params]
        # Initializing dict for printing function
        if progress_bar is None:
            progress_bar = console.ProgressBar()
        # Creating sample generator
        sample_gen, n_folds = cv_params.dataset.cv_generator(cv_params.pct_test, shuffle=True)
        # Calculating the total amount of iterations
        n_iter = n_folds * len(cv_params.model_params)
        # Setting up lists for results
        cv_result = CrossValResult(cv_params)
        # Evaluating all folds and parameters
        with Pool(processes=6) as pool:
            eval_map = pool.imap_unordered(
            #eval_map = map(
                ModelAnalyser._evaluate_model,
                map(ModelAnalyser._map_to_eval_parameters, itertools.product(zip(cv_params.ModelClass, cv_params.model_params, range(len(cv_params.model_params))), sample_gen))
            )
            for i in range(n_iter):
                # Printing progress bar
                progress_bar.add_bar('Cross-Validation', i, n_iter)
                progress_bar.print(postfix=f'Warnings: {cv_result.n_warnings}', is_end=False)
                # Evaluating and storing result
                eval_result = next(eval_map)
                cv_result.add_result(eval_result)
                # Printing progress bar
                progress_bar.add_bar('Cross-Validation', i, n_iter)
                progress_bar.print(postfix=f'Warnings: {cv_result.n_warnings}')
        return cv_result

    @staticmethod
    def cross_validate_stepwise(cv_params: CrossValParameters, n_splits: int, model_names: Optional[Iterable[str]]=None) -> None:
        """
        Shows model loss for different amount of training samples.
        """
        # Setting up data splits
        pct_test = np.linspace(0.1, 0.9, n_splits)
        # Setting up progress bar data
        progress_bar = console.ProgressBar()
        # Iterating through data splits
        cv_results = []
        for i, n in enumerate(pct_test):
            progress_bar.add_bar('Step', i, len(pct_test))
            cv_params.pct_test = n
            cv_results.append(ModelAnalyser.cross_validate(cv_params, progress_bar=progress_bar))
        console.print_cv_summary(cv_results)
        # Plotting results
        plot.plot_cv(pct_test, np.array([c.conf_ints for c in cv_results]), model_names=model_names)
        for i, cv_result in enumerate(cv_results):
            plot.plot_cv_grid(cv_result, model_names, f'Test Proportion: {str(pct_test[i])}')

    @staticmethod
    def mine_distances(ModelClass: type(Model), model_params: ModelParameters):
        dataset = Dataset()
        samples_train, samples_test = dataset.train_test_split(0.00001)
        model = ModelClass(model_params, samples_train)
        mine_labels = np.array([m._label for m in model._mines.values()])
        model_result = model.classify(samples_test[0], return_summary=True)
        score_pos = model_result.score[mine_labels == 1]
        score_neg = model_result.score[mine_labels == -1]
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
    def params_generator(all_combinations=False, **kwargs) -> (List[type(Model)], List[ModelParameters]):
        """
        Generates a list of Parameters from multiple parameters in form of a list.
        """
        # Wrap values in list if not already
        for k, v in kwargs.items():
            if not isinstance(v, list or tuple):
                kwargs[k] = [v]
        param_dicts = []
        # Fills up values to reach the length of the longest values
        if not all_combinations:
            # Find the longest value list
            len_max = max([len(v) for v in kwargs.values()])
            # Iterating through max length of values
            model_classes, params = [], []
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
                param_dicts.append(param_dict)
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
            # Creating dicts from kwargs tuples
            for c in combinations:
                param_dicts.append(dict(c))

        model_classes, params = [], []
        for param_dict in param_dicts:
            # Create Parameters and add to output
            mine_kwargs = param_dict.get('mine_kwargs')
            if mine_kwargs is None:
                mine_kwargs = dict()
            eval_kwargs = param_dict.get('eval_kwargs')
            if eval_kwargs is None:
                eval_kwargs = dict()

            model_classes.append(param_dict['ModelClass'])
            mine_class_name = param_dict['func_eval'].__str__().split(' ')[1].split('.')[0]
            params.append(
                ModelParameters(
                    MineClass=globals()[mine_class_name],
                    func_classification=param_dict['func_classification'],
                    threshold=param_dict['threshold'],
                    mine_params=MineParameters(
                        func_transform=param_dict['func_transform'],
                        func_eval=param_dict['func_eval'],
                        NormalizerClass=param_dict['NormalizerClass'],
                        EstimatorClass=param_dict['EstimatorClass'],
                        mine_kwargs=mine_kwargs,
                        eval_kwargs=eval_kwargs
                    )
                )
            )
        return model_classes, params

    @staticmethod
    def _map_to_eval_parameters(inputs: Tuple[tuple, Tuple[np.ndarray, np.ndarray]]):
        """ Function to map the generator of evaluation inputs to a class """
        return ModelEvaluationParameters(
            ModelClass=inputs[0][0],
            model_params=inputs[0][1],
            eval_idx=inputs[0][2],
            samples_train=inputs[1][0],
            samples_test=inputs[1][1]
        )

    @staticmethod
    def _evaluate_model(eval_params: ModelEvaluationParameters) -> ModelEvaluationResult:
        """ Wrapper function for model evaluation to be used in maps. """
        with logging.Logger(ModelEvaluationResult(eval_params)) as eval_result:
            # Creating model and test values
            model = eval_params.ModelClass(eval_params.model_params, eval_params.samples_train)
            # Iterating and evaluating all test values
            for k, sample_test in enumerate(eval_params.samples_test):
                eval_result.predictions[k] = model.classify(sample_test)
        return eval_result

    @staticmethod
    def compute_loss(func_loss: Callable, labels: Sequence[np.ndarray], predictions:Sequence[np.ndarray]) -> float or np.ndarray:
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
