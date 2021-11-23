import numpy as np
from typing import Callable, List
from classes.mines import Mine, OrigMine, AggregationUniMine
from classes.models import Model
from classes.dataset import Sample, Dataset
from classes.parameters import Parameters
from functions.evaluation import transform_none
from functions.plotting import plot_eval_results, plot_eval_results_2d


class EvalFuncAnalyser:
    @staticmethod
    def func_analysis(eval_func: List[Callable] or Callable, x_min:float=-5, x_max:float=5,
                      n_pts:int=100, normal_scale:float=1, n_samples:int=50):
        self = EvalFuncAnalyser
        if not isinstance(eval_func, list):
            eval_func = [eval_func]

        offsets_x = np.linspace(x_min, x_max, n_pts)
        results_all = []
        for func in eval_func:
            np.random.seed(0)

            attr_vals = np.random.normal(loc=1000, scale=normal_scale, size=n_samples)[:, None]
            sample_mine = self._create_sample(attr_vals)
            mine = self._create_mine(func)
            mine.add_sample(sample_mine)

            results_func = np.zeros_like(offsets_x)
            for i, x_off in enumerate(offsets_x):
                attr_vals_current = np.array(attr_vals)
                attr_vals_current[:, 0] += x_off
                sample_current = self._create_sample(attr_vals_current)
                results_func[i] = mine.eval_sample(sample_current)

            results_all.append(results_func)

        labels = [func.__qualname__ for func in eval_func]
        plot_eval_results(offsets_x, results_all, labels)

    @staticmethod
    def func_analysis_2d(eval_func: Callable, x_limits:tuple=(-5, 5), y_limits:tuple=(-5, 5)):
        self = EvalFuncAnalyser
        np.random.seed(0)

        attr_vals = np.random.multivariate_normal(mean=[1000, 1000], cov=[[1, 0], [0, 1]], size=50)
        sample_mine = self._create_sample(attr_vals)
        mine = self._create_mine(eval_func)
        mine.add_sample(sample_mine)

        steps_x = np.linspace(*x_limits, 100)
        steps_y = np.linspace(*y_limits, 100)

        offsets_x, offsets_y = np.meshgrid(steps_x, steps_y)
        results_eval = np.zeros_like(offsets_x)
        for i in range(len(offsets_x)):
            for j in range(len(offsets_y)):
                attr_current = np.array(attr_vals)
                attr_current[:, 0] += offsets_x[i, j]
                attr_current[:, 1] += offsets_y[i, j]
                sample_new = self._create_sample(attr_current)
                results_eval[i, j] = mine.eval_sample(sample_new)

        plot_eval_results_2d(offsets_x, offsets_y, results_eval)

    @staticmethod
    def _create_mine(eval_func: Callable) -> Mine:
        func_str = eval_func.__qualname__.split('.')
        mine = globals()[func_str[0]](
            coordinates=np.zeros(3),
            status=0,
            parameters=Parameters(
                MineClass=None,
                func_transform=transform_none,
                func_eval=eval_func,
                func_loss=transform_none,
                func_selection=transform_none
            )
        )
        return mine

    @staticmethod
    def _create_sample(attr_values: np.ndarray) -> Sample:
        sample = Sample(
            coordinates=np.zeros(3),
            label=0,
            attributes=attr_values,
            sample_id=0,
            mine_id=''
        )
        return sample

    @staticmethod
    def sample_evaluation(eval_func:Callable):
        self = EvalFuncAnalyser
        dataset = Dataset()
        mine_ids_unique, mine_ids_count = np.unique([smp.mine_id for smp in dataset], return_counts=True)
        idx_selected = np.argmax(mine_ids_count)
        samples_mine_selected = [sample for sample in dataset if sample.mine_id == mine_ids_unique[idx_selected]]
        mine = self._create_mine(eval_func)

        sample = samples_mine_selected[0]

        mine.add_sample(sample)
        multipliers = np.linspace(0.1, 1.9, 100)
        dataset_analysis = dataset[:5]
        eval_results = np.zeros((len(dataset_analysis), len(multipliers)))
        x_vals = np.zeros_like(eval_results)
        for s, sample in enumerate(dataset_analysis):
            attr_orig = np.array(sample.attributes)
            for m, multi in enumerate(multipliers):
                sample.attributes = np.array(attr_orig)
                sample.attributes[:, 0] *= multi
                eval_results[s, m] = mine.eval_sample(sample)
            x_vals[s, :] = attr_orig[:, 0].mean() * multipliers

        plot_eval_results(x_vals, eval_results)


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
