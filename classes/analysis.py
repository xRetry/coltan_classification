import numpy as np
from typing import Callable, List
from classes.mines import Mine, OrigMine, AggregationUniMine
from classes.dataset import Sample, Dataset
from classes.parameters import Parameters
from functions.evaluation import transform_none
from functions.plotting import plot_eval_results, plot_eval_results_2d


class EvalFuncAnalyser:
    @staticmethod
    def func_analysis(eval_func: List[Callable] or Callable):
        self = EvalFuncAnalyser
        if not isinstance(eval_func, list):
            eval_func = [eval_func]

        offsets_x = np.linspace(-5, 5, 100)
        results_all = []
        for func in eval_func:
            np.random.seed(0)

            attr_vals = np.random.normal(loc=1000, scale=1, size=50)[:, None]
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
    def func_analysis_2d(eval_func: Callable):
        self = EvalFuncAnalyser
        np.random.seed(0)

        attr_vals = np.random.multivariate_normal(mean=[1000, 1000], cov=[[1, 0], [0, 1]], size=50)
        sample_mine = self._create_sample(attr_vals)
        mine = self._create_mine(eval_func)
        mine.add_sample(sample_mine)

        x_steps = np.linspace(-5, 5, 100)
        y_steps = np.linspace(-5, 5, 100)

        x_offsets, y_offsets = np.meshgrid(x_steps, y_steps)
        eval_results = np.zeros_like(x_offsets)
        for i in range(len(x_offsets)):
            for j in range(len(y_offsets)):
                vals_current = np.array(attr_vals)
                vals_current[:, 0] += x_offsets[i, j]
                vals_current[:, 1] += y_offsets[i, j]
                sample_new = self._create_sample(vals_current)
                eval_results[i, j] = mine.eval_sample(sample_new)

        plot_eval_results_2d(x_offsets, y_offsets, eval_results)

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


if __name__ == '__main__':
    pass
