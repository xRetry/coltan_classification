from typing import List, Callable, Tuple, Dict

import numpy as np

from analysis import plotting, loss
from core.mines import Mine, MineParameters
from core.dataset import Sample, Dataset
from analysis.model import ModelAnalyser, CrossValParameters
from analysis.utils import console
import analysis.plotting.evaluation as plot


class EvalFuncAnalyser:
    @staticmethod
    def pure_shape(func_eval: Callable or Tuple[Callable, dict] or List[Callable or Tuple[Callable, dict]], x: np.ndarray=np.zeros(2),
                   x1_range:Tuple[float]=(-3, 3), x2_range:Tuple[float]=(-3, 3)):
        """
            Shows the shape of the evaluation functions. Only works if the function takes single values.
        """
        # Wrap in list if single function
        if not isinstance(func_eval, list):
            func_eval = [func_eval]
        # Setting up x value offsets
        loc = np.array([0, 0])
        x1_offset = np.linspace(*x1_range, 300)
        x2_offset = np.linspace(*x2_range, 300)
        x1_grid, x2_grid = np.meshgrid(x1_offset, x2_offset)
        # Iterate through all functions
        results = {}
        for func in func_eval:
            # Setting up function and addition argument
            if isinstance(func, tuple):
                kwargs = func[1]
                func = func[0]
            else:
                kwargs = dict()
            # Iterating through grid
            y = np.zeros_like(x1_grid)
            for i in range(x1_grid.shape[0]):
                for j in range(x1_grid.shape[1]):
                    # Shift x values by offset
                    x_shifted = np.array([
                        x[0] + x1_grid[i, j],
                        x[1] + x2_grid[i, j]
                    ])
                    # Evaluate function
                    y[i, j] = func(loc, x_shifted, **kwargs)
            # Adding to result
            key = '{}({})'.format(func.__qualname__, kwargs)
            results[key] = y
        # Plotting result
        plot.plot_eval_result(x1_grid, x2_grid, results)

    @staticmethod
    def mine_shape(mine_class: type(Mine) or List[type(Mine)], mine_params: MineParameters or List[MineParameters],
                   data_train: Dataset, x1_delta: float=1, x2_delta: float=1):
        """
            Shows the shape of an evaluation functions used in mines. Samples are generated from a normal distribution.
        """
        # Wrap in list if single parameters
        if not isinstance(mine_params, list):
            mine_params = [mine_params]
            mine_class = [mine_class]
        # Computes the centers of each attribute
        center = np.concatenate([attr for attr in data_train.attributes]).mean(axis=0)
        # Creating artificial sample attributes
        attr_test = np.array([
            np.array([-1, 0, 1]),
            np.array([-1, 0, 1]),
        ])
        # Creating meshgrid around centers
        x1_offset = np.linspace(center[0]-x1_delta, center[0]+x1_delta, 100)
        x2_offset = np.linspace(center[1]-x2_delta, center[1]+x2_delta, 100)
        x1_mesh, x2_mesh = np.meshgrid(x1_offset, x2_offset)
        # Iterating through evaluation functions
        result = {}
        for p_idx, (m_class, params) in enumerate(zip(mine_class, mine_params)):
            print(f'{p_idx+1} / {len(mine_params)}')
            # Creating mine with eval function
            mine = EvalFuncAnalyser._create_mine(m_class, params)
            # Adding training samples
            for sample in data_train:
                mine.add_sample(sample)
            # Iterating through mesh
            y = np.zeros_like(x1_mesh)
            for i in range(x1_mesh.shape[0]):
                for j in range(x1_mesh.shape[1]):
                    # Shifting attributes by offsets
                    attr_shifted = np.array([
                        attr_test[0] + x1_mesh[i, j],
                        attr_test[1] + x2_mesh[i, j],
                    ]).T
                    # Creating and evaluating test sample
                    sample_test = EvalFuncAnalyser._create_sample(attr_shifted)
                    y[i, j] = mine.eval_sample(sample_test)
            # Saving result
            result[f'Mine {p_idx+1}'] = y
        # Plotting result
        plot.plot_eval_result(x1_mesh, x2_mesh, result)

    @staticmethod
    def kwargs_loss(params: CrossValParameters, kwargs_vals: Dict[str, np.ndarray]):
        """
            Computes the loss of a function with kwargs and plots the result
        """
        # Setting accuracy as loss function
        params.func_loss = loss.accuracy
        # Loading dataset
        dataset = Dataset()
        # Converting kwargs values
        kw_vals = list(kwargs_vals.values())
        kw_keys = list(kwargs_vals.keys())
        # Checking if kwargs are value ranges
        is_range = [True if isinstance(v, np.ndarray) else False for v in kw_vals]
        # Setting up progress bar data
        progress_bar = console.ProgressBar()
        # Iterating over range if only one value range
        if sum(is_range) == 1:
            idx_range = np.where(is_range)[0][0]
            accs = np.zeros((len(kw_vals[idx_range]), 2), dtype=float)
            n_iter = len(kw_vals[idx_range])
            for i in range(n_iter):
                progress_bar.add_bar('KW-Args', i, n_iter)
                cv_result = ModelAnalyser.cross_validate(params, progress_bar=progress_bar)
                accs[i, :] = cv_result.conf_ints[0]
            plot.plot_kwargs_accs({kw_keys[idx_range]: kw_vals[idx_range]}, accs)
        # Creating and iterating grid for two value ranges
        else:
            x_grid, y_grid = np.meshgrid(*kw_vals)
            loss_grid = np.zeros_like(x_grid, dtype=float)
            n_i, n_j = len(x_grid), len(x_grid[0])
            for i in range(n_i):
                for j in range(n_j):
                    progress_bar.add_bar('Grid', i*n_j+j, n_i*n_j)
                    cv_result = ModelAnalyser.cross_validate(params, progress_bar=progress_bar)
                    loss_grid[i, j] = np.mean(cv_result.conf_ints)
            plot.plot_kwargs_accs({kw_keys[0]: x_grid, kw_keys[1]: y_grid}, loss_grid)

    @staticmethod
    def _create_mine(MineClass: type(Mine), mine_params: MineParameters) -> Mine:
        """
            Creates a mine from an evaluation function.
        """
        mine = MineClass(
            coordinates=np.zeros(3),
            label=0,
            mine_parameters=mine_params
        )
        return mine

    @staticmethod
    def _create_sample(attr_values: np.ndarray) -> Sample:
        """
            Creates a sample with provided attribute values.
        """
        sample = Sample(
            coordinates=np.zeros(3),
            label=0,
            attributes=attr_values,
            sample_id=0,
            mine_id=''
        )
        return sample


if __name__ == '__main__':
    pass
