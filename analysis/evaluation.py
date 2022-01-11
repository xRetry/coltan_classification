import numpy as np

from functions import plotting, transformation, loss
from classes.mines import Mine, AggregationMine
from classes.parameters import Parameters
from classes.dataset import Sample, Dataset
from classes.normalizers import Normalizer
from classes.estimators import MeanUniEstimator
from analysis.model import ModelAnalyser
from typing import List, Callable, Optional, Tuple, Dict


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
        plotting.plot_eval_result(x1_grid, x2_grid, results)

    @staticmethod
    def mine_shape(mine_params: Parameters or List[Parameters],
                   x1_range: Tuple[float, float]=(-1, 1),x2_range: Tuple[float, float]=(-1, 1),
                   std: float=1, n_train: int=3, n_sample: int=20, show_sample: bool=False):
        """
        Shows the shape of an evaluation functions used in mines. Samples are generated from a normal distribution.
        """
        # Wrap in list if single parameters
        if not isinstance(mine_params, list):
            mine_params = [mine_params]
        # Creating training samples
        samples_train = []
        for i in range(n_train):
            attr_train = np.array([
                np.random.normal(loc=0, scale=std, size=n_sample),
                np.random.normal(loc=0, scale=std, size=n_sample)
            ])
            samples_train.append(EvalFuncAnalyser._create_sample(attr_train.T))
        # Creating test sample
        attr_test = np.array([
            np.random.normal(loc=0, scale=std, size=n_sample),
            np.random.normal(loc=0, scale=std, size=n_sample)
        ])
        # Creating offset mesh
        x1_offset = np.linspace(*x1_range, 250)
        x2_offset = np.linspace(*x2_range, 250)
        x1_mesh, x2_mesh = np.meshgrid(x1_offset, x2_offset)
        # Iterating through evaluation functions
        result = {}
        for p_idx, params in enumerate(mine_params):
            print(f'{p_idx+1} / {len(mine_params)}')
            # Creating mine with eval function
            mine = EvalFuncAnalyser._create_mine(params)
            # Adding training samples
            for sample in samples_train:
                mine.add_sample(sample)
            # Iterating through mesh
            y = np.zeros_like(x1_mesh)
            for i in range(x1_mesh.shape[0]):
                for j in range(x1_mesh.shape[1]):
                    # Shift x values by offset
                    attr_shifted = np.array([
                        attr_test[0] + x1_mesh[i, j],
                        attr_test[1] + x2_mesh[i, j]
                    ])
                    # Creating and evaluating test sample
                    sample_test = EvalFuncAnalyser._create_sample(attr_shifted.T)
                    y[i, j] = mine.eval_sample(sample_test)
            # Saving result
            result[f'Mine {p_idx+1}'] = y
        # Disable plotting of sample
        if not show_sample:
            attr_test = None
        # Plotting result
        plotting.plot_eval_result(x1_mesh, x2_mesh, result, sample_test=attr_test)

    @staticmethod
    def kwargs_loss(params: Parameters, kwargs_vals: Dict[str, np.ndarray]):
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
        # Iterating over range if only one value range
        if sum(is_range) == 1:
            idx_range = np.where(is_range)[0][0]
            accs = np.zeros((len(kw_vals[idx_range]), 2), dtype=float)
            for i in range(len(kw_vals[idx_range])):
                conf_int = ModelAnalyser.cross_validate(params, dataset, 0.2, return_confint=True)[0]
                accs[i, :] = conf_int
            plotting.plot_kwargs_accs({kw_keys[idx_range]: kw_vals[idx_range]}, accs)
        # Creating and iterating grid for two value ranges
        else:
            x_grid, y_grid = np.meshgrid(*kw_vals)
            loss_grid = np.zeros_like(x_grid, dtype=float)
            for i in range(len(x_grid)):
                for j in range(len(x_grid[0])):
                    conf_int = ModelAnalyser.cross_validate(params, dataset, 0.2, return_confint=True)
                    loss_grid[i, j] = np.mean(conf_int)
            plotting.plot_kwargs_accs({kw_keys[0]: x_grid, kw_keys[1]: y_grid}, loss_grid)

    @staticmethod
    def _create_mine(params: Parameters) -> Mine:
        """
        Creates a mine from an evaluation function.
        """
        mine = params.MineClass(
            coordinates=np.zeros(3),
            status=0,
            parameters=params
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
