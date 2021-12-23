import numpy as np
from functions import plotting, transformation
from classes.mines import Mine, AggregationMine
from classes.parameters import Parameters
from classes.dataset import Sample, Dataset
from classes.normalizers import Normalization
from classes.estimators import MeanUniEstimator
from typing import List, Callable, Optional, Tuple


class EvalFuncAnalyser:
    @staticmethod
    def pure_analysis(func_eval: Callable or List[Callable or Tuple[Callable, dict]], x: np.ndarray=np.zeros(2),
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
        x1_mesh, x2_mesh = np.meshgrid(x1_offset, x2_offset)
        # Iterate through all functions
        results = {}
        for func in func_eval:
            # Setting up function and addition argument
            if isinstance(func, tuple):
                kwargs = func[1]
                func = func[0]
            else:
                kwargs = dict()
            if not isinstance(kwargs, list):
                kwargs = [kwargs]
            # Iterate through function arguments
            for kw in kwargs:
                y = np.zeros_like(x1_mesh)
                for i in range(x1_mesh.shape[0]):
                    for j in range(x1_mesh.shape[1]):
                        # Shift x values by offset
                        x_shifted = np.array([
                            x[0] + x1_mesh[i, j],
                            x[1] + x2_mesh[i, j]
                        ])
                        # Evaluate function
                        y[i, j] = func(loc, x_shifted, **kw)
                # Adding to result
                key = '{}({})'.format(func.__qualname__, kwargs)
                results[key] = y
        # Plotting result
        plotting.plot_eval_result(x1_mesh, x2_mesh, results)

    @staticmethod
    def mine_analysis(eval_func: Callable or List[Callable], x1_range: Tuple[float, float]=(-3, 3),
                      x2_range: Tuple[float, float]=(-3, 3), std: float=1, n_train: int=3):
        """
        Shows the shape of an evaluation functions used in mines. Samples are generated from a normal distribution.
        """
        # Wrap in list if single function
        if not isinstance(eval_func, list):
            eval_func = [eval_func]
        # Size of a sample
        n_sample = 20
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
        for func in eval_func:
            # Creating mine with eval function
            mine = EvalFuncAnalyser._create_mine(func)
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
            result[func.__qualname__] = y
        # Plotting result
        plotting.plot_eval_result(x1_mesh, x2_mesh, result, sample_test=attr_test)

    @staticmethod
    def _create_mine(eval_func: Callable) -> Mine:
        """
        Creates a mine from an evaluation function.
        """
        func_str = eval_func.__qualname__.split('.')
        mine = globals()[func_str[0]](
            coordinates=np.zeros(3),
            status=0,
            parameters=Parameters(
                MineClass=None,
                func_normalize=Normalization.none,
                func_transform=transformation.none,
                func_eval=eval_func,
                func_loss=transformation.none,
                func_selection=transformation.none,
                estimator=MeanUniEstimator
            )
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
