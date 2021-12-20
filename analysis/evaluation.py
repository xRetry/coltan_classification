import numpy as np
from functions import plotting, transformation
from classes.mines import Mine
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
            y_list = []
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
                y_list.append(y)
            # Adding to result
            results[func.__qualname__] = {
                'y': y_list,
                'kwargs': kwargs
            }
        # Plotting result
        plotting.plot_eval_results(x1_mesh, x2_mesh, results)

    @staticmethod
    def _create_mine(eval_func: Callable) -> Mine:
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
