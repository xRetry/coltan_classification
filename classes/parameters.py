from typing import Callable, Optional
from classes.estimators import Estimator


class Parameters:
    MineClass: object.__class__
    func_normalize: Callable
    func_transform: Callable
    func_eval: Callable
    eval_args: dict
    func_selection: Callable
    func_loss: Callable
    estimator: type(Estimator)
    mine_kwargs: dict

    def __init__(self, MineClass: object.__class__, func_transform: Callable, func_normalize: Callable,
                 func_eval: Callable, func_selection: Callable, func_loss: Callable,
                 estimator: type(Estimator), mine_kwargs:Optional[dict]=None, eval_args:Optional[dict]=None):
        self.MineClass = MineClass
        self.func_normalize = func_normalize
        self.func_transform = func_transform
        self.func_eval = func_eval
        self.func_selection = func_selection
        self.func_loss = func_loss
        self.estimator = estimator
        if mine_kwargs is None:
            mine_kwargs = {}
        self.mine_kwargs = mine_kwargs
        if eval_args is None:
            eval_args = {}
        self.eval_args = eval_args


if __name__ == '__main__':
    pass
