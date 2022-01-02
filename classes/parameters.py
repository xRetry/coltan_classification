from typing import Callable, Optional
from dataclasses import dataclass, field
from functions import transformation
from classes.normalizers import Normalization
from classes.estimators import Estimator


@dataclass
class Parameters:
    MineClass: object.__class__
    func_eval: Callable
    ModelClass: object.__class__ = None
    func_normalize: Callable = Normalization.none
    func_transform: Callable = transformation.none
    func_selection: Callable = None
    func_loss: Callable = None
    estimator: type(Estimator) = None
    eval_kwargs: dict = field(default_factory=dict)
    mine_kwargs: dict = field(default_factory=dict)


if __name__ == '__main__':
    pass
