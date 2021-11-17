from typing import Callable, Optional


class Parameters:
    MineClass: object.__class__
    func_transform: Callable
    func_distgen: Callable
    func_eval: Callable
    func_selection: Callable
    func_loss: Callable
    mine_kwargs: dict

    def __init__(self, MineClass: object.__class__, func_transform: Callable,
                 func_eval: Callable, func_selection: Callable, func_loss: Callable, mine_kwargs:Optional[dict]=None):
        self.MineClass = MineClass
        self.func_transform = func_transform
        self.func_eval = func_eval
        self.func_selection = func_selection
        self.func_loss = func_loss
        if mine_kwargs is None:
            mine_kwargs = {}
        self.mine_kwargs = mine_kwargs


if __name__ == '__main__':
    pass
