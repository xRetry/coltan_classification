import numpy as np
from functions.utils import verification


@verification('ba')
def none(x):
    """
    Applies no transformation to the input.
    """
    return x


@verification('ba', None)
def log(x: np.ndarray, base_10:bool=False) -> np.ndarray or list:
    if base_10:
        return np.log10(x, out=np.ones_like(x) * -20, where=x > 0)
    return np.log(x, out=np.ones_like(x)*-20, where=x > 0)  # TODO: Find appropriate value for negative inputs


@verification('ba')
def log10(x: np.ndarray):
    return log(x, base_10=True)


if __name__ == '__main__':
    pass
