import numpy as np


def none(x):
    """
    Applies no transformation to the input.
    """
    return x


def log(x: np.ndarray or list, base_10:bool=False) -> np.ndarray or list:
    is_list = True
    if not isinstance(x, list):
        is_list = False
        x = [x]
    for i in range(len(x)):
        x[i] = np.array(x[i])
        if base_10:
            x[i] = np.log10(x[i], out=np.ones_like(x[i]) * -20, where=x[i] > 0)
        else:
            x[i] = np.log(x[i], out=np.ones_like(x[i])*-20, where=x[i] > 0)  # TODO: Find appropriate value for negative inputs
    if is_list:
        return x
    return x[0]


def log10(x: np.ndarray or list):
    return log(x, base_10=True)


if __name__ == '__main__':
    pass
