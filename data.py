import pandas as pd
import numpy as np
from typing import Iterable


'''
    +++ DATA LOADING +++
'''


def load_data(remove_indices: Iterable = (2, 4, 11, 17, 19, 27, 29, 45, 25)) -> pd.DataFrame:
    """
    Loads data from .csv file and adds mine-ID.
    :return: Dataframe of Coltan data
    """
    data = pd.read_csv('ctpa-data.csv', sep=';')
    data['mineID'] = data['x'].astype(str) + data['y'].astype(str) + data['z'].astype(str)
    data = data.drop(columns=[f'Att{i}' for i in remove_indices])
    return data
    # Duplicate attributes:
    # 1, 2
    # 3, 4
    # 10, 11
    # 16, 17
    # 18, 19, 27, 29, 45
    # 23, 25


'''
    +++ DATA TRANSFORMATIONS +++
'''


def no_transform(x):
    return x


def log_transform(x: np.ndarray or list) -> np.ndarray or list:
    is_list = True
    if not isinstance(x, list):
        is_list = False
        x = [x]
    for i in range(len(x)):
        x[i] = np.array(x[i])
        x[i] = np.log(x[i], out=np.ones_like(x[i])*-20, where=x[i] > 0)
    if is_list:
        return x
    return x[0]


if __name__ == '__main__':
    pass
