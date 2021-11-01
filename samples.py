import numpy as np
from typing import List, Iterator
import pandas as pd


'''
    +++ SUPERCLASS +++
'''


class Samples:
    _samples: List[np.ndarray]

    def __init__(self, data: pd.DataFrame):
        self._samples = []
        sample_ids = pd.unique(data['smp'])
        for sample_id in sample_ids:
            attr_values = np.array(data[data['smp'] == sample_id].filter(regex='Att*'))
            self._samples.append(attr_values)

    @property
    def values(self) -> List[np.ndarray]:
        return self._samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, item) -> np.ndarray:
        return self._samples[item]

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter(self._samples)


'''
    +++ SUBCLASSES +++
'''


class TestSamples(Samples):
    _labels: np.ndarray

    def __init__(self, data: pd.DataFrame):
        sample_ids = pd.unique(data['smp'])
        labels = np.zeros(len(sample_ids))
        for i, sample_id in enumerate(sample_ids):
            labels[i] = data[data['smp'] == sample_id]['FP'].iloc[0]
        self._labels = labels
        super(TestSamples, self).__init__(data)

    @property
    def labels(self):
        return self._labels


if __name__ == '__main__':
    pass
