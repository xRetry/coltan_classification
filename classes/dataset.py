import numpy as np
import pandas as pd
import os
from typing import List, Iterator, Optional, Callable


class Sample:
    sample_id: int
    mine_id: str
    label: int
    coordinates: np.ndarray
    attributes: np.ndarray

    def __init__(self, attributes: np.ndarray, label: int=0, coordinates: np.ndarray=np.zeros(3), sample_id: int=0, mine_id: str=''):
        self.label = label
        self.coordinates = coordinates
        self.attributes = attributes
        self.sample_id = sample_id
        self.mine_id = mine_id

    def append(self, attr_values: np.ndarray):
        self.attributes = np.row_stack([self.attributes, attr_values])

    @property
    def n_attributes(self) -> int:
        return self.attributes.shape[1]

    def __len__(self) -> int:
        return len(self.attributes)


class Dataset:
    _samples: np.ndarray
    _attr_labels: np.ndarray

    def __init__(self, file:Optional[str]='\\data\\ctpa-data.csv'):
        if file is not None:
            self._load_from_file(file)

    def _load_from_file(self, file) -> None:
        """
        Loads data from .csv file, adds mine-ID and creates Samples based on sample ID.
        """

        # Loading .csv file
        path = os.path.dirname(os.path.dirname(__file__)) + file
        data = pd.read_csv(path, sep=';')
        # Adding unique identifier for each mine
        data['mineID'] = data['x'].astype(str) + data['y'].astype(str) + data['z'].astype(str)
        # Finding duplicates in dataset
        corr_mat = np.corrcoef(np.array(data.filter(regex='Att*')).T)
        is_equal = np.isclose(np.tril(corr_mat, -1), 1, rtol=1e-04, atol=1e-06)
        idx_duplicates = np.where(is_equal)[0]
        # Removing duplicates
        data = data.drop(columns=[f'Att{i}' for i in idx_duplicates])
        self._attr_labels = data.filter(regex='Att*').columns.values
        # Grouping values based on sample ID
        samples = []
        sample_ids = pd.unique(data['smp'])
        for sample_id in sample_ids:
            data_sample = data[data['smp'] == sample_id]
            sample = Sample(
                attributes=np.array(data_sample.filter(regex='Att*')),
                label=data_sample['FP'].iloc[0],
                coordinates=data_sample[['x', 'y', 'z']].iloc[0,:].values,
                sample_id=sample_id,
                mine_id=data_sample['mineID'].iloc[0]
            )
            samples.append(sample)
        self._samples = np.array(samples)

    def _set_parameters(self, samples, attr_labels):
        """
        Method to manually set the Dataset parameters.
        """
        self._samples = samples
        self._attr_labels = attr_labels

    def crossval_samples(self, n_folds:int, shuffle=True):
        # Copy samples
        samples_cv = np.array(self._samples)
        # Set up array with fold sizes
        n_samples = len(samples_cv)
        fold_sizes = n_samples // n_folds * np.ones(n_folds, dtype=int)
        fold_sizes[:(n_samples % n_folds)] += 1
        # Shuffle data if wanted
        if shuffle:
            samples_cv = samples_cv[np.random.permutation(len(samples_cv))]

        samples_train = []
        samples_test = []
        idx_begin = 0
        # Iterate through folds
        for fold_size in fold_sizes:
            # Setting up mask indices
            idx_end = idx_begin + fold_size
            # Creating sample mask
            mask = np.ones(n_samples).astype(np.bool)
            mask[idx_begin:idx_end] = False
            samples_train.append(samples_cv[mask])
            samples_test.append(samples_cv[np.invert(mask)])
            idx_begin += fold_size

        return list(zip(samples_train, samples_test))

    @property
    def attributes(self) -> np.ndarray:
        return np.array([sample.attributes for sample in self._samples])

    @property
    def attribute_labels(self) -> np.ndarray:
        return self._attr_labels

    @property
    def n_attributes(self) -> int:
        return self._samples[0].n_attributes

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, item):
        dataset_new = Dataset(None)
        dataset_new._set_parameters(
            samples=np.array(self._samples[item]),
            attr_labels=np.array(self._attr_labels[item])
        )
        return dataset_new

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._samples)


if __name__ == '__main__':
    pass
