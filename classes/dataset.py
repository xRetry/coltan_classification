import numpy as np
import pandas as pd
import os
from typing import List, Iterator, Optional, Callable, Generator, Tuple


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

    def cv_generator(self, proportion_test:float, shuffle=True, verbose: bool=False) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generates train and test samples for cross-validation according to proportion of test samples.
        """
        # Copy samples
        samples_cv = np.array(self._samples)
        # Shuffle data if wanted
        if shuffle:
            samples_cv = samples_cv[np.random.permutation(len(samples_cv))]
        # Iterate through folds
        for mask in self._cv_mask_generator(len(samples_cv), proportion_test, verbose=verbose):
            yield samples_cv[mask], samples_cv[np.invert(mask)]

    @staticmethod
    def _cv_mask_generator(n_samples: int, pct_test: float, verbose: bool=False) -> Generator[np.ndarray, None, None]:
        """
        Creates generator which gradually yields masks for cross-validation.
        """
        # Check input value
        if pct_test > 1 or pct_test < 0:
            raise ValueError('Invalid value for test proportion!')
        # Handling case of reverse cross-validation
        bool_array = np.ones
        if pct_test > 0.5:
            bool_array = np.zeros
            pct_test = 1 - pct_test
        # Determining amount of test samples per fold
        n_test = int(np.ceil(n_samples * pct_test))
        if n_test == 0:
            n_test = int(n_samples * pct_test)
        # Determining amount of folds
        n_folds = int(np.ceil(n_samples / n_test))
        # Iterating trough folds
        idx_start = 0
        for i in range(n_folds):
            # Creating mask
            mask = bool_array(n_samples, dtype=bool)
            # Determining end of selection
            idx_end = idx_start + n_test
            if idx_end > n_samples:
                idx_end = n_samples
            # Reversing bool of selection
            mask[idx_start:idx_end] = np.invert(mask[idx_start])
            # Shifting selection start
            idx_start += n_test
            if verbose:
                print('\r{}/{}'.format(i+1, n_folds), end='')
            # Yielding current mask
            yield mask

    @property
    def attributes(self) -> np.ndarray:
        return np.array([sample.attributes for sample in self._samples])

    @property
    def attribute_labels(self) -> np.ndarray:
        return self._attr_labels

    @property
    def n_attributes(self) -> int:
        return self._samples[0].n_attributes

    @property
    def labels(self) -> np.ndarray:
        lbls = []
        for sample in self._samples:
            lbls.append(np.repeat(sample.label, len(sample.attributes)))
        return np.concatenate(lbls)

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
