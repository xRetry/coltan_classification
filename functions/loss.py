import numpy as np
from functions.decorators import verification


@verification('a', 'a')
def accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    n_correct = (labels == predictions).sum()
    return n_correct / len(predictions)


@verification('a', 'a')
def error(labels: np.ndarray, predictions: np.ndarray) -> float:
    return 1-accuracy(labels, predictions)


@verification('a', 'a')
def acc_matrix(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    tp = ((labels == 1) & (predictions == 1)).sum()
    fp = ((labels == -1) & (predictions == 1)).sum()
    tn = ((labels == -1) & (predictions == -1)).sum()
    fn = ((labels == 1) & (predictions == -1)).sum()

    return np.array([
        [tn, fp],
        [fn, tp]
    ])


@verification('a', 'a')
def error_matrix(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    return 1-acc_matrix(labels, predictions)


if __name__ == '__main__':
    pass
