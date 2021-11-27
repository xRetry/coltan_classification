import numpy as np


class Normalization:
    @staticmethod
    def none(mine, x):
        return x


class Transformation:
    @staticmethod
    def none(x):
        """
        Applies no transformation to the input.
        """
        return x

    @staticmethod
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

    @staticmethod
    def log10(x: np.ndarray or list):
        return Transformation.log(x, base_10=True)


class Loss:
    @staticmethod
    def accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
        n_correct = (labels == predictions).sum()
        return n_correct / len(predictions)

    @staticmethod
    def error(labels: np.ndarray, predictions: np.ndarray) -> float:
        return 1-Loss.accuracy(labels, predictions)

    @staticmethod
    def acc_matrix(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        tp = ((labels == 1) & (predictions == 1)).sum()
        fp = ((labels == -1) & (predictions == 1)).sum()
        tn = ((labels == -1) & (predictions == -1)).sum()
        fn = ((labels == 1) & (predictions == -1)).sum()

        return np.array([
            [tn, fp],
            [fn, tp]
        ])

    @staticmethod
    def error_matrix(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        return 1-Loss.acc_matrix(labels, predictions)


class Selection:
    @staticmethod
    def mine(eval_results: np.ndarray, labels: np.ndarray) -> int:
        if np.all(np.isnan(eval_results)):
            return np.nan
        idx = np.nanargmax(eval_results)
        return labels[idx]

    @staticmethod
    def label(eval_results: np.ndarray, labels: np.ndarray) -> int:
        sum_full = np.nansum(eval_results)
        sum_pos = np.nansum(eval_results[labels == 1])
        if sum_full == 0:
            return np.nan
        p_pos = sum_pos/sum_full
        return (p_pos > 0.5) * 2 - 1


if __name__ == '__main__':
    pass
