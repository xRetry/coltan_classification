import numpy as np

'''
    +++ LOSS FUNCTIONS +++
'''


def accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    n_correct = (labels == predictions).sum()
    return n_correct / len(predictions)


def error(labels: np.ndarray, predictions: np.ndarray) -> float:
    acc = accuracy(labels, predictions)
    return 1-acc


'''
    +++ VALUE-TO-LABEL CONVERSIONS +++
'''


def best_mine(eval_results, mines):
    eval_results = np.array(eval_results)

    if len(eval_results.shape) == 1:
        eval_results = eval_results[:, None]

    idx = np.nanargmax(eval_results, axis=0)
    return np.array([mines[i].status for i in idx])


def best_label(eval_results, mines):
    labels = np.array([m.status for m in mines])
    sum_full = np.sum(eval_results, axis=0)
    sum_pos = np.sum(eval_results[labels == 1], axis=0)
    p_pos = np.divide(sum_pos, sum_full, out=np.zeros_like(sum_pos), where=sum_full != 0)
    return -np.ones(len(p_pos), dtype=int) + 2 * (p_pos > 0.5).astype(int)


if __name__ == '__main__':
    pass
