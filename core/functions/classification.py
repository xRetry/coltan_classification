import numpy as np
from core.utils import verification
from typing import Callable


@verification('a', 'a', '')
def mean_score(eval_scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> int:
    return _score_selection(eval_scores, labels, np.nanmean, threshold)


@verification('a', 'a', '')
def median_score(eval_scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> int:
    return _score_selection(eval_scores, labels, np.nanmedian, threshold)


@verification('a', 'a', '')
def max_score(eval_scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> int:
    return _score_selection(eval_scores, labels, np.nanmax, threshold)


@verification('a', 'a', '')
def proportional_score(eval_scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> int:
    """
    Chooses label based on the proportion of evaluation values.
    """
    # Convert type if not float
    if eval_scores.dtype != float:
        eval_scores = eval_scores.astype(float)
    # Replace inf values with nan
    eval_scores[eval_scores == np.inf] = np.nan
    # Sum up all evaluation values
    sum_full = np.nansum(eval_scores)
    # Check if all values are nan
    if sum_full == 0:
        return np.nan
    # Sum up positive evaluation values
    sum_pos = np.nansum(eval_scores[labels == 1])
    # Calculate proportion of positive evaluations
    p_pos = sum_pos / sum_full
    # Return 1 if above 0.5 else -1
    return (p_pos > threshold) * 2 - 1


def _score_selection(eval_scores: np.ndarray, labels: np.ndarray, func_select: Callable, threshold: float=0.5) -> int:
    scores_shifted = eval_scores - np.nanmin(eval_scores)
    scores_pos = scores_shifted[labels == 1]
    scores_neg = scores_shifted[labels == -1]
    is_nan_pos = np.alltrue(np.isnan(scores_pos))
    is_nan_neg = np.alltrue(np.isnan(scores_neg))
    if is_nan_pos and is_nan_neg:
        return np.nan
    if is_nan_pos and not is_nan_neg:
        return -1
    if not is_nan_pos and is_nan_neg:
        return 1
    score_pos = func_select(scores_pos)
    score_neg = func_select(scores_neg)
    score_sum = score_pos + score_neg
    if score_sum == 0:
        return np.nan  # TODO: Reconsider return value
    return 1 if score_pos / (score_pos + score_neg) > threshold else -1


if __name__ == '__main__':
    pass
