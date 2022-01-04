import numpy as np
from functions.decorators import verification
from typing import Callable


@verification('a', 'a')
def mean_score(eval_scores: np.ndarray, labels: np.ndarray) -> int:
    return _score_selection(eval_scores, labels, np.nanmean)


@verification('a', 'a')
def median_score(eval_scores: np.ndarray, labels: np.ndarray) -> int:
    return _score_selection(eval_scores, labels, np.nanmedian)


@verification('a', 'a')
def max_score(eval_scores: np.ndarray, labels: np.ndarray) -> int:
    return _score_selection(eval_scores, labels, np.nanmax)


@verification('a', 'a')
def max_score_old(eval_scores: np.ndarray, labels: np.ndarray) -> int:
    """
    Chooses label based on the mine with highest evaluation.
    """
    # Convert type if not float
    if eval_scores.dtype != float:
        eval_scores = eval_scores.astype(float)
    # Replace inf values with nan
    eval_scores[eval_scores == np.inf] = np.nan
    # Check if all values are nan
    if np.all(np.isnan(eval_scores)):
        return np.nan
    # Select highest evaluation label
    idx = np.nanargmax(eval_scores)
    return labels[idx]


@verification('a', 'a')
def proportional_score(eval_scores: np.ndarray, labels: np.ndarray) -> int:
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
    return (p_pos > 0.5) * 2 - 1


def _score_selection(eval_scores: np.ndarray, labels: np.ndarray, func_select: Callable) -> int:
    scores_pos = eval_scores[labels == 1]
    scores_neg = eval_scores[labels == -1]
    is_nan_pos = np.alltrue(np.isnan(scores_pos))
    is_nan_neg = np.alltrue(np.isnan(scores_neg))
    if is_nan_pos and is_nan_neg:
        return np.nan
    if is_nan_pos and not is_nan_neg:
        return -1
    if not is_nan_pos and is_nan_neg:
        return 1
    return 1 if func_select(scores_pos) > func_select(scores_neg) else -1


if __name__ == '__main__':
    pass
