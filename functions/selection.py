import numpy as np
from functions.decorators import verification


@verification('a', 'a')
def mine(eval_results: np.ndarray, labels: np.ndarray) -> int:
    """
    Chooses label based on the mine with highest evaluation.
    """
    # Convert type if not float
    if eval_results.dtype != float:
        eval_results = eval_results.astype(float)
    # Replace inf values with nan
    eval_results[eval_results == np.inf] = np.nan
    # Check if all values are nan
    if np.all(np.isnan(eval_results)):
        return np.nan
    # Select highest evaluation label
    idx = np.nanargmax(eval_results)
    return labels[idx]


@verification('a', 'a')
def label(eval_results: np.ndarray, labels: np.ndarray) -> int:
    """
    Chooses label based on the proportion of evaluation values.
    """
    # Convert type if not float
    if eval_results.dtype != float:
        eval_results = eval_results.astype(float)
    # Replace inf values with nan
    eval_results[eval_results == np.inf] = np.nan
    # Sum up all evaluation values
    sum_full = np.nansum(eval_results)
    # Check if all values are nan
    if sum_full == 0:
        return np.nan
    # Sum up positive evaluation values
    sum_pos = np.nansum(eval_results[labels == 1])
    # Calculate proportion of positive evaluations
    p_pos = sum_pos / sum_full
    # Return 1 if above 0.5 else -1
    return (p_pos > 0.5) * 2 - 1


if __name__ == '__main__':
    pass
