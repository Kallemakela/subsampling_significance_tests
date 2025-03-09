"""
https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html#comparing-two-models-frequentist-approach
"""

import numpy as np
from scipy.stats import t


def corrected_variance(x, n_train, n_test):
    """
    Corrected repeated cross-validation variance from:
    Nadeau, C., Bengio, Y. Inference for the Generalization Error.
    Machine Learning 52, 239–281 (2003).
    https://doi.org/10.1023/A:1024068626366
    """
    j = len(x)  # repeats * folds per repeat
    return np.var(x, ddof=1) * (1 / j + n_test / n_train)


def corrected_t_test(x, n_train, n_test, two_tailed=False):
    """
    Corrected repeated cross-validation t-test from:
    Nadeau, C., Bengio, Y. Inference for the Generalization Error.
    Machine Learning 52, 239–281 (2003).
    https://doi.org/10.1023/A:1024068626366
    """
    mu = np.mean(x)
    var = corrected_variance(x, n_train, n_test)
    std = np.max([np.sqrt(var), 1e-9])
    t_stat = mu / std
    p = t.sf(np.abs(t_stat), len(x) - 1)
    p = 2 * p if two_tailed else p
    return t_stat, p
