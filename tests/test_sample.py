from scipy.stats import ttest_1samp
import numpy as np
from sklearn.model_selection import RepeatedKFold

from subsampling_significance_tests import corrected_t_test


def test_match_vs_standard_ttest():
    np.random.seed(42)
    x = np.random.randn(30)
    t_stat_standard, p_standard = ttest_1samp(x, popmean=0)
    t_stat_corrected, p_corrected = corrected_t_test(
        x, n_train=30, n_test=0, two_tailed=True
    )
    np.testing.assert_allclose(t_stat_corrected, t_stat_standard, rtol=1e-6)
    np.testing.assert_allclose(p_corrected, p_standard, rtol=1e-6)


def test_mismatch_vs_standard_ttest():
    np.random.seed(42)
    x = np.random.randn(30)
    t_stat_standard, p_standard = ttest_1samp(x, popmean=0)
    t_stat_corrected, p_corrected = corrected_t_test(
        x, n_train=15, n_test=15, two_tailed=True
    )
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_allclose,
        t_stat_corrected,
        t_stat_standard,
        rtol=1e-6,
    )
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_allclose,
        p_corrected,
        p_standard,
        rtol=1e-6,
    )


def test_corrected_better_than_standard_ttest():

    np.random.seed(42)
    alpha = 0.05
    n_iters = 1000
    R, k = 10, 5
    n = 100

    n_test = int(n * (1 / k))
    n_train = n - n_test
    standard_rejections = 0
    corrected_rejections = 0
    for _ in range(n_iters):
        sample_effects = np.random.randn(n)
        diffs = []
        cv = RepeatedKFold(n_splits=k, n_repeats=R)
        for split_ix, (train_ix, test_ix) in enumerate(cv.split(sample_effects)):
            x = sample_effects[test_ix]
            diffs.append(x.sum())
        diffs = np.array(diffs)
        _, p_std = ttest_1samp(diffs, popmean=0)
        _, p_corr = corrected_t_test(
            diffs, n_train=n_train, n_test=n_test, two_tailed=True
        )
        if p_std < alpha:
            standard_rejections += 1
        if p_corr < alpha:
            corrected_rejections += 1

    rejection_rate_std = standard_rejections / n_iters
    rejection_rate_corr = corrected_rejections / n_iters
    assert rejection_rate_corr < rejection_rate_std
    assert np.abs(rejection_rate_corr - alpha) < np.abs(rejection_rate_std - alpha)
    assert np.abs(rejection_rate_corr - alpha) < 0.01
