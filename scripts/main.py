# %%
"""
Slower testing of corrected_t_test
"""
#!%load_ext autoreload
#!%autoreload 2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.stats import ttest_1samp
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression

from subsampling_significance_tests import corrected_t_test

# %%


def simulate_experiment(n, R, k, n_iters=1000, alpha=0.05):
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
    return rejection_rate_std, rejection_rate_corr


np.random.seed(42)
alpha = 0.05
n_iters = 10000
R, k = 5, 5
j = R * k
n = 100


rs = np.arange(1, 20, 4)
ks = np.arange(2, 20, 4)
rates_std = np.zeros((len(rs), len(ks)))
rates_corr = np.zeros((len(rs), len(ks)))
for ri, r in enumerate(rs):
    for ki, k in enumerate(ks):
        rejection_rate_std, rejection_rate_corr = simulate_experiment(
            n, r, k, n_iters=n_iters, alpha=alpha
        )
        rates_std[ri, ki] = rejection_rate_std
        rates_corr[ri, ki] = rejection_rate_corr

# %%

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# cmap = "YlOrRd"
# im1 = ax[0].matshow(rates_std, cmap=cmap, norm=colors.LogNorm(vmin=0.01, vmax=0.7))
# im2 = ax[1].matshow(rates_corr, cmap=cmap, norm=colors.LogNorm(vmin=0.01, vmax=0.7))

from matplotlib.colors import LinearSegmentedColormap

cmap = LinearSegmentedColormap.from_list(
    "custom_cmap", [(0, "blue"), (0.5, "white"), (1, "red")]
)
norm = colors.TwoSlopeNorm(vmin=0.01, vcenter=0.05, vmax=0.7)
im1 = ax[0].matshow(rates_std, cmap=cmap, norm=norm)
im2 = ax[1].matshow(rates_corr, cmap=cmap, norm=norm)

# Annotate heatmaps with values
for i in range(rates_std.shape[0]):
    for j in range(rates_std.shape[1]):
        ax[0].text(
            j, i, f"{rates_std[i, j]:.2f}", ha="center", va="center", color="black"
        )
        ax[1].text(
            j, i, f"{rates_corr[i, j]:.2f}", ha="center", va="center", color="black"
        )

for a in ax:
    a.set_xticks(np.arange(len(ks)))
    a.set_xticklabels(ks)
    a.set_yticks(np.arange(len(rs)))
    a.set_yticklabels(rs)

ax[0].set_title("Standard rejection rate")
ax[0].set_xlabel("k")
ax[0].set_ylabel("R")
ax[1].set_title("Corrected rejection rate")
ax[1].set_xlabel("k")

# Create one shared custom colorbar with ticks at 0, 0.05, and 0.5
cbar = fig.colorbar(im2, ax=ax.ravel().tolist(), ticks=[0, 0.05, 0.5])
cbar.ax.set_yticklabels(["0", "0.05", "0.5"])
plt.savefig("fig/heatmap.png")
plt.show()

# %% Model vs popmean

np.random.seed(42)
alpha = 0.05
n_iters = 1000
R, k = 5, 5
j = R * k
n = 100
n_test = int(n * (1 / k))
n_train = n - n_test
standard_rejections = 0
corrected_rejections = 0


n_class = 2
null_acc = 1 / n_class

for i in range(n_iters):
    model1 = LogisticRegression(random_state=i, solver="liblinear")
    X = np.random.randn(n, 10)
    y = np.random.randint(0, n_class, n)
    scores = []
    cv = RepeatedStratifiedKFold(n_splits=k, n_repeats=R)
    for split_ix, (train_ix, test_ix) in enumerate(cv.split(X, y)):
        pred1 = model1.fit(X[train_ix], y[train_ix]).predict(X[test_ix])
        score = (pred1 == y[test_ix]).mean()
        scores.append(score)
    scores = np.array(scores)
    scores = scores - null_acc
    _, p_std = ttest_1samp(scores, popmean=0)
    _, p_corr = corrected_t_test(
        scores, n_train=n_train, n_test=n_test, two_tailed=True
    )
    if p_std < alpha:
        standard_rejections += 1
    if p_corr < alpha:
        corrected_rejections += 1

rejection_rate_std = standard_rejections / n_iters
rejection_rate_corr = corrected_rejections / n_iters
print(f"Standard rejection rate: {rejection_rate_std}")
print(f"Corrected rejection rate: {rejection_rate_corr}")
# %% Model vs model

np.random.seed(42)
alpha = 0.05
n_iters = 1000
R, k = 10, 5
j = R * k
n = 100
n_test = int(n * (1 / k))
n_train = n - n_test
standard_rejections = 0
corrected_rejections = 0
n_class = 2

for i in range(n_iters):
    model1 = LogisticRegression(random_state=i, solver="saga", C=10)
    model2 = LogisticRegression(random_state=n_iters + i, solver="saga", C=0.1)
    X = np.random.randn(n, 10)
    y = np.random.randint(0, n_class, n)
    scores = []
    cv = RepeatedStratifiedKFold(n_splits=k, n_repeats=R)
    for split_ix, (train_ix, test_ix) in enumerate(cv.split(X, y)):
        pred1 = model1.fit(X[train_ix], y[train_ix]).predict(X[test_ix])
        pred2 = model2.fit(X[train_ix], y[train_ix]).predict(X[test_ix])
        diff = (pred1 == y[test_ix]).mean() - (pred2 == y[test_ix]).mean()
        scores.append(diff)
    scores = np.array(scores)
    _, p_std = ttest_1samp(scores, popmean=0)
    _, p_corr = corrected_t_test(
        scores, n_train=n_train, n_test=n_test, two_tailed=True
    )
    if p_std < alpha:
        standard_rejections += 1
    if p_corr < alpha:
        corrected_rejections += 1

rejection_rate_std = standard_rejections / n_iters
rejection_rate_corr = corrected_rejections / n_iters
print(f"Standard rejection rate: {rejection_rate_std}")
print(f"Corrected rejection rate: {rejection_rate_corr}")

# %%
