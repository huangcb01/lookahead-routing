from collections import defaultdict
import math
import numpy as np
import pandas as pd
from typing import Any, Sequence
from sklearn.linear_model import LogisticRegression


def mean(x: Sequence[float]) -> float:
    """Calculate the mean of a sequence of numbers.

    Args:
        x: The sequence of numbers.

    Returns:
        The mean of the sequence.
    """
    return sum(x) / len(x) if len(x) > 0 else 0


def argmax(x: Sequence[Any]) -> int:
    """Find the index of the maximum value in a sequence of numbers.

    Args:
        x: The sequence of numbers.

    Returns:
        The index of the maximum value.
    """
    return max(range(len(x)), key=lambda i: x[i])


def argmin(x: Sequence[Any]) -> int:
    """Find the index of the minimum value in a sequence of numbers.

    Args:
        x: The sequence of numbers.

    Returns:
        The index of the minimum value.
    """
    return min(range(len(x)), key=lambda i: x[i])


def compute_mle_elo(df, SCALE=400, BASE=10, INIT_RATING=1000, baseline_model="gpt-4-0314"):
    models = pd.concat([df["model_a"], df["model_b"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)

    # duplicate battles
    df = pd.concat([df, df], ignore_index=True)
    p = len(models.index)
    n = df.shape[0]

    X = np.zeros([n, p])
    X[np.arange(n), models[df["model_a"]]] = +math.log(BASE)
    X[np.arange(n), models[df["model_b"]]] = -math.log(BASE)

    # one A win => two A win
    Y = np.zeros(n)
    Y[df["winner"] == "model_a"] = 1.0

    # one tie => one A win + one B win
    # find tie + tie (both bad) index
    tie_idx = (df["winner"] == "tie") | (df["winner"] == "tie (bothbad)")
    tie_idx[len(tie_idx)//2:] = False
    Y[tie_idx] = 1.0

    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-8)
    lr.fit(X,Y)

    elo_scores = SCALE * lr.coef_[0] + INIT_RATING

    # set anchor as gpt-4-0314 = 1000
    if baseline_model in models.index:
        elo_scores += 1000 - elo_scores[models[baseline_model]]
    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)


def predict_win_rate(elo_ratings, SCALE=400, BASE=10, INIT_RATING=1000):
    names = sorted(list(elo_ratings.keys()))
    wins = defaultdict(lambda: defaultdict(lambda: 0))
    for a in names:
        for b in names:
            ea = 1 / (1 + BASE ** ((elo_ratings[b] - elo_ratings[a]) / SCALE))
            wins[a][b] = ea
            wins[b][a] = 1 - ea

    data = {
        a: [wins[a][b] if a != b else np.NAN for b in names]
        for a in names
    }

    df = pd.DataFrame(data, index=names)
    df.index.name = "model_a"
    df.columns.name = "model_b"
    return df.T
