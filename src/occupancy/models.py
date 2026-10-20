"""Baselines, Random Forest tuning, metrics and feature-group sensitivity."""
import itertools
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score


def scores(y, p):
    """Precision, recall, F1 and accuracy for binary labels.

    Parameters
    ----------
    y, p : array-like of int
        True and predicted labels (0/1). Undefined ratios are scored as 0.

    Returns
    -------
    dict
        Keys ``P, R, F1, Acc``.
    """
    y, p = np.asarray(y), np.asarray(p)
    return dict(P=precision_score(y, p, zero_division=0), R=recall_score(y, p, zero_division=0),
                F1=f1_score(y, p, zero_division=0), Acc=float((y == p).mean()))


def majority_baseline(train, n):
    """Baseline A: always predict the most common training label. Returns ``n`` predictions."""
    return np.full(n, train["target"].mode()[0])


def persistence_baseline(df):
    """Baseline B: predict that the room stays as it is now (``occupied`` column)."""
    return df["occupied"].to_numpy()


def hour_of_week_table(train):
    """Baseline C lookup: for each hour of the week, 1 if the room was occupied >= 50% of the time.

    Returns
    -------
    pandas.Series
        Index ``dayofweek * 24 + hour``, values 0/1.
    """
    return (train.assign(k=train.dayofweek * 24 + train.hour)
            .groupby("k").target.mean().ge(0.5).astype(int))


def predict_hour_of_week(df, table):
    """Apply the :func:`hour_of_week_table` rule; hours never seen in training predict 0."""
    return (df.dayofweek * 24 + df.hour).map(table).fillna(0).astype(int).to_numpy()


def tune_random_forest(train, val, features, grid=None, random_state=42):
    """Try Random Forest settings, scoring on the validation set only.

    Parameters
    ----------
    train, val : pandas.DataFrame
        With a ``target`` column and every column in ``features``.
    features : list of str
    grid : iterable of (n_estimators, max_depth, min_samples_leaf), optional
        Default is the 12 settings used in the project.
    random_state : int
        Seed, so results repeat.

    Returns
    -------
    pandas.DataFrame
        One row per setting with ``val_F1``, ``train_F1`` and ``fit_s``, best first.
    """
    if grid is None:
        grid = itertools.product([100, 200], [6, 12, 20], [10, 50])
    rows = []
    for n, md, leaf in grid:
        t0 = time.time()
        rf = RandomForestClassifier(n_estimators=n, max_depth=md, min_samples_leaf=leaf,
                                    n_jobs=-1, random_state=random_state).fit(train[features], train["target"])
        rows.append(dict(n_estimators=n, max_depth=md, min_samples_leaf=leaf,
                         val_F1=f1_score(val["target"], rf.predict(val[features])),
                         train_F1=f1_score(train["target"], rf.predict(train[features])),
                         fit_s=time.time() - t0))
    return pd.DataFrame(rows).sort_values("val_F1", ascending=False).reset_index(drop=True)


def feature_group_sensitivity(train, val, features, groups, params, random_state=42):
    """Retrain with one group of features removed at a time, scored on validation.

    Parameters
    ----------
    groups : dict
        ``{group name: [columns]}``.
    params : dict
        Random Forest settings (``n_estimators, max_depth, min_samples_leaf``).

    Returns
    -------
    pandas.DataFrame
        ``removed, val_F1, change`` (vs all features) and ``fit_s``.
    """
    def fit_eval(feats):
        t0 = time.time()
        m = RandomForestClassifier(**params, n_jobs=-1, random_state=random_state).fit(train[feats], train["target"])
        return f1_score(val["target"], m.predict(val[feats])), time.time() - t0

    full_f1, full_t = fit_eval(features)
    rows = [dict(removed="(none: all features)", val_F1=full_f1, change=0.0, fit_s=full_t)]
    for name, cols in groups.items():
        f, s = fit_eval([c for c in features if c not in cols])
        rows.append(dict(removed=name, val_F1=f, change=f - full_f1, fit_s=s))
    return pd.DataFrame(rows)
