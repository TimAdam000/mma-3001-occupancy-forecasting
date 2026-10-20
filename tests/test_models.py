"""Tests for occupancy.models: metrics, baselines, tuning, sensitivity."""
import numpy as np
import pandas as pd
import pytest

from occupancy.features import FEATURES, chronological_split, make_features
from occupancy.models import (feature_group_sensitivity, hour_of_week_table, majority_baseline,
                              persistence_baseline, predict_hour_of_week, scores,
                              tune_random_forest)


@pytest.fixture
def splits(occ_table):
    return chronological_split(make_features(occ_table))


def test_scores_known_values():
    s = scores([1, 1, 0, 0], [1, 0, 0, 1])
    assert s["P"] == 0.5 and s["R"] == 0.5 and s["F1"] == 0.5 and s["Acc"] == 0.5


def test_scores_all_negative_predictions_do_not_crash():
    s = scores([1, 0, 1], [0, 0, 0])
    assert s["P"] == 0 and s["R"] == 0 and s["F1"] == 0


def test_majority_baseline_predicts_most_common_label(splits):
    train, val, _ = splits
    p = majority_baseline(train, len(val))
    assert len(p) == len(val) and set(p) == {train.target.mode()[0]}


def test_persistence_copies_current_state(splits):
    _, val, _ = splits
    assert (persistence_baseline(val) == val["occupied"].to_numpy()).all()


def test_hour_of_week_rule_learns_the_weekday_pattern(splits):
    train, val, _ = splits
    table = hour_of_week_table(train)
    pred = predict_hour_of_week(val, table)
    assert scores(val["target"], pred)["Acc"] > 0.95      # the synthetic pattern is exactly weekly


def test_hour_of_week_unseen_hours_predict_zero(splits):
    train, _, _ = splits
    table = hour_of_week_table(train)
    unseen = pd.DataFrame({"dayofweek": [6], "hour": [3]}).assign(dayofweek=99)
    assert predict_hour_of_week(unseen, table).tolist() == [0]


def test_tune_random_forest_returns_sorted_results(splits):
    train, val, _ = splits
    res = tune_random_forest(train, val, FEATURES, grid=[(10, 3, 10), (10, 6, 10)], random_state=0)
    assert len(res) == 2
    assert res["val_F1"].is_monotonic_decreasing
    assert {"val_F1", "train_F1", "fit_s"} <= set(res.columns)


def test_tuning_is_repeatable_with_same_seed(splits):
    train, val, _ = splits
    a = tune_random_forest(train, val, FEATURES, grid=[(10, 4, 10)], random_state=1)["val_F1"].iloc[0]
    b = tune_random_forest(train, val, FEATURES, grid=[(10, 4, 10)], random_state=1)["val_F1"].iloc[0]
    assert a == b


def test_feature_group_sensitivity_structure(splits):
    train, val, _ = splits
    out = feature_group_sensitivity(train, val, FEATURES, {"room": ["room_code"], "time": ["hour_sin", "hour_cos"]},
                                    dict(n_estimators=10, max_depth=4, min_samples_leaf=10), random_state=0)
    assert out["removed"].tolist() == ["(none: all features)", "room", "time"]
    assert out.loc[0, "change"] == 0
    assert np.isclose(out["val_F1"] - out.loc[0, "val_F1"], out["change"]).all()
