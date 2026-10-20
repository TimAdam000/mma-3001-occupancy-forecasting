"""Tests for occupancy.features: target alignment, leakage, split."""
import numpy as np
import pandas as pd

from occupancy.features import FEATURES, build_dataset, chronological_split, make_features


def test_target_is_occupancy_one_hour_later(occ_table):
    d = make_features(occ_table, horizon_bins=4)
    lookup = occ_table.set_index(["room", "timestamp"])["occupied"]
    for _, r in d.sample(50, random_state=0).iterrows():
        assert r["target"] == lookup[(r["room"], r["timestamp"] + pd.Timedelta(hours=1))]


def test_gap_gives_dropped_row_not_wrong_target(occ_table):
    gapped = occ_table.drop(occ_table.index[(occ_table.room == "R1") &
                                            (occ_table.timestamp == occ_table.timestamp.iloc[200])])
    d = make_features(gapped)
    missing = occ_table.timestamp.iloc[200] - pd.Timedelta(hours=1)
    assert d[(d.room == "R1") & (d.timestamp == missing)].empty


def test_features_present_and_no_nans(occ_table):
    d = make_features(occ_table)
    assert set(FEATURES + ["target"]) <= set(d.columns)
    assert not d[FEATURES + ["target"]].isna().any().any()


def test_features_use_only_the_past(occ_table):
    """Changing the future must not change today's features (no leakage)."""
    base = make_features(occ_table).set_index(["room", "timestamp"])
    changed = occ_table.copy()
    cutoff = changed.timestamp.iloc[500]
    changed.loc[changed.timestamp > cutoff, ["occupied", "headcount"]] = 1
    new = make_features(changed).set_index(["room", "timestamp"])
    past = base.index[base.index.get_level_values("timestamp") <= cutoff]
    cols = ["occupied", "headcount", "occ_mean_1h", "occ_mean_4h", "occ_last_week"]
    pd.testing.assert_frame_equal(base.loc[past, cols], new.loc[past, cols])


def test_last_week_feature_is_minus_one_when_unknown(occ_table):
    d = make_features(occ_table)
    first_week = d[d.timestamp < occ_table.timestamp.min() + pd.Timedelta(days=7)]
    assert (first_week["occ_last_week"] == -1).all()
    assert d["occ_last_week"].isin([-1, 0, 1]).all()


def test_time_features_use_local_time(occ_table):
    d = make_features(occ_table)
    assert np.allclose(d["hour_sin"] ** 2 + d["hour_cos"] ** 2, 1)
    assert (d["is_weekend"] == (d["dayofweek"] >= 5)).all()
    assert (d["hour"] == d["local"].dt.hour).all()


def test_chronological_split_is_ordered_and_60_20_20(occ_table):
    d = make_features(occ_table)
    tr, va, te = chronological_split(d)
    assert tr.timestamp.max() < va.timestamp.min() <= va.timestamp.max() < te.timestamp.min()
    assert len(tr) + len(va) + len(te) == len(d)
    assert abs(len(tr) / len(d) - 0.6) < 0.02 and abs(len(va) / len(d) - 0.2) < 0.02


def test_build_dataset_matches_manual_pipeline(occ_table):
    tr, va, te = build_dataset(occ_table.assign(occupied=-9))   # occupied column is recomputed
    tr2, va2, te2 = chronological_split(make_features(occ_table))
    assert (len(tr), len(va), len(te)) == (len(tr2), len(va2), len(te2))


def test_include_recent_changes_label():
    ts = pd.date_range("2024-06-03", periods=3 * 96, freq="15min", tz="UTC")
    occ = pd.DataFrame({"room": "R", "timestamp": ts, "local": ts.tz_convert("Australia/Melbourne"),
                        "status": 1.0, "headcount": 1.0})
    only_current = build_dataset(occ, include_recent=False)
    with_recent = build_dataset(occ, include_recent=True)
    assert sum(p.target.sum() for p in only_current) == 0
    assert all(p.target.eq(1).all() for p in with_recent)
