"""Tests for occupancy.data: binning, gap filling and JSON parsing."""
import pandas as pd

from occupancy.data import (STATUS_CODE, aggregate_chunk, combine_chunks,
                            fill_time_grid, parse_env_json)


def test_aggregate_chunk_takes_last_status_and_max_headcount(raw_events):
    out = aggregate_chunk(raw_events)
    first = out[out["bin"] == pd.Timestamp("2024-05-01 00:00", tz="UTC")].iloc[0]
    assert first["status"] == STATUS_CODE["CurrentlyOccupied"]   # last event in the bin
    assert first["headcount"] == 3                               # max of 0, 2, 3
    assert first["n_events"] == 3


def test_aggregate_chunk_drops_unknown_status(raw_events):
    out = aggregate_chunk(raw_events)
    assert out["n_events"].sum() == 4          # the "Bogus" row is gone


def test_aggregate_chunk_drops_unparseable_time(raw_events):
    raw_events.loc[0, "collecteddate"] = "not a date"
    assert aggregate_chunk(raw_events)["n_events"].sum() == 3


def test_combine_chunks_merges_a_bin_split_across_chunks(raw_events):
    a, b = aggregate_chunk(raw_events.iloc[:2]), aggregate_chunk(raw_events.iloc[2:4])
    out = combine_chunks([a, b])
    row = out[out["bin"] == pd.Timestamp("2024-05-01 00:00", tz="UTC")].iloc[0]
    assert row["n_events"] == 3 and row["headcount"] == 3


def test_fill_time_grid_carries_status_forward_at_most_limit():
    bins = pd.to_datetime(["2024-05-01 00:00", "2024-05-01 05:00"], utc=True)  # 5 h gap
    occ = pd.DataFrame({"floorspaceid": "A", "bin": bins, "status": [2.0, 0.0],
                        "headcount": [1.0, 0.0], "n_events": [1, 1]})
    g = fill_time_grid(occ, "15min", max_ffill_bins=8).set_index("timestamp")
    assert g.loc["2024-05-01 02:00", "status"] == 2.0         # 8 bins = 2 h: still filled
    assert pd.isna(g.loc["2024-05-01 02:15", "status"])       # beyond the limit: left empty
    assert g.loc["2024-05-01 05:00", "status"] == 0.0


def test_fill_time_grid_is_regular_and_per_space():
    bins = pd.to_datetime(["2024-05-01 00:00", "2024-05-01 01:00"], utc=True)
    occ = pd.concat([pd.DataFrame({"floorspaceid": s, "bin": bins, "status": [0.0, 0.0],
                                   "headcount": [0.0, 0.0], "n_events": [1, 1]}) for s in "AB"])
    g = fill_time_grid(occ)
    assert len(g) == 2 * 5                                    # 5 bins per space
    assert g.groupby("floorspaceid")["timestamp"].apply(lambda s: s.diff().dropna().nunique()).eq(1).all()


def test_parse_env_json_good_and_bad():
    js = '[{"variable": {"name": "Temp"}, "values": [{"value": 21.5, "time": 1700000000}]}]'
    assert parse_env_json(js) == {"Temp": 21.5, "_t": 1700000000}
    assert parse_env_json("{broken") is None
    assert parse_env_json(None) is None
