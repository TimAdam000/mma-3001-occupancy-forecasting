"""Shared fixtures: a tiny synthetic occupancy log so the tests need no real data."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_events():
    """Raw event rows in the same format as the occupancy CSV (one space, UTC times)."""
    return pd.DataFrame({
        "floorspaceid": ["A"] * 5,
        "occupancystatus": ["NotOccupied", "CurrentlyOccupied", "CurrentlyOccupied",
                            "RecentlyOccupied", "Bogus"],
        "headcount": [0, 2, 3, 1, 9],
        "collecteddate": ["2024-05-01T00:01:00Z", "2024-05-01T00:05:00Z",
                          "2024-05-01T00:10:00Z", "2024-05-01T00:20:00Z",
                          "2024-05-01T00:25:00Z"],
    })


@pytest.fixture
def occ_table():
    """Binned table for two rooms over 3 weeks, occupied 9:00-17:00 local on weekdays."""
    ts = pd.date_range("2024-06-03", periods=21 * 96, freq="15min", tz="UTC")  # starts Monday
    frames = []
    for room in ("R1", "R2"):
        local = ts.tz_convert("Australia/Melbourne")
        busy = (local.dayofweek < 5) & (local.hour >= 9) & (local.hour < 17)
        frames.append(pd.DataFrame({
            "room": room, "timestamp": ts, "local": local,
            "status": np.where(busy, 2, 0).astype(float),
            "headcount": np.where(busy, 2.0, 0.0),
            "occupied": busy.astype(int)}))
    return pd.concat(frames, ignore_index=True)
