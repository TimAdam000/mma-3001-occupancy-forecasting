"""Build the forecasting target, past-only features and the chronological split."""
import numpy as np
import pandas as pd

#: Model input columns. All are computed from information available at the forecast time.
FEATURES = ["occupied", "headcount", "occ_mean_1h", "occ_mean_4h", "occ_last_week",
            "hour_sin", "hour_cos", "dayofweek", "is_weekend", "room_code"]


def make_features(occ, horizon_bins=4, bin_freq="15min"):
    """Add the target and features to a regular occupancy table.

    The target is the occupancy ``horizon_bins`` bins ahead. It is matched on real
    timestamps, so a gap in the data gives a missing target and the row is dropped, rather
    than silently pairing the wrong row. Features use the present and past only (no leakage):
    current state, rolling means over the last 1 h and 4 h, the same time one week earlier
    (``-1`` if unknown), and sine/cosine of local time of day, day of week and room.

    Parameters
    ----------
    occ : pandas.DataFrame
        Needs columns ``room, timestamp, local, occupied, headcount``.
    horizon_bins : int
        Forecast horizon in bins (4 x 15 min = 1 hour).
    bin_freq : str
        Bin width, must match the table.

    Returns
    -------
    pandas.DataFrame
        Rows with a known target, with ``target`` and every column in :data:`FEATURES`.
    """
    step = pd.Timedelta(bin_freq)
    d = (occ[["room", "timestamp", "local", "occupied", "headcount"]]
         .sort_values(["room", "timestamp"]).reset_index(drop=True))

    tgt = d[["room", "timestamp", "occupied"]].copy()
    tgt["timestamp"] -= horizon_bins * step
    d = d.merge(tgt.rename(columns={"occupied": "target"}), on=["room", "timestamp"], how="left")

    wk = d[["room", "timestamp", "occupied"]].copy()
    wk["timestamp"] += pd.Timedelta(days=7)
    d = d.merge(wk.rename(columns={"occupied": "occ_last_week"}), on=["room", "timestamp"], how="left")

    d["occ_mean_1h"] = d.groupby("room")["occupied"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    d["occ_mean_4h"] = d.groupby("room")["occupied"].transform(lambda s: s.rolling(16, min_periods=1).mean())

    hr = d["local"].dt.hour + d["local"].dt.minute / 60
    d["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    d["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    d["hour"] = d["local"].dt.hour
    d["dayofweek"] = d["local"].dt.dayofweek
    d["is_weekend"] = (d["dayofweek"] >= 5).astype(int)
    d["room_code"] = d["room"].astype("category").cat.codes

    d = d.dropna(subset=["target"] + [f for f in FEATURES if f != "occ_last_week"]).reset_index(drop=True)
    d["target"] = d["target"].astype(int)
    d["occ_last_week"] = d["occ_last_week"].fillna(-1)
    return d


def chronological_split(d, fractions=(0.6, 0.8)):
    """Split by time, never at random: the model is tested on a later period than it trained on.

    Parameters
    ----------
    d : pandas.DataFrame
        Needs a ``timestamp`` column.
    fractions : tuple of float
        Timestamp quantiles ending train and validation (0.6, 0.8 gives 60/20/20).

    Returns
    -------
    train, val, test : pandas.DataFrame
    """
    t1, t2 = d["timestamp"].quantile(list(fractions))
    return (d[d.timestamp <= t1],
            d[(d.timestamp > t1) & (d.timestamp <= t2)],
            d[d.timestamp > t2])


def build_dataset(occ, horizon_bins=4, include_recent=False, bin_freq="15min"):
    """Features, target and 60/20/20 split from the binned table, in one call.

    Parameters
    ----------
    occ : pandas.DataFrame
        Output of ``load_occupancy`` (needs a ``status`` column of 0/1/2).
    horizon_bins : int
        Forecast horizon in bins.
    include_recent : bool
        If True, ``RecentlyOccupied`` also counts as occupied.
    bin_freq : str
        Bin width.

    Returns
    -------
    train, val, test : pandas.DataFrame
    """
    x = occ[["room", "timestamp", "local", "status", "headcount"]].copy()
    x["occupied"] = (x["status"] >= 1).astype(int) if include_recent else (x["status"] == 2).astype(int)
    return chronological_split(make_features(x, horizon_bins, bin_freq))
