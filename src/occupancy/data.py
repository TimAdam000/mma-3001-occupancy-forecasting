"""Load the occupancy event log and turn it into a regular 15-minute table per room.

The raw file has about 9 million rows (one per sensor event), so it is read in chunks and
aggregated to time bins as it is read. Bins are in UTC; local time (Australia/Melbourne)
is only added afterwards, which avoids daylight-saving ambiguity.
"""
import json
import os

import pandas as pd

#: Numeric code for each sensor status string.
STATUS_CODE = {"NotOccupied": 0, "RecentlyOccupied": 1, "CurrentlyOccupied": 2}


def load_space_lookup(map_path):
    """Read the sensor lookup spreadsheet.

    Parameters
    ----------
    map_path : str
        Path to ``Sensor ID and Locations.xlsx``. Sheet 1 maps environmental sensors to
        rooms, sheet 2 maps occupancy ``SpaceId`` values to rooms.

    Returns
    -------
    env_map : pandas.DataFrame
        Environmental sensor lookup (first sheet).
    space_to_room : dict
        ``{SpaceId: room name}`` for the occupancy sensors.
    """
    sheets = pd.read_excel(map_path, sheet_name=None, dtype=str)
    env_map, occ_map = list(sheets.values())[0], list(sheets.values())[1]
    space_to_room = (occ_map.dropna(subset=["SpaceId"])
                     .groupby("SpaceId")["Floor/Room"].first().to_dict())
    return env_map, space_to_room


def aggregate_chunk(chunk, bin_freq="15min"):
    """Aggregate one chunk of raw events to one row per (space, time bin).

    The status kept for a bin is the *last* event in it; headcount is the maximum.

    Parameters
    ----------
    chunk : pandas.DataFrame
        Columns ``floorspaceid``, ``occupancystatus``, ``headcount``, ``collecteddate``.
    bin_freq : str
        Pandas frequency string for the bin width.

    Returns
    -------
    pandas.DataFrame
        Columns ``floorspaceid, bin, status, headcount, n_events, t_last``.
        Rows with an unparseable time or an unknown status are dropped.
    """
    t = pd.to_datetime(chunk["collecteddate"], utc=True, errors="coerce")
    ch = chunk.assign(t=t).dropna(subset=["t"])
    ch["bin"] = ch["t"].dt.floor(bin_freq)
    ch["status"] = ch["occupancystatus"].map(STATUS_CODE)
    ch = ch.dropna(subset=["status"]).sort_values("t")
    return ch.groupby(["floorspaceid", "bin"]).agg(
        status=("status", "last"), headcount=("headcount", "max"),
        n_events=("status", "size"), t_last=("t", "max")).reset_index()


def combine_chunks(parts):
    """Merge the per-chunk aggregates (a bin can be split across two chunks).

    Parameters
    ----------
    parts : list of pandas.DataFrame
        Outputs of :func:`aggregate_chunk`.

    Returns
    -------
    pandas.DataFrame
        One row per (space, bin): last status, maximum headcount, total event count.
    """
    occ = pd.concat(parts).sort_values("t_last")
    return occ.groupby(["floorspaceid", "bin"]).agg(
        status=("status", "last"), headcount=("headcount", "max"),
        n_events=("n_events", "sum")).reset_index()


def fill_time_grid(occ, bin_freq="15min", max_ffill_bins=8):
    """Put each space on a regular time grid, carrying the last status over short gaps.

    A sensor only reports when something changes, so empty bins are normal. The last known
    status and headcount are carried forward for at most ``max_ffill_bins`` bins (8 x 15 min
    = 2 h). Longer gaps stay empty and are dropped, rather than guessed.

    Parameters
    ----------
    occ : pandas.DataFrame
        Output of :func:`combine_chunks`.
    bin_freq : str
        Bin width.
    max_ffill_bins : int
        Maximum number of consecutive empty bins to fill.

    Returns
    -------
    pandas.DataFrame
        Columns ``timestamp, status, headcount, n_events, floorspaceid``.
    """
    grids = []
    for sid, g in occ.groupby("floorspaceid"):
        full = pd.date_range(g["bin"].min(), g["bin"].max(), freq=bin_freq)
        g = g.set_index("bin").reindex(full)
        g.index.name = "timestamp"
        g["n_events"] = g["n_events"].fillna(0)
        g[["status", "headcount"]] = g[["status", "headcount"]].ffill(limit=max_ffill_bins)
        g["floorspaceid"] = sid
        grids.append(g.reset_index())
    return pd.concat(grids, ignore_index=True)


def load_occupancy(occ_path, space_to_room, bin_freq="15min", tz="Australia/Melbourne",
                   occupied_states=("CurrentlyOccupied",), max_ffill_bins=8,
                   chunksize=1_000_000):
    """Full pipeline: raw event log -> regular 15-minute occupancy table per room.

    Parameters
    ----------
    occ_path : str
        Path to the occupancy CSV.
    space_to_room : dict
        ``{SpaceId: room}`` from :func:`load_space_lookup`. Unknown spaces are named
        ``unmapped_<first 8 characters of the id>``.
    bin_freq, tz, max_ffill_bins, chunksize
        Bin width, local time zone, maximum gap to fill, rows read per chunk.
    occupied_states : iterable of str
        Statuses that count as occupied (default only ``CurrentlyOccupied``).

    Returns
    -------
    pandas.DataFrame
        Columns include ``timestamp`` (UTC), ``local`` (tz-aware local time), ``room``,
        ``status`` (0/1/2), ``headcount`` and ``occupied`` (0/1).
    """
    reader = pd.read_csv(occ_path,
                         usecols=["floorspaceid", "occupancystatus", "headcount", "collecteddate"],
                         chunksize=chunksize)
    occ = combine_chunks([aggregate_chunk(ch, bin_freq) for ch in reader])
    occ = fill_time_grid(occ, bin_freq, max_ffill_bins)
    occ["room"] = occ["floorspaceid"].map(space_to_room).fillna(
        "unmapped_" + occ["floorspaceid"].str[:8])
    occ = occ.dropna(subset=["status"])
    code_to_name = {v: k for k, v in STATUS_CODE.items()}
    occ["occupied"] = occ["status"].map(code_to_name).isin(set(occupied_states)).astype(int)
    occ["local"] = occ["timestamp"].dt.tz_convert(tz)
    return occ


def parse_env_json(js):
    """Parse the JSON text in the environmental sensor ``jsondata`` column.

    Parameters
    ----------
    js : str
        JSON list of ``{"variable": {"name": ...}, "values": [{"value": ..., "time": ...}]}``.

    Returns
    -------
    dict or None
        ``{variable name: value, "_t": unix time}``, or ``None`` if the text is malformed.
    """
    out = {}
    try:
        for item in json.loads(js):
            v = item["values"][0]
            out[item["variable"]["name"]] = v["value"]
            out["_t"] = v["time"]
    except Exception:
        return None
    return out
