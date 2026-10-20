"""Occupancy forecasting package (MMA3001 individual project).

Forecast whether a room will be occupied one hour ahead from a sensor event log.

Modules
-------
data      load the raw event log, aggregate it to 15-minute bins, parse environmental JSON
features  build the target, past-only features and the chronological train/val/test split
models    baselines, Random Forest tuning, metrics and feature-group sensitivity
"""
from .data import (STATUS_CODE, aggregate_chunk, combine_chunks, fill_time_grid,
                   load_occupancy, load_space_lookup, parse_env_json)
from .features import FEATURES, build_dataset, chronological_split, make_features
from .models import (hour_of_week_table, majority_baseline, persistence_baseline,
                     predict_hour_of_week, scores, tune_random_forest,
                     feature_group_sensitivity)

__all__ = [
    "STATUS_CODE", "aggregate_chunk", "combine_chunks", "fill_time_grid", "load_occupancy",
    "load_space_lookup", "parse_env_json", "FEATURES", "build_dataset", "chronological_split",
    "make_features", "hour_of_week_table", "majority_baseline", "persistence_baseline",
    "predict_hour_of_week", "scores", "tune_random_forest", "feature_group_sensitivity",
]
