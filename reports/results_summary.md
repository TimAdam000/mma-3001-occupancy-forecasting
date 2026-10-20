# MMA3001 Occupancy Forecasting: Saved Results

All numbers below were read from the Colab outputs of the project notebook (Cells 1 to 7). Random Forest results can shift in the third decimal place when the notebook is rerun on a different machine or library version, so if you rerun, update this file from the new outputs.

## Settings used

| Setting | Value |
|---|---|
| Time bin | 15 minutes (binned in UTC, local time = Australia/Melbourne) |
| Occupied label | status = `CurrentlyOccupied` |
| Forecast horizon | 4 bins = 1 hour ahead |
| Gap filling | last status carried forward up to 8 bins (2 hours); longer gaps dropped |
| Split | chronological 60 / 20 / 20 (train / validation / test) |
| Random seed | 42 |
| Features | occupied, headcount, occ_mean_1h, occ_mean_4h, occ_last_week, hour_sin, hour_cos, dayofweek, is_weekend, room_code |
| Tuned Random Forest | 100 trees, max_depth 20, min_samples_leaf 50 |

## Cell 1: Binned occupancy dataset

Final table shape: 274,014 rows x 8 columns.

| Room | Rows (15-min bins) | Occupied rate | First bin (UTC) | Last bin (UTC) |
|---|---|---|---|---|
| G.20 | 49,161 | 0.475 | 2024-03-03 04:00 | 2026-04-22 12:45 |
| G.25 | 68,422 | 0.379 | 2024-03-03 04:00 | 2026-04-22 12:00 |
| unmapped_0f3f37f8 | 67,408 | 0.410 | 2023-11-23 13:30 | 2026-04-22 13:00 |
| unmapped_3e5a14df | 20,162 | 0.604 | 2024-04-30 14:00 | 2024-12-30 12:00 |
| unmapped_941f0352 | 68,861 | 0.412 | 2023-11-23 13:30 | 2026-04-22 13:00 |

## Cell 2: Environmental sensor file and join check

| Sensor ID | Room | First bin (UTC) | Last bin (UTC) | 15-min bins |
|---|---|---|---|---|
| 6012002000227 | none (not in lookup) | 2023-12-11 05:00 | 2023-12-11 05:00 | 1 |
| 6012002000241 | none (not in lookup) | 2023-12-11 05:45 | 2023-12-11 05:45 | 1 |
| 6012002000326 | none (not in lookup) | 2023-12-11 05:00 | 2023-12-11 05:00 | 1 |
| 6012002000777 | none (not in lookup) | 2023-05-05 22:30 | 2023-05-05 22:30 | 1 |
| 6012002000869 | G.38 | 2024-02-29 06:00 | 2024-06-13 17:45 | 9,650 |

- Rooms with both environmental and occupancy data: **none** (`set()`).
- Duplicate ID in the lookup spreadsheet: 6012003000691 is assigned to both G.52 (ES2G14) and G.61 (ES2G01).
- Conclusion: environmental features cannot be joined to occupancy labels with this dataset, so the project uses occupancy-only forecasting.

## Cell 3: Split and baselines

| Block | Rows | Dates | Occupied rate (target) |
|---|---|---|---|
| Train | 155,801 | 2023-11-23 to 2025-04-25 | 46.05% |
| Validation | 51,935 | 2025-04-25 to 2025-10-16 | 44.06% |
| Test | 51,933 | 2025-10-16 to 2026-04-22 | 38.96% |

Baselines on the validation set:

| Baseline | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| A: majority class | 0.000 | 0.000 | 0.000 | 0.559 |
| B: persistence (same as now) | 0.720 | 0.735 | 0.727 | 0.757 |
| C: hour-of-week rule | 0.716 | 0.682 | 0.699 | 0.741 |

## Cell 4: Random Forest tuning, test results, importance

Top 8 of 12 settings tried (scored on validation only):

| n_estimators | max_depth | min_samples_leaf | Validation F1 | Train F1 | Fit time (s) |
|---|---|---|---|---|---|
| 100 | 20 | 50 | 0.7641 | 0.7815 | 12.8 |
| 200 | 20 | 50 | 0.7638 | 0.7814 | 22.6 |
| 200 | 12 | 10 | 0.7635 | 0.7922 | 25.7 |
| 100 | 12 | 10 | 0.7635 | 0.7925 | 9.9 |
| 100 | 12 | 50 | 0.7623 | 0.7799 | 12.4 |
| 200 | 12 | 50 | 0.7623 | 0.7800 | 21.1 |
| 100 | 20 | 10 | 0.7619 | 0.8078 | 15.6 |
| 200 | 20 | 10 | 0.7619 | 0.8083 | 37.3 |

Best settings: 100 trees, max_depth 20, min_samples_leaf 50.

**Test set (final, unseen later period, Oct 2025 to Apr 2026):**

| Model | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| A: majority class | 0.000 | 0.000 | 0.000 | 0.610 |
| B: persistence (same as now) | 0.683 | 0.698 | 0.691 | 0.756 |
| C: hour-of-week rule | 0.634 | 0.694 | 0.662 | 0.725 |
| D: logistic regression | 0.725 | 0.700 | 0.712 | 0.780 |
| **Random Forest** | **0.730** | **0.723** | **0.727** | **0.788** |

Overfitting check (Random Forest F1): train 0.781, validation 0.764, test 0.727.

Confusion matrix on the test set (rows = actual, columns = predicted):

| | Predicted unoccupied | Predicted occupied |
|---|---|---|
| Actual unoccupied | 26,303 | 5,396 |
| Actual occupied | 5,612 | 14,622 |

Permutation importance (drop in validation F1 when the feature is scrambled):

| Feature | Importance |
|---|---|
| hour_cos | 0.0604 |
| occ_mean_1h | 0.0260 |
| occupied | 0.0155 |
| occ_mean_4h | 0.0094 |
| headcount | 0.0086 |
| hour_sin | 0.0076 |
| room_code | 0.0035 |
| occ_last_week | 0.0021 |
| dayofweek | 0.0021 |
| is_weekend | 0.0017 |

## Cell 5: Error analysis and feature-removal sensitivity

Test-set performance by room (room names inferred from the alphabetical order of the output; `unmapped_3e5a14df` has no test rows because it stopped reporting in Dec 2024):

| Room | Test rows | Occupied rate | RF F1 | Persistence F1 | Error rate |
|---|---|---|---|---|---|
| G.20 | 10,294 | 0.498 | 0.789 | 0.765 | 0.212 |
| G.25 | 16,552 | 0.320 | 0.509 | 0.539 | 0.281 |
| unmapped_0f3f37f8 | 11,631 | 0.370 | 0.758 | 0.682 | 0.190 |
| unmapped_941f0352 | 13,456 | 0.409 | 0.827 | 0.773 | 0.145 |

Error by hour and weekday (read from the charts, approximate): error peaks around 7 to 8 am and 5 pm local time. Weekday error is about 0.22 and weekend error is about 0.18.

Feature-group removal (retrained with the best settings, scored on validation):

| Feature group removed | Validation F1 | Change | Fit time (s) |
|---|---|---|---|
| none (all features) | 0.7641 | 0.0000 | 10.8 |
| current state (occupied, headcount) | 0.7573 | -0.0069 | 9.5 |
| rolling history (1 h, 4 h) | 0.7522 | -0.0119 | 6.9 |
| same time last week | 0.7633 | -0.0008 | 12.0 |
| time of day/week | 0.7384 | -0.0258 | 5.1 |
| room identity | 0.7607 | -0.0035 | 12.3 |

## Cell 6: Exploration (status mix per room)

| Room | NotOccupied | RecentlyOccupied | CurrentlyOccupied |
|---|---|---|---|
| G.20 | 0.505 | 0.020 | 0.475 |
| G.25 | 0.581 | 0.040 | 0.379 |
| unmapped_0f3f37f8 | 0.535 | 0.055 | 0.410 |
| unmapped_3e5a14df | 0.364 | 0.032 | 0.604 |
| unmapped_941f0352 | 0.553 | 0.035 | 0.412 |

Plot observations: weekdays about 9 am to 5 pm are the most occupied; weekends are much lower; nights are near empty. G.25 has the flattest daily profile (about 0.23 to 0.25 overnight, lowest daytime peak near 0.6). Headcount is mostly 1 to 3. The monthly occupied rate varies between about 0.27 and 0.56 with no clear semester pattern.

## Cell 7: Sensitivity runs and runtime

Forecast horizon (validation set):

| Horizon | RF F1 | Persistence F1 | RF gain | Validation rows |
|---|---|---|---|---|
| 15 min | 0.810 | 0.793 | 0.017 | 53,990 |
| 1 hour | 0.764 | 0.727 | 0.037 | 51,935 |
| 4 hours | 0.722 | 0.579 | 0.143 | 47,700 |

Label definition, 1 hour ahead (validation set):

| Occupied means | Occupied rate | Precision | Recall | RF F1 | Persistence F1 |
|---|---|---|---|---|---|
| CurrentlyOccupied only | 0.441 | 0.771 | 0.757 | 0.764 | 0.727 |
| Currently or RecentlyOccupied | 0.482 | 0.791 | 0.782 | 0.786 | 0.752 |

Runtime and model size (validation set; timings vary on shared Colab hardware):

| Model | Fit time (s) | Predict time (s) | Microseconds per row | Model size (MB) | Validation F1 |
|---|---|---|---|---|---|
| Random Forest, 10 trees | 1.42 | 0.089 | 1.72 | 2.07 | 0.7635 |
| Random Forest, 50 trees | 6.11 | 0.217 | 4.17 | 10.30 | 0.7638 |
| Random Forest, 100 trees | 8.22 | 0.710 | 13.68 | 20.55 | 0.7641 |
| Random Forest, 200 trees | 22.07 | 0.819 | 15.76 | 41.14 | 0.7638 |
| Logistic regression | 0.24 | 0.010 | 0.20 | 0.0016 | 0.7552 |

Binned occupancy table in memory: 55.2 MB (274,014 rows), compared with a raw event log of 9,091,732 rows (about 1.6 GB on disk).

## Caveats to state in the report

- Results apply to these rooms in this building; the test period is one later block of time.
- Environmental sensors could not be linked to any occupancy room, so no conclusion is possible about temperature, humidity or CO2 as predictors.
- Three of the five occupancy spaces have no room name in the lookup.
- The sensor labels are treated as ground truth; G.25's overnight occupancy is the main doubt.
- The occupied rate falls from 46% (train) to 39% (test), which explains part of the drop in F1 between train and test.
- The Random Forest beats persistence by 0.036 F1 and logistic regression by 0.015 on the test set, so the improvement is real but modest.
