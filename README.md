# Forecasting Room Occupancy One Hour Ahead

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TimAdam000/mma-3001-occupancy-forecasting/blob/main/notebooks/occupancy_forecasting.ipynb)

Individual project for **MMA3001 Numerical Methods and Machine Learning**.
Author: Timothy Adam (34990143)

## Purpose

Building services (HVAC, ventilation and lighting) are usually run on fixed
schedules, so energy is spent conditioning rooms nobody uses. This project asks:

> Can past occupancy readings and time-of-day/week patterns predict whether a
> room will be occupied one hour ahead?

A reliable short-term forecast would let a building management system
pre-condition only the spaces that will be used. The solution is a Random Forest
classifier, compared against simple baselines and logistic regression using a
chronological train / validation / test split.

**Scope limits:** predictions are per room, at 15-minute resolution, for the five
spaces covered by the occupancy sensors in this dataset.

## Data

Monash Smart Infrastructure occupancy and environmental data (Dataset 2),
supplied for MMA3001. The data files are **not included** in this repository
(the occupancy log is about 1.6 GB, over GitHub's file size limit) and are
excluded by `.gitignore`.

| File | Contents |
|---|---|
| `5occupancySensor_MayToDec2024_9MRows.csv` (supplied as `.zip`) | Occupancy event log: 9,091,732 rows covering Nov 2023 to Apr 2026. Columns used: `floorspaceid`, `occupancystatus`, `headcount`, `collecteddate` |
| `5EnvSensor_MayToDec2024_180kRows.csv` (supplied as `.zip`) | Environmental sensor log: 179,447 rows. Readings (temperature, humidity, CO2, etc.) are stored as JSON text in `jsondata` |
| `Sensor ID and Locations.xlsx` | Sensor-to-room lookup (sheets `ENV` and `OCC`) |

**Data limitations found during the project**

- None of the environmental sensors share a room with an occupancy sensor, so
  environmental features cannot be joined to the occupancy labels. The project
  therefore forecasts occupancy from past occupancy and time features only
  (`src/cell2_env_join_check.py` produces the evidence).
- Only 2 of the 5 occupancy spaces map to a room name in the lookup (G.20 and
  G.25). The other three are kept as anonymous rooms (`unmapped_...`).
- The file names say "May to Dec 2024", but the occupancy log covers
  Nov 2023 to Apr 2026. All of it is used.

## Inputs and outputs

| | Description |
|---|---|
| **Input** | Per room, per 15-minute bin: current occupied flag (0/1), headcount (people), rolling mean occupancy over the last 1 h and 4 h, occupancy at the same time one week earlier, hour of day (sine and cosine, local Melbourne time), day of week, weekend flag, room code |
| **Output** | Predicted occupancy (1 = occupied, 0 = unoccupied) for the room one hour later |
| **Label** | Occupied means status `CurrentlyOccupied` (the effect of also counting `RecentlyOccupied` is tested in Cell 7) |
| **Missing data** | The last known status is carried forward up to 2 hours; longer gaps are dropped, not guessed. An unknown last-week value is coded as -1 |

Only information available at the prediction time is used as input, and the
target is strictly in the future, which prevents data leakage.

## Environment and setup

Developed and tested in **Google Colab** (Python 3.13). Dependencies are listed in
`requirements.txt`: pandas, numpy, scikit-learn, matplotlib, seaborn, openpyxl,
plus pytest, pytest-html and pdoc for testing and documentation.

## How to run (reproducible workflow)

1. Open a new Colab notebook.
2. Upload the three data files into Colab's `/content` folder (sidebar folder
   icon, then the upload icon). Do not place them inside the repo folder.
3. Clone the repo and install the dependencies:

```python
   !git clone https://github.com/TimAdam000/mma-3001-occupancy-forecasting.git
   %cd /content/mma-3001-occupancy-forecasting
   !pip install -q -r requirements.txt
   %matplotlib inline
```

4. Run the scripts in order, one per cell. The `-i` flag is required because the
   scripts share variables with each other:

```python
   %run -i src/cell0_unzip_data.py            # only if the data files are .zip
   %run -i src/cell1_load_occupancy.py        # load and bin the occupancy log
   %run -i src/cell2_env_join_check.py        # parse env data, check join with occupancy
   %run -i src/cell3_features_baselines.py    # features, chronological split, baselines
   %run -i src/cell4_random_forest.py         # tune, train, test, feature importance
   %run -i src/cell5_error_sensitivity.py     # error analysis, feature-removal sensitivity
   %run -i src/cell6_exploration.py           # exploratory plots
   %run -i src/cell7_horizon_label_runtime.py # horizon, label definition, runtime
```

The full notebook is `notebooks/occupancy_forecasting.ipynb`. Cell 1 takes a few
minutes because it reads the 9-million-row log in chunks. Cells 4 and 7 each
train several models.

**Checks that the run worked**

| Script | Expected output |
|---|---|
| Cell 1 | `(274014, 8)` and five rooms in the table |
| Cell 3 | Train / validation / test rows: 155,801 / 51,935 / 51,933 |
| Cell 4 | Test Random Forest F1 of about 0.73 |
| Cell 7 | `Reproduces Cell 3? True True True` |

Random Forest scores can differ in the third decimal place between machines and
library versions. Timings depend on the Colab hardware.

**Common problems**

- `NameError`: a script was run without `-i`, or out of order.
- `AssertionError: Not found in /content`: the data is not in `/content`, or has
  not been unzipped. Run `cell0_unzip_data.py`.

## Method

- **Preparation:** the event log is aggregated to 15-minute bins per room
  (binned in UTC, local time Australia/Melbourne is used for time features), and
  each room gets a regular time grid.
- **Split:** chronological 60 / 20 / 20 into train, validation and test (never
  shuffled), so the model is always tested on a later period than it trained on.
- **Models compared:** majority class, persistence (predict the current state),
  hour-of-week rule, logistic regression, and Random Forest.
- **Random Forest settings:** 100 trees, maximum depth 20, minimum leaf size 50,
  chosen on the validation set only. The test set is used once for the final score.
- **Metrics:** precision, recall and F1 on the occupied class, and a confusion
  matrix. Accuracy is reported but not relied on.

## Results

Test set (Oct 2025 to Apr 2026):

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Majority class | 0.000 | 0.000 | 0.000 |
| Persistence | 0.683 | 0.698 | 0.691 |
| Hour-of-week rule | 0.634 | 0.694 | 0.662 |
| Logistic regression | 0.725 | 0.700 | 0.712 |
| **Random Forest** | **0.730** | **0.723** | **0.727** |

Key findings:

- The Random Forest is the best model, but only modestly: +0.036 F1 over
  persistence and +0.015 over logistic regression.
- Time of day is the most important input (permutation importance, and the largest
  F1 drop when removed).
- The Random Forest's advantage over persistence grows with the forecast horizon
  (validation F1: +0.017 at 15 minutes, +0.037 at 1 hour, +0.143 at 4 hours).
- Counting `RecentlyOccupied` as occupied changes the scores but not the conclusion.
- 10 trees score the same as 200 trees on validation, at a fraction of the training
  time and model size.
- Room G.25 is the hardest to predict; the model does slightly worse than
  persistence there.

All saved numbers for every script are in
[`reports/results_summary.md`](reports/results_summary.md).

## Limitations

- Environmental sensors could not be linked to any occupancy room, so nothing here
  says whether temperature, humidity or CO2 would improve predictions.
- Three of the five occupancy spaces have no room name.
- Results apply to these rooms in one building, and the test set is a single later
  block of time. The occupied rate falls from 46% (train) to 39% (test), which
  explains part of the drop in F1.
- Sensor labels are treated as ground truth and are not independently checked.
  Room G.25 shows unexplained overnight occupancy.

## Repository structure

```
notebooks/
    occupancy_forecasting.ipynb   full Colab notebook
src/
    cell0_unzip_data.py           unzip the data files and check they are present
    cell1_load_occupancy.py       load, bin and map the occupancy data
    cell2_env_join_check.py       parse environmental data, check room overlap
    cell3_features_baselines.py   features, split, baselines
    cell4_random_forest.py        tune, train, test, importance
    cell5_error_sensitivity.py    error analysis and feature-removal sensitivity
    cell6_exploration.py          exploratory plots
    cell7_horizon_label_runtime.py  horizon, label and runtime experiments
reports/
    results_summary.md            saved numbers from the runs
requirements.txt
LICENSE                           MIT
ai_use_log.md                     AI use disclosure
```

<!-- Switch these on once they exist:
tests/                  pytest tests
reports/test_report.html  pytest test report
docs/                   HTML documentation generated with pdoc
-->

## AI use

AI tools (Claude, Gemini in Colab) were used during this project. What they
were used for, how the outputs were checked, and the errors found are described
in [`ai_use_log.md`](ai_use_log.md).

## Licence

MIT, see [`LICENSE`](LICENSE).
