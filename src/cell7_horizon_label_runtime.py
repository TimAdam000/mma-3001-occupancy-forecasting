"""Cell 7: sensitivity to forecast horizon and label definition, plus runtime and model size.

Needs (from Cells 1 to 4): occ, train, val, test, FEATURES, BEST, RANDOM_STATE.
Defines: build_dataset(), which rebuilds features and the split for any horizon and label definition.
Prints: a check that build_dataset reproduces Cell 3, then horizon, label and runtime tables.
"""

import time, pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, recall_score, f1_score

def build_dataset(occ_in, horizon_bins=4, include_recent=False):
    """Rebuild features, target and chronological split from the binned occupancy table.

    horizon_bins: forecast horizon in 15-minute bins (4 = 1 hour ahead).
    include_recent: if True, 'RecentlyOccupied' also counts as occupied.
    Returns (train, val, test) DataFrames using the global FEATURES list.
    """
    step = pd.Timedelta(BIN)
    x = occ_in[['room', 'timestamp', 'local', 'status', 'headcount']].copy()
    x['occupied'] = (x['status'] >= 1).astype(int) if include_recent else (x['status'] == 2).astype(int)
    x = x.sort_values(['room', 'timestamp']).reset_index(drop=True)

    tgt = x[['room', 'timestamp', 'occupied']].copy(); tgt['timestamp'] -= horizon_bins * step
    x = x.merge(tgt.rename(columns={'occupied': 'target'}), on=['room', 'timestamp'], how='left')
    wk = x[['room', 'timestamp', 'occupied']].copy(); wk['timestamp'] += pd.Timedelta(days=7)
    x = x.merge(wk.rename(columns={'occupied': 'occ_last_week'}), on=['room', 'timestamp'], how='left')
    x['occ_mean_1h'] = x.groupby('room')['occupied'].transform(lambda s: s.rolling(4, min_periods=1).mean())
    x['occ_mean_4h'] = x.groupby('room')['occupied'].transform(lambda s: s.rolling(16, min_periods=1).mean())
    hr = x['local'].dt.hour + x['local'].dt.minute / 60
    x['hour_sin'] = np.sin(2 * np.pi * hr / 24); x['hour_cos'] = np.cos(2 * np.pi * hr / 24)
    x['hour'] = x['local'].dt.hour; x['dayofweek'] = x['local'].dt.dayofweek
    x['is_weekend'] = (x['dayofweek'] >= 5).astype(int)
    x['room_code'] = x['room'].astype('category').cat.codes
    x = x.dropna(subset=['target'] + [f for f in FEATURES if f != 'occ_last_week']).reset_index(drop=True)
    x['target'] = x['target'].astype(int); x['occ_last_week'] = x['occ_last_week'].fillna(-1)
    t1, t2 = x['timestamp'].quantile([0.6, 0.8])
    return x[x.timestamp <= t1], x[(x.timestamp > t1) & (x.timestamp <= t2)], x[x.timestamp > t2]

# Sanity check: the function must reproduce the datasets from Cell 3
tr0, va0, te0 = build_dataset(occ, 4, False)
print("Reproduces Cell 3?", len(tr0) == len(train), len(va0) == len(val), len(te0) == len(test))

# ---------- 7a. Forecast-horizon sensitivity (scored on VALIDATION; test stays untouched) ----------
rows = []
for name, h in [('15 min', 1), ('1 hour', 4), ('4 hours', 16)]:
    tr, va, _ = build_dataset(occ, h, False)
    m = RandomForestClassifier(**BEST, n_jobs=-1, random_state=RANDOM_STATE).fit(tr[FEATURES], tr['target'])
    f_rf = f1_score(va['target'], m.predict(va[FEATURES]))
    f_pe = f1_score(va['target'], va['occupied'])
    rows.append(dict(horizon=name, RF_F1=f_rf, persistence_F1=f_pe, RF_gain=f_rf - f_pe, val_rows=len(va)))
horizon_res = pd.DataFrame(rows); display(horizon_res.round(3))

# ---------- 7b. Label definition: does "RecentlyOccupied" count as occupied? (1 h ahead) ----------
rows = []
for name, inc in [('Occupied = Currently only', False), ('Occupied = Currently or Recently', True)]:
    tr, va, _ = build_dataset(occ, 4, inc)
    m = RandomForestClassifier(**BEST, n_jobs=-1, random_state=RANDOM_STATE).fit(tr[FEATURES], tr['target'])
    p = m.predict(va[FEATURES])
    rows.append(dict(label=name, occupied_rate=va['target'].mean(), precision=precision_score(va['target'], p),
                     recall=recall_score(va['target'], p), RF_F1=f1_score(va['target'], p),
                     persistence_F1=f1_score(va['target'], va['occupied'])))
label_res = pd.DataFrame(rows); display(label_res.round(3))

# ---------- 7c. Runtime and memory ----------
rows = []
for n in [10, 50, 100, 200]:
    t0 = time.time(); m = RandomForestClassifier(**{**BEST, 'n_estimators': n}, n_jobs=-1, random_state=RANDOM_STATE).fit(train[FEATURES], train['target'])
    fit_s = time.time() - t0
    t0 = time.time(); p = m.predict(val[FEATURES]); pred_s = time.time() - t0
    rows.append(dict(model=f'RandomForest n={n}', fit_s=fit_s, predict_s=pred_s,
                     us_per_row=1e6 * pred_s / len(val), model_MB=len(pickle.dumps(m)) / 1e6, val_F1=f1_score(val['target'], p)))
t0 = time.time(); lr_m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(train[FEATURES], train['target']); fit_s = time.time() - t0
t0 = time.time(); p = lr_m.predict(val[FEATURES]); pred_s = time.time() - t0
rows.append(dict(model='Logistic regression', fit_s=fit_s, predict_s=pred_s, us_per_row=1e6 * pred_s / len(val),
                 model_MB=len(pickle.dumps(lr_m)) / 1e6, val_F1=f1_score(val['target'], p)))
runtime_res = pd.DataFrame(rows); display(runtime_res.round(4))
print(f"\nBinned occupancy table in memory: {occ.memory_usage(deep=True).sum() / 1e6:.1f} MB "
      f"({len(occ):,} rows) vs raw event log of 9,091,732 rows (~1.6 GB on disk)")
