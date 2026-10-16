from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

HORIZON_BINS = 4                      # 4 x 15 min = predict 1 hour ahead
STEP = pd.Timedelta(BIN)

d = occ[['room', 'timestamp', 'local', 'occupied', 'headcount']].sort_values(['room', 'timestamp']).reset_index(drop=True)

# Target = occupancy HORIZON_BINS ahead, matched on real timestamps (gaps give NaN, not a wrong row)
tgt = d[['room', 'timestamp', 'occupied']].copy()
tgt['timestamp'] -= HORIZON_BINS * STEP
d = d.merge(tgt.rename(columns={'occupied': 'target'}), on=['room', 'timestamp'], how='left')

# Same time last week (past only)
wk = d[['room', 'timestamp', 'occupied']].copy()
wk['timestamp'] += pd.Timedelta(days=7)
d = d.merge(wk.rename(columns={'occupied': 'occ_last_week'}), on=['room', 'timestamp'], how='left')

# Past-only rolling occupancy
d['occ_mean_1h'] = d.groupby('room')['occupied'].transform(lambda s: s.rolling(4,  min_periods=1).mean())
d['occ_mean_4h'] = d.groupby('room')['occupied'].transform(lambda s: s.rolling(16, min_periods=1).mean())

# Time features from LOCAL Melbourne time
hr = d['local'].dt.hour + d['local'].dt.minute / 60
d['hour_sin'] = np.sin(2 * np.pi * hr / 24)
d['hour_cos'] = np.cos(2 * np.pi * hr / 24)
d['hour'] = d['local'].dt.hour
d['dayofweek'] = d['local'].dt.dayofweek
d['is_weekend'] = (d['dayofweek'] >= 5).astype(int)
d['room_code'] = d['room'].astype('category').cat.codes

FEATURES = ['occupied', 'headcount', 'occ_mean_1h', 'occ_mean_4h', 'occ_last_week',
            'hour_sin', 'hour_cos', 'dayofweek', 'is_weekend', 'room_code']
d = d.dropna(subset=['target'] + [f for f in FEATURES if f != 'occ_last_week']).reset_index(drop=True)
d['target'] = d['target'].astype(int)
d['occ_last_week'] = d['occ_last_week'].fillna(-1)     # -1 = unknown

# Chronological 60/20/20 split
t1, t2 = d['timestamp'].quantile([0.6, 0.8])
train = d[d.timestamp <= t1]
val   = d[(d.timestamp > t1) & (d.timestamp <= t2)]
test  = d[d.timestamp > t2]
for n, p in [('train', train), ('val', val), ('test', test)]:
    print(f"{n:5s} {len(p):7d} rows {p.timestamp.min():%Y-%m-%d} -> {p.timestamp.max():%Y-%m-%d} occupied={p.target.mean():.2%}")

def report(name, y, p):
    print(f"{name:30s} P={precision_score(y,p,zero_division=0):.3f} R={recall_score(y,p,zero_division=0):.3f} "
          f"F1={f1_score(y,p,zero_division=0):.3f} Acc={(y==p).mean():.3f}")

report('A: majority class',            val.target, np.full(len(val), train.target.mode()[0]))
report('B: persistence (same as now)', val.target, val['occupied'])
how = train.assign(k=train.dayofweek * 24 + train.hour).groupby('k').target.mean().ge(0.5).astype(int)
report('C: hour-of-week rule',         val.target, (val.dayofweek * 24 + val.hour).map(how).fillna(0).astype(int))
