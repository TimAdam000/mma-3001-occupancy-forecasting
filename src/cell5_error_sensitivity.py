# ---- Error analysis on the TEST set
test = test.copy(); test['pred'] = rf.predict(Xte); test['wrong'] = (test['pred'] != test['target']).astype(int)

by_room = test.groupby('room').apply(lambda g: pd.Series(dict(
    n=len(g), occupied_rate=g.target.mean(), F1=f1_score(g.target, g.pred, zero_division=0),
    persistence_F1=f1_score(g.target, g.occupied, zero_division=0), error_rate=g.wrong.mean())))
display(by_room.round(3))

by_hour = test.groupby('hour').apply(lambda g: pd.Series(dict(
    error_rate=g.wrong.mean(), occupied_rate=g.target.mean())))
fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
by_hour['error_rate'].plot(kind='bar', ax=ax[0], title='Test error rate by local hour')
test.groupby('is_weekend')['wrong'].mean().rename({0: 'weekday', 1: 'weekend'}).plot(kind='bar', ax=ax[1], title='Test error rate: weekday vs weekend')
plt.tight_layout(); plt.show()

# ---- Sensitivity analysis: remove one feature group, retrain, score on VALIDATION (test stays untouched)
GROUPS = {
    'current state (occupied, headcount)': ['occupied', 'headcount'],
    'rolling history (1 h, 4 h)':          ['occ_mean_1h', 'occ_mean_4h'],
    'same time last week':                 ['occ_last_week'],
    'time of day/week':                    ['hour_sin', 'hour_cos', 'dayofweek', 'is_weekend'],
    'room identity':                       ['room_code'],
}
def fit_eval(feats):
    t0 = time.time()
    m = RandomForestClassifier(**BEST, n_jobs=-1, random_state=RANDOM_STATE).fit(train[feats], ytr)
    return f1_score(yva, m.predict(val[feats])), time.time() - t0

full_f1, full_t = fit_eval(FEATURES)
res = [dict(removed='(none: all features)', val_F1=full_f1, change=0.0, fit_s=full_t)]
for name, cols in GROUPS.items():
    f1v, secs = fit_eval([f for f in FEATURES if f not in cols])
    res.append(dict(removed=name, val_F1=f1v, change=f1v - full_f1, fit_s=secs))
display(pd.DataFrame(res).round(4))
