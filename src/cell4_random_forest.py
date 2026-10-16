import matplotlib.pyplot as plt
import time, itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42
Xtr, ytr = train[FEATURES], train['target']
Xva, yva = val[FEATURES],   val['target']
Xte, yte = test[FEATURES],  test['target']

def scores(y, p):
    return dict(P=precision_score(y, p, zero_division=0), R=recall_score(y, p, zero_division=0),
                F1=f1_score(y, p, zero_division=0), Acc=(y == p).mean())

# ---- Tune a small grid on the VALIDATION set only
grid = list(itertools.product([100, 200], [6, 12, 20], [10, 50]))   # n_estimators, max_depth, min_samples_leaf
rows = []
for n, md, leaf in grid:
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=n, max_depth=md, min_samples_leaf=leaf,
                                n_jobs=-1, random_state=RANDOM_STATE).fit(Xtr, ytr)
    s = scores(yva, rf.predict(Xva))
    rows.append(dict(n_estimators=n, max_depth=md, min_samples_leaf=leaf, val_F1=s['F1'],
                     train_F1=f1_score(ytr, rf.predict(Xtr)), fit_s=time.time() - t0))
tune = pd.DataFrame(rows).sort_values('val_F1', ascending=False)
display(tune.head(8))
best = tune.iloc[0]
BEST = dict(n_estimators=int(best.n_estimators), max_depth=int(best.max_depth), min_samples_leaf=int(best.min_samples_leaf))
print("Best settings:", BEST)

# ---- Final model: train on TRAIN only; the test set is used once, here
rf = RandomForestClassifier(**BEST, n_jobs=-1, random_state=RANDOM_STATE).fit(Xtr, ytr)
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xtr, ytr)

def line(name, y, p):
    s = scores(y, p); print(f"{name:34s} P={s['P']:.3f} R={s['R']:.3f} F1={s['F1']:.3f} Acc={s['Acc']:.3f}")

print("\n=== TEST SET (final, unseen later period) ===")
how = train.assign(k=train.dayofweek * 24 + train.hour).groupby('k').target.mean().ge(0.5).astype(int)
line('A: majority class',            yte, np.full(len(test), train.target.mode()[0]))
line('B: persistence (same as now)', yte, test['occupied'])
line('C: hour-of-week rule',         yte, (test.dayofweek * 24 + test.hour).map(how).fillna(0).astype(int))
line('D: logistic regression',       yte, lr.predict(Xte))
line('Random Forest',                yte, rf.predict(Xte))

print("\n=== Overfitting check (Random Forest F1) ===")
for n, X, y in [('train', Xtr, ytr), ('val', Xva, yva), ('test', Xte, yte)]:
    print(f"{n:5s} F1={f1_score(y, rf.predict(X)):.3f}")

print("\nConfusion matrix, test (rows = actual 0/1, cols = predicted 0/1):")
print(confusion_matrix(yte, rf.predict(Xte)))

# ---- Feature importance: permutation importance on validation F1
pi = permutation_importance(rf, Xva, yva, scoring='f1', n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values()
imp.plot(kind='barh', figsize=(7, 4), title='Permutation importance (drop in validation F1)')
plt.tight_layout(); plt.show()
print(imp.sort_values(ascending=False).round(4))
