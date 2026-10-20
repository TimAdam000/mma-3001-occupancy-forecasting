"""Cell 8: repeatability across random seeds, profiler output and a cost estimate.

Run after cells 1, 3 and 4 (needs train, val, test, FEATURES and BEST in memory):
    %run -i src/cell8_profile_repeatability.py
"""
import cProfile, pstats, io, math, time
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# ---- 8a. Repeatability: refit the tuned Random Forest with 5 different random seeds
SEEDS = [0, 1, 2, 3, 4]
rows = []
for s in SEEDS:
    m = RandomForestClassifier(**BEST, n_jobs=-1, random_state=s).fit(train[FEATURES], train['target'])
    rows.append(dict(seed=s,
                     val_F1=f1_score(val['target'], m.predict(val[FEATURES])),
                     test_F1=f1_score(test['target'], m.predict(test[FEATURES]))))
seed_res = pd.DataFrame(rows)
print(seed_res.round(4).to_string(index=False))
pers = f1_score(test['target'], test['occupied'])
print(f"\nTest F1 over {len(SEEDS)} seeds: mean {seed_res.test_F1.mean():.4f}, "
      f"std {seed_res.test_F1.std():.4f}, min {seed_res.test_F1.min():.4f}, max {seed_res.test_F1.max():.4f}")
print(f"Persistence test F1 (no randomness): {pers:.4f}")
print(f"Smallest gain over persistence across seeds: {seed_res.test_F1.min() - pers:+.4f}")

# ---- 8b. Profiler: where does the time go? (single thread so cProfile can see the work)
rf1 = RandomForestClassifier(**BEST, n_jobs=1, random_state=42)
pr = cProfile.Profile(); pr.enable()
rf1.fit(train[FEATURES], train['target'])
pr.disable()
buf = io.StringIO()
pstats.Stats(pr, stream=buf).sort_stats('cumulative').print_stats(8)
print("\n=== cProfile: Random Forest fit (top 8 by cumulative time) ===")
print(buf.getvalue())

pr = cProfile.Profile(); pr.enable()
rf1.predict(val[FEATURES])
pr.disable()
buf = io.StringIO()
pstats.Stats(pr, stream=buf).sort_stats('cumulative').print_stats(6)
print("=== cProfile: Random Forest predict on validation (top 6) ===")
print(buf.getvalue())

# ---- 8c. Rough operation-count estimate (orders of magnitude, not exact FLOPs)
n, p, T, D = len(train), len(FEATURES), BEST['n_estimators'], BEST['max_depth']
per_split = max(1, int(math.sqrt(p)))                     # features tried per split (sqrt rule)
fit_ops = T * per_split * n * math.log2(n)                # ~ trees x features tried x n log n
pred_ops_per_row = T * D                                  # one comparison per tree level, worst case
lr_fit_ops = 100 * n * p * 2                              # ~100 iterations x n x p multiply-adds
print("=== Rough cost estimate ===")
print(f"Random Forest fit:      ~{fit_ops:.1e} basic operations (n={n:,}, p={p}, {T} trees, {per_split} features per split)")
print(f"Random Forest predict:  ~{pred_ops_per_row:,} comparisons per row (worst case: {T} trees x depth {D})")
print(f"Logistic regression fit: ~{lr_fit_ops:.1e} multiply-adds (about 100 iterations)")

