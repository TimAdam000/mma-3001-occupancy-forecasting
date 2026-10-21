"""Cell 6: exploratory plots of the binned occupancy data.

Needs (from Cell 1): occ.
Shows: data coverage per room, occupied rate by hour and weekday, daily profile per room, monthly pattern, headcount distribution, and the status mix per room.
"""

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid')

e = occ.copy()
e['hour'] = e['local'].dt.hour
e['dow'] = e['local'].dt.dayofweek
e['month'] = e['local'].dt.tz_localize(None).dt.to_period('M').dt.to_timestamp()

# 3.1 Data coverage: how many 15-min bins each room has per month (dark = no data)
cov = e.groupby(['room', 'month']).size().unstack(0).fillna(0)
fig, ax = plt.subplots(figsize=(12, 3.8))
sns.heatmap(cov.T, cmap='viridis', ax=ax, cbar_kws={'label': '15-min bins per month'},
            xticklabels=[m.strftime('%Y-%m') if i % 3 == 0 else '' for i, m in enumerate(cov.index)])
ax.set_title('Data coverage per room (dark = no data)'); ax.set_xlabel('month'); plt.tight_layout(); plt.show()

# 3.2 Occupied rate by hour x weekday (all rooms)
pivot = e.pivot_table(index='dow', columns='hour', values='occupied', aggfunc='mean')
pivot.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
fig, ax = plt.subplots(figsize=(12, 3.4))
sns.heatmap(pivot, cmap='YlOrRd', ax=ax, vmin=0, vmax=1, cbar_kws={'label': 'fraction of time occupied'})
ax.set_title('Occupied rate by weekday and local hour'); plt.tight_layout(); plt.show()

# 3.3 Daily profile for each room
fig, ax = plt.subplots(figsize=(10, 3.8))
for r, g in e.groupby('room'):
    g.groupby('hour')['occupied'].mean().plot(ax=ax, label=r)
ax.set(title='Occupied rate by hour of day, per room', ylabel='fraction occupied', xlabel='local hour'); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

# 3.4 Occupied rate over the months (semester / holiday pattern)
fig, ax = plt.subplots(figsize=(12, 3.5))
e.groupby('month')['occupied'].mean().plot(ax=ax, marker='o')
ax.set(title='Fraction of time occupied, by month (all rooms)', ylabel='fraction occupied'); plt.tight_layout(); plt.show()

# 3.5 Room comparison and headcount distribution
fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
e.groupby('room')['occupied'].mean().sort_values().plot(kind='barh', ax=ax[0], title='Occupied rate by room')
e['headcount'].clip(upper=20).plot(kind='hist', bins=21, ax=ax[1], title='Headcount distribution (clipped at 20)')
plt.tight_layout(); plt.show()

# 3.6 Status mix per room (Not / Recently / Currently occupied)
mix = e.groupby('room')['status'].value_counts(normalize=True).unstack().rename(
    columns={0: 'NotOccupied', 1: 'RecentlyOccupied', 2: 'CurrentlyOccupied'})
display(mix.round(3))
