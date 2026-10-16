import os, json
import numpy as np, pandas as pd

DATA_DIR = '/content'
OCC_FILE = '5occupancySensor_MayToDec2024_9MRows.csv'   # or .csv if unzipped
ENV_FILE = '5EnvSensor_MayToDec2024_180kRows.csv'       # or .csv if unzipped
MAP_FILE = 'Sensor ID and Locations.xlsx'

# Check the files are where we expect before doing anything heavy
print(os.listdir(DATA_DIR))
for f in (OCC_FILE, ENV_FILE, MAP_FILE):
    assert os.path.exists(os.path.join(DATA_DIR, f)), f"Not found in {DATA_DIR}: {f}"

BIN = '15min'
TZ = 'Australia/Melbourne'
OCCUPIED_STATES = {'CurrentlyOccupied'}   # add 'RecentlyOccupied' if you decide it counts
START, END = None, None                   # e.g. '2024-05-01', '2025-01-01'
MAX_FFILL_BINS = 8                        # carry last known status forward up to 2 h

# ---- Lookup: occupancy SpaceId -> room name
sheets = pd.read_excel(os.path.join(DATA_DIR, MAP_FILE), sheet_name=None, dtype=str)
env_map, occ_map = list(sheets.values())[0], list(sheets.values())[1]
space_to_room = occ_map.dropna(subset=['SpaceId']).groupby('SpaceId')['Floor/Room'].first().to_dict()

# ---- Occupancy: chunked aggregation of the event log to 15-min bins (UTC; local time added later)
STATUS_CODE = {'NotOccupied': 0, 'RecentlyOccupied': 1, 'CurrentlyOccupied': 2}
parts = []
reader = pd.read_csv(os.path.join(DATA_DIR, OCC_FILE),
                     usecols=['floorspaceid', 'occupancystatus', 'headcount', 'collecteddate'],
                     chunksize=1_000_000)
for ch in reader:
    t = pd.to_datetime(ch['collecteddate'], utc=True, errors='coerce')
    ch = ch.assign(t=t).dropna(subset=['t'])
    ch['bin'] = ch['t'].dt.floor(BIN)
    ch['status'] = ch['occupancystatus'].map(STATUS_CODE)
    ch = ch.dropna(subset=['status']).sort_values('t')
    parts.append(ch.groupby(['floorspaceid', 'bin']).agg(
        status=('status', 'last'), headcount=('headcount', 'max'),
        n_events=('status', 'size'), t_last=('t', 'max')).reset_index())

occ = pd.concat(parts).sort_values('t_last')
occ = occ.groupby(['floorspaceid', 'bin']).agg(
    status=('status', 'last'), headcount=('headcount', 'max'),
    n_events=('n_events', 'sum')).reset_index()

# ---- Regular time grid per space, carrying the last known status forward over short gaps
grids = []
for sid, g in occ.groupby('floorspaceid'):
    full = pd.date_range(g['bin'].min(), g['bin'].max(), freq=BIN)
    g = g.set_index('bin').reindex(full)
    g.index.name = 'timestamp'
    g['n_events'] = g['n_events'].fillna(0)
    g[['status', 'headcount']] = g[['status', 'headcount']].ffill(limit=MAX_FFILL_BINS)
    g['floorspaceid'] = sid
    grids.append(g.reset_index())
occ = pd.concat(grids, ignore_index=True)
occ['room'] = occ['floorspaceid'].map(space_to_room).fillna('unmapped_' + occ['floorspaceid'].str[:8])
occ = occ.dropna(subset=['status'])
occ['occupied'] = occ['status'].map({v: k for k, v in STATUS_CODE.items()}).isin(OCCUPIED_STATES).astype(int)
if START: occ = occ[occ['timestamp'] >= pd.Timestamp(START, tz=TZ).tz_convert('UTC')]
if END:   occ = occ[occ['timestamp'] <  pd.Timestamp(END,   tz=TZ).tz_convert('UTC')]
occ['local'] = occ['timestamp'].dt.tz_convert(TZ)

print(occ.shape)
print(occ.groupby('room')['occupied'].agg(rows='size', occupied_rate='mean'))
print(occ.groupby('room')['timestamp'].agg(['min', 'max']))
