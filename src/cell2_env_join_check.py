env = pd.read_csv(os.path.join(DATA_DIR, ENV_FILE), usecols=['sensorid', 'jsondata'], dtype={'sensorid': str})

def parse(js):
    out = {}
    try:
        for item in json.loads(js):
            v = item['values'][0]
            out[item['variable']['name']] = v['value']
            out['_t'] = v['time']
    except Exception:
        return None
    return out

parsed = env['jsondata'].map(parse)
ok = parsed.notna()
wide = pd.DataFrame(parsed[ok].tolist(), index=env.index[ok])
env = pd.concat([env.loc[ok, ['sensorid']], wide], axis=1)
env['timestamp'] = pd.to_datetime(env['_t'], unit='s', utc=True).dt.floor(BIN)
env = env.drop(columns='_t')
ENV_NUM = [c for c in env.columns if c not in ('sensorid', 'timestamp')]
env = env.groupby(['sensorid', 'timestamp'])[ENV_NUM].mean().reset_index()

# Excel stores IDs as numbers; convert back to the 13-digit strings used in the CSV
id_to_room = (env_map.assign(ID=env_map['ID'].astype(float).astype('int64').astype(str))
              .drop_duplicates('ID').set_index('ID')['Floor/Room'].to_dict())
env['room'] = env['sensorid'].map(id_to_room)

print(env.groupby('sensorid').agg(room=('room', 'first'), start=('timestamp', 'min'),
                                  end=('timestamp', 'max'), bins=('timestamp', 'size')))
print("Rooms with BOTH env and occupancy data:", set(env['room'].dropna()) & set(occ['room']))
print("Duplicate IDs in lookup:\n", env_map[env_map['ID'].duplicated(keep=False)])
