"""Unzip the zipped data files into DATA_DIR and check the three files the workflow needs.

Run this first when the occupancy and environmental data are supplied as .zip files.
Safe to run twice: files that already exist are skipped.
"""
import os, glob, zipfile, shutil

DATA_DIR = '/content'

zips = sorted(glob.glob(os.path.join(DATA_DIR, '*.zip')))
if not zips:
    print(f"No .zip files found in {DATA_DIR}. Upload the zipped data files first (sidebar folder icon -> upload).")

for z in zips:
    with zipfile.ZipFile(z) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            # keep only the file name, so nothing can be written outside DATA_DIR
            target = os.path.join(DATA_DIR, os.path.basename(member.filename))
            if os.path.exists(target):
                print(f"Skipping (already exists): {os.path.basename(target)}")
                continue
            with zf.open(member) as src, open(target, 'wb') as dst:
                shutil.copyfileobj(src, dst)
            print(f"Extracted: {os.path.basename(target)}")

# Check the three files Cell 1 needs are now present
expected = ['5occupancySensor_MayToDec2024_9MRows.csv',
            '5EnvSensor_MayToDec2024_180kRows.csv',
            'Sensor ID and Locations.xlsx']
for f in expected:
    p = os.path.join(DATA_DIR, f)
    print(f"{'OK     ' if os.path.exists(p) else 'MISSING'} {f}",
          f"({os.path.getsize(p) / 1e6:.0f} MB)" if os.path.exists(p) else '')
