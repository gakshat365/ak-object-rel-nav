import os
from pathlib import Path
import datetime

d = Path("visualisations/traversability_analysis")
files = sorted(d.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
for f in files:
    mtime = datetime.datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f"{mtime} | {size_mb:6.1f} MB | {f.name}")
