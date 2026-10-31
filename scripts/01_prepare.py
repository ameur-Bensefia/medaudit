"""Load every dataset through its adapter (feature extraction / embedding) and save it once.
Usage: python scripts/01_prepare.py [dataset ...]"""
import sys
import time
import traceback

import pandas as pd

from _common import DATASETS, adapter_args, load_cfg, out
from medaudit import adapters, io

cfg = load_cfg()
names = sys.argv[1:] or DATASETS
rows = []
for name in names:
    t = time.time()
    try:
        ad, opts = adapter_args(cfg, name)
        ds = adapters.load(ad, cfg["root"], **opts)
        ds.name = name
        io.save(ds, out(cfg, "datasets", "x")[:-2])
        s = ds.summary()
        s["seconds"] = round(time.time() - t)
        rows.append(s)
        print(f"[ok] {name}: {s}", flush=True)
    except Exception as e:
        print(f"[skip] {name}: {e.__class__.__name__}: {e}", flush=True)
        traceback.print_exc(limit=1)
if rows:
    p = out(cfg, "tables", "datasets_summary.csv")
    old = pd.read_csv(p) if __import__("os").path.exists(p) else pd.DataFrame()
    new = pd.concat([old[~old.get("dataset", pd.Series(dtype=str)).isin([r["dataset"] for r in rows])] if len(old) else old,
                     pd.DataFrame(rows)], ignore_index=True)
    new.to_csv(p, index=False)
    print(new.to_string(index=False))
