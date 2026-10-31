"""A4 with the fully nested procedure (the audited model is retrained inside each outer fold, so no score used by the
second level was produced by a model that saw an outer-test participant). Resumable: one row per dataset and model.
Usage: python scripts/10_nested_a4.py [dataset ...]"""
import json
import os
import sys
import time

import pandas as pd

from _common import load_cfg, out
from medaudit import io
from medaudit.audits.a4_incremental import incremental_value_nested

DEFAULT = ["hw_pahaw", "abide", "ptbxl", "ham10000", "chexpert", "nih_cxr"]
cfg = load_cfg()
dsdir = out(cfg, "datasets", "x")[:-2]
path = out(cfg, "tables", "a4_nested.csv")
done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["dataset", "model"])
key = set(zip(done.dataset, done.model))
rows = done.to_dict("records")
for name in sys.argv[1:] or DEFAULT:
    if not os.path.exists(os.path.join(dsdir, name + ".npz")):
        print(f"[skip] {name}: not prepared")
        continue
    ds = io.load(name, dsdir)
    repeats = 3 if len(set(ds.group)) < 1000 else 1
    for m in ("logreg", "mlp", "rf"):
        if (name, m) in key:
            continue
        t = time.time()
        r = incremental_value_nested(ds, m, repeats=repeats, n_boot=2000)
        r.update(dataset=name, model=m, repeats=repeats, seconds=round(time.time() - t))
        rows.append(r)
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"  {name} {m}: dAUC {r['delta_auc']:+.3f} [{r['delta_auc_lo']:+.3f}, {r['delta_auc_hi']:+.3f}] "
              f"({r['seconds']}s)", flush=True)
print(json.dumps({"saved": path}))
