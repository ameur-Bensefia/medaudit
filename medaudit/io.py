"""Save and load Datasets (npz + csv) so that feature extraction runs once."""
import json
import os

import numpy as np
import pandas as pd

from .schema import Dataset


def save(ds, folder):
    os.makedirs(folder, exist_ok=True)
    base = os.path.join(folder, ds.name)
    np.savez_compressed(base + ".npz", X=ds.X, y=ds.y, group=ds.group, sample_id=ds.sample_id,
                        site=ds.site if ds.site is not None else np.array([]),
                        paths=ds.paths if ds.paths is not None else np.array([]))
    if ds.confounders is not None:
        ds.confounders.to_csv(base + "_confounders.csv", index=False)
    notes = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in ds.notes.items()}
    with open(base + ".json", "w") as f:
        json.dump(dict(task=ds.task, notes=notes), f, default=str)


def load(name, folder):
    base = os.path.join(folder, name)
    z = np.load(base + ".npz", allow_pickle=True)
    meta = json.load(open(base + ".json"))
    conf = pd.read_csv(base + "_confounders.csv") if os.path.exists(base + "_confounders.csv") else None
    site = z["site"] if z["site"].size else None
    paths = z["paths"] if z["paths"].size else None
    notes = meta.get("notes", {})
    if "known_shared_groups" in notes:
        notes["known_shared_groups"] = [tuple(x) for x in notes["known_shared_groups"]]
    return Dataset(name, z["X"], z["y"], z["group"], z["sample_id"], site, conf, paths, meta.get("task", ""), notes)
