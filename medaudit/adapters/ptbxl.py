"""PTB-XL 12-lead ECG (requires ptbxl_database.csv). Task: myocardial infarction vs normal ECG (diagnostic
superclasses MI vs NORM, single-superclass records). Group: patient. Site: recording site. Confounders: age, sex."""
import ast
import os

import numpy as np
import pandas as pd

from ..features.signals import lead_stats
from ._util import folder
from ..schema import Dataset


def load(root, cache_dir="cache", max_records=None, seed=0, **_):
    import wfdb
    base = folder(root, "PTB-XL")
    dbp = os.path.join(base, "ptbxl_database.csv")
    if not os.path.exists(dbp):
        raise FileNotFoundError("PTB-XL needs ptbxl_database.csv (patient IDs and labels); download it from PhysioNet")
    db = pd.read_csv(dbp)
    st = pd.read_csv(os.path.join(base, "scp_statements.csv"), index_col=0)
    diag = st[st["diagnostic"] == 1]["diagnostic_class"]
    sup = db["scp_codes"].map(lambda s: {diag[c] for c in ast.literal_eval(s) if c in diag.index})
    db = db[sup.map(lambda s: s in ({"NORM"}, {"MI"}))].copy()
    db["y"] = sup[db.index].map(lambda s: int(s == {"MI"}))
    if max_records and len(db) > max_records:
        db = db.sample(max_records, random_state=seed)
    db = db.reset_index(drop=True)
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"ptbxl_features_{len(db)}.npy")
    if os.path.exists(cache):
        X = np.load(cache)
    else:
        X = np.vstack([lead_stats(wfdb.rdsamp(os.path.join(base, f))[0], 100) for f in db["filename_lr"]])
        np.save(cache, X)
    conf = pd.DataFrame({"age": pd.to_numeric(db["age"], errors="coerce").where(lambda a: a < 120),
                         "sex": db["sex"].astype(str)})
    return Dataset("ptbxl", X, db["y"].to_numpy(), db["patient_id"].astype(int).astype(str).to_numpy(),
                   db["ecg_id"].astype(str).to_numpy(), db["site"].astype(str).to_numpy(), conf,
                   task="myocardial infarction vs normal ECG, 12 leads")
