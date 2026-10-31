"""Handwriting biomarkers of Parkinson's disease computed directly from the raw data shared by the authors.

hw_pahaw         PaHaW: one .svc file per participant and task (columns x, y, time, pen-down, azimuth, altitude,
                 pressure); diagnosis, age and sex from corpus_PaHaW.xlsx. Group: participant.
hw_newhandpdsig  NewHandPD: biometric pen signals in Signal_Patient / Signal_Healthy (files like sigSp1-P12.txt).
                 Group: participant code (P12, H3). Pen-signal files are numerically identical across all 12 tasks
                 for P1 = P4, P2 = P25 = P26, P9 = P10 and P15 = P17 (verified on the raw files); they are reported
                 to A3 as documented shared groups. Use merge_duplicates=True to merge them into one participant."""
import glob
import os
import re

import numpy as np
import pandas as pd

from ._util import col
from ..features.pen import multichannel_features, tablet_features
from ..schema import Dataset


def _read_svc(path):
    with open(path, errors="ignore") as fh:
        lines = fh.read().split("\n")
    rows = [list(map(float, ln.split())) for ln in lines[1:] if len(ln.split()) >= 7]
    return np.asarray(rows)[:, :7] if rows else None


def load_pahaw(root):
    base = os.path.join(root, "PaHaW")
    xl = glob.glob(os.path.join(base, "**", "corpus_PaHaW.xlsx"), recursive=True)
    if not xl:
        raise FileNotFoundError("corpus_PaHaW.xlsx not found under PaHaW/")
    meta = pd.read_excel(xl[0])
    idc = col(meta, "ID")
    dis = col(meta, "Disease", "Diagnosis", "Group")
    meta["_id"] = meta[idc].astype(str).str.extract(r"(\d+)")[0].astype(int)
    meta = meta.set_index("_id")
    rows, ys, gs, ids, ages, sexes = [], [], [], [], [], []
    for p in sorted(glob.glob(os.path.join(base, "**", "*.svc"), recursive=True)):
        m = re.match(r"(\d+)__(\d+)_(\d+)\.svc$", os.path.basename(p))
        if not m:
            continue
        pid, task = int(m.group(1)), int(m.group(2))
        if pid not in meta.index:
            continue
        a = _read_svc(p)
        if a is None or len(a) < 20:
            continue
        t = a[:, 2] - a[:, 2][0]
        if np.median(np.diff(t)) > 1:              # milliseconds -> seconds
            t = t / 1000.0
        f = tablet_features(a[:, 0], a[:, 1], t, a[:, 3], a[:, 6], a[:, 4], a[:, 5])
        f["task"] = task
        rows.append(f)
        ys.append(int(str(meta.loc[pid, dis]).strip().upper().startswith("PD")))
        gs.append(f"pahaw_{pid:05d}")
        ids.append(os.path.basename(p)[:-4])
        ages.append(meta.loc[pid].get(col(meta, "Age", required=False), np.nan) if col(meta, "Age", required=False) else np.nan)
        sx = col(meta, "Sex", "Gender", required=False)
        sexes.append(str(meta.loc[pid, sx]) if sx else "NA")
    X = pd.DataFrame(rows)
    X = pd.get_dummies(X, columns=["task"], dtype=float)
    conf = pd.DataFrame({"age": pd.to_numeric(pd.Series(ages), errors="coerce"), "sex": sexes})
    return Dataset("hw_pahaw", X.to_numpy(float), np.array(ys), np.array(gs), np.array(ids), None, conf,
                   task="Parkinson's disease vs control, handwriting signals (PaHaW)")


def _read_signal(path):
    rows = []
    with open(path, errors="ignore") as fh:
        for ln in fh:
            parts = re.split(r"[\s,;]+", ln.strip())
            try:
                rows.append([float(v) for v in parts if v != ""])
            except ValueError:
                continue                            # header or comment line
    if not rows:
        return None
    k = max(set(map(len, rows)), key=[len(r) for r in rows].count)
    M = np.asarray([r for r in rows if len(r) == k], float)
    return M if len(M) >= 20 else None


SIGNAL_DUPLICATES = {"P4": "P1", "P25": "P2", "P26": "P2", "P10": "P9", "P17": "P15"}


def load_newhandpd(root, merge_duplicates=False):
    base = os.path.join(root, "NewHandPD")
    rows, ys, gs, ids = [], [], [], []
    for p in sorted(glob.glob(os.path.join(base, "Signal_*", "*.txt"))):
        name = os.path.basename(p)[:-4]
        m = re.search(r"-([PH])(\d+)$", name)
        if not m:
            continue
        M = _read_signal(p)
        if M is None:
            continue
        task = re.sub(r"\d*-[PH]\d+$", "", name)
        f = multichannel_features(M)
        f["task"] = task
        rows.append(f)
        ys.append(int(m.group(1) == "P"))
        code = f"{m.group(1)}{int(m.group(2))}"
        gs.append(SIGNAL_DUPLICATES.get(code, code) if merge_duplicates else code)
        ids.append(name)
    X = pd.get_dummies(pd.DataFrame(rows), columns=["task"], dtype=float)
    ds = Dataset("hw_newhandpdsig", X.to_numpy(float), np.array(ys), np.array(gs), np.array(ids),
                 task="Parkinson's disease vs control, biometric pen signals (NewHandPD)")
    ds.notes["known_shared_groups"] = [("P1", "P4"), ("P2", "P25"), ("P2", "P26"), ("P9", "P10"), ("P15", "P17")]
    return ds


def load(root, which="pahaw", merge_duplicates=False, **_):
    return load_pahaw(root) if which == "pahaw" else load_newhandpd(root, merge_duplicates)
