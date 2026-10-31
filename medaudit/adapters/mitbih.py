"""MIT-BIH Arrhythmia Database (48 half-hour two-lead records). Task: ventricular ectopic (V) vs normal (N)
beats, AAMI grouping. Group: subject. Paced records (102, 104, 107, 217) are excluded as usual.
Records 201 and 202 come from the same subject according to the database documentation; they are merged into
one group and also reported to A3 as a documented shared group (verify in mitdbdir/records.htm)."""
import glob
import os

import numpy as np

from ..features.signals import ecg_beat, rr_features
from ._util import folder
from ..schema import Dataset

N_SYM, V_SYM = set("NLRej"), set("VE")
PACED = {"102", "104", "107", "217"}
SAME_SUBJECT = {"202": "201"}


def load(root, max_n_per_record=400, seed=0, **_):
    import wfdb
    base = folder(root, "mit-bih")
    recs = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(base, "[0-9][0-9][0-9].hea")))
    rng = np.random.default_rng(seed)
    X, y, g, ids, rec_of = [], [], [], [], []
    for r in recs:
        if r in PACED:
            continue
        rec = wfdb.rdrecord(os.path.join(base, r))
        ann = wfdb.rdann(os.path.join(base, r), "atr")
        fs = rec.fs
        sig = rec.p_signal[:, 0]
        locs = np.array(ann.sample)
        syms = np.array(ann.symbol)
        beat = np.array([s in N_SYM or s in V_SYM for s in syms])
        locs, syms = locs[beat], syms[beat]
        n_idx = np.where([s in N_SYM for s in syms])[0]
        v_idx = np.where([s in V_SYM for s in syms])[0]
        if len(n_idx) > max_n_per_record:
            n_idx = rng.choice(n_idx, max_n_per_record, replace=False)
        for i in np.sort(np.concatenate([n_idx, v_idx])).astype(int):
            b = ecg_beat(sig, locs[i], fs)
            if b is None:
                continue
            X.append(np.concatenate([b, rr_features(locs, i, fs)]))
            y.append(int(syms[i] in V_SYM))
            g.append(SAME_SUBJECT.get(r, r))
            ids.append(f"{r}_{locs[i]}")
            rec_of.append(r)
    ds = Dataset("mitbih", np.vstack(X), np.array(y), np.array(g), np.array(ids),
                 task="ventricular ectopic vs normal beats (AAMI)")
    ds.notes.update(known_shared_groups=[("201", "202")], records=np.array(rec_of))
    return ds
