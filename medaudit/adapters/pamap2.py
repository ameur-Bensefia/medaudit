"""PAMAP2 physical activity monitoring (9 participants, 3 IMUs + heart rate, 100 Hz). Task: activity
(multi-class) from 5-s windows. Group: participant."""
import glob
import io
import os
import zipfile

import numpy as np
import pandas as pd

from ..features.signals import window_stats
from ..schema import Dataset

ACTIVITIES = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]     # protocol activities
IMU_START = (3, 20, 37)                                      # hand, chest, ankle (0-based column of temperature)


def _channels():
    cols = [2]                                               # heart rate
    for s in IMU_START:
        cols += list(range(s + 1, s + 4)) + list(range(s + 7, s + 10)) + list(range(s + 10, s + 13))  # acc16, gyro, mag
    return cols


def _subject_files(root):
    """(subject, reader) pairs from the extracted folder or, failing that, from the zip."""
    files = sorted(glob.glob(os.path.join(root, "pamap2", "**", "Protocol", "subject*.dat"), recursive=True))
    if files:
        return [(os.path.basename(f)[:-4], f) for f in files]
    zp = os.path.join(root, "pamap2", "PAMAP2_Dataset.zip")
    if os.path.exists(zp):
        z = zipfile.ZipFile(zp)
        return [(os.path.basename(m)[:-4], io.BytesIO(z.read(m))) for m in sorted(z.namelist())
                if "/Protocol/subject" in m and m.endswith(".dat")]
    raise FileNotFoundError("no Protocol/subject*.dat in pamap2/ (folder or PAMAP2_Dataset.zip)")


def load(root, win_s=5.0, step_s=2.5, max_windows_per_class_subject=60, seed=0, **_):
    rng = np.random.default_rng(seed)
    rows, ys, gs, ids = [], [], [], []
    for subj, src in _subject_files(root):
        d = pd.read_csv(src, sep=r"\s+", header=None).to_numpy(float)
        act = d[:, 1].astype(int)
        sig = d[:, _channels()]
        sig[:, 0] = pd.Series(sig[:, 0]).ffill().bfill().to_numpy()       # heart rate sampled at ~9 Hz
        w, st = int(win_s * 100), int(step_s * 100)
        for a in ACTIVITIES:
            idx = np.where(act == a)[0]
            if len(idx) < w:
                continue
            runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
            starts = [r[0] + k for r in runs for k in range(0, len(r) - w + 1, st)]
            if len(starts) > max_windows_per_class_subject:
                starts = rng.choice(starts, max_windows_per_class_subject, replace=False)
            for s0 in starts:
                rows.append(window_stats(sig[s0:s0 + w]))
                ys.append(ACTIVITIES.index(a))
                gs.append(subj)
                ids.append(f"{subj}_{a}_{s0}")
    return Dataset("pamap2", np.vstack(rows), np.array(ys), np.array(gs), np.array(ids),
                   task="physical activity (12 classes), 5-s IMU windows")
