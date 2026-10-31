"""CHB-MIT Scalp EEG (paediatric, 23 bipolar channels, 256 Hz). Task: seizure vs non-seizure 4-s windows.
Group: patient. Case chb21 was recorded from the same subject as chb01 (1.5 years later) according to the
database documentation; they are merged and reported to A3 (verify in SUBJECT-INFO)."""
import glob
import os
import re

import numpy as np

from ..features.signals import band_powers
from ._util import folder
from ..schema import Dataset

CHANNELS = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4",
            "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FZ-CZ", "CZ-PZ"]
SAME_SUBJECT = {"chb21": "chb01"}


def parse_summary(path):
    out, cur = {}, None
    for line in open(path, errors="ignore"):
        m = re.match(r"File Name:\s*(\S+)", line)
        if m:
            cur = m.group(1)
            out[cur] = []
            continue
        m = re.search(r"Seizure\s*\d*\s*Start Time:\s*(\d+)", line)
        if m and cur:
            out[cur].append([int(m.group(1)), None])
        m = re.search(r"Seizure\s*\d*\s*End Time:\s*(\d+)", line)
        if m and cur and out[cur]:
            out[cur][-1][1] = int(m.group(1))
    return out


def _read(path, start_s, dur_s):
    import pyedflib
    f = pyedflib.EdfReader(path)
    try:
        fs = f.getSampleFrequency(0)
        labels = [lab.upper().strip() for lab in f.getSignalLabels()]
        n = int(dur_s * fs)
        x = np.zeros((len(CHANNELS), n))
        present = 0
        for k, ch in enumerate(CHANNELS):
            if ch in labels:
                j = labels.index(ch)
                x[k] = f.readSignal(j, int(start_s * fs), n)
                present += 1
        return x, fs, present
    finally:
        f.close()


def _duration(path):
    import pyedflib
    f = pyedflib.EdfReader(path)
    try:
        return float(f.getFileDuration())
    finally:
        f.close()


def load(root, win_s=4, nonseizure_ratio=3, max_seizure_windows_per_patient=150, seed=0, **_):
    rng = np.random.default_rng(seed)
    base = folder(root, "CHB-MIT")
    if not glob.glob(os.path.join(base, "chb[0-9][0-9]", "*.edf")):
        raise FileNotFoundError(f"no EEG (.edf) files under {base}; restore the CHB-MIT recordings from PhysioNet")
    X, y, g, ids = [], [], [], []
    skipped = 0
    for pdir in sorted(glob.glob(os.path.join(base, "chb[0-9][0-9]"))):
        pid = os.path.basename(pdir)
        summ = glob.glob(os.path.join(pdir, "*summary*.txt"))
        if not summ:
            continue
        seiz = parse_summary(summ[0])
        windows = []
        for fname, ev in seiz.items():
            for s, e in ev:
                if e:
                    windows += [(fname, t, 1) for t in range(s, e - win_s + 1, win_s)]
        if not windows:
            continue
        if len(windows) > max_seizure_windows_per_patient:
            windows = [windows[i] for i in rng.choice(len(windows), max_seizure_windows_per_patient, replace=False)]
        n_non = nonseizure_ratio * len(windows)
        files = [f for f in seiz if os.path.exists(os.path.join(pdir, f))]
        dur = {}
        for f in files:
            try:
                dur[f] = _duration(os.path.join(pdir, f))
            except Exception:
                pass
        files = [f for f in files if dur.get(f, 0) > win_s + 1]
        if not files:
            continue
        for _ in range(n_non):
            f = files[rng.integers(len(files))]
            t = int(rng.integers(0, int(dur[f] - win_s)))
            if any(s - 60 <= t <= e + 60 for s, e in seiz[f] if e):
                continue
            windows.append((f, t, 0))
        for f, t, lab in windows:
            path = os.path.join(pdir, f)
            if not os.path.exists(path):
                continue
            try:
                x, fs, present = _read(path, t, win_s)
            except Exception:
                continue
            if present < len(CHANNELS) * 0.75 or np.any(np.std(x[np.std(x, axis=1) > 0], axis=1) == 0):
                skipped += 1                   # different montage (e.g. chb12_27-29, chb17b_69): not comparable
                continue
            X.append(band_powers(x, fs))
            y.append(lab)
            g.append(SAME_SUBJECT.get(pid, pid))
            ids.append(f"{f}_{t}_{lab}")
    ids = np.array(ids)
    _, keep = np.unique(ids, return_index=True)
    ds = Dataset("chbmit", np.vstack(X)[keep], np.array(y)[keep], np.array(g)[keep], ids[keep],
                 task="seizure vs non-seizure 4-s EEG windows")
    ds.notes.update(known_shared_groups=[("chb01", "chb21")], skipped_windows_other_montage=skipped)
    print(f"  chbmit: {skipped} windows skipped (montage without the standard channels)")
    return ds
