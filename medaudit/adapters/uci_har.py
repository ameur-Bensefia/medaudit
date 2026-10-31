"""UCI Human Activity Recognition (smartphone, 30 participants, 6 activities, 10,299 windows). Uses the 561
features released with the dataset (2.56-s windows, 50% overlap); train and test files are pooled so that the
audit controls the splits. Group: participant. Overlapping windows make sample-wise splits leak by construction."""
import os

import numpy as np
import pandas as pd

from ._util import folder
from ..schema import Dataset


def load(root, **_):
    base = folder(root, "UCI-HAR Dataset")
    X, y, g = [], [], []
    for part in ("train", "test"):
        d = os.path.join(base, part)
        X.append(pd.read_csv(os.path.join(d, f"X_{part}.txt"), sep=r"\s+", header=None).to_numpy(float))
        y.append(np.loadtxt(os.path.join(d, f"y_{part}.txt"), dtype=int) - 1)
        g.append(np.loadtxt(os.path.join(d, f"subject_{part}.txt"), dtype=int))
    X, y, g = np.vstack(X), np.concatenate(y), np.concatenate(g)
    return Dataset("uci_har", X, y, np.char.add("s", g.astype(str)), np.array([f"w{i:05d}" for i in range(len(y))]),
                   task="activity recognition (6 classes), smartphone windows")
