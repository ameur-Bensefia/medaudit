"""DARWIN handwriting (one row per participant). Negative control: no group can cross splits."""
import os

import pandas as pd

from ._util import col
from ..schema import Dataset


def load(root, **_):
    df = pd.read_csv(os.path.join(root, "Darwin", "data.csv"))
    idc, yc = col(df, "ID"), col(df, "class")
    y = df[yc].astype(str).str.upper().str.startswith("P").astype(int).to_numpy()
    X = df.drop(columns=[idc, yc]).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    ids = df[idc].astype(str).to_numpy()
    return Dataset("darwin", X, y, ids, ids, task="Alzheimer's disease vs healthy, handwriting (1 sample/participant)",
                   notes=dict(role="negative control"))
