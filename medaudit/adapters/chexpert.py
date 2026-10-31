"""CheXpert. Task: pleural effusion (label 1) vs no finding (label 1), frontal radiographs; uncertain labels
excluded. Group: patient. Confounders: age, sex, AP/PA. Same task definition as nih_cxr for external validation."""
import os
import re

import numpy as np
import pandas as pd

from ._util import col
from ..features.images import embed
from ..schema import Dataset


def load(root, cache_dir="cache", max_images=20000, seed=0, **_):
    base = os.path.join(root, "chexpert")
    df = pd.concat([pd.read_csv(os.path.join(base, f)) for f in ("train.csv", "valid.csv")
                    if os.path.exists(os.path.join(base, f))], ignore_index=True)
    fl = col(df, "Frontal/Lateral")
    df = df[df[fl] == "Frontal"].copy()
    eff, nof = col(df, "Pleural Effusion"), col(df, "No Finding")
    pos = df[eff] == 1
    neg = (df[nof] == 1) & ~(df[eff] == 1) & ~(df[eff] == -1)
    df = df[pos | neg].copy()
    df["y"] = (df[eff] == 1).astype(int)
    pth = col(df, "Path")
    df["rel"] = df[pth].str.replace(r"^[^/]*CheXpert[^/]*/", "", regex=True)
    df["local"] = base + os.sep + df["rel"]
    df["patient"] = df["rel"].map(lambda s: re.search(r"(patient\d+)", s).group(1))
    df = df[df["local"].map(os.path.exists)].drop_duplicates("rel").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    pats = df["patient"].unique()
    rng.shuffle(pats)
    sizes = df.groupby("patient").size()
    keep, n = [], 0
    for p in pats:
        if n >= max_images:
            break
        keep.append(p)
        n += sizes[p]
    df = df[df["patient"].isin(keep)].reset_index(drop=True)
    os.makedirs(cache_dir, exist_ok=True)
    X = embed(list(df["local"]), os.path.join(cache_dir, f"emb_chexpert_{max_images}_{seed}.npy"))
    appa = col(df, "AP/PA", required=False)
    conf = pd.DataFrame({"age": pd.to_numeric(df[col(df, "Age")], errors="coerce"),
                         "sex": df[col(df, "Sex")].astype(str),
                         "view": df[appa].astype(str) if appa else "NA"})
    return Dataset("chexpert", X, df["y"].to_numpy(), "chx_" + df["patient"].to_numpy(), df["rel"].to_numpy(),
                   np.full(len(df), "CheXpert"), conf, df["local"].to_numpy(),
                   task="pleural effusion vs no finding, chest radiographs (ResNet-50 embeddings)")
