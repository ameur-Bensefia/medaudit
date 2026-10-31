"""NIH ChestX-ray14. Task: pleural effusion vs no finding (frontal radiographs). Group: patient. Confounders:
age, sex, view position. A random subset of patients (max_images) is used to keep the embedding step feasible."""
import os

import numpy as np
import pandas as pd

from ._util import col
from ..features.images import embed
from ..schema import Dataset


def load(root, cache_dir="cache", max_images=20000, seed=0, **_):
    base = os.path.join(root, "NIH ChestX")
    df = pd.read_csv(os.path.join(base, "Data_Entry_2017.csv"))
    lab = df[col(df, "Finding Labels")].astype(str)
    df = df[(lab == "No Finding") | lab.str.contains("Effusion")].copy()
    df["y"] = df[col(df, "Finding Labels")].str.contains("Effusion").astype(int)
    pid = col(df, "Patient ID")
    rng = np.random.default_rng(seed)
    pats = df[pid].unique()
    rng.shuffle(pats)
    sizes = df.groupby(pid).size()
    keep, n = [], 0
    for p in pats:
        if n >= max_images:
            break
        keep.append(p)
        n += sizes[p]
    df = df[df[pid].isin(keep)].reset_index(drop=True)
    paths = {}
    for d in sorted(os.listdir(base)):
        sub = os.path.join(base, d, "images")
        if d.startswith("images") and os.path.isdir(sub):
            for f in os.listdir(sub):
                paths[f] = os.path.join(sub, f)
    df = df[df[col(df, "Image Index")].isin(paths)].reset_index(drop=True)
    os.makedirs(cache_dir, exist_ok=True)
    items = [paths[f] for f in df[col(df, "Image Index")]]
    X = embed(items, os.path.join(cache_dir, f"emb_nih_{max_images}_{seed}.npy"))
    conf = pd.DataFrame({"age": pd.to_numeric(df[col(df, "Patient Age")], errors="coerce").clip(0, 100),
                         "sex": df[col(df, "Patient Gender")].astype(str),
                         "view": df[col(df, "View Position")].astype(str)})
    return Dataset("nih_cxr", X, df["y"].to_numpy(), "nih_" + df[pid].astype(str).to_numpy(),
                   df[col(df, "Image Index")].to_numpy(), np.full(len(df), "NIH"), conf, np.array(items),
                   task="pleural effusion vs no finding, chest radiographs (ResNet-50 embeddings)")
