"""ABIDE I and II, preprocessed structural MRI (one image per participant). Task: autism vs typical control.
Group: participant. Site: acquisition site (many sites -> leave-one-site-out). Confounders: age, sex.
Works with 3D volumes or 2D slices (some public copies provide one 224 x 224 axial slice per participant);
the image is intensity-normalized and downsampled to a coarse grid (cached)."""
import glob
import os

import numpy as np
import pandas as pd

from ._util import col, read_table
from ..schema import Dataset


def _feat(path, shape=(20, 24, 20)):
    import nibabel as nib
    from scipy.ndimage import zoom
    v = np.asarray(nib.load(path).get_fdata(), np.float32)
    if v.ndim > 3:
        v = v[..., 0]
    m = v > np.percentile(v, 5)
    v = (v - v[m].mean()) / (v[m].std() + 1e-6)
    return zoom(v, [s / d for s, d in zip(shape, v.shape)], order=1).ravel()


SITE_FILES = ("Phenotypic_V1_0b_preprocessed1.csv", "Phenotypic_V1_0b.csv", "ABIDEII_Composite_Phenotypic.csv",
              "ABIDE_II_Composite_Phenotypic.csv")


def _add_sites(base, df):
    """Merge SITE_ID from the official phenotype files when the tables used here do not contain it."""
    if col(df, "SITE_ID", "site", "SITE", required=False):
        return df
    idc = col(df, "SUB_ID", "subject", "Subject ID", "subject_id", "ID")
    maps = []
    for f in SITE_FILES:
        p = os.path.join(base, f)
        if os.path.exists(p):
            ph = pd.read_csv(p, encoding="latin-1")
            i, s = col(ph, "SUB_ID"), col(ph, "SITE_ID", required=False)
            if s:
                maps.append(ph[[i, s]].rename(columns={i: "_sid", s: "SITE_ID"}))
    if not maps:
        return df
    m = pd.concat(maps).drop_duplicates("_sid")
    m["_sid"] = m["_sid"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["_sid"] = df[idc].astype(str).str.replace(r"\.0$", "", regex=True)
    return df.merge(m, on="_sid", how="left").drop(columns="_sid")


# Duplicates documented in the public Kaggle copy of ABIDE (byte-identical files; same participant in both releases, pixel r = 0.999)
KNOWN_DUPLICATES = [("ABIDE1_50279", "ABIDE1_50286"), ("ABIDE2_29241", "ABIDE2_29245"),
                    ("ABIDE1_50778", "ABIDE2_29351"), ("ABIDE1_50811", "ABIDE2_29357")]


def load(root, cache_dir="cache", merge_duplicates=False, **_):
    base = os.path.join(root, "ABIDE")
    rows = []
    for rel, folder in (("abide1_data.csv", "ABIDE1_Preprocess"), ("abide2_data.csv", "ABIDE2_Preprocess")):
        ph = read_table(os.path.join(base, rel))
        idc = col(ph, "SUB_ID", "subject", "Subject ID", "subject_id", "ID")
        files = {os.path.basename(p)[:-4]: p for p in glob.glob(os.path.join(base, folder, "*.nii"))}
        ph["path"] = ph[idc].astype(str).str.replace(r"\.0$", "", regex=True).map(files)
        ph["release"] = folder[:6]
        rows.append(ph.dropna(subset=["path"]))
    df = pd.concat(rows, ignore_index=True)
    df = _add_sites(base, df)
    dx = col(df, "DX_GROUP", "dx_group", "DX")
    df = df[df[dx].isin([1, 2])].reset_index(drop=True)
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, "abide_features.npz")
    paths = df["path"].astype(str).to_numpy()
    X = None
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if len(z["ids"]) == len(df) and list(map(str, z["ids"])) == list(paths):
            X = z["X"]
    if X is None:
        X = np.vstack([_feat(p) for p in paths])
        np.savez_compressed(cache, X=X, ids=paths.astype("U"))
    sc = col(df, "SITE_ID", "site", "SITE", required=False)
    if sc is None or df[sc].isna().all():
        print("WARNING: ABIDE phenotype tables have no SITE_ID column; site analyses (A2 site, A6, predictive "
              "validity) are disabled. Add Phenotypic_V1_0b_preprocessed1.csv (ABIDE I) and "
              "ABIDEII_Composite_Phenotypic.csv (ABIDE II) to the ABIDE folder.")
        site = None
    else:
        site = df[sc].fillna("unknown").astype(str).to_numpy()
    sex = col(df, "SEX", "sex", required=False)
    conf = pd.DataFrame({"age": pd.to_numeric(df[col(df, "AGE_AT_SCAN", "age")], errors="coerce"),
                         "sex": df[sex].astype(str) if sex else "NA"})
    ids = df["release"] + "_" + df["path"].map(lambda p: os.path.basename(p)[:-4])
    groups = ids.to_numpy().copy()
    if merge_duplicates:
        mp = {b: a for a, b in KNOWN_DUPLICATES}
        groups = np.array([mp.get(x, x) for x in groups])
    ds = Dataset("abide", X, (df[dx] == 1).astype(int).to_numpy(), groups, ids.to_numpy(), site, conf,
                 task="autism vs typical control, structural MRI (2D slices in the copy used)")
    ds.notes["known_shared_groups"] = KNOWN_DUPLICATES
    return ds
