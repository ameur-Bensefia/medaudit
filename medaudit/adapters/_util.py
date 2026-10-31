import os

import pandas as pd


def find(root, *cands, required=True):
    for c in cands:
        p = os.path.join(root, c)
        if os.path.exists(p):
            return p
    if required:
        raise FileNotFoundError(f"none of {cands} found under {root}")
    return None


def col(df, *cands, required=True):
    """Case-insensitive column lookup."""
    low = {c.lower().strip(): c for c in df.columns}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    if required:
        raise KeyError(f"none of {cands} in columns {list(df.columns)[:25]}")
    return None


def read_table(path):
    with open(path, "rb") as f:
        head = f.read(4096).decode("utf-8", "ignore")
    sep = "\t" if head.count("\t") > head.count(",") else ","
    return pd.read_csv(path, sep=sep)


ALIASES = {
    "UCI_PD_Voice": ["UCI_PD_Voice", "UCI-Voice PD", "UCI Voice PD"],
    "mit-bih": ["mit-bih", "mit-bh", "mitdb"],
    "PTB-XL": ["PTB-XL", "ptb-xl", "ptbxl"],
    "CHB-MIT": ["CHB-MIT", "CHB", "chb-mit", "chbmit"],
    "UCI-HAR Dataset": ["UCI-HAR Dataset", "UCI HAR Dataset", "UCI-HAR"],
}


def folder(root, name):
    """Return root/<name> or the first existing alias of it."""
    for cand in ALIASES.get(name, [name]):
        p = os.path.join(root, cand)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(f"folder {name} (or aliases {ALIASES.get(name, [])}) not found under {root}")


def image_index(base, names):
    """image_id -> path or (zip, member), searching folders first, then zip archives with the given stems."""
    import glob
    import zipfile
    idx = {}
    for n in names:
        d = os.path.join(base, n)
        if os.path.isdir(d):
            for p in glob.glob(os.path.join(d, "**", "*.jpg"), recursive=True):
                idx.setdefault(os.path.basename(p)[:-4], p)
    for n in names:
        z = os.path.join(base, n + ".zip")
        if os.path.exists(z):
            with zipfile.ZipFile(z) as zf:
                for m in zf.namelist():
                    if m.lower().endswith(".jpg"):
                        idx.setdefault(os.path.basename(m)[:-4], (z, m))
    return idx
