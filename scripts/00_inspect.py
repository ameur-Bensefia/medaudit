"""Check the Python environment and every dataset, using the same folder aliases as the loaders.
Run this first; every line should say ok."""
import glob
import importlib
import os
import sys
import warnings

warnings.filterwarnings("ignore")
from _common import load_cfg  # noqa: E402

print(f"python {sys.version.split()[0]}")
try:
    import numpy
    print(f"  numpy {numpy.__version__}" + ("   <-- must be < 2 with torch 2.2.2 (Intel Mac)" if numpy.__version__.startswith("2") else ""))
except Exception as e:
    print("  numpy MISSING", e)
for p in ["pandas", "sklearn", "scipy", "yaml", "PIL", "torch", "torchvision", "wfdb", "pyedflib", "nibabel", "matplotlib"]:
    try:
        m = importlib.import_module(p)
        print(f"  {p:12s} ok {getattr(m, '__version__', '')}")
    except Exception as e:
        print(f"  {p:12s} BROKEN ({e.__class__.__name__}: {str(e)[:80]})")
try:
    import torch
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu only")
    print(f"  torch device: {dev}")
except Exception:
    pass

from medaudit.adapters._util import folder  # noqa: E402

cfg = load_cfg()
root = cfg["root"]
print("\nroot:", root, "| exists:", os.path.isdir(root))


def count(pattern):
    return len(glob.glob(pattern, recursive=True))


def check(name, fn):
    try:
        msg = fn()
        print(f"  ok      {name:16s} {msg}")
    except Exception as e:
        print(f"  MISSING {name:16s} {e}")


check("uci_voice", lambda: f"{os.path.exists(os.path.join(folder(root, 'UCI_PD_Voice'), 'parkinsons.data'))} parkinsons.data")
check("uci_har", lambda: f"{count(os.path.join(folder(root, 'UCI-HAR Dataset'), '*', 'X_*.txt'))} feature files")
check("pamap2", lambda: f"{count(os.path.join(root, 'pamap2', '**', 'Protocol', 'subject*.dat')) or count(os.path.join(root, 'pamap2', '*.zip'))} protocol files (or zip)")
check("mitbih", lambda: f"{count(os.path.join(folder(root, 'mit-bih'), '[0-9][0-9][0-9].atr'))} annotated records")
check("chbmit", lambda: f"{count(os.path.join(folder(root, 'CHB-MIT'), 'chb*', '*.edf'))} EDF files, "
                        f"{count(os.path.join(folder(root, 'CHB-MIT'), 'chb*', '*summary*.txt'))} summaries")
check("ptbxl", lambda: f"{os.path.exists(os.path.join(folder(root, 'PTB-XL'), 'ptbxl_database.csv'))} database, "
                       f"{count(os.path.join(folder(root, 'PTB-XL'), 'records100', '*', '*.hea'))} records")
check("ham10000", lambda: f"{count(os.path.join(root, 'HAM10000', 'HAM10000_images_part_*', '*.jpg'))} images")
check("isic2018_test", lambda: f"{count(os.path.join(root, 'HAM10000', 'ISIC2018_Task3_Test_Images', '**', '*.jpg'))} images")
check("nih_cxr", lambda: f"{count(os.path.join(root, 'NIH ChestX', 'images_*', 'images', '*.png'))} images")
check("chexpert", lambda: f"{os.path.exists(os.path.join(root, 'chexpert', 'train.csv'))} train.csv")
check("darwin", lambda: f"{os.path.exists(os.path.join(root, 'Darwin', 'data.csv'))} data.csv")


def abide():
    import pandas as pd
    base = os.path.join(root, "ABIDE")
    cols = list(pd.read_csv(os.path.join(base, "abide1_data.csv"), nrows=1).columns)
    extra = [f for f in ("Phenotypic_V1_0b_preprocessed1.csv", "ABIDEII_Composite_Phenotypic.csv") if os.path.exists(os.path.join(base, f))]
    site = any(c.strip().upper() == "SITE_ID" for c in cols) or len(extra) == 2
    return (f"{count(os.path.join(base, 'ABIDE*_Preprocess', '*.nii'))} volumes; site information: "
            + ("yes" if site else "NO -> add Phenotypic_V1_0b_preprocessed1.csv and ABIDEII_Composite_Phenotypic.csv"))


check("abide", abide)
check("hw_pahaw", lambda: f"{count(os.path.join(root, 'PaHaW', '**', '*.svc'))} signals, corpus table "
                          f"{bool(glob.glob(os.path.join(root, 'PaHaW', '**', 'corpus_PaHaW.xlsx'), recursive=True))}")
check("hw_newhandpdsig", lambda: f"{count(os.path.join(root, 'NewHandPD', 'Signal_*', '*.txt'))} signals")
