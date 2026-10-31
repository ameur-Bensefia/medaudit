"""Calibration of the A3 near-duplicate search, per dataset (modality).
False positives: for every sample, the highest cosine similarity to a sample of ANOTHER group, after merging the
groups documented as identical; pairs above a threshold are apparent duplicates between distinct participants
(an upper bound on the false-positive count, since undiscovered duplicates would also be counted).
Sensitivity: copies of randomly chosen samples are injected with Gaussian noise of increasing size (fraction of
each feature's standard deviation) and the proportion recovered as mutual nearest neighbours above the threshold
is recorded.
Usage: python scripts/08_duplicate_calibration.py [dataset ...]"""
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from _common import load_cfg, out
from medaudit import io
from medaudit.audits.a2_identity import _prep

THRESHOLDS = (0.99, 0.995, 0.999)
NOISE = (0.0, 0.01, 0.05, 0.10)
DEFAULT = ["uci_voice", "hw_pahaw", "hw_newhandpdsig", "darwin", "mitbih", "chbmit", "pamap2", "uci_har", "ptbxl",
           "ham10000", "nih_cxr", "chexpert", "abide"]
cfg = load_cfg()
dsdir = out(cfg, "datasets", "x")[:-2]
rows = []
for name in sys.argv[1:] or DEFAULT:
    try:
        ds = io.load(name, dsdir)
    except FileNotFoundError:
        continue
    rng = np.random.default_rng(0)
    idx = np.arange(len(ds.y)) if len(ds.y) <= 20000 else rng.choice(len(ds.y), 20000, replace=False)
    g = ds.group[idx].astype(str)
    for a, b in ds.notes.get("known_shared_groups", []) or []:
        g[g == str(b)] = str(a)
    Z = _prep(ds.X[idx], n_comp=128)
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    nn = NearestNeighbors(n_neighbors=min(30, len(Zn) - 1)).fit(Zn)
    dist, j = nn.kneighbors(Zn)
    best = np.full(len(Zn), -1.0)
    for i in range(len(Zn)):                       # nearest neighbour belonging to another group
        for dd, jj in zip(dist[i, 1:], j[i, 1:]):
            if g[jj] != g[i]:
                best[i] = 1 - dd ** 2 / 2
                break
    for t in THRESHOLDS:
        rows.append(dict(dataset=name, kind="false_positive", threshold=t, noise=np.nan, n=len(Zn),
                         rate=float(np.mean(best >= t)), count=int(np.sum(best >= t))))
    sd = np.nan_to_num(np.nanstd(ds.X[idx], axis=0)) + 1e-12      # ignore missing values
    pick = rng.choice(len(idx), min(200, len(idx)), replace=False)
    for noise in NOISE:
        Xc = ds.X[idx][pick] + rng.standard_normal((len(pick), ds.X.shape[1])) * sd * noise
        Zall = _prep(np.vstack([ds.X[idx], Xc]), n_comp=128)
        Zall /= np.linalg.norm(Zall, axis=1, keepdims=True) + 1e-12
        nn2 = NearestNeighbors(n_neighbors=5).fit(Zall)
        _, j2 = nn2.kneighbors(Zall[len(idx):])
        selfidx = len(idx) + np.arange(len(pick))
        top3 = [[x for x in row if x != si][:3] for row, si in zip(j2, selfidx)]
        in_top = np.array([pk in t for pk, t in zip(pick, top3)])     # original among the 3 nearest neighbours
        cos = np.sum(Zall[len(idx):] * Zall[pick], axis=1)           # similarity between copy and original
        for t in THRESHOLDS:
            ok = in_top & (cos >= t)
            rows.append(dict(dataset=name, kind="sensitivity", threshold=t, noise=noise, n=len(pick),
                             rate=float(np.mean(ok)), count=int(np.sum(ok))))
    print(f"  {name}: done", flush=True)
res = pd.DataFrame(rows)
res.to_csv(out(cfg, "tables", "duplicate_calibration.csv"), index=False)
fp = res[res.kind == "false_positive"].pivot_table(index="dataset", columns="threshold", values="rate")
se = res[res.kind == "sensitivity"].pivot_table(index=["dataset", "noise"], columns="threshold", values="rate")
print("false-positive rate (apparent duplicates between distinct participants):\n", fp.round(4).to_string())
print("\nsensitivity (injected copies recovered) by noise level:\n", se.round(3).to_string())
