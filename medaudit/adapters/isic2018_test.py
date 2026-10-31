"""ISIC 2018 Task 3 test set (1,512 images), external dataset for HAM10000 (melanoma vs nevus).
Two ground-truth formats are supported: the newer metadata format (lesion_id, image_id, dx, age, sex, ...), which
gives lesion identifiers, and the original one-hot format (image, MEL, NV, ...), where each image is its own group.
Lesion identifiers shared with HAM10000 are counted by the cross-dataset check of A3."""
import os

import numpy as np
import pandas as pd

from ._util import col, image_index
from ..features.images import embed
from ..schema import Dataset


def load(root, cache_dir="cache", **_):
    base = os.path.join(root, "HAM10000")
    gt = pd.read_csv(os.path.join(base, "ISIC2018_Task3_Test_GroundTruth.csv"))
    conf = None
    if col(gt, "dx", required=False):
        dx, img = col(gt, "dx"), col(gt, "image_id", "image")
        gt = gt[gt[dx].isin(["mel", "nv"])].reset_index(drop=True)
        y = (gt[dx] == "mel").astype(int)
        les = col(gt, "lesion_id", required=False)
        src = col(gt, "dataset", required=False)
    else:
        img, mel, nv = col(gt, "image", "image_id"), col(gt, "MEL"), col(gt, "NV")
        gt = gt[(gt[mel] == 1) | (gt[nv] == 1)].reset_index(drop=True)
        y, les, src = (gt[mel] == 1).astype(int), None, None
    members = image_index(base, ["ISIC2018_Task3_Test_Images"])
    keep = gt[img].isin(members).to_numpy()
    gt, y = gt[keep].reset_index(drop=True), y[keep].to_numpy()
    os.makedirs(cache_dir, exist_ok=True)
    X = embed([members[i] for i in gt[img]], os.path.join(cache_dir, f"emb_isic2018_test_{len(gt)}.npy"))
    ids = gt[img].astype(str).to_numpy()
    groups = gt[les].astype(str).to_numpy() if les else ids
    if col(gt, "age", required=False):
        conf = pd.DataFrame({"age": pd.to_numeric(gt[col(gt, "age")], errors="coerce"),
                             "sex": gt[col(gt, "sex")].astype(str) if col(gt, "sex", required=False) else "NA",
                             "localization": gt[col(gt, "localization")].astype(str)
                             if col(gt, "localization", required=False) else "NA"})
    site = gt[src].astype(str).to_numpy() if src else np.full(len(gt), "ISIC2018-test")
    return Dataset("isic2018_test", X, y, groups, ids, site, conf, task="melanoma vs nevus, external test set")
