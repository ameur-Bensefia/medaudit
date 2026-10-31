"""HAM10000 dermoscopy. Task: melanoma vs melanocytic nevus. Group: lesion_id (several images of one lesion).
Site: image source (`dataset` column when present). Confounders: age, sex, body site.
Images are read directly from the zip archives and embedded with a frozen ImageNet ResNet-50."""
import os

import pandas as pd

from ._util import col, find, image_index, read_table
from ..features.images import embed
from ..schema import Dataset


def load(root, cache_dir="cache", **_):
    base = os.path.join(root, "HAM10000")
    meta = read_table(find(base, "HAM10000_metadata", "HAM10000_metadata.csv", "HAM10000_metadata.tab"))
    dx, les, img = col(meta, "dx"), col(meta, "lesion_id"), col(meta, "image_id")
    meta = meta[meta[dx].isin(["mel", "nv"])].reset_index(drop=True)
    members = image_index(base, ["HAM10000_images_part_1", "HAM10000_images_part_2"])
    meta = meta[meta[img].isin(members)].reset_index(drop=True)
    os.makedirs(cache_dir, exist_ok=True)
    X = embed([members[i] for i in meta[img]], os.path.join(cache_dir, f"emb_ham10000_{len(meta)}.npy"))
    src = col(meta, "dataset", required=False)
    conf = pd.DataFrame({"age": pd.to_numeric(meta[col(meta, "age")], errors="coerce"),
                         "sex": meta[col(meta, "sex")].astype(str),
                         "localization": meta[col(meta, "localization")].astype(str)})
    return Dataset("ham10000", X, (meta[dx] == "mel").astype(int).to_numpy(), meta[les].to_numpy(),
                   meta[img].to_numpy(), meta[src].astype(str).to_numpy() if src else None, conf,
                   task="melanoma vs nevus, dermoscopy (ResNet-50 embeddings)")
