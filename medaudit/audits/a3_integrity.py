"""A3 data integrity: duplicated samples, groups sharing identical data, and duplicates across splits."""
import hashlib

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .a2_identity import _prep


def exact_duplicates(ds, decimals=5):
    keys = [hashlib.md5(np.round(r, decimals).tobytes()).hexdigest() for r in np.nan_to_num(ds.X)]
    s = pd.Series(keys)
    dup = s[s.duplicated(keep=False)]
    sets = dup.groupby(dup).groups
    cross_group = sum(1 for idx in sets.values() if len(set(ds.group[list(idx)])) > 1)
    return dict(exact_dup_samples=int(len(dup)), exact_dup_sets=int(len(sets)), exact_dup_sets_cross_group=int(cross_group))


def near_duplicates(ds, sim=0.999, max_n=50000, seed=0):
    """Mutual nearest neighbours with cosine similarity >= sim (on standardized features)."""
    idx = np.arange(len(ds.y))
    if len(idx) > max_n:
        idx = np.random.default_rng(seed).choice(idx, max_n, replace=False)
    Z = _prep(ds.X[idx], n_comp=128)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    nn = NearestNeighbors(n_neighbors=2).fit(Z)
    dist, j = nn.kneighbors(Z)
    j1, d1 = j[:, 1], dist[:, 1]
    cos = 1 - d1 ** 2 / 2
    mutual = j[j1, 1] == np.arange(len(idx))
    pairs = [(idx[a], idx[j1[a]]) for a in np.where(mutual & (cos >= sim))[0] if a < j1[a]]
    cross = [(a, b) for a, b in pairs if ds.group[a] != ds.group[b]]
    return dict(near_dup_pairs=len(pairs), near_dup_pairs_cross_group=len(cross),
                near_dup_examples=[(ds.sample_id[a], ds.sample_id[b]) for a, b in cross[:10]])


def cross_dataset_duplicates(ds_a, ds_b, sim=0.995, max_n=50000, seed=0):
    """Samples of ds_b whose nearest neighbour in ds_a has cosine similarity >= sim (features standardized with
    ds_a statistics). Detects images or recordings shared between public datasets."""
    rng = np.random.default_rng(seed)
    ia = np.arange(len(ds_a.y)) if len(ds_a.y) <= max_n else rng.choice(len(ds_a.y), max_n, replace=False)
    A, B = np.nan_to_num(ds_a.X[ia]), np.nan_to_num(ds_b.X)
    mu, sd = A.mean(0), A.std(0) + 1e-9
    A, B = (A - mu) / sd, (B - mu) / sd
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-12
    d, j = NearestNeighbors(n_neighbors=1).fit(A).kneighbors(B)
    cos = 1 - d[:, 0] ** 2 / 2
    hit = np.where(cos >= sim)[0]
    shared_groups = sorted(set(map(str, ds_a.group)) & set(map(str, ds_b.group)))
    return dict(cross_dataset=f"{ds_a.name}|{ds_b.name}", cross_dup_samples=int(len(hit)),
                shared_group_ids=len(shared_groups), shared_group_examples=shared_groups[:10],
                cross_dup_frac=float(len(hit) / max(len(B), 1)),
                cross_dup_examples=[(ds_b.sample_id[i], ds_a.sample_id[ia[j[i, 0]]], float(cos[i])) for i in hit[:10]])


def duplicates_across_splits(ds, preds, decimals=5):
    """Exact duplicates where one copy is in a test fold and another copy in the corresponding training set."""
    keys = pd.Series([hashlib.md5(np.round(r, decimals).tobytes()).hexdigest() for r in np.nan_to_num(ds.X)],
                     index=ds.sample_id)
    n_leak, n_test = 0, 0
    for (_, _), te in preds.groupby(["seed", "fold"]):
        test_ids = set(te["sample_id"])
        train_keys = set(keys[~keys.index.isin(test_ids)])
        tk = keys.loc[list(test_ids)]
        n_leak += int(tk.isin(train_keys).sum())
        n_test += len(tk)
    return dict(dup_test_samples_in_train=n_leak, dup_test_frac=n_leak / max(n_test, 1))


def label_consistency(ds):
    """Groups whose samples carry different labels (relevant when the label is a property of the group)."""
    s = pd.DataFrame({"g": ds.group, "y": ds.y}).groupby("g")["y"].nunique()
    return dict(groups_with_mixed_labels=int((s > 1).sum()), groups_total=int(len(s)))


def audit_integrity(ds, preds=None, sim=0.999, known_shared_groups=None):
    """known_shared_groups: list of (group_a, group_b) documented as the same person (e.g. by the dataset
    authors); these are reported and should be merged before splitting."""
    out = dict(module="A3")
    out.update(exact_duplicates(ds))
    out.update(near_duplicates(ds, sim=sim))
    if preds is not None:
        out.update(duplicates_across_splits(ds, preds))
    out.update(label_consistency(ds))
    out["known_shared_groups"] = known_shared_groups or []
    leak = out.get("dup_test_samples_in_train", 0)
    cross = out["exact_dup_sets_cross_group"] + out["near_dup_pairs_cross_group"]
    problems = []
    if leak:
        problems.append(f"{leak} test samples duplicated in training")
    if cross:
        problems.append(f"{cross} duplicate sets span different groups")
    if out["known_shared_groups"]:
        problems.append(f"{len(out['known_shared_groups'])} documented shared groups")
    if leak or out["exact_dup_sets_cross_group"] or out["known_shared_groups"]:
        rating = "red"
    elif cross or out["exact_dup_samples"]:
        rating = "amber"
        if not problems:
            problems.append(f"{out['exact_dup_samples']} duplicated samples within groups")
    else:
        rating = "green"
    out.update(rating=rating, reason="; ".join(problems) if problems else "no duplicates found")
    return out


# ------------------------------------------------------------------ image duplicates (optional, Pillow only)
def dhash(path, size=16):
    from PIL import Image
    im = Image.open(path).convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def image_duplicates(paths_a, paths_b, max_bits=12, margin=6, min_corr=0.95):
    """Mutual nearest dhash neighbours between two image lists, confirmed by pixel correlation."""
    from PIL import Image
    Ha = np.array([dhash(p) for p in paths_a])
    Hb = np.array([dhash(p) for p in paths_b])
    D = (Ha[:, None, :] != Hb[None, :, :]).sum(2)
    ab, ba = D.argmin(1), D.argmin(0)
    srt = np.sort(D, 1)
    out = []
    for i in range(len(paths_a)):
        j = ab[i]
        if ba[j] == i and D[i, j] <= max_bits and (srt[i, 1] - srt[i, 0] if D.shape[1] > 1 else 99) >= margin:
            a = np.asarray(Image.open(paths_a[i]).convert("L").resize((128, 128)), float).ravel()
            b = np.asarray(Image.open(paths_b[j]).convert("L").resize((128, 128)), float).ravel()
            r = np.corrcoef(a, b)[0, 1]
            if r >= min_corr:
                out.append((i, j, int(D[i, j]), float(r)))
    return out
