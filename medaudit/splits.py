"""Evaluation protocols. Each returns a list of (train_idx, test_idx)."""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

PROTOCOLS = ("sample", "group", "site")


def sample_wise(y, k=5, seed=0):
    k = min(k, int(np.bincount(y).min()))
    return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def group_wise(y, groups, k=5, seed=0):
    n_groups = len(set(groups))
    k = min(k, n_groups)
    return list(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups))


def site_wise(y, sites, min_test=20):
    """Leave-one-site-out; sites with fewer than min_test samples or a single class are skipped as test sites."""
    out = []
    for s in sorted(set(sites)):
        te = np.where(sites == s)[0]
        tr = np.where(sites != s)[0]
        if len(te) >= min_test and len(set(y[te])) > 1 and len(set(y[tr])) > 1:
            out.append((tr, te))
    return out


def make_folds(ds, protocol, k=5, seed=0):
    if protocol == "sample":
        return sample_wise(ds.y, k, seed)
    if protocol == "group":
        return group_wise(ds.y, ds.group, k, seed)
    if protocol == "site":
        if ds.site is None:
            raise ValueError(f"{ds.name}: no site information for the site protocol")
        return site_wise(ds.y, ds.site)
    raise ValueError(protocol)


def overlap(ds, folds):
    """Fraction of test groups (and test samples) whose group also appears in the training set, per fold."""
    rows = []
    for f, (tr, te) in enumerate(folds):
        seen = set(ds.group[tr])
        g_te = set(ds.group[te])
        rows.append(dict(fold=f, test_groups=len(g_te), shared_groups=len(g_te & seen),
                         shared_group_frac=len(g_te & seen) / max(len(g_te), 1),
                         shared_sample_frac=float(np.isin(ds.group[te], list(seen)).mean())))
    return rows
