"""A1 split integrity: do groups cross training and test sets, and how much does it inflate performance?"""
import numpy as np
import pandas as pd

from ..bootstrap import bacc_diff_ci, group_bootstrap
from ..metrics import metric


def overlap_from_predictions(ds, preds):
    """Overlap computed from the prediction table: in cross-validation every sample not in a fold's test set
    is assumed to be in its training set."""
    all_groups = pd.Series(ds.group)
    rows = []
    for (seed, fold), te in preds.groupby(["seed", "fold"]):
        test_ids = set(te["sample_id"])
        train_groups = set(all_groups[~np.isin(ds.sample_id, list(test_ids))])
        g_te = set(te["group"])
        rows.append(dict(seed=seed, fold=fold, shared_group_frac=len(g_te & train_groups) / max(len(g_te), 1),
                         shared_sample_frac=float(te["group"].isin(train_groups).mean())))
    return pd.DataFrame(rows)


def audit_split(ds, preds_reported, preds_group=None, metric_name="bacc", n_boot=2000,
                amber=0.0, red=0.05, level=None, margin=0.02):
    """A1 assesses INFLATION (gap = reported - participant-wise, at the evaluation level of the task):
      red          lower bound of the 95% interval > 0 and gap >= `red` (default 5 points): inflation detected
      amber        lower bound of the 95% interval > 0 and gap < `red`: small inflation detected
      green        no group crosses splits, or the upper bound of the 90% interval <= `margin` (one-sided test at
                   alpha = 0.05 that inflation is smaller than the margin, default 2 points); negative gaps, where the
                   conventional estimate is lower, are therefore green when their interval stays below the margin
      inconclusive no inflation detected, but the upper bound exceeds the margin: inflation not excluded"""
    """preds_reported: predictions under the protocol to audit (e.g. sample-wise);
    preds_group: predictions of the same model under a group-wise protocol (optional, needed for the gap)."""
    lv = level or ds.default_level()          # captured before the loop below reuses the name `level`
    ov = overlap_from_predictions(ds, preds_reported)
    out = dict(module="A1", shared_group_frac=float(ov["shared_group_frac"].mean()),
               shared_sample_frac=float(ov["shared_sample_frac"].mean()))
    for lev in ("sample", "group"):
        rep = metric(preds_reported, metric_name, lev)
        out[f"{lev}_reported"] = rep
        if preds_group is not None:
            grp = metric(preds_group, metric_name, lev)
            if metric_name == "bacc":
                lo, hi = bacc_diff_ci(preds_reported, preds_group, lev, n_boot)
                elo, ehi = bacc_diff_ci(preds_reported, preds_group, lev, n_boot, alpha=0.10)
                out.update({f"{lev}_delta_eq_lo": elo, f"{lev}_delta_eq_hi": ehi})
            else:
                lo, hi, _ = group_bootstrap(lambda a, b, L=lev: metric(a, metric_name, L) - metric(b, metric_name, L),
                                            [preds_reported, preds_group], n_boot)
            out.update({f"{lev}_groupwise": grp, f"{lev}_delta": rep - grp,
                        f"{lev}_delta_lo": lo, f"{lev}_delta_hi": hi})
    # rating on the gap at the level where the label is defined (group if the label is a property of the group)
    out["rating_level"] = lv
    if preds_group is None:
        rating = "green" if out["shared_group_frac"] == 0 else "amber"
        reason = "no group crosses splits" if rating == "green" else "groups cross splits; gap not measured"
    elif out["shared_group_frac"] == 0:
        rating, reason = "green", "no group crosses training and test sets"
    elif not out[f"{lv}_delta_lo"] > 0:
        elo, ehi = out.get(f"{lv}_delta_eq_lo", out[f"{lv}_delta_lo"]), out.get(f"{lv}_delta_eq_hi", out[f"{lv}_delta_hi"])
        if ehi <= margin:
            rating = "green"
            reason = (f"inflation smaller than {100 * margin:.0f} points (one-sided test; 90% CI {100 * elo:.1f} to "
                      f"{100 * ehi:.1f})")
        else:
            rating = "inconclusive"
            reason = (f"inflation neither detected nor excluded: upper bound {100 * ehi:.1f} points exceeds the "
                      f"{100 * margin:.0f}-point margin")
    elif out[f"{lv}_delta"] >= red:
        rating, reason = "red", f"gap {100 * out[f'{lv}_delta']:.1f} points ({lv} level) with interval excluding zero"
    else:
        rating, reason = "amber", f"gap {100 * out[f'{lv}_delta']:.1f} points ({lv} level) with interval excluding zero"
    out.update(rating=rating, reason=reason, equivalence_margin=margin)
    return out
