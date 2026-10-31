"""A4 confounds: can confounders alone explain the result, and does the model add information beyond them?"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..bootstrap import bacc_ci
from ..splits import group_wise


def _conf_model(conf: pd.DataFrame):
    num = [c for c in conf.columns if pd.api.types.is_numeric_dtype(conf[c])]
    cat = [c for c in conf.columns if c not in num]
    tr = []
    if num:
        tr.append(("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num))
    if cat:
        tr.append(("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                        OneHotEncoder(handle_unknown="ignore")), cat))
    return make_pipeline(ColumnTransformer(tr), LogisticRegression(max_iter=3000, class_weight="balanced"))


def confound_only(ds, seeds=(0, 1, 2), k=5, n_boot=1000):
    """Group-wise cross-validated performance of a model that sees only the confounders."""
    conf = ds.confounders
    rows = []
    for seed in seeds:
        for f, (tr, te) in enumerate(group_wise(ds.y, ds.group, k, seed)):
            m = _conf_model(conf).fit(conf.iloc[tr], ds.y[tr])
            p = m.predict_proba(conf.iloc[te])
            d = pd.DataFrame({"sample_id": ds.sample_id[te], "group": ds.group[te], "y_true": ds.y[te],
                              "seed": seed, "fold": f})
            if p.shape[1] == 2:
                d["score"] = p[:, 1]
            else:
                for c in range(p.shape[1]):
                    d[f"p{m.classes_[c]}"] = p[:, c]
                d["score"] = p.max(1)
            rows.append(d)
    pr = pd.concat(rows, ignore_index=True)
    out = {}
    for level in ("sample", "group"):
        est, lo, hi = bacc_ci(pr, level, n_boot)
        out.update({f"confound_only_{level}": est, f"confound_only_{level}_lo": lo, f"confound_only_{level}_hi": hi})
    return out


def audit_confounds(ds, preds_group=None, alpha=0.05, n_tests=1):
    """A4 covariates. Reports (i) the participant-wise performance of a model that sees only the covariates (age,
    sex, view, ...) and (ii) whether the audited model adds information beyond them. A predictive covariate is not
    by itself a bias: age may be clinically relevant. The module flags situations that must be interpreted, not
    causal confounding, which requires a causal argument outside the data.
    n_tests: number of models audited together (Bonferroni adjustment of alpha)."""
    if ds.confounders is None or ds.confounders.shape[1] == 0:
        return dict(module="A4", rating="grey", reason="no covariates available")
    out = dict(module="A4", confounders=",".join(ds.confounders.columns))
    out.update(confound_only(ds))
    if preds_group is not None:
        from .a4_incremental import incremental_value
        out.update(incremental_value(ds, preds_group))
    chance = 1.0 / ds.n_classes
    above = out["confound_only_group_lo"] > chance
    cov = out["confounders"]
    if "delta_auc_lo" in out and not out["delta_auc_lo"] > 0:
        rating = "red"
        reason = (f"no detectable incremental value of the model beyond {cov} on held-out participants (dAUC "
                  f"{out['delta_auc']:+.3f}, 95% CI {out['delta_auc_lo']:+.3f} to {out['delta_auc_hi']:+.3f})")
    elif above:
        rating = "amber"
        reason = (f"{cov} alone reach {100 * out['confound_only_group']:.1f}% balanced accuracy; the model adds "
                  "information beyond them; interpret whether these covariates are clinically legitimate predictors")
    else:
        rating, reason = "green", "no association detected between the covariates alone and the label"
    out.update(rating=rating, reason=reason)
    return out


# ---------------------------------------------------------------- matched cohorts (used by the scripts)
def match_groups(ds, confounder, caliper):
    """1:1 global matching of groups (binary task) on one numeric confounder (Hungarian algorithm)."""
    from scipy.optimize import linear_sum_assignment
    g = pd.DataFrame({"g": ds.group, "y": ds.y, "c": ds.confounders[confounder].to_numpy()}).groupby("g").agg(
        y=("y", lambda v: v.mode().iloc[0]), c=("c", "mean")).dropna().sort_index()
    a, b = g[g.y == 1], g[g.y == 0]
    D = np.abs(a.c.to_numpy()[:, None] - b.c.to_numpy()[None, :])
    cost = np.where(D <= caliper, D, 1e6)
    r, c = linear_sum_assignment(cost)
    keep = cost[r, c] < 1e6
    return pd.DataFrame({"pair": np.arange(keep.sum()), "g1": a.index[r[keep]], "g0": b.index[c[keep]],
                         "diff": a.c.to_numpy()[r[keep]] - b.c.to_numpy()[c[keep]]})
