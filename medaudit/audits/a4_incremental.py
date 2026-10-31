"""Incremental value of the model beyond the covariates, evaluated on held-out participants.

At participant level, two logistic models are cross-fitted (stratified K-fold over participants, repeated):
  (0) label ~ covariates            (1) label ~ covariates + logit(score)
where score is the audited model's out-of-fold score averaged over seeds. Every prediction is made for participants
not used to fit the second-level model, so the comparison does not reuse the same participants for fitting and
testing. Effect size: dAUC = AUC(1) - AUC(0) on the pooled held-out predictions; 95% interval from a paired participant
bootstrap of the held-out predictions."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def participant_table(ds, preds):
    conf = ds.confounders.copy()
    conf["group"] = ds.group
    num = [c for c in ds.confounders.columns if pd.api.types.is_numeric_dtype(ds.confounders[c])]
    cat = [c for c in ds.confounders.columns if c not in num]
    g = conf.groupby("group").agg({**{c: "mean" for c in num}, **{c: lambda v: v.mode().iloc[0] for c in cat}})
    X = pd.get_dummies(g, columns=cat, drop_first=True, dtype=float)
    X = X.fillna(X.median())
    s = preds.groupby("group").agg(y=("y_true", lambda v: v.mode().iloc[0]), score=("score", "mean"))
    t = s.join(X, how="inner").dropna()
    sc = np.clip(t["score"].to_numpy(), 1e-4, 1 - 1e-4)
    t["logit_score"] = np.log(sc / (1 - sc))
    return t, list(X.columns)


def incremental_value(ds, preds, k=5, repeats=5, n_boot=2000, seed=0):
    if ds.n_classes != 2:
        return dict(inc_note="multi-class task: not applied")
    t, cov = participant_table(ds, preds)
    y = t["y"].to_numpy().astype(int)
    if len(set(y)) < 2 or len(t) < 20:
        return dict(inc_note="not enough participants")
    X0 = t[cov].to_numpy()
    X1 = t[cov + ["logit_score"]].to_numpy()
    p0 = np.zeros((repeats, len(y)))
    p1 = np.zeros((repeats, len(y)))
    for r in range(repeats):
        for tr, te in StratifiedKFold(min(k, int(np.bincount(y).min())), shuffle=True,
                                      random_state=seed + r).split(X0, y):
            mu, sd = X1[tr].mean(0), X1[tr].std(0) + 1e-9          # standardization fitted on training participants
            m0 = LogisticRegression(max_iter=5000).fit((X0[tr] - mu[:-1]) / sd[:-1], y[tr])
            m1 = LogisticRegression(max_iter=5000).fit((X1[tr] - mu) / sd, y[tr])
            p0[r, te] = m0.predict_proba((X0[te] - mu[:-1]) / sd[:-1])[:, 1]
            p1[r, te] = m1.predict_proba((X1[te] - mu) / sd)[:, 1]
    q0, q1 = p0.mean(0), p1.mean(0)                                  # average over repeats for each participant
    auc0, auc1 = roc_auc_score(y, q0), roc_auc_score(y, q1)
    # Cross-validated predictions of an uninformative model have an AUC that tends to fall below 0.5, which would
    # make any addition look like a gain; the baseline is therefore floored at chance: dAUC = AUC1 - max(AUC0, 0.5).
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], q1[i]) - max(roc_auc_score(y[i], q0[i]), 0.5))
    lo, hi = np.quantile(d, [0.025, 0.975])
    return dict(auc_covariates_heldout=float(auc0), auc_covariates_plus_model_heldout=float(auc1),
                delta_auc=float(auc1 - max(auc0, 0.5)), delta_auc_lo=float(lo), delta_auc_hi=float(hi),
                inc_participants=int(len(y)))


def incremental_value_nested(ds, model, outer_k=5, inner_k=5, repeats=1, n_boot=2000, seed=0):
    """Fully nested version of the incremental-value test (requires retraining the audited model, level 3).

    For each outer fold over participants, the audited model is retrained using the outer-training participants only:
    inner participant-wise cross-validation gives out-of-fold scores for the outer-training participants, and a model
    fitted on all outer-training participants scores the outer-test participants. The two second-level logistic models
    (covariates; covariates + logit(score)) are fitted on the outer-training participants and predict the outer-test
    participants. No score used to fit the second level was produced by a model that saw an outer-test participant.
    With repeats > 1, held-out probabilities are averaged per participant over repeats before computing the AUCs."""
    from ..train import run_cv, run_external
    if ds.n_classes != 2:
        return dict(inc_note="multi-class task: not applied")
    groups = np.array(sorted(set(ds.group)))
    gy = pd.Series(ds.y, index=ds.group).groupby(level=0).agg(lambda v: v.mode().iloc[0]).loc[groups].to_numpy()
    P0 = np.full((repeats, len(groups)), np.nan)
    P1 = np.full((repeats, len(groups)), np.nan)
    for r in range(repeats):
        for otr, ote in StratifiedKFold(outer_k, shuffle=True, random_state=seed + r).split(groups, gy):
            gtr, gte = groups[otr], groups[ote]
            dtr, dte = ds.subset(np.isin(ds.group, gtr)), ds.subset(np.isin(ds.group, gte))
            ptr = run_cv(dtr, model, "group", seeds=(seed + r,), k=inner_k)           # inner OOF scores
            pte = run_external(dtr, dte, model, seeds=(seed + r,))                   # scores for outer test
            ttr, cov = participant_table(dtr, ptr)
            tte, cov_te = participant_table(dte, pte)
            for c in cov:                                                            # align dummy columns
                if c not in tte:
                    tte[c] = 0.0
            X0tr, X1tr = ttr[cov].to_numpy(), ttr[cov + ["logit_score"]].to_numpy()
            X0te, X1te = tte[cov].to_numpy(), tte[cov + ["logit_score"]].to_numpy()
            mu, sd = X1tr.mean(0), X1tr.std(0) + 1e-9
            ytr = ttr["y"].to_numpy().astype(int)
            m0 = LogisticRegression(max_iter=5000).fit((X0tr - mu[:-1]) / sd[:-1], ytr)
            m1 = LogisticRegression(max_iter=5000).fit((X1tr - mu) / sd, ytr)
            idx = np.searchsorted(groups, tte.index.to_numpy())
            P0[r, idx] = m0.predict_proba((X0te - mu[:-1]) / sd[:-1])[:, 1]
            P1[r, idx] = m1.predict_proba((X1te - mu) / sd)[:, 1]
    ok = ~np.isnan(P0).any(0)
    y, q0, q1 = gy[ok].astype(int), np.nanmean(P0, 0)[ok], np.nanmean(P1, 0)[ok]
    auc0, auc1 = roc_auc_score(y, q0), roc_auc_score(y, q1)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], q1[i]) - max(roc_auc_score(y[i], q0[i]), 0.5))
    lo, hi = np.quantile(d, [0.025, 0.975])
    return dict(nested=True, auc_covariates_heldout=float(auc0), auc_covariates_plus_model_heldout=float(auc1),
                delta_auc=float(auc1 - max(auc0, 0.5)), delta_auc_lo=float(lo), delta_auc_hi=float(hi),
                inc_participants=int(len(y)))
