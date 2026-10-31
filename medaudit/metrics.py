"""Metrics at sample and group level. Binary: `score` = P(y=1). Multi-class: per-class probabilities in
columns p0..pK (and `score` = probability of the predicted class)."""
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def _pcols(df):
    return [c for c in df.columns if len(c) > 1 and c[0] == "p" and c[1:].isdigit()]


def _pred_label(df):
    pc = _pcols(df)
    if pc:
        return df[pc].to_numpy().argmax(1)
    return (df["score"].to_numpy() >= 0.5).astype(int)


def bacc(df) -> float:
    if len(df) == 0 or df["y_true"].nunique() < 2:
        return float("nan")
    return float(balanced_accuracy_score(df["y_true"], _pred_label(df)))


def auc(df) -> float:
    if len(df) == 0 or df["y_true"].nunique() < 2:
        return float("nan")
    pc = _pcols(df)
    try:
        if pc:
            present = sorted(df["y_true"].unique())
            if len(present) < len(pc):
                return float("nan")
            return float(roc_auc_score(df["y_true"], df[pc].to_numpy(), multi_class="ovr", average="macro"))
        return float(roc_auc_score(df["y_true"], df["score"]))
    except ValueError:
        return float("nan")


def to_group_level(df) -> pd.DataFrame:
    """One row per (seed, group): mean score / probabilities, majority true label."""
    agg = {"y_true": lambda s: s.mode().iloc[0], "score": "mean"}
    agg.update({c: "mean" for c in _pcols(df)})
    keys = [k for k in ("seed", "group") if k in df.columns]
    return df.groupby(keys, sort=False).agg(agg).reset_index()


def metric(df, name="bacc", level="sample") -> float:
    """Metric computed per seed, then averaged over seeds."""
    f = {"bacc": bacc, "auc": auc}[name]
    d = to_group_level(df) if level == "group" else df
    if "seed" in d.columns and d["seed"].nunique() > 1:
        return float(np.nanmean([f(g) for _, g in d.groupby("seed")]))
    return f(d)
