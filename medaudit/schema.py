"""Core data structures.

A Dataset holds one row per sample. `group` identifies the unit that must not cross splits (participant,
patient, lesion); `site` the acquisition site, device or source; `confounders` variables such as age and sex.
Predictions are long-format tables with one row per (seed, fold, test sample).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    name: str
    X: np.ndarray                      # features or embeddings, n x d
    y: np.ndarray                      # labels (int)
    group: np.ndarray                  # unit that must not be split (str)
    sample_id: np.ndarray              # unique sample identifier (str)
    site: np.ndarray | None = None     # acquisition site / source (str)
    confounders: pd.DataFrame | None = None
    paths: np.ndarray | None = None    # raw files, for duplicate search (A3)
    task: str = ""
    notes: dict = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.y)
        self.X = np.asarray(self.X, dtype=np.float32)
        self.y = np.asarray(self.y).astype(int)
        self.group = np.asarray(self.group).astype(str)
        self.sample_id = np.asarray(self.sample_id).astype(str)
        if self.site is not None:
            self.site = np.asarray(self.site).astype(str)
        assert self.X.shape[0] == n == len(self.group) == len(self.sample_id), "inconsistent lengths"
        assert len(set(self.sample_id)) == n, "sample_id must be unique"
        if self.confounders is not None:
            self.confounders = self.confounders.reset_index(drop=True)
            assert len(self.confounders) == n

    @property
    def n_classes(self):
        return int(len(np.unique(self.y)))

    def summary(self) -> dict:
        per = pd.Series(self.group).value_counts()
        out = dict(dataset=self.name, task=self.task, samples=len(self.y), groups=int(per.size),
                   samples_per_group_median=float(per.median()), samples_per_group_max=int(per.max()),
                   classes=self.n_classes, features=int(self.X.shape[1]),
                   sites=int(len(set(self.site))) if self.site is not None else 0)
        lab = pd.Series(self.y).value_counts().sort_index()
        out["class_counts"] = ";".join(f"{k}:{v}" for k, v in lab.items())
        return out

    def default_level(self, purity=0.95) -> str:
        """'group' when the label is a property of the group (constant within >=95% of groups), else 'sample'."""
        s = pd.DataFrame({"g": self.group, "y": self.y}).groupby("g")["y"].agg(
            lambda v: v.value_counts(normalize=True).iloc[0])
        return "group" if float((s >= purity).mean()) >= 0.95 else "sample"

    def subset(self, mask) -> "Dataset":
        m = np.asarray(mask)
        return Dataset(self.name, self.X[m], self.y[m], self.group[m], self.sample_id[m],
                       None if self.site is None else self.site[m],
                       None if self.confounders is None else self.confounders[m].reset_index(drop=True),
                       None if self.paths is None else np.asarray(self.paths)[m], self.task, dict(self.notes))


def check_predictions(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in ["sample_id", "group", "y_true", "score"] if c not in df.columns]
    if missing:
        raise ValueError(f"prediction table lacks columns {missing}")
    df = df.copy()
    for c, d in (("fold", 0), ("seed", 0), ("protocol", "reported"), ("model", "model"), ("site", "")):
        if c not in df.columns:
            df[c] = d
    df["group"] = df["group"].astype(str)
    return df
