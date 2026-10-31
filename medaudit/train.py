"""Cross-validated training under a protocol, returning the long prediction table used by the audit."""
import numpy as np
import pandas as pd

from .models import make_model
from .splits import make_folds


def _rebalance_weights(y):
    counts = np.bincount(y)
    w = len(y) / (len(counts) * np.maximum(counts, 1))
    return w[y]


def _predict(model, X):
    return model.predict_proba(X)


def _new_model(model, n_features, seed):
    """`model` is a reference model name or a callable (n_features, seed) -> unfitted scikit-learn estimator."""
    return model(n_features, seed) if callable(model) else make_model(model, n_features, seed)


def _name(model):
    return getattr(model, "__name__", "custom") if callable(model) else model


def run_cv(ds, model_name, protocol, seeds=(0,), k=5, max_train=None):
    """Train `model_name` with folds from `protocol`; return predictions for every test sample."""
    rows = []
    for seed in seeds:
        folds = make_folds(ds, protocol, k, seed)
        if not folds:
            raise ValueError(f"{ds.name}: no valid folds for protocol '{protocol}'")
        for f, (tr, te) in enumerate(folds):
            if max_train and len(tr) > max_train:
                tr = np.random.default_rng(seed + 31 * f).choice(tr, max_train, replace=False)
            model = _new_model(model_name, ds.X.shape[1], seed)
            model.fit(ds.X[tr], ds.y[tr])
            rows.append(_frame(ds, te, _predict(model, ds.X[te]), model.classes_, f, seed, protocol, _name(model_name)))
    return pd.concat(rows, ignore_index=True)


def run_external(train_ds, test_ds, model_name, seeds=(0,), max_train=None, tag=None):
    rows = []
    for seed in seeds:
        tr = np.arange(len(train_ds.y))
        if max_train and len(tr) > max_train:
            tr = np.random.default_rng(seed).choice(tr, max_train, replace=False)
        model = _new_model(model_name, train_ds.X.shape[1], seed)
        model.fit(train_ds.X[tr], train_ds.y[tr])
        te = np.arange(len(test_ds.y))
        rows.append(_frame(test_ds, te, _predict(model, test_ds.X), model.classes_, 0, seed,
                           tag or f"external:{train_ds.name}->{test_ds.name}", _name(model_name)))
    return pd.concat(rows, ignore_index=True)


def _frame(ds, te, proba, classes, fold, seed, protocol, model_name):
    n_cls = max(ds.n_classes, len(classes))
    d = pd.DataFrame({"sample_id": ds.sample_id[te], "group": ds.group[te],
                      "site": ds.site[te] if ds.site is not None else "",
                      "y_true": ds.y[te], "fold": fold, "seed": seed, "protocol": protocol, "model": model_name})
    if n_cls <= 2:
        pos = list(classes).index(1) if 1 in classes else proba.shape[1] - 1
        d["score"] = proba[:, pos]
    else:
        full = np.zeros((len(te), n_cls))
        for j, c in enumerate(classes):
            full[:, int(c)] = proba[:, j]
        for c in range(n_cls):
            d[f"p{c}"] = full[:, c]
        d["score"] = full.max(1)
    return d
