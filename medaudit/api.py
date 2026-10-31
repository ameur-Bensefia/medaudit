"""One-call entry point for researchers auditing their own model.

    import medaudit
    results = medaudit.audit(dataset, preds_reported, preds_groupwise, preds_external, html="card.html")

`dataset`          medaudit.Dataset with features (or embeddings), labels, group IDs, optional site/confounders
`preds_reported`   predictions under the protocol used in the study being audited (CSV or DataFrame)
`preds_groupwise`  predictions of the same model under group-wise cross-validation (optional but recommended)
`preds_external`   predictions on held-out sites or an external dataset (optional)
Prediction tables need columns sample_id, group, y_true, score (+ fold, seed when available).
"""
import pandas as pd

from .audits import (audit_confounds, audit_identity, audit_integrity, audit_split, audit_transfer,
                     audit_uncertainty)
from .report import to_html, to_json, verdict
from .schema import check_predictions


def _read(p):
    if p is None or isinstance(p, pd.DataFrame):
        return None if p is None else check_predictions(p)
    return check_predictions(pd.read_csv(p, dtype={"group": str, "sample_id": str}))


def audit(dataset, preds_reported, preds_groupwise=None, preds_external=None, n_boot=2000, n_perm=200,
          known_shared_groups=None, html=None, json_path=None, meta=None):
    pr, pg, pe = _read(preds_reported), _read(preds_groupwise), _read(preds_external)
    res = [audit_split(dataset, pr, pg, n_boot=n_boot),
           audit_identity(dataset, n_perm=n_perm),
           audit_integrity(dataset, pr, known_shared_groups=known_shared_groups)]
    res.append(audit_confounds(dataset, pg if pg is not None else pr))
    res.append(audit_uncertainty(dataset, pg if pg is not None else pr, n_boot=n_boot))
    if pe is not None:
        res.append(audit_transfer(pg if pg is not None else pr, pe, level=dataset.default_level(), n_boot=n_boot))
    meta = dict(meta or {}, dataset=dataset.name, samples=len(dataset.y))
    if html:
        to_html(res, meta, html)
    if json_path:
        to_json(res, meta, json_path)
    v, msg = verdict(res)
    print(f"Overall: {v}. {msg}")
    for r in res:
        print(f"  {r['module']} {r['rating']:5s} {r.get('reason', '')}")
    return res
