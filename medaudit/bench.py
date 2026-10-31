"""Benchmark mode (MedAudit-Bench): run a researcher's model on the benchmark suite and report, for every dataset,
the score a conventional (sample-wise) evaluation would report next to the audited score.

    from medaudit.bench import run_benchmark
    board = run_benchmark(my_factory, "my_model", dataset_dir, out_dir)

`my_factory(n_features, seed)` returns an unfitted scikit-learn-compatible classifier with predict_proba.
"""
import os

import pandas as pd

from . import io
from .audits import (audit_confounds, audit_identity, audit_integrity, audit_split, audit_transfer,
                     audit_uncertainty)
from .metrics import metric
from .report import to_html, to_json, verdict
from .train import run_cv, run_external

SUITE = ["uci_voice", "uci_har", "pamap2", "mitbih", "chbmit", "ptbxl", "ham10000", "nih_cxr", "chexpert",
         "abide", "hw_pahaw", "hw_newhandpdsig", "darwin"]
EXTERNAL = {"nih_cxr": "chexpert", "chexpert": "nih_cxr", "ham10000": "isic2018_test"}


def _available(dataset_dir, name):
    return os.path.exists(os.path.join(dataset_dir, name + ".npz"))


def benchmark_one(ds, factory, name, dataset_dir, out_dir, seeds=(0, 1, 2), n_boot=1000, n_perm=100):
    ps, pg = run_cv(ds, factory, "sample", seeds), run_cv(ds, factory, "group", seeds)
    lv = ds.default_level()
    res = [audit_split(ds, ps, pg, n_boot=n_boot), audit_identity(ds, n_perm=n_perm),
           audit_integrity(ds, ps, known_shared_groups=ds.notes.get("known_shared_groups")),
           audit_confounds(ds, pg), audit_uncertainty(ds, pg, n_boot=n_boot)]
    ext = []
    if ds.site is not None and len(set(ds.site)) >= 3:
        try:
            r = audit_transfer(pg, run_cv(ds, factory, "site", seeds[:1]), level=lv, n_boot=n_boot)
            r["target"] = "held-out sites"
            res.append(r)
            ext.append(r["external"])
        except ValueError:
            pass
    if ds.name in EXTERNAL and _available(dataset_dir, EXTERNAL[ds.name]):
        pe = run_external(ds, io.load(EXTERNAL[ds.name], dataset_dir), factory, seeds[:1])
        r = audit_transfer(pg, pe, level=lv, n_boot=n_boot)
        r["target"] = f"external dataset {EXTERNAL[ds.name]}"
        res.append(r)
        ext.append(r["external"])
    meta = dict(dataset=ds.name, model=name, task=ds.task, evaluation_level=lv)
    os.makedirs(out_dir, exist_ok=True)
    to_html(res, meta, os.path.join(out_dir, f"card__{ds.name}__{name}.html"))
    to_json(res, meta, os.path.join(out_dir, f"card__{ds.name}__{name}.json"))
    row = dict(dataset=ds.name, model=name, level=lv, conventional=metric(ps, "bacc", lv),
               audited=metric(pg, "bacc", lv), gap=res[0].get(f"{lv}_delta"), gap_lo=res[0].get(f"{lv}_delta_lo"),
               gap_hi=res[0].get(f"{lv}_delta_hi"), external=min(ext) if ext else None, verdict=verdict(res)[0])
    for r in res:
        if r["module"] not in row or r["rating"] == "red":
            row[r["module"]] = r["rating"]
    return row


def run_benchmark(factory, name, dataset_dir, out_dir, datasets=None, **kw):
    rows = []
    for d in datasets or SUITE:
        if not _available(dataset_dir, d):
            print(f"[skip] {d}: not prepared")
            continue
        row = benchmark_one(io.load(d, dataset_dir), factory, name, dataset_dir, out_dir, **kw)
        rows.append(row)
        print(f"  {d:16s} conventional {100 * row['conventional']:5.1f}  audited {100 * row['audited']:5.1f}  "
              f"verdict {row['verdict']}", flush=True)
    board = pd.DataFrame(rows)
    board.to_csv(os.path.join(out_dir, f"leaderboard__{name}.csv"), index=False)
    return board
