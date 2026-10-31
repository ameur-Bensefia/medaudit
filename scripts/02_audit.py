"""Run the full audit on each prepared dataset and reference model.
Usage: python scripts/02_audit.py [dataset ...]   (resumable)"""
import json
import os
import sys
import time

import pandas as pd

from _common import DATASETS, load_cfg, out
from medaudit import io
from medaudit.audits import (audit_confounds, audit_identity, audit_integrity, audit_split, audit_transfer,
                             audit_uncertainty)
from medaudit.audits.a3_integrity import cross_dataset_duplicates
from medaudit.report import _clean, to_html, to_json
from medaudit.train import run_cv, run_external

EXTERNAL = {"nih_cxr": "chexpert", "chexpert": "nih_cxr", "ham10000": "isic2018_test"}
EXTERNAL_ONLY = {"isic2018_test"}                    # used only as an external test set
cfg = load_cfg()
seeds = tuple(cfg.get("seeds", [0, 1, 2]))
models = cfg.get("models", ["logreg", "rf", "mlp"])
n_boot, n_perm = cfg.get("n_boot", 2000), cfg.get("n_perm", 200)
dsdir = out(cfg, "datasets", "x")[:-2]


def preds(ds, model, protocol, s=seeds):
    p = out(cfg, "predictions", f"{ds.name}__{model}__{protocol}.csv.gz")
    if os.path.exists(p):
        return pd.read_csv(p, dtype={"group": str, "sample_id": str, "site": str})
    t = time.time()
    d = run_cv(ds, model, protocol, seeds=s)
    d.to_csv(p, index=False)
    print(f"    {model}/{protocol}: {len(d)} predictions in {time.time() - t:.0f}s", flush=True)
    return d


def cached_json(path, fn):
    if os.path.exists(path):
        return json.load(open(path))
    r = _clean(fn())
    json.dump(r, open(path, "w"), indent=1, default=str)
    return r


rows = []
for name in [n for n in (sys.argv[1:] or DATASETS) if n not in EXTERNAL_ONLY]:
    if not os.path.exists(os.path.join(dsdir, name + ".npz")):
        print(f"[skip] {name}: not prepared")
        continue
    ds = io.load(name, dsdir)
    print(f"== {name}: {ds.summary()}", flush=True)
    has_site = ds.site is not None and len(set(ds.site)) >= 3
    # dataset-level modules (independent of the model)
    a2 = cached_json(out(cfg, "modules", f"{name}__A2.json"), lambda: audit_identity(ds, n_perm=n_perm))
    p0 = preds(ds, models[0], "sample")
    a3 = cached_json(out(cfg, "modules", f"{name}__A3.json"),
                     lambda: audit_integrity(ds, p0, known_shared_groups=ds.notes.get("known_shared_groups")))
    if name in EXTERNAL and os.path.exists(os.path.join(dsdir, EXTERNAL[name] + ".npz")):
        xd = cached_json(out(cfg, "modules", f"{name}__{EXTERNAL[name]}__A3cross.json"),
                         lambda: cross_dataset_duplicates(ds, io.load(EXTERNAL[name], dsdir)))
        print(f"    cross-dataset duplicates {xd['cross_dataset']}: {xd['cross_dup_samples']} images, "
              f"{xd.get('shared_group_ids', 0)} shared group identifiers", flush=True)
    for model in models:
        try:
            ps, pg = preds(ds, model, "sample"), preds(ds, model, "group")
            a1 = cached_json(out(cfg, "modules", f"{name}__{model}__A1.json"),
                             lambda: audit_split(ds, ps, pg, n_boot=n_boot))
            a4 = cached_json(out(cfg, "modules", f"{name}__{model}__A4.json"),
                             lambda: audit_confounds(ds, pg, n_tests=len(models)))
            a5 = cached_json(out(cfg, "modules", f"{name}__{model}__A5.json"),
                             lambda: audit_uncertainty(ds, pg, n_boot=n_boot))
            res = [a1, a2, a3, a4, a5]
            if has_site:
                try:
                    pst = preds(ds, model, "site", seeds[:1])
                    a6 = cached_json(out(cfg, "modules", f"{name}__{model}__A6.json"),
                                     lambda: audit_transfer(pg, pst, level=ds.default_level(), n_boot=n_boot))
                    a6["target"] = "held-out sites"
                    res.append(a6)
                except ValueError as e:
                    print(f"    A6 skipped: {e}")
            if name in EXTERNAL and os.path.exists(os.path.join(dsdir, EXTERNAL[name] + ".npz")):
                ext_name = EXTERNAL[name]
                pe_path = out(cfg, "predictions", f"{name}__{model}__external_{ext_name}.csv.gz")
                if os.path.exists(pe_path):
                    pe = pd.read_csv(pe_path, dtype={"group": str, "sample_id": str, "site": str})
                else:
                    pe = run_external(ds, io.load(ext_name, dsdir), model, seeds=seeds)
                    pe.to_csv(pe_path, index=False)
                ax = cached_json(out(cfg, "modules", f"{name}__{model}__A6ext.json"),
                                 lambda: audit_transfer(pg, pe, level=ds.default_level(), n_boot=n_boot))
                ax["target"] = f"external dataset {ext_name}"
                res.append(ax)
            meta = dict(dataset=name, task=ds.task, model=model, samples=len(ds.y),
                        groups=int(pd.Series(ds.group).nunique()))
            to_json(res, meta, out(cfg, "cards", f"{name}__{model}.json"))
            to_html(res, meta, out(cfg, "cards", f"{name}__{model}.html"))
            row = dict(dataset=name, model=model)
            for r in res:
                row[r["module"]] = r["rating"]
            row.update(level=a1.get("rating_level"), sample_reported=a1.get("sample_reported"), sample_groupwise=a1.get("sample_groupwise"),
                       sample_delta=a1.get("sample_delta"), group_delta=a1.get("group_delta"),
                       group_delta_lo=a1.get("group_delta_lo"), group_delta_hi=a1.get("group_delta_hi"),
                       nn_ratio=a2.get("nn_ratio"), site_bacc=a2.get("site_bacc"),
                       confound_only=a4.get("confound_only_group"), ci_width_ratio=a5.get("width_ratio"),
                       transfer_drop=max([r.get("drop", float("-inf")) for r in res if r["module"] == "A6"],
                                         default=None))
            a6s = [r["rating"] for r in res if r["module"] == "A6"]
            if a6s:
                row["A6"] = "red" if "red" in a6s else ("amber" if "amber" in a6s else "green")
            rows.append(row)
            print("   ", model, " ".join(f"{r['module']}:{r['rating']}" for r in res), flush=True)
        except Exception as e:
            print(f"   [error] {name}/{model}: {e.__class__.__name__}: {e}", flush=True)
if rows:
    p = out(cfg, "tables", "audit_summary.csv")
    old = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame(columns=["dataset", "model"])
    keep = old[~old.set_index(["dataset", "model"]).index.isin([(r["dataset"], r["model"]) for r in rows])]
    pd.concat([keep, pd.DataFrame(rows)], ignore_index=True).to_csv(p, index=False)
    print(pd.read_csv(p).to_string(index=False))
