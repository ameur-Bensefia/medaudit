"""Recompute modules A1 (equivalence rule), A4 (covariates, seed-averaged test) and A5 (descriptive) from the saved
predictions, without retraining; optionally A2 with --perm N permutations. Writes modules/*.json and
tables/audit_summary_modules.csv.  Usage: python scripts/09_recompute_modules.py [--perm 1000] [dataset ...]"""
import argparse
import glob
import json
import os

import pandas as pd

from _common import load_cfg, out
from medaudit import io
from medaudit.audits import audit_confounds, audit_identity, audit_split, audit_uncertainty
from medaudit.report import _clean

ap = argparse.ArgumentParser()
ap.add_argument("--perm", type=int, default=0, help="recompute A2 with this many permutations (0 = skip)")
ap.add_argument("datasets", nargs="*")
args = ap.parse_args()
cfg = load_cfg()
dsdir, pdir = out(cfg, "datasets", "x")[:-2], out(cfg, "predictions", "x")[:-2]
names = args.datasets or sorted({os.path.basename(f).split("__")[0] for f in glob.glob(os.path.join(pdir, "*__group.csv.gz"))})
rows = []
rd = lambda f: pd.read_csv(f, dtype={"group": str, "sample_id": str, "site": str})  # noqa: E731
for name in names:
    if not os.path.exists(os.path.join(dsdir, name + ".npz")):
        print(f"[skip] {name}: no prepared dataset with this name")
        continue
    ds = io.load(name, dsdir)
    if args.perm:
        a2 = _clean(audit_identity(ds, n_perm=args.perm))
        json.dump(a2, open(out(cfg, "modules", f"{name}__A2.json"), "w"), indent=1, default=str)
        print(f"  {name} A2: {a2['rating']} | {a2['reason']}", flush=True)
    models = sorted({os.path.basename(f).split("__")[1] for f in glob.glob(os.path.join(pdir, f"{name}__*__group.csv.gz"))})
    for m in models:
        ps, pg = rd(os.path.join(pdir, f"{name}__{m}__sample.csv.gz")), rd(os.path.join(pdir, f"{name}__{m}__group.csv.gz"))
        a1 = _clean(audit_split(ds, ps, pg, n_boot=2000))
        a4 = _clean(audit_confounds(ds, pg, n_tests=len(models)))
        a5 = _clean(audit_uncertainty(ds, pg, n_boot=2000))
        for k, r in (("A1", a1), ("A4", a4), ("A5", a5)):
            json.dump(r, open(out(cfg, "modules", f"{name}__{m}__{k}.json"), "w"), indent=1, default=str)
        lv = a1["rating_level"]
        rows.append(dict(dataset=name, model=m, level=lv, A1=a1["rating"], gap=a1[f"{lv}_delta"],
                         gap_lo=a1[f"{lv}_delta_lo"], gap_hi=a1[f"{lv}_delta_hi"], gap_eq_lo=a1.get(f"{lv}_delta_eq_lo"),
                         gap_eq_hi=a1.get(f"{lv}_delta_eq_hi"), A4=a4["rating"],
                         covariate_only=a4.get("confound_only_group"), lr_p=a4.get("lr_p"),
                         auc_cov=a4.get("auc_covariates"), auc_cov_model=a4.get("auc_covariates_plus_model"),
                         A5_width_ratio=a5["width_ratio"]))
        print(f"  {name} {m}: A1 {a1['rating']} ({a1['reason']}) | A4 {a4['rating']}", flush=True)
p = out(cfg, "tables", "audit_summary_modules.csv")
old = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame(columns=["dataset", "model"])
new = pd.DataFrame(rows)
old = old[~old.set_index(["dataset", "model"]).index.isin(list(zip(new.dataset, new.model)))]
pd.concat([old, new], ignore_index=True).to_csv(p, index=False)
