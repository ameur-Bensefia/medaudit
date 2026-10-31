"""Validation by simulation: inject controlled amounts of each threat and measure how often each module
flags it (power) and how often it flags clean data (false alarms).
Usage: python scripts/03_simulation.py [--reps 50] [--base DATASET]  (resumable)"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from _common import load_cfg, out
from medaudit import io
from medaudit.audits import audit_confounds, audit_identity, audit_integrity, audit_split, audit_transfer
from medaudit.sim import simulate
from medaudit.train import run_cv, run_external

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=200)
ap.add_argument("--base", default=None, help="prepared dataset whose features serve as background noise")
ap.add_argument("--n_perm", type=int, default=100)
args = ap.parse_args()
cfg = load_cfg()
X_base = io.load(args.base, out(cfg, "datasets", "x")[:-2]).X if args.base else None

GRID = []
for v in (0.0, 0.25, 0.5, 1.0, 1.5):
    GRID.append(("identity", v, dict(sigma_id=v)))
for v in (0.0, 0.5, 0.8, 1.0):
    GRID.append(("site", v, dict(site_corr=v, sigma_site=1.0 if v > 0 else 0.0)))
for v in (0.0, 0.01, 0.05, 0.10):
    GRID.append(("duplicates", v, dict(dup_frac=v)))
for v in (0.0, 0.3, 0.6, 0.9):
    GRID.append(("confound_with_signal", v, dict(conf_corr=v, conf_effect=1.0 if v > 0 else 0.0)))
    GRID.append(("confound_no_signal", v, dict(conf_corr=v, conf_effect=1.0 if v > 0 else 0.0, delta=0.0)))
# combined threats: several threats injected together, at increasing strength
for v in (0.0, 0.5, 1.0):
    GRID.append(("combined", v, dict(sigma_id=0.5 * v, site_corr=0.8 * v, sigma_site=1.0 * v, dup_frac=0.01 * v,
                                     conf_corr=0.6 * v, conf_effect=1.0 * v)))

path = out(cfg, "simulation", f"runs{'_' + args.base if args.base else ''}.csv")
done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["scenario", "level", "rep"])
key = set(zip(done["scenario"], done["level"].round(4), done["rep"]))
rows = []
t0 = time.time()
for scen, level, kw in GRID:
    for rep in range(args.reps):
        if (scen, round(level, 4), rep) in key:
            continue
        ds, _ = simulate(seed=1000 * rep + 7, X_base=X_base, **kw)
        r = dict(scenario=scen, level=level, rep=rep)
        ps = run_cv(ds, "logreg", "sample", seeds=(rep,))
        pg = run_cv(ds, "logreg", "group", seeds=(rep,))
        a1 = audit_split(ds, ps, pg, n_boot=500)
        r.update(A1=a1["rating"], group_delta=a1["group_delta"], sample_delta=a1["sample_delta"])
        a2 = audit_identity(ds, n_perm=args.n_perm, seed=rep)
        r.update(A2=a2["rating"], nn_ratio=a2["nn_ratio"], site_bacc=a2["site_bacc"],
                 perm_null_mean=a2.get("perm_null_mean"), perm_p=a2.get("perm_p"))
        a3 = audit_integrity(ds, ps)
        r.update(A3=a3["rating"], dup_test_in_train=a3.get("dup_test_samples_in_train"))
        a4 = audit_confounds(ds, pg)
        r.update(A4=a4["rating"], confound_only=a4["confound_only_group"], delta_auc=a4.get("delta_auc"),
                 delta_auc_lo=a4.get("delta_auc_lo"))
        if scen == "site" or (scen == "combined" and kw.get("sigma_site", 0) > 0):
            # external hospital: new site signatures, site unrelated to the label
            ext, _ = simulate(seed=1000 * rep + 99991, X_base=X_base, site_corr=0.0, sigma_site=kw["sigma_site"])
            ext.site = np.char.add(ext.site.astype(str), "_ext")
            a6 = audit_transfer(pg, run_external(ds, ext, "logreg", seeds=(rep,)), n_boot=300)
            r.update(A6=a6["rating"], transfer_drop=a6["drop"])
        rows.append(r)
        if len(rows) % 20 == 0:
            pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)
            print(f"  {len(rows)} runs, {time.time() - t0:.0f}s", flush=True)
allr = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
allr.to_csv(path, index=False)

# ---------------------------------------------------------------- summaries with 95% Wilson intervals
def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


TARGET = {"identity": ["A1", "A2"], "site": ["A2", "A6"], "duplicates": ["A3"],
          "confound_with_signal": ["A4"], "confound_no_signal": ["A4"], "combined": ["A1", "A2", "A3", "A4", "A6"]}
summ = []
DET = ("amber", "red")                     # detection of a threat; "inconclusive" = insufficient evidence
for (scen, level), g in allr.groupby(["scenario", "level"]):
    mods = [m for m in ("A1", "A2", "A3", "A4", "A6") if m in g and g[m].notna().any()]
    for m in mods + ["overall"]:
        if m == "overall":
            v = g[mods].apply(lambda r: "red" if any(x in DET for x in r if isinstance(x, str)) else
                              ("inconclusive" if any(x == "inconclusive" for x in r) else "green"), axis=1)
        else:
            v = g[m].dropna()
        k = int(v.isin(DET).sum())
        ki = int((v == "inconclusive").sum())
        kr = int((v == "red").sum())
        lo, hi = wilson(k, len(v))
        ilo, ihi = wilson(ki, len(v))
        summ.append(dict(scenario=scen, level=level, module=m, target=m in TARGET[scen], n=len(v), detections=k,
                         detection_rate=k / len(v), detection_lo=lo, detection_hi=hi, inconclusive=ki,
                         inconclusive_rate=ki / len(v), inconclusive_lo=ilo, inconclusive_hi=ihi,
                         red_rate=kr / len(v)))
s = pd.DataFrame(summ)
tag = "_" + args.base if args.base else ""
s.to_csv(out(cfg, "simulation", f"flag_rates{tag}.csv"), index=False)
params = []
for scen, level, kw in GRID:
    d = dict(n_groups=120, per_group=8, n_features=40, n_signal=5, delta=0.35, sigma_id=0.0, n_sites=4,
             site_corr=0.0, sigma_site=0.0, dup_frac=0.0, conf_corr=0.0, conf_effect=0.0)
    d.update(kw)
    params.append(dict(scenario=scen, level=level, **d))
pd.DataFrame(params).to_csv(out(cfg, "simulation", f"parameters{tag}.csv"), index=False)
print(s[s.target].pivot_table(index=["scenario", "level"], columns="module", values="detection_rate").round(3))
print("\nclean data (level 0): detections and inconclusive results [95% CI]:")
z = s[(s.level == 0)]
for _, r in z.iterrows():
    print(f"  {r['scenario']:22s} {r['module']:8s} detected {int(r['detections']):3d}/{int(r['n']):3d} "
          f"[{r['detection_lo']:.3f}, {r['detection_hi']:.3f}]  inconclusive {int(r['inconclusive']):3d}")
