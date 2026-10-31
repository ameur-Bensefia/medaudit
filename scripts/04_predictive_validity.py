"""Predictive validity: do audit scores computed on internal data predict the performance drop on held-out
sites? For every dataset with sites and every held-out site: compute the audit scores on the remaining
(internal) data, train on the internal data, test on the held-out site.
Usage: python scripts/04_predictive_validity.py [dataset ...]  (resumable)"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from _common import load_cfg, out
from medaudit import io
from medaudit.audits.a2_identity import nn_same_group, site_probe
from medaudit.audits.a4_confounds import confound_only
from medaudit.metrics import metric
from medaudit.train import run_cv, run_external

cfg = load_cfg()
dsdir = out(cfg, "datasets", "x")[:-2]
models = cfg.get("models", ["logreg", "rf", "mlp"])
CANDIDATES = sys.argv[1:] or ["abide", "ham10000", "ptbxl", "nih_chexpert"]
path = out(cfg, "predictive", "configurations.csv")
done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["dataset", "heldout", "model"])
key = set(zip(done["dataset"], done["heldout"].astype(str), done["model"]))
rows = []


def pooled_cxr():
    a, b = io.load("nih_cxr", dsdir), io.load("chexpert", dsdir)
    from medaudit.schema import Dataset
    conf = pd.concat([a.confounders, b.confounders], ignore_index=True) if a.confounders is not None else None
    return Dataset("nih_chexpert", np.vstack([a.X, b.X]), np.concatenate([a.y, b.y]),
                   np.concatenate([a.group, b.group]), np.concatenate([a.sample_id, b.sample_id]),
                   np.concatenate([a.site, b.site]), conf)


for name in CANDIDATES:
    try:
        ds = pooled_cxr() if name == "nih_chexpert" else io.load(name, dsdir)
    except Exception as e:
        print(f"[skip] {name}: {e}")
        continue
    if ds.site is None:
        continue
    sites = [s for s in sorted(set(ds.site)) if (ds.site == s).sum() >= 20 and len(set(ds.y[ds.site == s])) > 1]
    print(f"== {name}: {len(sites)} held-out sites", flush=True)
    for s in sites:
        internal, external = ds.subset(ds.site != s), ds.subset(ds.site == s)
        if len(set(internal.y)) < 2:
            continue
        shared = {}
        nn, ch = nn_same_group(internal)
        shared.update(nn_ratio=nn / max(ch, 1e-9))
        sp = site_probe(internal) if len(set(internal.site)) >= 2 else {}
        shared.update(site_bacc_internal=sp.get("site_bacc"), site_chance=sp.get("site_chance"),
                      site_label_v=sp.get("site_label_cramers_v"))
        if internal.confounders is not None:
            shared["confound_only"] = confound_only(internal, seeds=(0,), n_boot=200)["confound_only_group"]
        for m in models:
            if (name, str(s), m) in key:
                continue
            pg = run_cv(internal, m, "group", seeds=(0,))
            ps = run_cv(internal, m, "sample", seeds=(0,))
            pe = run_external(internal, external, m, seeds=(0,))
            lv = ds.default_level()
            r = dict(dataset=name, heldout=s, model=m, level=lv, n_internal=len(internal.y),
                     n_external=len(external.y), internal_group=metric(pg, "bacc", lv),
                     internal_sample_cv=metric(ps, "bacc", lv), external=metric(pe, "bacc", lv), **shared)
            r["drop"] = r["internal_group"] - r["external"]
            r["naive_optimism"] = r["internal_sample_cv"] - r["external"]
            r["a1_delta"] = r["internal_sample_cv"] - r["internal_group"]
            rows.append(r)
            print(f"   {s} {m}: internal {r['internal_group']:.3f} external {r['external']:.3f}", flush=True)
            pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)

allr = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
if len(allr) >= 8:
    res = []
    for score, target in (("site_bacc_internal", "drop"), ("site_label_v", "drop"), ("nn_ratio", "drop"),
                          ("confound_only", "drop"), ("a1_delta", "naive_optimism")):
        d = allr[[score, target, "dataset", "heldout"]].dropna()
        if len(d) >= 8:
            rho, p = spearmanr(d[score], d[target])
            # cluster bootstrap over held-out units (the models evaluated on one site are not independent)
            d = d.assign(unit=d["dataset"] + "|" + d["heldout"].astype(str))
            units = d["unit"].unique()
            idx = d.groupby("unit").indices
            boots = []
            rng = np.random.default_rng(0)
            for _ in range(2000):
                pick = rng.choice(units, len(units), replace=True)
                b = d.iloc[np.concatenate([idx[u] for u in pick])]
                boots.append(spearmanr(b[score], b[target])[0])
            res.append(dict(score=score, target=target, n=len(d), units=len(units), spearman=rho, p=p,
                            lo=np.nanquantile(boots, 0.025), hi=np.nanquantile(boots, 0.975)))
    pd.DataFrame(res).to_csv(out(cfg, "predictive", "correlations.csv"), index=False)
    print(pd.DataFrame(res).round(3).to_string(index=False))
