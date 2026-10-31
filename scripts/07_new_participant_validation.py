"""Do conventional or participant-wise cross-validation estimates predict performance on NEW participants?
For each dataset: hold out a random 30% of participants (never used in development); on the remaining 70%, run
conventional (sample-wise) and participant-wise cross-validation; train on the 70% and test on the held-out
participants. Repeated over several random hold-outs. Usage: python scripts/07_new_participant_validation.py [ds ...]"""
import argparse
import os

import numpy as np
import pandas as pd

from _common import load_cfg, out
from medaudit import io
from medaudit.metrics import metric
from medaudit.train import run_cv, run_external

DEFAULT = ["uci_voice", "hw_pahaw", "hw_newhandpdsig", "mitbih", "chbmit", "pamap2", "uci_har", "ptbxl", "darwin"]
ap = argparse.ArgumentParser()
ap.add_argument("--holdouts", type=int, default=20)
ap.add_argument("datasets", nargs="*")
args = ap.parse_args()
cfg = load_cfg()
dsdir = out(cfg, "datasets", "x")[:-2]
models = cfg.get("models", ["logreg", "rf", "mlp"])
path = out(cfg, "predictive", "new_participants.csv")
done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["dataset", "model", "rep"])
key = set(zip(done.dataset, done.model, done.rep))
rows = []
for name in args.datasets or DEFAULT:
    if not os.path.exists(os.path.join(dsdir, name + ".npz")):
        continue
    ds = io.load(name, dsdir)
    lv = ds.default_level()
    groups = np.array(sorted(set(ds.group)))
    for rep in range(args.holdouts):
        rng = np.random.default_rng(100 + rep)
        test_g = rng.choice(groups, max(2, int(round(0.3 * len(groups)))), replace=False)
        te = np.isin(ds.group, test_g)
        dev, new = ds.subset(~te), ds.subset(te)
        if len(set(new.y)) < 2 or len(set(dev.y)) < 2:
            continue
        for m in models:
            if (name, m, rep) in key:
                continue
            ps = run_cv(dev, m, "sample", seeds=(rep,))
            pg = run_cv(dev, m, "group", seeds=(rep,))
            pn = run_external(dev, new, m, seeds=(rep,))
            # size-matched control: train on 80% of development participants, the training size of a CV fold
            rs = np.random.default_rng(1000 + rep)
            dg = np.array(sorted(set(dev.group)))
            sub_g = rs.choice(dg, int(round(0.8 * len(dg))), replace=False)
            dsub = dev.subset(np.isin(dev.group, sub_g))
            pm = run_external(dsub, new, m, seeds=(rep,)) if len(set(dsub.y)) > 1 else None
            r = dict(dataset=name, model=m, rep=rep, level=lv, n_dev_groups=len(set(dev.group)),
                     n_new_groups=len(set(new.group)), conventional=metric(ps, "bacc", lv),
                     participant_wise=metric(pg, "bacc", lv), new_participants=metric(pn, "bacc", lv),
                     new_participants_matched=metric(pm, "bacc", lv) if pm is not None else np.nan)
            rows.append(r)
            print(f"  {name} {m} rep{rep}: conventional {r['conventional']:.3f} participant-wise "
                  f"{r['participant_wise']:.3f} new participants {r['new_participants']:.3f}", flush=True)
            pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)
allr = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
allr.to_csv(path, index=False)
# ---------------------------------------------------------------- summary with uncertainty
# For each hold-out h, models are averaged first; then
#   err_conv(h) = conventional(h) - new(h),  err_pw(h) = participant_wise(h) - new(h)
#   gain(h)     = |err_conv(h)| - |err_pw(h)|    (positive = participant-wise estimate closer to the truth)
#   diff(h)     = conventional(h) - participant_wise(h)   (equivalence of the two estimates, margin 2 points)
# The table reports means over hold-outs with corrected resampled t intervals (Nadeau and Bengio, 2003; n - 1 degrees
# of freedom); the 90% interval of diff gives the two one-sided tests of equivalence.
from scipy import stats as _st
out_rows = []
for target in ("new_participants", "new_participants_matched"):
    if target not in allr or allr[target].isna().all():
        continue
    d = allr.dropna(subset=[target])
    for name, g in d.groupby("dataset"):
        h = g.groupby("rep")[["conventional", "participant_wise", target]].mean()
        h["err_conv"] = h.conventional - h[target]
        h["err_pw"] = h.participant_wise - h[target]
        h["gain"] = h.err_conv.abs() - h.err_pw.abs()
        h["diff"] = h.conventional - h.participant_wise
        n = len(h)
        row = dict(target=target, dataset=name, holdouts=n, conventional=h.conventional.mean(),
                   participant_wise=h.participant_wise.mean(), new=h[target].mean())
        for c in ("err_conv", "err_pw", "gain", "diff"):
            # corrected resampled t (Nadeau and Bengio, 2003): repeated hold-outs share most of their data, so the
            # variance of the mean is inflated by (1/J + n_test/n_train) instead of 1/J
            m = h[c].mean()
            se = np.sqrt((1.0 / n + 0.3 / 0.7) * h[c].var(ddof=1))
            t95, t90 = _st.t.ppf(0.975, n - 1), _st.t.ppf(0.95, n - 1)
            row.update({c: m, c + "_lo": m - t95 * se, c + "_hi": m + t95 * se})
            if c == "diff":
                row.update(diff_lo90=m - t90 * se, diff_hi90=m + t90 * se,
                           equivalent_2pts=bool(m - t90 * se >= -0.02 and m + t90 * se <= 0.02))
        out_rows.append(row)
summary = pd.DataFrame(out_rows)
summary.to_csv(out(cfg, "predictive", "new_participants_summary.csv"), index=False)
print((summary.set_index(["target", "dataset"])[["err_conv", "err_pw", "gain", "gain_lo", "gain_hi"]] * 100).round(1))
