"""Example: audit your own model in three steps (runs on simulated data as-is).

1. Put your data in a medaudit.Dataset (features, labels, participant IDs, optional sites and confounders).
2. Produce predictions with your own training code, under the protocol you report AND under a group-wise
   protocol (medaudit.train.run_cv can do both for scikit-learn models).
3. Call medaudit.audit(...) and open the HTML report card.
"""
import sys
sys.path.insert(0, ".")

import medaudit
from medaudit.sim import simulate
from medaudit.train import run_cv

# 1. a dataset with a participant signature (replace with your own Dataset)
ds, _ = simulate(sigma_id=1.0, seed=0)

# 2. predictions (replace with predictions from your own model; columns: sample_id, group, y_true, score, fold, seed)
reported = run_cv(ds, "logreg", "sample", seeds=(0, 1))     # conventional (sample-wise) protocol
groupwise = run_cv(ds, "logreg", "group", seeds=(0, 1))     # what should be reported

# 3. audit
medaudit.audit(ds, reported, groupwise, n_boot=500, n_perm=50, html="report_card.html")
print("open report_card.html")
