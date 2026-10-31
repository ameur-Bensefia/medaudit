"""Fast unit tests: python -m pytest -q tests"""
import sys

import numpy as np

sys.path.insert(0, ".")
from medaudit.audits import audit_integrity, audit_split  # noqa: E402
from medaudit.bootstrap import bacc_draws, group_bootstrap  # noqa: E402
from medaudit.metrics import metric  # noqa: E402
from medaudit.sim import simulate  # noqa: E402
from medaudit.train import run_cv  # noqa: E402


def test_fast_bootstrap_matches_reference():
    ds, _ = simulate(sigma_id=0.8, seed=3)
    p = run_cv(ds, "logreg", "sample", seeds=(0,))
    fast = bacc_draws([p], "sample", 2000, seed=1)[0]
    _, _, slow = group_bootstrap(lambda d: metric(d, "bacc", "sample"), [p], 400, seed=1)
    assert abs(fast.mean() - slow.mean()) < 0.01 and abs(fast.std() - slow.std()) < 0.01


def test_identity_leakage_is_flagged_and_clean_is_not():
    ds, _ = simulate(sigma_id=1.0, seed=1)
    a = audit_split(ds, run_cv(ds, "logreg", "sample"), run_cv(ds, "logreg", "group"), n_boot=300)
    assert a["rating"] == "red" and a["group_delta"] > 0.05
    ds0, _ = simulate(seed=1)
    b = audit_split(ds0, run_cv(ds0, "logreg", "sample"), run_cv(ds0, "logreg", "group"), n_boot=300)
    assert b["rating"] in ("green", "inconclusive")          # clean data is never flagged as inflated


def test_equivalence_reached_with_enough_participants():
    ds, _ = simulate(n_groups=1500, per_group=2, seed=4)
    a = audit_split(ds, run_cv(ds, "logreg", "sample"), run_cv(ds, "logreg", "group"), n_boot=300)
    assert a["rating"] == "green" and "smaller than" in a["reason"]


def test_duplicates_are_flagged():
    ds, _ = simulate(dup_frac=0.05, seed=2)
    r = audit_integrity(ds, run_cv(ds, "logreg", "sample"))
    assert r["rating"] == "red" and r["exact_dup_sets_cross_group"] > 0


def test_label_level_detection():
    ds, _ = simulate(seed=0)
    assert ds.default_level() == "group"
    ds.y = np.random.default_rng(0).integers(0, 2, len(ds.y))
    assert ds.default_level() == "sample"
