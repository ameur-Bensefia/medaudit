"""Benchmark mode: evaluate YOUR model on the MedAudit-Bench suite.
Edit `my_model` (any scikit-learn-compatible classifier with predict_proba), then run:
    python3 scripts/06_benchmark_model.py [dataset ...]"""
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from _common import load_cfg, out
from medaudit.bench import run_benchmark


def my_model(n_features, seed):
    """Replace with your own model (must provide fit / predict_proba). The benchmark trains it many times
    (folds x seeds x protocols): prefer models that train in seconds."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))


cfg = load_cfg()
board = run_benchmark(my_model, "my_model", out(cfg, "datasets", "x")[:-2], out(cfg, "benchmark", "x")[:-2],
                      datasets=sys.argv[1:] or None, seeds=tuple(cfg.get("seeds", [0, 1, 2])),
                      n_boot=cfg.get("n_boot", 1000), n_perm=cfg.get("n_perm", 100))
print(board.round(3).to_string(index=False))
