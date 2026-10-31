"""A6 transfer: does internal performance hold on held-out sites or an external dataset?"""
import numpy as np

from ..bootstrap import bacc_draws, group_bootstrap
from ..metrics import metric


def audit_transfer(preds_internal, preds_external, metric_name="bacc", level="group", n_boot=1000,
                   amber=0.05, red=0.10):
    """preds_internal: group-wise CV predictions on the development data; preds_external: predictions of a
    model trained on the development data and applied to new sites/datasets. Groups are resampled
    independently in the two sets."""
    internal = metric(preds_internal, metric_name, level)
    external = metric(preds_external, metric_name, level)
    if metric_name == "bacc":
        a = bacc_draws([preds_internal], level, n_boot, seed=1)[0]
        b = bacc_draws([preds_external], level, n_boot, seed=2)[0]
    else:
        _, _, a = group_bootstrap(lambda d: metric(d, metric_name, level), [preds_internal], n_boot, seed=1)
        _, _, b = group_bootstrap(lambda d: metric(d, metric_name, level), [preds_external], n_boot, seed=2)
    n = min(len(a), len(b))
    diff = a[:n] - b[:n]
    out = dict(module="A6", internal=internal, external=external, drop=internal - external,
               drop_lo=float(np.quantile(diff, 0.025)), drop_hi=float(np.quantile(diff, 0.975)))
    chance = 1.0 / max(preds_external["y_true"].nunique(), 2)
    if out["drop"] >= red or external <= chance + 0.02:
        rating = "red"
    elif out["drop"] >= amber and out["drop_lo"] > 0:
        rating = "amber"
    else:
        rating = "green"
    out.update(rating=rating, reason=f"{100 * internal:.1f}% internally, {100 * external:.1f}% externally "
                                     f"(drop {100 * out['drop']:.1f} points)")
    return out
