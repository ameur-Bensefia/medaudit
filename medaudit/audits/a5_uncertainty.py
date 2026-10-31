"""A5 uncertainty: are the error bars realistic for grouped data?"""
import numpy as np

from ..bootstrap import bacc_ci, metric_ci, sample_bootstrap_ci
from ..metrics import metric


def audit_uncertainty(ds, preds, metric_name="bacc", n_boot=1000, min_ratio_amber=0.7, min_ratio_red=0.5):
    """Compare the width of a naive (sample) bootstrap interval with a group bootstrap interval, and report
    the variability over training seeds."""
    out = dict(module="A5", n_groups=int(preds["group"].nunique()), n_samples=int(preds["sample_id"].nunique()))
    if metric_name == "bacc":
        est, glo, ghi = bacc_ci(preds, "sample", n_boot)
        # naive bootstrap = every sample its own group
        naive = preds.assign(group=preds["sample_id"].astype(str))   # every sample its own group
        _, nlo, nhi = bacc_ci(naive, "sample", n_boot)
    else:
        est, glo, ghi = metric_ci(preds, metric_name, "sample", n_boot)
        nlo, nhi = sample_bootstrap_ci(preds, metric_name, "sample", n_boot)
    out.update(estimate=est, group_ci=(glo, ghi), naive_ci=(nlo, nhi),
               group_ci_width=ghi - glo, naive_ci_width=nhi - nlo,
               width_ratio=(nhi - nlo) / max(ghi - glo, 1e-9))
    if preds["seed"].nunique() > 1:
        per_seed = [metric(g, metric_name, "sample") for _, g in preds.groupby("seed")]
        out["seed_sd"] = float(np.std(per_seed, ddof=1))
    else:
        out["seed_sd"] = float("nan")
    # A5 is descriptive: it reports how much a naive interval understates the participant-level uncertainty, and
    # does not contribute to the overall verdict (percentile intervals from n_boot participant resamples)
    rating = "info"
    r = out["width_ratio"]
    out["n_boot"] = n_boot
    reason = (f"naive interval is {100 * r:.0f}% as wide as the participant-level interval "
              f"({out['n_groups']} groups, {out['n_samples']} samples)")
    if np.isfinite(out["seed_sd"]):
        reason += f"; seed SD {100 * out['seed_sd']:.1f} points"
    out.update(rating=rating, reason=reason)
    return out
