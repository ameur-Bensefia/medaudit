"""Group (cluster) bootstrap. Groups are resampled with replacement; each resample is applied to all seeds
and, for paired comparisons, to every prediction set (sets must share their groups)."""
import numpy as np

from .metrics import metric


def group_bootstrap(stat, dfs, n_boot=2000, seed=0, alpha=0.05):
    """stat(*resampled_dfs) -> float. Returns (lo, hi, draws)."""
    rng = np.random.default_rng(seed)
    groups = np.array(sorted(set(dfs[0]["group"])))
    idx = [d.groupby("group").indices for d in dfs]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        res = []
        for d, ix in zip(dfs, idx):
            rows, newg = [], []
            for k, gi in enumerate(pick):
                r = ix.get(groups[gi])
                if r is not None:
                    rows.append(r)
                    newg.append(np.full(len(r), k))
            if rows:
                sub = d.iloc[np.concatenate(rows)].copy()
                sub["group"] = np.concatenate(newg).astype(str)   # repeated groups stay distinct
            else:
                sub = d.iloc[:0]
            res.append(sub)
        v = stat(*res)
        if np.isfinite(v):
            vals.append(v)
    vals = np.asarray(vals)
    if vals.size == 0:
        return float("nan"), float("nan"), vals
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)), vals


def metric_ci(df, name="bacc", level="sample", n_boot=2000, seed=0):
    est = metric(df, name, level)
    lo, hi, _ = group_bootstrap(lambda d: metric(d, name, level), [df], n_boot, seed)
    return est, lo, hi


def sample_bootstrap_ci(df, name="bacc", level="sample", n_boot=2000, seed=0, alpha=0.05):
    """Naive bootstrap over rows, ignoring groups (used by A5 to show how much it understates uncertainty)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        v = metric(df.iloc[rng.integers(0, len(df), len(df))], name, level)
        if np.isfinite(v):
            vals.append(v)
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


# ---------------------------------------------------------------- fast path for balanced accuracy
def _bacc_tables(df, groups, level):
    """Per seed: arrays n[g, c] (samples of class c in group g) and k[g, c] (correctly predicted)."""
    from .metrics import _pred_label, to_group_level
    d = to_group_level(df) if level == "group" else df
    pos = {g: i for i, g in enumerate(groups)}
    classes = np.sort(df["y_true"].unique())
    cidx = {c: j for j, c in enumerate(classes)}
    out = []
    for _, s in (d.groupby("seed") if "seed" in d.columns else [(0, d)]):
        gi = s["group"].map(pos).to_numpy()
        ok = ~np.isnan(gi.astype(float))
        gi = gi[ok].astype(int)
        ci = s["y_true"].map(cidx).to_numpy()[ok].astype(int)
        corr = (_pred_label(s) == s["y_true"].to_numpy())[ok]
        n = np.zeros((len(groups), len(classes)))
        k = np.zeros((len(groups), len(classes)))
        np.add.at(n, (gi, ci), 1)
        np.add.at(k, (gi, ci), corr)
        out.append((n, k))
    return out


def bacc_draws(dfs, level="sample", n_boot=2000, seed=0, chunk=250):
    """Group-bootstrap draws of balanced accuracy for several prediction sets with the SAME resampling
    weights (paired). Returns an array (len(dfs), n_boot). Equivalent to resampling groups with replacement."""
    groups = np.array(sorted(set(dfs[0]["group"])))
    G = len(groups)
    tables = [_bacc_tables(d, groups, level) for d in dfs]
    rng = np.random.default_rng(seed)
    out = np.zeros((len(dfs), n_boot))
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        W = rng.multinomial(G, np.full(G, 1.0 / G), size=b).astype(float)      # b x G
        for i, tab in enumerate(tables):
            per_seed = []
            for n, k in tab:
                num, den = W @ k, W @ n                                          # b x C
                with np.errstate(invalid="ignore", divide="ignore"):
                    rec = np.where(den > 0, num / den, np.nan)
                per_seed.append(np.nanmean(rec, axis=1))
            out[i, done:done + b] = np.nanmean(np.vstack(per_seed), axis=0)
        done += b
    return out


def bacc_ci(df, level="sample", n_boot=2000, seed=0, alpha=0.05):
    from .metrics import metric
    d = bacc_draws([df], level, n_boot, seed)[0]
    return metric(df, "bacc", level), float(np.nanquantile(d, alpha / 2)), float(np.nanquantile(d, 1 - alpha / 2))


def bacc_diff_ci(df_a, df_b, level="sample", n_boot=2000, seed=0, alpha=0.05):
    """Paired difference A - B (same groups resampled for both)."""
    d = bacc_draws([df_a, df_b], level, n_boot, seed)
    diff = d[0] - d[1]
    return float(np.nanquantile(diff, alpha / 2)), float(np.nanquantile(diff, 1 - alpha / 2))
