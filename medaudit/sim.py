"""Semi-synthetic data with controlled threats, used to validate the audit modules.

Each participant (group) has a label, an identity signature, a site and a confounder; each sample adds noise.
Threat parameters (all zero = clean data):
  sigma_id        strength of the participant signature                     -> A1 gap, A2
  site_corr       probability that the site is chosen from the label        -> A2 site, A6
  sigma_site      strength of the site signature
  dup_frac        fraction of samples copied under another participant ID  -> A3
  conf_corr       correlation between the confounder and the label          -> A4
  conf_effect     effect of the confounder on the features
"""
import numpy as np
import pandas as pd

from .schema import Dataset


def simulate(n_groups=120, per_group=8, n_features=40, n_signal=5, delta=0.35, sigma_id=0.0, n_sites=4,
             site_corr=0.0, sigma_site=0.0, dup_frac=0.0, conf_corr=0.0, conf_effect=0.0, X_base=None, seed=0):
    rng = np.random.default_rng(seed)
    yg = rng.permutation(np.arange(n_groups) % 2)
    # sites: with probability site_corr the site is drawn from the label's half of the sites
    half = max(n_sites // 2, 1)
    site_g = np.where(rng.random(n_groups) < site_corr,
                      np.where(yg == 1, rng.integers(0, half, n_groups), rng.integers(half, n_sites, n_groups)),
                      rng.integers(0, n_sites, n_groups))
    conf_g = conf_corr * (2 * yg - 1) + np.sqrt(max(1 - conf_corr ** 2, 0)) * rng.standard_normal(n_groups)
    id_off = sigma_id * rng.standard_normal((n_groups, n_features))
    site_off = sigma_site * rng.standard_normal((n_sites, n_features))
    g = np.repeat(np.arange(n_groups), per_group)
    n = len(g)
    if X_base is not None:      # real features as noise background (semi-synthetic)
        Xb = np.asarray(X_base, float)
        Xb = (Xb - Xb.mean(0)) / (Xb.std(0) + 1e-9)
        # rows drawn WITHOUT replacement (with replacement would create exact duplicates across participants)
        rows = rng.choice(len(Xb), n, replace=len(Xb) < n)
        X = Xb[rows][:, :n_features].copy()
        if X.shape[1] < n_features:
            X = np.column_stack([X, rng.standard_normal((n, n_features - X.shape[1]))])
    else:
        X = rng.standard_normal((n, n_features))
    y = yg[g]
    X[:, :n_signal] += delta * (2 * y[:, None] - 1)
    X += id_off[g] + site_off[site_g[g]]
    if conf_effect:
        X[:, n_signal:n_signal + 5] += conf_effect * conf_g[g][:, None]
    group = np.array([f"P{i:04d}" for i in g])
    site = np.array([f"S{s}" for s in site_g[g]])
    conf = pd.DataFrame({"age": conf_g[g]})
    sid = np.array([f"x{i:06d}" for i in range(n)])
    truth = dict(sigma_id=sigma_id, site_corr=site_corr, sigma_site=sigma_site, dup_frac=dup_frac,
                 conf_corr=conf_corr, conf_effect=conf_effect)
    if dup_frac > 0:                     # copies of samples under a different participant ID
        k = int(round(dup_frac * n))
        src = rng.choice(n, k, replace=False)
        X = np.vstack([X, X[src] + 1e-6 * rng.standard_normal((k, n_features))])
        y = np.concatenate([y, y[src]])
        group = np.concatenate([group, np.char.add(group[src], "_copy")])
        site = np.concatenate([site, site[src]])
        conf = pd.concat([conf, conf.iloc[src]], ignore_index=True)
        sid = np.concatenate([sid, np.array([f"d{i:06d}" for i in range(k)])])
    ds = Dataset("simulation", X, y, group, sid, site, conf, task="simulated binary")
    return ds, truth
