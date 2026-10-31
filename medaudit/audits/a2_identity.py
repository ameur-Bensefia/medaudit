"""A2 identity and site shortcuts: how much group (participant) and site information does a representation carry?"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def _prep(X, n_comp=64):
    Z = SimpleImputer(strategy="median").fit_transform(X)
    Z = StandardScaler().fit_transform(Z)
    k = min(n_comp, Z.shape[0] - 1, Z.shape[1])
    if Z.shape[1] > k >= 1:
        Z = PCA(n_components=k, random_state=0).fit_transform(Z)
    return Z


def _subsample(ds, max_n, seed=0):
    """Subsample whole groups, so that within-group structure is preserved."""
    if len(ds.y) <= max_n:
        return ds
    rng = np.random.default_rng(seed)
    groups = np.array(sorted(set(ds.group)))
    rng.shuffle(groups)
    sizes = pd.Series(ds.group).value_counts()
    keep, n = [], 0
    for g in groups:
        if n >= max_n:
            break
        keep.append(g)
        n += sizes[g]
    return ds.subset(np.isin(ds.group, keep))


class _FoldPrep:
    """Imputation, standardization and PCA fitted on the training part only."""

    def __init__(self, n_comp=64):
        self.n_comp = n_comp

    def fit(self, X):
        self.imp = SimpleImputer(strategy="median").fit(X)
        Z = self.imp.transform(X)
        self.sc = StandardScaler().fit(Z)
        Z = self.sc.transform(Z)
        k = min(self.n_comp, Z.shape[0] - 1, Z.shape[1])
        self.pca = PCA(n_components=k, random_state=0).fit(Z) if Z.shape[1] > k >= 1 else None
        return self

    def transform(self, X):
        Z = self.sc.transform(self.imp.transform(X))
        return self.pca.transform(Z) if self.pca is not None else Z


def nn_same_group(ds, k=5, seed=0, max_n=20000, class_conditional=None):
    """Same-participant nearest-neighbor rate under the conventional (sample-wise) split, with preprocessing fitted on
    the training fold. Chance level: for each test sample, the fraction of training samples from its own participant
    (the probability that a randomly chosen training sample shares its participant), averaged over test samples."""
    d = _subsample(ds, max_n, seed)
    if class_conditional is None:                 # when the label is a property of the participant
        class_conditional = labels_constant_within_groups(d)
    rates, chances = [], []
    for tr, te in StratifiedKFold(min(k, int(np.bincount(d.y).min())), shuffle=True, random_state=seed).split(d.X, d.y):
        fp = _FoldPrep().fit(d.X[tr])
        nn = NearestNeighbors(n_neighbors=1).fit(fp.transform(d.X[tr]))
        j = nn.kneighbors(fp.transform(d.X[te]), return_distance=False)[:, 0]
        rates.append(np.mean(d.group[tr][j] == d.group[te]))
        cnt = pd.Series(d.group[tr]).value_counts()
        own = pd.Series(d.group[te]).map(cnt).fillna(0).to_numpy()
        if class_conditional:
            # chance among training samples of the SAME class: samples of one participant share its label, so a
            # label signal alone makes same-participant neighbors more likely than the unconditional chance level
            ncls = pd.Series(d.y[tr]).value_counts()
            denom = pd.Series(d.y[te]).map(ncls).fillna(len(tr)).to_numpy()
        else:
            denom = np.full(len(te), len(tr))
        chances.append(float(np.mean(own / denom)))
    return float(np.mean(rates)), float(np.mean(chances))


def labels_constant_within_groups(ds, purity=0.95):
    s = pd.DataFrame({"g": ds.group, "y": ds.y}).groupby("g")["y"].agg(lambda v: v.value_counts(normalize=True).iloc[0])
    return float((s >= purity).mean()) >= 0.95


def group_permutation(ds, n_perm=200, k=5, seed=0, max_n=20000):
    """Chaibub Neto-type test: permute labels across groups (constant within groups), fit a linear model
    under a sample-wise split, compare with the true labels."""
    d = _subsample(ds, max_n, seed)
    Z = _prep(d.X)
    rng = np.random.default_rng(seed)
    g_lab = pd.DataFrame({"g": d.group, "y": d.y}).groupby("g")["y"].agg(lambda v: v.mode().iloc[0])

    def score(y):
        kk = min(k, int(np.bincount(y).min()))
        if kk < 2:
            return np.nan
        accs = []
        for tr, te in StratifiedKFold(kk, shuffle=True, random_state=seed).split(Z, y):
            m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Z[tr], y[tr])
            accs.append(balanced_accuracy_score(y[te], m.predict(Z[te])))
        return float(np.mean(accs))

    observed = score(d.y)
    null = []
    for _ in range(n_perm):
        perm = pd.Series(rng.permutation(g_lab.to_numpy()), index=g_lab.index)
        null.append(score(perm.loc[d.group].to_numpy()))
    null = np.asarray([v for v in null if np.isfinite(v)])
    chance = 1.0 / d.n_classes
    k = int(np.sum(null >= observed))
    from scipy.stats import beta
    lo_p = 0.0 if k == 0 else float(beta.ppf(0.025, k, len(null) - k + 1))
    hi_p = 1.0 if k == len(null) else float(beta.ppf(0.975, k + 1, len(null) - k))
    return dict(perm_observed=observed, perm_null_mean=float(null.mean()), perm_null_sd=float(null.std()),
                perm_null_p95=float(np.quantile(null, 0.95)), perm_null_le_chance=float(np.mean(null <= chance)),
                perm_p=float((1 + k) / (1 + len(null))), perm_p_mc_lo=lo_p, perm_p_mc_hi=hi_p,
                perm_n=int(len(null)), perm_note="labels permuted across groups (constant within groups); "
                "sample-wise stratified 5-fold CV with the same seed for every permutation; linear model on "
                "standardized PCA features; p = (1 + #null >= observed) / (1 + n_perm)")


def identity_probe(ds, seed=0, max_groups=300):
    """Linear probe predicting the group from held-out samples of training groups."""
    rng = np.random.default_rng(seed)
    sizes = pd.Series(ds.group).value_counts()
    eligible = sizes[sizes >= 2].index.to_numpy()
    if len(eligible) < 2:
        return dict(identity_top1=np.nan, identity_chance=np.nan, identity_groups=0)
    if len(eligible) > max_groups:
        eligible = rng.choice(eligible, max_groups, replace=False)
    m = np.isin(ds.group, eligible)
    X, g = ds.X[m], ds.group[m]
    held = np.zeros(len(g), bool)
    for gg in eligible:
        held[rng.choice(np.where(g == gg)[0])] = True
    fp = _FoldPrep().fit(X[~held])                    # fitted on the training samples only
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(fp.transform(X[~held]), g[~held])
    return dict(identity_top1=float(np.mean(clf.predict(fp.transform(X[held])) == g[held])),
                identity_chance=1.0 / len(eligible), identity_groups=int(len(eligible)))


def site_probe(ds, k=5, seed=0, max_n=20000):
    """Balanced accuracy of predicting the site under a group-wise split, and the site–label association."""
    if ds.site is None or len(set(ds.site)) < 2:
        return dict(site_bacc=np.nan, site_chance=np.nan, site_label_cramers_v=np.nan)
    d = _subsample(ds, max_n, seed)
    s = pd.Series(d.site).astype("category").cat.codes.to_numpy()
    kk = min(k, len(set(d.group)))
    preds = np.zeros(len(s), int)
    for tr, te in StratifiedGroupKFold(kk, shuffle=True, random_state=seed).split(d.X, s, d.group):
        fp = _FoldPrep().fit(d.X[tr])                 # fitted on the training fold only
        m = LogisticRegression(max_iter=3000, class_weight="balanced").fit(fp.transform(d.X[tr]), s[tr])
        preds[te] = m.predict(fp.transform(d.X[te]))
    ct = pd.crosstab(d.site, d.y).to_numpy().astype(float)
    n = ct.sum()
    exp = ct.sum(1, keepdims=True) * ct.sum(0, keepdims=True) / n
    chi2 = ((ct - exp) ** 2 / np.maximum(exp, 1e-9)).sum()
    v = np.sqrt(chi2 / (n * max(min(ct.shape) - 1, 1)))
    return dict(site_bacc=float(balanced_accuracy_score(s, preds)), site_chance=1.0 / len(set(s)),
                site_label_cramers_v=float(v), n_sites=int(len(set(s))))


def audit_identity(ds, n_perm=1000, seed=0, run_perm=True):
    out = dict(module="A2")
    out["nn_same_group"], out["nn_chance"] = nn_same_group(ds, seed=seed)
    out["nn_ratio"] = out["nn_same_group"] / max(out["nn_chance"], 1e-9)
    out["labels_constant_within_groups"] = labels_constant_within_groups(ds)
    if run_perm and out["labels_constant_within_groups"]:
        out.update(group_permutation(ds, n_perm=n_perm, seed=seed))
    out.update(identity_probe(ds, seed=seed))
    out.update(site_probe(ds, seed=seed))
    # rating
    chance = 1.0 / ds.n_classes
    # decision independent of the number of permutations: observed must exceed the null 95th percentile
    red = "perm_null_p95" in out and out["perm_observed"] <= out["perm_null_p95"]
    amber = out["nn_ratio"] > 2 or ("perm_null_mean" in out and out["perm_null_mean"] > chance + 0.03) or \
        (np.isfinite(out.get("site_bacc", np.nan)) and out["site_bacc"] > out["site_chance"] + 0.10)
    if red:
        rating = "red"
        reason = (f"observed score ({100 * out['perm_observed']:.1f}%) not above the 95th percentile of scores obtained "
                  f"with random participant-level labels ({100 * out['perm_null_p95']:.1f}%; p = {out['perm_p']:.3f})")
    elif amber:
        bits = []
        if out["nn_ratio"] > 2:
            bits.append(f"same-group neighbour {out['nn_ratio']:.0f}x chance")
        if "perm_null_mean" in out and out["perm_null_mean"] > chance + 0.03:
            bits.append(f"random labels fitted at {100 * out['perm_null_mean']:.1f}%")
        if np.isfinite(out.get("site_bacc", np.nan)) and out["site_bacc"] > out["site_chance"] + 0.10:
            bits.append(f"site predicted at {100 * out['site_bacc']:.1f}% (chance {100 * out['site_chance']:.1f}%)")
        rating, reason = "amber", "; ".join(bits)
    else:
        rating, reason = "green", "little identity or site information"
    a2b = ("red" if red else "green") if "perm_null_p95" in out else "n/a"
    out.update(rating=rating, reason=reason, rating_a2a=("amber" if amber else "green"), rating_a2b=a2b)
    return out
