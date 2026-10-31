"""Model factories (scikit-learn compatible). The audit itself is model-agnostic; these are the reference
models used in the case studies. Each factory returns a fresh, unfitted estimator."""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class SafePCA(BaseEstimator, TransformerMixin):
    """PCA whose size adapts to the training set (never more components than samples - 1)."""

    def __init__(self, n_components=256, random_state=0):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        k = max(1, min(self.n_components, X.shape[0] - 1, X.shape[1]))
        self.pca_ = PCA(n_components=k, random_state=self.random_state).fit(X)
        return self

    def transform(self, X):
        return self.pca_.transform(X)


def _pre(n_features, pca):
    steps = [SimpleImputer(strategy="median"), StandardScaler()]
    if pca and n_features > pca:
        steps.append(SafePCA(pca))
    return steps


def make_model(name, n_features, seed=0, pca=256):
    if name == "logreg":
        return make_pipeline(*_pre(n_features, pca), LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced"))
    if name == "rf":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(300, class_weight="balanced_subsample", n_jobs=-1,
                                                    min_samples_leaf=2, random_state=seed))
    if name == "mlp":
        return make_pipeline(*_pre(n_features, pca), MLPClassifier((128,), alpha=1e-3, max_iter=300,
                                                                   early_stopping=True, random_state=seed))
    if name == "xgb":
        from xgboost import XGBClassifier
        return make_pipeline(SimpleImputer(strategy="median"),
                             XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                                           colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbosity=0))
    raise ValueError(name)


MODELS = ("logreg", "rf", "mlp")
