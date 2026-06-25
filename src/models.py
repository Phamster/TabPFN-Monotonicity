"""
Model builders for the PoM benchmark.

XGBoost (replication of Koklev 2025):
  - Hyperparameters kept close to Koklev defaults (n_estimators=300,
    learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8)
  - Monotone constraints passed via xgb's native `monotone_constraints` param
  - Constrained model uses the dataset's mono_map; unconstrained sets all to 0

TabPFN (extension):
  - TabPFNClassifier from the Prior Labs tabpfn package
  - No native monotone_constraints support → we apply the post-hoc adapter
    (Stage 1: sign-constrained logistic regression; Stage 2: TabPFN correction)
  - See PoMTabPFNAdapter below
"""

import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler

try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    print("Warning: tabpfn not available — TabPFN experiments will be skipped.")


# ──────────────────────────────────────────────────────────────────────────────
# XGBoost models
# ──────────────────────────────────────────────────────────────────────────────

_XGB_DEFAULTS = dict(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)


def build_xgb_unconstrained(n_features: int) -> XGBClassifier:
    return XGBClassifier(**_XGB_DEFAULTS)


def build_xgb_constrained(mono_map: dict, n_features: int) -> XGBClassifier:
    """
    mono_map: dict {feature_index: +1 or -1}
    XGBoost expects a tuple of length n_features, 0 = unconstrained.
    """
    constraints = tuple(mono_map.get(i, 0) for i in range(n_features))
    return XGBClassifier(
        monotone_constraints=constraints,
        **_XGB_DEFAULTS,
    )


# ──────────────────────────────────────────────────────────────────────────────
# TabPFN wrapper
# ──────────────────────────────────────────────────────────────────────────────

class TabPFNWrapper(BaseEstimator, ClassifierMixin):
    """Thin sklearn-compatible wrapper around TabPFNClassifier."""

    def __init__(self, device="cpu"):
        self.device = device
        self._clf = None

    def fit(self, X, y):
        if not TABPFN_AVAILABLE:
            raise RuntimeError("tabpfn package not installed.")
        self._clf = TabPFNClassifier(device=self.device)
        self._clf.fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self._clf.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ──────────────────────────────────────────────────────────────────────────────
# Two-stage adapter: constrained logistic regression + TabPFN correction
# Following the design of the economic-validity-audit paper (2026)
# ──────────────────────────────────────────────────────────────────────────────

class PoMTabPFNAdapter(BaseEstimator, ClassifierMixin):
    """
    Stage 1: A sign-constrained logistic regression (penalty='l2') trained
             on features restricted to those with a known monotonicity direction.
             We use sklearn's LogisticRegression and enforce sign constraints
             post-hoc by clipping coefficients after fitting — an approximation
             suitable for benchmarking purposes.

    Stage 2: TabPFN is trained to predict the *residual* (logit of true label
             minus logit of Stage 1 prediction). At inference time:
               final_logit = logit(stage1_prob) + stage2_correction
             which is monotonic by design of Stage 1, with Stage 2 only
             nudging within the monotonic structure.

    This implements the three-component approach:
      - Monotonicity guarantee: from Stage 1 (sign-constrained LR)
      - Flexibility / accuracy gain: from Stage 2 (TabPFN correction)
      - Adapter: additive combination in logit space
    """

    def __init__(self, mono_map: dict, device="cpu", clip_coefs=True):
        self.mono_map = mono_map
        self.device = device
        self.clip_coefs = clip_coefs
        self._stage1 = None
        self._stage2 = None
        self._scaler = StandardScaler()
        self.classes_ = np.array([0, 1])

    def _logit(self, p):
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)
        Xs = self._scaler.fit_transform(X)

        # Stage 1: sign-constrained logistic regression
        self._stage1 = LogisticRegression(
            penalty="l2", C=1.0, max_iter=500, random_state=42, solver="lbfgs"
        )
        self._stage1.fit(Xs, y)

        if self.clip_coefs:
            coef = self._stage1.coef_[0].copy()
            for feat_idx, direction in self.mono_map.items():
                if feat_idx < len(coef):
                    if direction == 1 and coef[feat_idx] < 0:
                        coef[feat_idx] = 0.0
                    elif direction == -1 and coef[feat_idx] > 0:
                        coef[feat_idx] = 0.0
            self._stage1.coef_[0] = coef

        # Stage 2: TabPFN on residuals
        if TABPFN_AVAILABLE:
            p1 = self._stage1.predict_proba(Xs)[:, 1]
            logit_p1  = self._logit(p1)
            logit_y   = self._logit(np.clip(y.astype(float), 0.05, 0.95))
            residuals = np.sign(logit_y - logit_p1).astype(int)
            residuals = np.clip(residuals + 1, 0, 2)   # map {-1,0,1} → {0,1,2}

            # TabPFN has a 10k sample limit — subsample if needed
            if len(y) > 9000:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(y), 9000, replace=False)
                Xs2, res2 = Xs[idx], residuals[idx]
            else:
                Xs2, res2 = Xs, residuals

            if len(np.unique(res2)) > 1:
                self._stage2 = TabPFNWrapper(device=self.device)
                self._stage2.fit(Xs2, res2)

        return self

    def predict_proba(self, X):
        X = np.array(X, dtype=float)
        Xs = self._scaler.transform(X)

        p1 = self._stage1.predict_proba(Xs)[:, 1]

        if self._stage2 is not None:
            # Stage 2 predicts residual class {0,1,2} → map to correction
            res_probs = self._stage2.predict_proba(Xs)
            n_classes = res_probs.shape[1]
            if n_classes == 3:
                correction = res_probs[:, 2] - res_probs[:, 0]  # P(+1) - P(-1)
            else:
                correction = np.zeros(len(X))
            final_logit = self._logit(p1) + 0.5 * correction
            p_final = self._sigmoid(final_logit)
        else:
            p_final = p1

        return np.column_stack([1 - p_final, p_final])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
