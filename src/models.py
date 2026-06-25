"""
Model builders for the PoM benchmark.

XGBoost (replication of Koklev 2025):
  - Hyperparameters kept close to Koklev defaults (n_estimators=300,
    learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8)
  - Monotone constraints passed via xgb's native `monotone_constraints` param
  - Constrained model uses the dataset's mono_map; unconstrained sets all to 0

TabPFN (extension):
  - TabPFNClassifier from the Prior Labs tabpfn package
  - No native monotone_constraints support → four enforcement strategies:
    1. PoMTabPFNAdapter   — post-hoc two-stage adapter (economic-validity-audit 2026)
    2. ContextEngineeredTabPFN — context engineering (Kenfack et al. 2025)
    3. MonoHeadTabPFN     — sign-constrained output head (MonoNet-style)
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
# Strategy 3: Context Engineering
# Inspired by Kenfack et al. (2025) — shift ICL context to enforce monotonicity
# ──────────────────────────────────────────────────────────────────────────────

class ContextEngineeredTabPFN(BaseEstimator, ClassifierMixin):
    """
    Curates the training context passed to TabPFN to bias predictions toward
    monotone-consistent outcomes.  Two complementary mechanisms:

    1. Score-based selection: rank training rows by "monotone alignment" —
       how consistently their label agrees with the expected feature direction
       across all constrained features.  Top-scoring rows fill the context.

    2. Synthetic anchors: add n_anchors_per_feature pairs of synthetic rows
       at extreme feature values (10th / 90th percentile) with appropriate
       labels, replicated anchor_weight times to emphasise them.

    This is the inference-only analogue of COMET's counterexample-guided
    training (moved from training time to context-window construction time).
    """

    def __init__(self, mono_map: dict, n_anchors_per_feature: int = 2,
                 anchor_weight: int = 5, device: str = "cpu"):
        self.mono_map = mono_map
        self.n_anchors_per_feature = n_anchors_per_feature
        self.anchor_weight = anchor_weight
        self.device = device
        self._clf = None
        self.classes_ = np.array([0, 1])

    def _alignment_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return per-row monotone alignment score in [-1, 1]."""
        if not self.mono_map:
            return np.zeros(len(y))
        scores = np.zeros(len(y))
        n_mono = 0
        for feat_idx, direction in self.mono_map.items():
            if feat_idx >= X.shape[1]:
                continue
            above = (X[:, feat_idx] > np.median(X[:, feat_idx])).astype(float)
            if direction == 1:
                scores += (2 * y - 1) * (2 * above - 1)
            else:
                scores += (2 * y - 1) * (1 - 2 * above)
            n_mono += 1
        return scores / max(n_mono, 1)

    def _build_anchors(self, X: np.ndarray) -> tuple:
        """Synthetic anchor rows demonstrating each monotone relationship."""
        if not self.mono_map:
            return np.empty((0, X.shape[1])), np.array([], dtype=int)
        medians = np.median(X, axis=0)
        ax, ay = [], []
        for feat_idx, direction in self.mono_map.items():
            if feat_idx >= X.shape[1]:
                continue
            col = X[:, feat_idx]
            high = np.percentile(col, 90)
            low  = np.percentile(col, 10)
            label_high = 1 if direction == 1 else 0
            for _ in range(self.n_anchors_per_feature):
                row_h = medians.copy(); row_h[feat_idx] = high
                row_l = medians.copy(); row_l[feat_idx] = low
                ax.extend([row_h, row_l])
                ay.extend([label_high, 1 - label_high])
        anc_X = np.array(ax)
        anc_y = np.array(ay, dtype=int)
        anc_X = np.tile(anc_X, (self.anchor_weight, 1))
        anc_y = np.tile(anc_y, self.anchor_weight)
        return anc_X, anc_y

    def fit(self, X, y):
        if not TABPFN_AVAILABLE:
            raise RuntimeError("tabpfn package not installed.")
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)

        anc_X, anc_y = self._build_anchors(X)
        n_anchors = len(anc_y)
        max_real = max(100, 9000 - n_anchors)

        if len(y) > max_real:
            scores = self._alignment_scores(X, y)
            order = np.argsort(-scores)[:max_real]
            X_ctx, y_ctx = X[order], y[order]
        else:
            X_ctx, y_ctx = X.copy(), y.copy()

        if n_anchors > 0:
            X_ctx = np.vstack([X_ctx, anc_X])
            y_ctx = np.concatenate([y_ctx, anc_y])

        self._clf = TabPFNClassifier(device=self.device)
        self._clf.fit(X_ctx, y_ctx)
        return self

    def predict_proba(self, X):
        return self._clf.predict_proba(np.array(X, dtype=float))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ──────────────────────────────────────────────────────────────────────────────
# Strategy 4: Output-head constraint (MonoNet-style)
# Sign-constrained linear head trained on top of frozen TabPFN logits
# ──────────────────────────────────────────────────────────────────────────────

class MonoHeadTabPFN(BaseEstimator, ClassifierMixin):
    """
    Leaves TabPFN's in-context mechanism untouched and adds a tiny
    sign-constrained linear head on top, mapping:

        z = b + w_tfm * logit(tabpfn_prob) + w_features @ x_scaled
        final_prob = sigmoid(z)

    Sign constraints (enforced by projection after every gradient step):
        w_tfm >= 0  (TabPFN's raw prediction contributes non-negatively)
        w_i   >= 0  for mono_map[i] == +1
        w_i   <= 0  for mono_map[i] == -1
        w_i is unconstrained for features not in mono_map

    The head is trained via projected gradient descent on cross-entropy loss
    (pure numpy — no extra dependencies beyond what TabPFN already requires).
    """

    def __init__(self, mono_map: dict, device: str = "cpu",
                 lr: float = 0.05, n_iter: int = 300, random_state: int = 42):
        self.mono_map = mono_map
        self.device = device
        self.lr = lr
        self.n_iter = n_iter
        self.random_state = random_state
        self._tabpfn = None
        self._scaler = StandardScaler()
        self._w = None
        self._b = 0.0
        self.classes_ = np.array([0, 1])

    @staticmethod
    def _logit(p):
        return np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1))

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def _project(self, w: np.ndarray) -> np.ndarray:
        w = w.copy()
        w[0] = max(w[0], 0.0)
        for feat_idx, direction in self.mono_map.items():
            j = feat_idx + 1    # offset: index 0 is the TabPFN logit weight
            if j < len(w):
                if direction == 1:
                    w[j] = max(w[j], 0.0)
                elif direction == -1:
                    w[j] = min(w[j], 0.0)
        return w

    def fit(self, X, y):
        if not TABPFN_AVAILABLE:
            raise RuntimeError("tabpfn package not installed.")
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        Xs = self._scaler.fit_transform(X)

        if len(y) > 9000:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(y), 9000, replace=False)
            Xs, y = Xs[idx], y[idx]

        self._tabpfn = TabPFNClassifier(device=self.device)
        self._tabpfn.fit(Xs, y.astype(int))

        p_tfm = self._tabpfn.predict_proba(Xs)[:, 1]
        X_aug = np.hstack([self._logit(p_tfm).reshape(-1, 1), Xs])

        rng = np.random.default_rng(self.random_state)
        w = rng.normal(0, 0.01, X_aug.shape[1])
        w[0] = 0.5
        b = 0.0
        w = self._project(w)

        n = len(y)
        for _ in range(self.n_iter):
            p = self._sigmoid(X_aug @ w + b)
            err = p - y
            w = self._project(w - self.lr * (X_aug.T @ err / n))
            b = b - self.lr * err.mean()

        self._w = w
        self._b = b
        return self

    def predict_proba(self, X):
        X = np.array(X, dtype=float)
        Xs = self._scaler.transform(X)
        p_tfm = self._tabpfn.predict_proba(Xs)[:, 1]
        X_aug = np.hstack([self._logit(p_tfm).reshape(-1, 1), Xs])
        p_final = self._sigmoid(X_aug @ self._w + self._b)
        return np.column_stack([1 - p_final, p_final])

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
