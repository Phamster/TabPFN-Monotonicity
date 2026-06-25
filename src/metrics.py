"""
Price of Monotonicity (PoM) metric — following Koklev (2025).

PoM = (metric_unconstrained - metric_constrained) / metric_unconstrained

Estimated via paired bootstrap: train both models on B bootstrap samples of
the training set, evaluate on a held-out test set, compute PoM for each
bootstrap draw, then report mean ± 95% CI.

Metrics: AUC-ROC (discrimination) and Brier score (calibration).
Note: for Brier, lower is better, so PoM = (Brier_constrained - Brier_unconstrained)
      / Brier_unconstrained (positive PoM = constrained is worse).
"""

import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy import stats


def compute_pom(
    model_unconstrained,
    model_constrained,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> dict:
    """
    Paired bootstrap estimate of the Price of Monotonicity.

    Returns a dict with keys:
      auc_unconstrained, auc_constrained,
      brier_unconstrained, brier_constrained,
      pom_auc_mean, pom_auc_ci95,
      pom_brier_mean, pom_brier_ci95,
      pom_auc_pct, pom_brier_pct        ← percentage versions (headline numbers)
    """
    rng = np.random.default_rng(random_state)
    n = len(y_train)

    pom_auc_samples   = np.zeros(n_bootstrap)
    pom_brier_samples = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        Xb, yb = X_train[idx], y_train[idx]

        # skip degenerate bootstrap draws (only one class)
        if len(np.unique(yb)) < 2:
            pom_auc_samples[b]   = np.nan
            pom_brier_samples[b] = np.nan
            continue

        model_unconstrained.fit(Xb, yb)
        model_constrained.fit(Xb, yb)

        p_unc = model_unconstrained.predict_proba(X_test)[:, 1]
        p_con = model_constrained.predict_proba(X_test)[:, 1]

        auc_unc   = roc_auc_score(y_test, p_unc)
        auc_con   = roc_auc_score(y_test, p_con)
        brier_unc = brier_score_loss(y_test, p_unc)
        brier_con = brier_score_loss(y_test, p_con)

        # PoM: how much does constraining cost?
        # AUC: higher is better → cost = (unc - con) / unc
        pom_auc_samples[b] = (auc_unc - auc_con) / auc_unc if auc_unc > 0 else np.nan
        # Brier: lower is better → cost = (con - unc) / unc
        pom_brier_samples[b] = (brier_con - brier_unc) / brier_unc if brier_unc > 0 else np.nan

    # drop NaN draws
    auc_draws   = pom_auc_samples[~np.isnan(pom_auc_samples)]
    brier_draws = pom_brier_samples[~np.isnan(pom_brier_samples)]

    def ci95(arr):
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return (lo, hi)

    # point estimates on full training set
    model_unconstrained.fit(X_train, y_train)
    model_constrained.fit(X_train, y_train)
    p_unc_full = model_unconstrained.predict_proba(X_test)[:, 1]
    p_con_full = model_constrained.predict_proba(X_test)[:, 1]

    auc_unc_full   = roc_auc_score(y_test, p_unc_full)
    auc_con_full   = roc_auc_score(y_test, p_con_full)
    brier_unc_full = brier_score_loss(y_test, p_unc_full)
    brier_con_full = brier_score_loss(y_test, p_con_full)

    return {
        "auc_unconstrained":  auc_unc_full,
        "auc_constrained":    auc_con_full,
        "brier_unconstrained": brier_unc_full,
        "brier_constrained":   brier_con_full,
        "pom_auc_mean":   float(np.mean(auc_draws)),
        "pom_auc_ci95":   ci95(auc_draws),
        "pom_auc_pct":    float(np.mean(auc_draws)) * 100,
        "pom_brier_mean": float(np.mean(brier_draws)),
        "pom_brier_ci95": ci95(brier_draws),
        "pom_brier_pct":  float(np.mean(brier_draws)) * 100,
        "n_bootstrap_valid_auc":   int(len(auc_draws)),
        "n_bootstrap_valid_brier": int(len(brier_draws)),
    }


def format_pom_row(dataset_name: str, model_name: str, result: dict) -> dict:
    """Flatten a result dict into a table row."""
    lo_auc, hi_auc     = result["pom_auc_ci95"]
    lo_brier, hi_brier = result["pom_brier_ci95"]
    return {
        "dataset":            dataset_name,
        "model":              model_name,
        "auc_unconstrained":  round(result["auc_unconstrained"], 4),
        "auc_constrained":    round(result["auc_constrained"], 4),
        "pom_auc_pct":        round(result["pom_auc_pct"], 3),
        "pom_auc_ci95_lo":    round(lo_auc * 100, 3),
        "pom_auc_ci95_hi":    round(hi_auc * 100, 3),
        "brier_unconstrained": round(result["brier_unconstrained"], 4),
        "brier_constrained":   round(result["brier_constrained"], 4),
        "pom_brier_pct":      round(result["pom_brier_pct"], 3),
        "pom_brier_ci95_lo":  round(lo_brier * 100, 3),
        "pom_brier_ci95_hi":  round(hi_brier * 100, 3),
    }
