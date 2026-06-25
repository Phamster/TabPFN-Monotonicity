"""
Main experiment runner for the Price of Monotonicity benchmark.

Phase 1 (replication):  XGBoost unconstrained vs constrained
                        on Koklev's original 5 datasets
                        (UCI ones downloaded automatically;
                         Kaggle ones require manual download)

Phase 2 (extension):   Same XGBoost PoM on South German Credit (corrected)
                        + TabPFN two-stage adapter PoM on South German Credit

Run with:
    python experiments/run_benchmark.py --phase 1
    python experiments/run_benchmark.py --phase 2
    python experiments/run_benchmark.py --phase all
"""

import sys
import os
import argparse
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# project imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.datasets import UCI_DATASETS, KAGGLE_DATASETS, load_south_german_credit
from src.metrics  import compute_pom, format_pom_row
from src.models   import (build_xgb_unconstrained, build_xgb_constrained,
                           TabPFNWrapper, PoMTabPFNAdapter, TABPFN_AVAILABLE)

warnings.filterwarnings("ignore")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TEST_SIZE    = 0.2
N_BOOTSTRAP  = 500     # 1000 per Koklev; 500 for speed — increase for final runs
RANDOM_STATE = 42


def run_xgb_experiment(loader_fn, dataset_key, rows=None):
    print(f"\n{'─'*60}")
    print(f"  Loading: {dataset_key}")
    try:
        X, y, col_names, mono_map, label = loader_fn()
    except FileNotFoundError as e:
        print(f"  SKIP — {e}")
        return []
    except Exception as e:
        print(f"  ERROR loading {dataset_key}: {e}")
        return []

    # optional row cap (for quick testing)
    if rows and len(y) > rows:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(y), rows, replace=False)
        X, y = X[idx], y[idx]

    n_constrained = len(mono_map)
    n_features    = X.shape[1]
    default_rate  = y.mean()
    print(f"  Shape: {X.shape}  |  Default rate: {default_rate:.1%}  "
          f"|  Constrained features: {n_constrained}/{n_features}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    unc = build_xgb_unconstrained(n_features)
    con = build_xgb_constrained(mono_map, n_features)

    t0 = time.time()
    result = compute_pom(unc, con, X_train, y_train, X_test, y_test,
                         n_bootstrap=N_BOOTSTRAP)
    elapsed = time.time() - t0

    row = format_pom_row(label, "XGBoost", result)
    print(f"  PoM AUC:   {row['pom_auc_pct']:+.3f}%  "
          f"95% CI [{row['pom_auc_ci95_lo']:+.3f}%, {row['pom_auc_ci95_hi']:+.3f}%]")
    print(f"  PoM Brier: {row['pom_brier_pct']:+.3f}%  "
          f"95% CI [{row['pom_brier_ci95_lo']:+.3f}%, {row['pom_brier_ci95_hi']:+.3f}%]")
    print(f"  Elapsed: {elapsed:.1f}s  (bootstrap n={N_BOOTSTRAP})")
    return [row]


def run_tabpfn_experiment(loader_fn, dataset_key, max_train=8000):
    if not TABPFN_AVAILABLE:
        print("  SKIP TabPFN — package not available")
        return []

    print(f"\n{'─'*60}")
    print(f"  Loading (TabPFN): {dataset_key}")
    try:
        X, y, col_names, mono_map, label = loader_fn()
    except Exception as e:
        print(f"  ERROR: {e}")
        return []

    # TabPFN works best on small-medium data; cap train at max_train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    if len(y_train) > max_train:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(y_train), max_train, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    n_features = X.shape[1]
    print(f"  Train size (capped): {len(y_train)}  |  Test size: {len(y_test)}")

    rows = []

    # 2a. Raw TabPFN (unconstrained baseline for TabPFN)
    print("  Running: TabPFN (raw, unconstrained) vs XGBoost constrained")
    unc_tfm = TabPFNWrapper()
    con_xgb = build_xgb_constrained(mono_map, n_features)

    t0 = time.time()
    result_raw = compute_pom(unc_tfm, con_xgb,
                             X_train, y_train, X_test, y_test,
                             n_bootstrap=min(N_BOOTSTRAP, 200))
    elapsed = time.time() - t0
    row_raw = format_pom_row(label + " [TabPFN raw vs XGB-constrained]",
                             "TabPFN_raw", result_raw)
    print(f"  PoM AUC (TabPFN raw vs XGB-con): {row_raw['pom_auc_pct']:+.3f}%  "
          f"(elapsed {elapsed:.1f}s)")
    rows.append(row_raw)

    # 2b. TabPFN with two-stage adapter (the actual thesis contribution)
    print("  Running: TabPFN two-stage adapter (constrained) vs raw TabPFN")
    raw_tfm     = TabPFNWrapper()
    adapted_tfm = PoMTabPFNAdapter(mono_map=mono_map)

    t0 = time.time()
    result_adapted = compute_pom(raw_tfm, adapted_tfm,
                                 X_train, y_train, X_test, y_test,
                                 n_bootstrap=min(N_BOOTSTRAP, 200))
    elapsed = time.time() - t0
    row_adapted = format_pom_row(label + " [TabPFN raw vs TabPFN+adapter]",
                                 "TabPFN_adapter", result_adapted)
    print(f"  PoM AUC (adapter cost): {row_adapted['pom_auc_pct']:+.3f}%  "
          f"(elapsed {elapsed:.1f}s)")
    rows.append(row_adapted)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["1", "2", "all"], default="all")
    parser.add_argument("--quick", action="store_true",
                        help="Cap rows at 5000 and bootstrap at 100 for fast testing")
    args = parser.parse_args()

    global N_BOOTSTRAP
    rows_cap = None
    if args.quick:
        N_BOOTSTRAP = 100
        rows_cap    = 5000
        print("QUICK MODE: n_bootstrap=100, max rows=5000")

    all_results = []

    # ── Phase 1: XGBoost replication ──────────────────────────────────────────
    if args.phase in ("1", "all"):
        print("\n" + "═"*60)
        print("  PHASE 1 — XGBoost PoM replication (Koklev 2025 datasets)")
        print("═"*60)

        for key, loader in UCI_DATASETS.items():
            if key == "south_german_credit":
                continue   # saved for phase 2
            all_results += run_xgb_experiment(loader, key, rows=rows_cap)

        for key, loader in KAGGLE_DATASETS.items():
            all_results += run_xgb_experiment(loader, key, rows=rows_cap)

    # ── Phase 2: Extension ────────────────────────────────────────────────────
    if args.phase in ("2", "all"):
        print("\n" + "═"*60)
        print("  PHASE 2 — Extension: South German Credit + TabPFN")
        print("═"*60)

        # XGBoost on corrected South German Credit
        all_results += run_xgb_experiment(
            load_south_german_credit, "south_german_credit", rows=rows_cap
        )

        # TabPFN experiments on South German Credit
        all_results += run_tabpfn_experiment(
            load_south_german_credit, "south_german_credit"
        )

    # ── Save results ──────────────────────────────────────────────────────────
    if all_results:
        df = pd.DataFrame(all_results)
        out_path = os.path.join(RESULTS_DIR, "pom_results.csv")
        df.to_csv(out_path, index=False)
        print(f"\n{'═'*60}")
        print(f"  Results saved → {out_path}")
        print(f"{'═'*60}")
        print(df.to_string(index=False))
    else:
        print("\nNo results collected — check dataset availability.")

    return all_results


if __name__ == "__main__":
    main()
