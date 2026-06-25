# Price of Monotonicity — Tabular Foundation Models for Credit PD

**Master's thesis project** · Humboldt-Universität zu Berlin

Replicates and extends Koklev (2025) *"What's the Price of Monotonicity?"*
to tabular foundation models (TabPFN), comparing four monotonicity enforcement
strategies in the context of credit probability-of-default (PD) modeling.

---

## Research question

> What is the Price of Monotonicity for tabular foundation models on credit PD data,
> and does it exceed the GBM-established baseline — particularly on small datasets
> where TFMs claim their advantage?

---

## Structure

```
pom-thesis/
├── src/
│   ├── datasets.py      # Dataset loaders (6 datasets; synthetic fallback if CSV absent)
│   ├── metrics.py       # PoM metric with paired bootstrap CI
│   └── models.py        # XGBoost, TabPFN wrapper, two-stage adapter
├── experiments/
│   └── run_benchmark.py # Main experiment runner
├── notebooks/
│   └── plot_results.py  # Summary figure (AUC + Brier)
├── data/                # Put downloaded CSVs here (gitignored)
└── results/             # Output CSVs and figures (gitignored)
```

---

## Datasets

| Dataset | Size | Source |
|---|---|---|
| German Credit (original) | 1,000 | [UCI](https://archive.ics.uci.edu/dataset/144) |
| South German Credit (corrected) | 1,000 | [UCI](https://archive.ics.uci.edu/dataset/522) |
| Taiwan Credit | 30,000 | [UCI](https://archive.ics.uci.edu/dataset/350) |
| Polish Bankruptcy | ~7,600 | [UCI](https://archive.ics.uci.edu/dataset/365) |
| Give Me Some Credit | ~150,000 | [Kaggle](https://www.kaggle.com/competitions/GiveMeSomeCredit) → save as `data/gmsc.csv` |
| Lending Club | ~150,000 | [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club) → save as `data/lending_club.csv` |

Download CSVs manually and place in `data/`. Loaders fall back to synthetic data if files are absent (labelled `[SYNTHETIC]` in results — not valid for the thesis).

---

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate.bat
pip install -r requirements.txt
```

### TabPFN authentication (required for Phase 2)
```bash
huggingface-cli login           # paste your HF token
# then accept terms at: huggingface.co/Prior-Labs/tabpfn_3
```

---

## Running experiments

```bash
# Phase 1: XGBoost replication (Koklev 5 datasets)
python experiments/run_benchmark.py --phase 1

# Phase 2: South German Credit + TabPFN extension
python experiments/run_benchmark.py --phase 2

# Both phases
python experiments/run_benchmark.py --phase all

# Quick smoke test (n_bootstrap=100, max 5k rows)
python experiments/run_benchmark.py --phase all --quick
```

Results are written to `results/pom_results.csv`.

```bash
# Generate summary figure
python notebooks/plot_results.py
```

---

## Monotonicity enforcement strategies

Four strategies are implemented and compared (see `src/models.py`):

| Strategy | Touches TFM? | Evidence base | Implemented |
|---|---|---|---|
| 1. Post-hoc adapter (Stage 1 LR + Stage 2 TabPFN correction) | No | Economic validity audit (2026) | ✅ `PoMTabPFNAdapter` |
| 2. Prior modification (FairPFN-style) | Yes — requires retraining | FairPFN (ICML 2025) | 🔲 HPC required |
| 3. Context engineering | No | Kenfack et al. (2025) | ✅ `ContextEngineeredTabPFN` |
| 4. Output-head constraint (MonoNet-style) | Partial — head only | MonoNet (2023) | ✅ `MonoHeadTabPFN` |

---

## Key references

- Koklev, P. (2025). *What's the Price of Monotonicity?* arXiv:2512.17945
- Hollmann et al. (2025). *TabPFN v2.* Nature.
- Robertson et al. (2025). *FairPFN.* ICML. arXiv:2506.07049
- (2026). *Auditing and Fixing Economic Validity in TFMs for Discrete Choice.* arXiv:2605.26559
- Grömping, U. (2019). *South German Credit Data: Correcting a Widely Used Data Set.*

---

## Status

- [x] XGBoost PoM replication pipeline
- [x] South German Credit (corrected) substitution
- [x] Two-stage adapter (Strategy 1) — `PoMTabPFNAdapter`
- [x] Context engineering (Strategy 3) — `ContextEngineeredTabPFN`
- [x] MonoNet-style output head (Strategy 4) — `MonoHeadTabPFN`
- [ ] Real data download and full run (500 bootstrap)
- [ ] TabPFN authentication and Phase 2 run
- [ ] Final figures and write-up
