"""
Dataset loaders for the Price of Monotonicity replication study.

Koklev (2025) five original datasets:
  1. German Credit       — UCI Statlog, 1000 rows, binary
  2. Give Me Some Credit — Kaggle GMSC, ~150k rows, binary
  3. Taiwan Credit       — UCI, 30k rows, binary
  4. Polish Bankruptcy   — UCI, ~7.6k rows (year-1), binary
  5. Lending Club        — Kaggle subset, ~150k rows, binary

Extension dataset:
  6. South German Credit — UCI corrected version (Grömping 2019)

HOW TO GET THE REAL DATA:
  UCI datasets  → download from https://archive.ics.uci.edu and save to data/
  Kaggle GMSC   → https://www.kaggle.com/competitions/GiveMeSomeCredit
                  save cs-training.csv as data/gmsc.csv
  Lending Club  → https://www.kaggle.com/datasets/wordsforthewise/lending-club
                  save accepted_2007_to_2018q4.csv as data/lending_club.csv
  South German  → https://archive.ics.uci.edu/dataset/522/south+german+credit
                  save as data/south_german_credit.csv

If the CSVs are not present the loaders fall back to synthetic data and label
the result accordingly — safe for pipeline testing, NOT for the thesis.

For each dataset we define:
  - feature matrix X (numeric, no missing)
  - binary target y  (1 = default / bad)
  - monotone_map : dict {col_index -> +1 or -1}
    where +1 means "higher value → higher PD" (e.g. past-due count)
    and  -1 means "higher value → lower PD" (e.g. income)
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _synthetic_fallback(name, n, n_features, weights, cols, target_col, seed=42):
    warnings.warn(
        f"\n⚠️  SYNTHETIC DATA in use for '{name}' — "
        f"download the real CSV to data/ before final experiments.\n",
        UserWarning, stacklevel=3
    )
    X_arr, y_arr = make_classification(
        n_samples=n, n_features=n_features, n_informative=max(4, n_features//2),
        n_redundant=min(4, n_features//4), weights=weights, random_state=seed
    )
    df = pd.DataFrame(X_arr, columns=cols[:n_features])
    df[target_col] = y_arr
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 1. German Credit (original Statlog)
# ──────────────────────────────────────────────────────────────────────────────

def load_german_credit():
    path = os.path.join(DATA_DIR, "german_credit.csv")

    if os.path.exists(path):
        df = pd.read_csv(path)
        target_col = [c for c in df.columns if c.lower() in ("class","kredit","target","y")][-1]
        y = df[target_col].values
        X = df.drop(columns=[target_col])
        for col in X.select_dtypes(include="object").columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        X = X.fillna(X.median()).astype(float)
        # In UCI Statlog: class 1=good, 2=bad → recode so default=1
        if set(np.unique(y)) == {1, 2}:
            y = (y == 2).astype(int)
        else:
            y = y.astype(int)
        synthetic = False
    else:
        cols = ['checking_status','duration','credit_history','purpose','credit_amount',
                'savings_status','employment','installment_rate','personal_status',
                'other_parties','residence_since','property_magnitude','age',
                'other_payment_plans','housing','existing_credits','job',
                'num_dependents','own_telephone','foreign_worker']
        df = _synthetic_fallback("German Credit", 1000, 20, [0.7,0.3], cols, "class")
        y = df["class"].values.astype(int)
        X = df.drop(columns=["class"]).astype(float)
        synthetic = True

    col_names = list(X.columns)
    mono = {}
    pos_features = ["duration", "credit_amount", "installment_rate", "num_dependents"]
    neg_features = ["age", "present_residence", "existing_credits", "residence_since"]
    for i, c in enumerate(col_names):
        if c in pos_features:   mono[i] = 1
        elif c in neg_features: mono[i] = -1

    label = "German Credit (original)" + (" [SYNTHETIC]" if synthetic else "")
    return X.values, y, col_names, mono, label


# ──────────────────────────────────────────────────────────────────────────────
# 2. South German Credit — corrected (Grömping 2019)
# ──────────────────────────────────────────────────────────────────────────────

def load_south_german_credit():
    path = os.path.join(DATA_DIR, "south_german_credit.csv")

    if os.path.exists(path):
        df = pd.read_csv(path)
        target_col = [c for c in df.columns if c.lower() in ("kredit","class","target","y")][-1]
        y = df[target_col].values
        X = df.drop(columns=[target_col])
        for col in X.select_dtypes(include="object").columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        X = X.fillna(X.median()).astype(float)
        # In South German Credit: 0=bad, 1=good → recode so default=1
        if set(np.unique(y)).issubset({0, 1}):
            y = (y == 0).astype(int)
        synthetic = False
    else:
        cols = ['laufzeit','hoehe','tilgung','verw','hoehe_spareing','bishdauer',
                'rate','famges','bisher_kredit','wohnzeit','wohn','bisher_bank',
                'beruf','pers','telef','gastarb','alter','eigenh','besitz','kredit_art']
        df = _synthetic_fallback("South German Credit", 1000, 20, [0.7,0.3], cols, "kredit", seed=43)
        y = df["kredit"].values.astype(int)
        X = df.drop(columns=["kredit"]).astype(float)
        synthetic = True

    col_names = list(X.columns)
    mono = {}
    pos_features = ["laufzeit", "hoehe", "rate", "duration", "credit_amount"]
    neg_features = ["alter", "wohnzeit", "bisher_kredit", "age"]
    for i, c in enumerate(col_names):
        if c in pos_features:   mono[i] = 1
        elif c in neg_features: mono[i] = -1

    label = "South German Credit (corrected)" + (" [SYNTHETIC]" if synthetic else "")
    return X.values, y, col_names, mono, label


# ──────────────────────────────────────────────────────────────────────────────
# 3. Taiwan Credit
# ──────────────────────────────────────────────────────────────────────────────

def load_taiwan_credit():
    path = os.path.join(DATA_DIR, "taiwan_credit.csv")

    if os.path.exists(path):
        df = pd.read_csv(path, header=1) if "ID" in open(path).read(100) else pd.read_csv(path)
        # drop ID column if present
        df = df[[c for c in df.columns if c.upper() not in ("ID",)]]
        target_col = [c for c in df.columns
                      if "default" in c.lower() or c.lower() == "y"][-1]
        y = df[target_col].values.astype(int)
        X = df.drop(columns=[target_col]).fillna(0).astype(float)
        synthetic = False
    else:
        cols = ['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE',
                'PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6',
                'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6',
                'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
        df = _synthetic_fallback("Taiwan Credit", 30000, 23, [0.78,0.22],
                                 cols, "default.payment.next.month")
        y = df["default.payment.next.month"].values.astype(int)
        X = df.drop(columns=["default.payment.next.month"]).astype(float)
        synthetic = True

    col_names = list(X.columns)
    mono = {}
    for i, c in enumerate(col_names):
        cu = c.upper()
        if cu.startswith("PAY_") and not cu.startswith("PAY_AMT"): mono[i] = 1
        elif cu.startswith("PAY_AMT"):                              mono[i] = -1
        elif cu.startswith("BILL_AMT"):                             mono[i] = 1
        elif cu == "LIMIT_BAL":                                     mono[i] = -1

    label = "Taiwan Credit" + (" [SYNTHETIC]" if synthetic else "")
    return X.values, y, col_names, mono, label


# ──────────────────────────────────────────────────────────────────────────────
# 4. Polish Bankruptcy
# ──────────────────────────────────────────────────────────────────────────────

def load_polish_bankruptcy():
    path = os.path.join(DATA_DIR, "polish_bankruptcy.csv")

    if os.path.exists(path):
        df = pd.read_csv(path)
        target_col = [c for c in df.columns if c.lower() in ("class","bankrupt","y","target")][-1]
        y = df[target_col].values.astype(int)
        X = df.drop(columns=[target_col]).fillna(df.drop(columns=[target_col]).median()).astype(float)
        synthetic = False
    else:
        cols = [f"Attr{i+1}" for i in range(40)]
        df = _synthetic_fallback("Polish Bankruptcy", 7000, 40, [0.97,0.03], cols, "class")
        y = df["class"].values.astype(int)
        X = df.drop(columns=["class"]).astype(float)
        synthetic = True

    col_names = list(X.columns)
    mono = {}
    leverage = {"Attr6","Attr13","Attr29","Attr46","Attr38"}
    profit   = {"Attr1","Attr2","Attr3","Attr8"}
    liquid   = {"Attr4","Attr5","Attr18","Attr19"}
    for i, c in enumerate(col_names):
        if c in leverage: mono[i] = 1
        elif c in profit: mono[i] = -1
        elif c in liquid: mono[i] = -1

    label = "Polish Bankruptcy" + (" [SYNTHETIC]" if synthetic else "")
    return X.values, y, col_names, mono, label


# ──────────────────────────────────────────────────────────────────────────────
# 5. Give Me Some Credit (Kaggle)
# ──────────────────────────────────────────────────────────────────────────────

def load_give_me_some_credit():
    path = os.path.join(DATA_DIR, "gmsc.csv")

    if os.path.exists(path):
        df = pd.read_csv(path)
        # handle optional index column
        if df.columns[0] in ("Unnamed: 0", ""):
            df = df.iloc[:, 1:]
        df = df.dropna()
        y = df["SeriousDlqin2yrs"].values.astype(int)
        X = df.drop(columns=["SeriousDlqin2yrs"]).astype(float)
        synthetic = False
    else:
        cols = ['RevolvingUtilizationOfUnsecuredLines','age',
                'NumberOfTime30-59DaysPastDueNotWorse','DebtRatio','MonthlyIncome',
                'NumberOfOpenCreditLinesAndLoans','NumberOfTimes90DaysLate',
                'NumberRealEstateLoansOrLines','NumberOfTime60-89DaysPastDueNotWorse',
                'NumberOfDependents']
        df = _synthetic_fallback("Give Me Some Credit", 150000, 10, [0.93,0.07],
                                 cols, "SeriousDlqin2yrs")
        y = df["SeriousDlqin2yrs"].values.astype(int)
        X = df.drop(columns=["SeriousDlqin2yrs"]).astype(float)
        synthetic = True

    col_names = list(X.columns)
    mono = {}
    pos = ["NumberOfTime30-59DaysPastDueNotWorse","NumberOfTimes90DaysLate",
           "NumberOfTime60-89DaysPastDueNotWorse","DebtRatio"]
    neg = ["MonthlyIncome","age"]
    for i, c in enumerate(col_names):
        if c in pos:   mono[i] = 1
        elif c in neg: mono[i] = -1

    label = "Give Me Some Credit" + (" [SYNTHETIC]" if synthetic else "")
    return X.values, y, col_names, mono, label


# ──────────────────────────────────────────────────────────────────────────────
# 6. Lending Club
# ──────────────────────────────────────────────────────────────────────────────

def load_lending_club():
    path = os.path.join(DATA_DIR, "lending_club.csv")

    USEFUL_COLS = ["loan_amnt","int_rate","installment","annual_inc","dti",
                   "delinq_2yrs","fico_range_low","fico_range_high","open_acc",
                   "pub_rec","revol_bal","revol_util","total_acc","out_prncp",
                   "total_pymnt","loan_status","default"]

    if os.path.exists(path):
        # try to read with loan_status first (raw Kaggle), fall back to pre-processed
        try:
            df = pd.read_csv(path, usecols=[c for c in USEFUL_COLS if c != "default"],
                             low_memory=False)
            df = df[df["loan_status"].isin(["Fully Paid","Charged Off"])].copy()
            df["default"] = (df["loan_status"] == "Charged Off").astype(int)
            df = df.drop(columns=["loan_status"])
        except Exception:
            df = pd.read_csv(path, low_memory=False)
        for c in ["int_rate","revol_util"]:
            if c in df.columns:
                df[c] = df[c].astype(str).str.replace("%","",regex=False).astype(float)
        df = df.dropna(subset=["default"])
        df = df.fillna(df.median(numeric_only=True))
        from sklearn.utils import resample as _resample
        df = _resample(df, n_samples=min(150_000, len(df)),
                       stratify=df["default"], random_state=42)
        y = df["default"].values.astype(int)
        X = df.drop(columns=["default"]).select_dtypes(include="number").astype(float)
        synthetic = False
    else:
        cols = ["loan_amnt","int_rate","installment","annual_inc","dti",
                "delinq_2yrs","fico_range_low","fico_range_high","open_acc",
                "pub_rec","revol_bal","revol_util","total_acc","out_prncp","total_pymnt"]
        df = _synthetic_fallback("Lending Club", 150000, 15, [0.80,0.20], cols, "default")
        y = df["default"].values.astype(int)
        X = df.drop(columns=["default"]).astype(float)
        synthetic = True

    col_names = list(X.columns)
    mono = {}
    pos = ["int_rate","dti","delinq_2yrs","pub_rec","revol_util"]
    neg = ["annual_inc","fico_range_low","fico_range_high","total_pymnt"]
    for i, c in enumerate(col_names):
        if c in pos:   mono[i] = 1
        elif c in neg: mono[i] = -1

    label = "Lending Club" + (" [SYNTHETIC]" if synthetic else "")
    return X.values, y, col_names, mono, label


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

UCI_DATASETS = {
    "german_credit":       load_german_credit,
    "south_german_credit": load_south_german_credit,
    "taiwan_credit":       load_taiwan_credit,
    "polish_bankruptcy":   load_polish_bankruptcy,
}

KAGGLE_DATASETS = {
    "give_me_some_credit": load_give_me_some_credit,
    "lending_club":        load_lending_club,
}
