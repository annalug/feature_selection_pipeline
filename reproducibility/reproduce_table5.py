#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduce Table 5: generalisation across classifier families on the reduced MH100K subset.

Table 5 evaluates FOUR classifiers from distinct inductive families — a single decision
tree, a random forest, a distance-based classifier (k-NN), and a linear model (logistic
regression) — on the MH100K reduced subset (93 features), under stratified 5-fold CV
(paper Section 6.3).

Unlike Protocol P1 (reproduce_p1.py), where selection runs INSIDE each fold, Table 5 selects
the subset ONCE over the whole dataset and reuses the same subset in every fold. This is a
documented, deliberate choice, not a bug — the paper says so explicitly (Sec. 6.3): "these
figures come from cross-validating a subset selected once over the whole dataset rather than
inside each fold as in Protocol P1 ... the absolute values are optimistically biased": 0.7502
for Random Forest under in-fold selection (Table 3) vs. 0.8587 here, a gap of 0.109. This
script reproduces that exact, documented protocol — bias included, by design.

GradientBoosting — the fifth classifier in pipeline_multiclf.py's broader robustness check,
which additionally reselects features inside every fold — is NOT one of Table 5's four; see
"Note on Table 5" in reproducibility/README.md for that distinction.

Random Forest configuration matches paper Sec. 5.2 exactly: n_estimators=100, gini,
max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='sqrt',
bootstrap=True, random_state=42, n_jobs=-1. Features are standardised (StandardScaler fit
on training data only) for the two classifiers that need it: k-NN and Logistic Regression.

Usage
-----
    # from an already-reduced CSV, e.g. the one imported into MalDataGen's Datasets/ for P2:
    python reproduce_table5.py --reduced-csv Maldatagen_additional_metrics/Datasets/statistical_mh100k.csv \
        --out-dir ./table5

    # or select once from the full dataset (slow: runs chi2/MI/RF ranking over 24,833 features
    # on mh100k's 101,934 rows — can take a long time; see main README's runtime notes):
    python reproduce_table5.py --data-dir ./data/Originais --dataset mh100k --out-dir ./table5
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.getcwd(), _HERE, os.path.dirname(_HERE)):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, matthews_corrcoef, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42
CLASSIFIER_NAMES = ("Decision Tree", "k-NN", "Random Forest", "Logistic Regression")
SCALED_CLASSIFIERS = {"k-NN", "Logistic Regression"}


def make_classifier(name: str, seed: int = SEED):
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=seed)
    if name == "k-NN":
        return KNeighborsClassifier()
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=100, criterion="gini", max_depth=None,
            min_samples_split=2, min_samples_leaf=1, max_features="sqrt",
            bootstrap=True, random_state=seed, n_jobs=-1,
        )
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=seed)
    raise ValueError(name)


def select_once(df: pd.DataFrame, target_col: str, correlation_threshold: float) -> list[str]:
    """Table 5's protocol: select the subset ONCE over the whole dataset — distinct from
    reproduce_p1.py's select_features(), which is called inside each training fold."""
    try:
        from main import MalwareFeatureSelector
    except ImportError:
        from feature_selector import MalwareFeatureSelector

    selector = MalwareFeatureSelector(df, target_col=target_col, variance_threshold=0.01)
    with open(os.devnull, "w") as fnull, contextlib.redirect_stdout(fnull):
        selected = selector.auto_select(
            target_features=None, correlation_threshold=correlation_threshold
        )
    return list(selected)


def evaluate(X: pd.DataFrame, y: pd.Series, selected: list[str],
            n_folds: int, seed: int) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        X_tr, X_te = X.iloc[tr][selected], X.iloc[te][selected]
        y_tr, y_te = y.iloc[tr], y.iloc[te]

        for name in CLASSIFIER_NAMES:
            clf = make_classifier(name, seed)
            Xtr, Xte = X_tr, X_te
            if name in SCALED_CLASSIFIERS:
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(Xtr)
                Xte = scaler.transform(Xte)

            t0 = time.perf_counter()
            clf.fit(Xtr, y_tr)
            fit_s = time.perf_counter() - t0

            pred = clf.predict(Xte)
            rows.append(dict(
                classifier=name, fold=fold,
                recall=recall_score(y_te, pred, zero_division=0),
                f1=f1_score(y_te, pred, zero_division=0),
                mcc=matthews_corrcoef(y_te, pred),
                fit_seconds=fit_s,
            ))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reduced-csv", default=None,
                    help="an already-reduced CSV (numeric features + binary target column) "
                         "to use as-is, instead of selecting from --data-dir/--dataset")
    ap.add_argument("--data-dir", default="./data/Originais",
                    help="directory holding the FULL dataset (used only if --reduced-csv "
                         "is not given)")
    ap.add_argument("--dataset", default="mh100k",
                    help="dataset stem inside --data-dir to select from (default: mh100k, "
                         "the paper's large-scale case)")
    ap.add_argument("--out-dir", default="./table5")
    ap.add_argument("--target-col", default="class")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--correlation-threshold", type=float, default=0.95)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.reduced_csv:
        df = pd.read_csv(args.reduced_csv)
        if args.target_col not in df.columns:
            raise SystemExit(f"target column '{args.target_col}' not in {args.reduced_csv}")
        X_full = df.drop(columns=[args.target_col])
        y_full = df[args.target_col].astype(int)
        selected = list(X_full.columns)   # already reduced: every remaining column counts
        print(f"Using pre-reduced CSV: {args.reduced_csv} ({len(selected)} features)")
    else:
        path = os.path.join(args.data_dir, f"{args.dataset}.csv")
        if not os.path.isfile(path):
            raise SystemExit(f"{path} not found — pass --reduced-csv instead, or check "
                             f"--data-dir/--dataset")
        df = pd.read_csv(path)
        if args.target_col not in df.columns:
            raise SystemExit(f"target column '{args.target_col}' not in {path}")
        X_full = df.drop(columns=[args.target_col])
        y_full = df[args.target_col].astype(int)
        print(f"Selecting once over the whole dataset ({X_full.shape[1]:,} features) — this "
             f"runs chi2/MI/RF ranking on every row and can take a long time on mh100k...")
        selected = select_once(df, args.target_col, args.correlation_threshold)
        print(f"Selected {len(selected)} features (paper: 93 for mh100k)")

    per_fold = evaluate(X_full, y_full, selected, args.n_folds, args.seed)
    per_fold_path = os.path.join(args.out_dir, "table5_per_fold.csv")
    per_fold.to_csv(per_fold_path, index=False)

    summary = (per_fold.groupby("classifier")
               .agg(recall=("recall", "mean"), sd=("recall", "std"),
                    f1=("f1", "mean"), mcc=("mcc", "mean"),
                    train_s=("fit_seconds", "mean"))
               .reindex(CLASSIFIER_NAMES)
               .round(4))
    summary_path = os.path.join(args.out_dir, "table5.csv")
    summary.to_csv(summary_path)

    print(f"\n=== Table 5 — {len(selected)} features, {args.n_folds}-fold CV ===")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(summary.to_string())
    rng = summary[["recall", "f1", "mcc"]].max() - summary[["recall", "f1", "mcc"]].min()
    print("\nRange (max-min across classifiers):", dict(rng.round(4)))
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {per_fold_path}")
    print("\nNote: this subset was selected ONCE over the whole dataset (not per-fold, unlike "
         "Protocol P1) — the absolute values are optimistically biased vs. Table 3's in-fold "
         "numbers, by design (paper Sec. 6.3). See this script's own docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
