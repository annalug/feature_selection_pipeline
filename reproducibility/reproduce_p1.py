#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduce Protocol P1 (Table 3): direct classification, original vs. reduced feature sets.

For each dataset, under stratified 5-fold cross-validation:
  * feature selection (Statistical Ranking) runs INSIDE each training fold only, so the
    selector never sees the test partition (this removes the optimistic bias of
    cross-validating an already-reduced dataset);
  * a fixed Random Forest is trained on the full feature set and on the reduced set;
  * recall and F1 are reported for both, plus the paired Wilcoxon p-value over the 5 folds.

The Wilcoxon p-value is reported for completeness only: at n=5 folds its exact two-sided
floor is 0.0625 > 0.05 (see audit_pvalue.py and paper §5.3).

Usage:
    python reproduce_p1.py --data-dir ./data --out-dir ./p1_out
    python reproduce_p1.py --datasets adroit,drebin215 --data-dir ./data --out-dir ./p1_test

Note: large datasets (e.g. MH100K, 24,833 features) train the full-dimensional model on the
raw feature space and can be slow / memory-heavy; use --datasets to scope a quick check.
"""

import os
import sys
import glob
import argparse
import warnings
import contextlib

# Make the repo's MalwareFeatureSelector importable whether this script is run from the
# repo root or from inside reproducibility/ (main.py lives in the repo root).
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.getcwd(), _HERE, os.path.dirname(_HERE)):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, f1_score


# --- Fixed classifier configuration (paper §5.2) ----------------------------

def make_rf(seed=42):
    return RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        random_state=seed,
        n_jobs=-1,
    )


# --- Selection (paper pipeline) ---------------------------------------------
# MalwareFeatureSelector.auto_select() IS the paper pipeline exactly:
#   variance filter (0.01) -> correlation filter (0.95, computed AFTER variance = v1)
#   -> ensemble vote (theta=2 of 3) with the adaptive budget min(100, max(10, p//10)),
#   where p is the raw feature count.
# It is called on the training fold only, so the selector never sees the test partition.
#
# NOTE: auto_select() must be used rather than calling the filter methods directly —
# remove_low_variance() and correlation_filter() return feature lists but do NOT mutate
# self.X, so only auto_select() actually applies the cleaning stages before the vote.

def select_features(train_df, target_col, correlation_threshold=0.95):
    # The class lives in this repo. The file is main.py; the README also refers to it
    # as feature_selector — import works for either name.
    try:
        from main import MalwareFeatureSelector
    except ImportError:
        from feature_selector import MalwareFeatureSelector

    selector = MalwareFeatureSelector(
        train_df, target_col=target_col, variance_threshold=0.01
    )
    with open(os.devnull, "w") as fnull, contextlib.redirect_stdout(fnull):
        selected = selector.auto_select(
            target_features=None, correlation_threshold=correlation_threshold
        )
    return list(selected)


# --- Per-dataset evaluation -------------------------------------------------

def evaluate_dataset(path, target_col, n_folds, seed):
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"target column '{target_col}' not in {os.path.basename(path)}")

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    p = X.shape[1]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rows = []

    for fold, (tr, te) in enumerate(skf.split(X, y)):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]

        # Full-dimensional model
        rf_full = make_rf(seed)
        rf_full.fit(X_tr, y_tr)
        pred_full = rf_full.predict(X_te)
        rec_full = recall_score(y_te, pred_full, pos_label=1, zero_division=0)
        f1_full = f1_score(y_te, pred_full, pos_label=1, zero_division=0)

        # In-fold selection, then reduced model
        train_df = X_tr.copy()
        train_df[target_col] = y_tr.values
        selected = select_features(train_df, target_col)
        selected = [c for c in selected if c in X.columns]
        if not selected:                            # degenerate fold guard
            warnings.warn(f"{os.path.basename(path)} fold {fold}: no features selected")
            continue

        rf_red = make_rf(seed)
        rf_red.fit(X_tr[selected], y_tr)
        pred_red = rf_red.predict(X_te[selected])
        rec_red = recall_score(y_te, pred_red, pos_label=1, zero_division=0)
        f1_red = f1_score(y_te, pred_red, pos_label=1, zero_division=0)

        rows.append({
            "fold": fold,
            "n_selected": len(selected),
            "reduction_pct": 100.0 * (1 - len(selected) / p),
            "recall_orig": rec_full, "recall_red": rec_red,
            "f1_orig": f1_full, "f1_red": f1_red,
        })

    return p, pd.DataFrame(rows)


def paired_wilcoxon(a, b):
    """Two-sided paired Wilcoxon on recall; returns None if undefined (all ties)."""
    try:
        return float(wilcoxon(a, b)[1])
    except ValueError:
        return None


# --- Main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Reproduce Protocol P1 (Table 3).")
    ap.add_argument("--data-dir", default="./data",
                    help="directory with dataset CSVs (numeric features + binary class)")
    ap.add_argument("--out-dir", default="./p1_out")
    ap.add_argument("--datasets", default=None,
                    help="comma-separated dataset stems to include (default: all CSVs)")
    ap.add_argument("--target-col", default="class")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.data_dir, "*.csv")))
    if args.datasets:
        wanted = {s.strip() for s in args.datasets.split(",")}
        paths = [p for p in paths
                 if os.path.splitext(os.path.basename(p))[0] in wanted]
    if not paths:
        raise SystemExit(f"no matching CSVs in {args.data_dir}")

    summary, per_fold_all = [], []

    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"[P1] {name} ...", flush=True)
        try:
            p, folds = evaluate_dataset(path, args.target_col, args.folds, args.seed)
        except Exception as e:                       # keep going on a bad dataset
            print(f"    skipped ({e})")
            continue
        if folds.empty:
            print("    skipped (no valid folds)")
            continue

        folds.insert(0, "dataset", name)
        per_fold_all.append(folds)

        rec_o, rec_r = folds["recall_orig"].mean(), folds["recall_red"].mean()
        pval = paired_wilcoxon(folds["recall_orig"].values, folds["recall_red"].values)
        summary.append({
            "dataset": name,
            "features": p,
            "orig_recall": round(rec_o, 4),
            "delta_recall": round(rec_r - rec_o, 4),   # Red - Orig
            "red_recall": round(rec_r, 4),
            "orig_f1": round(folds["f1_orig"].mean(), 4),
            "pvalue_recall": ("—" if pval is None else round(pval, 4)),
            "red_f1": round(folds["f1_red"].mean(), 4),
            "reduction_pct": round(folds["reduction_pct"].mean(), 2),
            "n_selected": int(round(folds["n_selected"].mean())),
        })

    table = pd.DataFrame(summary)
    table_path = os.path.join(args.out_dir, "table3_p1.csv")
    perfold_path = os.path.join(args.out_dir, "per_fold_p1.csv")
    table.to_csv(table_path, index=False)
    if per_fold_all:
        pd.concat(per_fold_all, ignore_index=True).to_csv(perfold_path, index=False)

    print("\n=== Protocol P1 — Table 3 ===")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(table.to_string(index=False))
    print(f"\nSaved: {table_path}")
    print(f"Saved: {perfold_path}")
    print("Note: p-values are bounded below by 0.0625 at n=5 folds and support no inference "
          "(paper §5.3; see audit_pvalue.py).")


if __name__ == "__main__":
    main()