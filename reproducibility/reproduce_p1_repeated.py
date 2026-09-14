#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduce Protocol P1 under repeated cross-validation (Table tab:p1-extended, journal
extension, Section "Restoring statistical power").

reproduce_p1.py runs stratified 5-fold CV with a single seed, giving n=5 paired blocks per
dataset — whose exact Wilcoxon floor is p=0.0625 and can never reject at any conventional
level (see audit_pvalue.py). This script repeats the SAME protocol (selection runs INSIDE
every training fold, exactly as in reproduce_p1.py) over multiple random seeds, so that
n = n_folds * n_seeds paired blocks are available per dataset — the paper's extension uses
5 seeds x 5 folds = 25 blocks, which lowers the attainable p-value by roughly seven orders
of magnitude.

It also reports the additional metrics the journal extension adds beyond recall/F1:
precision, specificity, Matthews correlation (MCC), and balanced accuracy — computed the
same way for both the full and the reduced model, every block.

This is intentionally a separate script rather than a --seeds flag bolted onto
reproduce_p1.py: reproduce_p1.py's Table 3 numbers are the conference-version, single-seed
protocol and must stay reproducible exactly as published; this script is the newer,
statistically stronger protocol described in the journal extension.

USES reproduce_p1.py's own make_rf()/select_features() unmodified, so both protocols share
the exact same classifier configuration and selection pipeline — only the resampling
(number of seeds) differs.

Usage
-----
    # the ten benchmark datasets used in Table tab:p1-extended (MH100K excluded, as in the
    # ablation study — the full-dimensional model does not scale to 25 repeats on 24,833
    # features):
    python reproduce_p1_repeated.py --data-dir ./data/Originais --out-dir ./p1_repeated

    # a quick check on the three datasets missing from the paper's current draft:
    python reproduce_p1_repeated.py --data-dir ./data/Originais --out-dir ./p1_repeated \
        --datasets defensedroid_apicalls_closeness,defensedroid_apicalls_degree,defensedroid_apicalls_katz

    # fewer seeds/folds for a fast smoke test before committing to the full run:
    python reproduce_p1_repeated.py --data-dir ./data/Originais --datasets adroit \
        --seeds 42,43 --folds 2 --out-dir ./p1_repeated_smoke

Runtime: selection re-runs the full chi2/MI/RF ranking pipeline inside every one of the
n_folds*n_seeds blocks, so the total cost is exactly n_seeds times reproduce_p1.py's own
cost. The three DefenseDroid API-call datasets (4,274-6,002 raw features) are the most
expensive to select on and dominate the total runtime; consider running them separately
and/or in the background.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.getcwd(), _HERE, os.path.dirname(_HERE)):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

# Reuse reproduce_p1.py's own classifier config and in-fold selection unmodified, so both
# protocols are guaranteed to run the identical pipeline — only the resampling differs.
from reproduce_p1 import make_rf, select_features

# The ten benchmark datasets used throughout the paper's repeated-CV/ablation analyses.
# MH100K is excluded here for the same reason it is excluded from the ablation study
# (§6.4): the full-dimensional model does not scale to n_seeds repeats of in-fold selection
# at 24,833 features within a practical budget.
TEN_BENCHMARK_DATASETS = [
    "adroit", "android_permissions", "androcrawl", "drebin215",
    "kronodroid_real_device", "kronodroid_emulator", "defensedroid_prs",
    "defensedroid_apicalls_closeness", "defensedroid_apicalls_degree",
    "defensedroid_apicalls_katz",
]

METRICS = ("recall", "precision", "f1", "mcc", "specificity", "balanced_accuracy")


def _specificity(y_true, y_pred) -> float:
    """Recall of the negative class — sklearn has no dedicated specificity metric."""
    return recall_score(y_true, y_pred, pos_label=0, zero_division=0)


def _compute_metrics(y_true, y_pred) -> dict:
    return dict(
        recall=recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        precision=precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        f1=f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        mcc=matthews_corrcoef(y_true, y_pred),
        specificity=_specificity(y_true, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
    )


def evaluate_dataset_repeated(path, target_col, n_folds, seeds):
    """Same protocol as reproduce_p1.evaluate_dataset(), repeated over every seed in
    `seeds` — selection runs inside every training fold, exactly as in Protocol P1.
    Returns (p, per_block_DataFrame) with n_folds*len(seeds) rows."""
    df = pd.read_csv(path)
    if target_col not in df.columns:
        raise ValueError(f"target column '{target_col}' not in {os.path.basename(path)}")

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    p = X.shape[1]

    rows = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            X_tr, X_te = X.iloc[tr], X.iloc[te]
            y_tr, y_te = y.iloc[tr], y.iloc[te]

            # Full-dimensional model
            rf_full = make_rf(seed)
            rf_full.fit(X_tr, y_tr)
            pred_full = rf_full.predict(X_te)
            m_full = _compute_metrics(y_te, pred_full)

            # In-fold selection (reproduce_p1.select_features — same pipeline as main.py),
            # then the reduced model
            train_df = X_tr.copy()
            train_df[target_col] = y_tr.values
            selected = select_features(train_df, target_col)
            selected = [c for c in selected if c in X.columns]
            if not selected:                            # degenerate block guard
                continue

            rf_red = make_rf(seed)
            rf_red.fit(X_tr[selected], y_tr)
            pred_red = rf_red.predict(X_te[selected])
            m_red = _compute_metrics(y_te, pred_red)

            row = dict(seed=seed, fold=fold, n_selected=len(selected),
                       reduction_pct=100.0 * (1 - len(selected) / p))
            for m in METRICS:
                row[f"{m}_full"] = m_full[m]
                row[f"{m}_reduced"] = m_red[m]
            rows.append(row)

    return p, pd.DataFrame(rows)


def paired_wilcoxon(a, b):
    """Two-sided paired Wilcoxon; returns None if undefined (all ties)."""
    try:
        return float(wilcoxon(a, b)[1])
    except ValueError:
        return None


def summarise(name, p, blocks) -> dict:
    n = len(blocks)
    out = dict(dataset=name, n_total_features=p, n_blocks=n,
               n_selected=round(blocks["n_selected"].mean(), 1),
               reduction_pct=round(blocks["reduction_pct"].mean(), 2))
    for m in METRICS:
        full = blocks[f"{m}_full"].values
        red = blocks[f"{m}_reduced"].values
        out[f"{m}_full"] = round(float(np.mean(full)), 4)
        out[f"{m}_reduced"] = round(float(np.mean(red)), 4)
        pval = paired_wilcoxon(full, red)
        out[f"{m}_pvalue"] = ("—" if pval is None else pval)
    return out


LATEX_ROW = r"""\multirow{{3}}{{*}}{{{name}}} & full & {rf:.4f} & {pf:.4f} & {f1f:.4f} & {mccf:.4f} & {sf:.4f} & {baf:.4f} & {feat} \\
 & reduced & {rr:.4f} & {pr:.4f} & {f1r:.4f} & {mccr:.4f} & {sr:.4f} & {bar:.4f} & {nsel} \\
 & $p$ & {rp} & & {f1p} & {mccp} & & {bap} & \\
\addlinespace[1.5pt]"""


def _fmt_p(v):
    """Matches main.tex's p-value style, e.g. $6.0\\!\\times\\!10^{-8}$."""
    if v == "—" or v is None:
        return "---"
    mantissa, exp = f"{v:.1e}".split("e")
    return f"${mantissa}\\!\\times\\!10^{{{int(exp)}}}$"


def to_latex_row(s: dict) -> str:
    return LATEX_ROW.format(
        name=s["dataset"], feat=s["n_total_features"], nsel=s["n_selected"],
        rf=s["recall_full"], pf=s["precision_full"], f1f=s["f1_full"],
        mccf=s["mcc_full"], sf=s["specificity_full"], baf=s["balanced_accuracy_full"],
        rr=s["recall_reduced"], pr=s["precision_reduced"], f1r=s["f1_reduced"],
        mccr=s["mcc_reduced"], sr=s["specificity_reduced"], bar=s["balanced_accuracy_reduced"],
        rp=_fmt_p(s["recall_pvalue"]), f1p=_fmt_p(s["f1_pvalue"]),
        mccp=_fmt_p(s["mcc_pvalue"]), bap=_fmt_p(s["balanced_accuracy_pvalue"]),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="./data/Originais",
                    help="directory with dataset CSVs (numeric features + binary class)")
    ap.add_argument("--out-dir", default="./p1_repeated")
    ap.add_argument("--datasets", default=None,
                    help="comma-separated dataset stems (default: the ten benchmark "
                         "datasets, MH100K excluded — see TEN_BENCHMARK_DATASETS)")
    ap.add_argument("--target-col", default="class")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", default="42,43,44,45,46",
                    help="comma-separated seeds (default: 5 seeds, giving n=25 blocks "
                         "with the default --folds 5)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    wanted = ([d.strip() for d in args.datasets.split(",")] if args.datasets
              else TEN_BENCHMARK_DATASETS)
    paths = []
    for name in wanted:
        matches = sorted(glob.glob(os.path.join(args.data_dir, f"{name}.csv")))
        if matches:
            paths.append(matches[0])
        else:
            print(f"[aviso] CSV não encontrado para '{name}' em {args.data_dir}",
                  file=sys.stderr)
    if not paths:
        raise SystemExit(f"nenhum CSV encontrado em {args.data_dir} para {wanted}")

    print(f"Protocolo repetido: {len(seeds)} seeds x {args.folds} folds = "
         f"{len(seeds) * args.folds} blocos pareados por dataset")
    print(f"Seeds: {seeds}")

    summary_rows, per_block_all = [], []
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"\n[P1-repeated] {name} ...", flush=True)
        try:
            p, blocks = evaluate_dataset_repeated(path, args.target_col, args.folds, seeds)
        except Exception as e:                          # keep going on a bad dataset
            print(f"    skipped ({e})")
            continue
        if blocks.empty:
            print("    skipped (no valid blocks)")
            continue

        blocks.insert(0, "dataset", name)
        per_block_all.append(blocks)
        s = summarise(name, p, blocks)
        summary_rows.append(s)
        print(f"    recall  full={s['recall_full']:.4f} reduced={s['recall_reduced']:.4f} "
             f"p={s['recall_pvalue']}")
        print(f"    mcc     full={s['mcc_full']:.4f} reduced={s['mcc_reduced']:.4f} "
             f"p={s['mcc_pvalue']}")

    if not summary_rows:
        raise SystemExit("nenhum resultado gerado")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.out_dir, "table_p1_extended.csv")
    perfold_path = os.path.join(args.out_dir, "per_block_p1_extended.csv")
    latex_path = os.path.join(args.out_dir, "table_p1_extended.tex")

    summary_df.to_csv(summary_path, index=False)
    pd.concat(per_block_all, ignore_index=True).to_csv(perfold_path, index=False)
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("%<<<TAB_P1_EXTENDED>>>\n")
        for _, row in summary_df.iterrows():
            f.write(to_latex_row(row.to_dict()) + "\n")
        f.write("%<<<END_TAB_P1_EXTENDED>>>\n")

    print("\n=== Table (repeated protocol) ===")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summary_df.to_string(index=False))
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {perfold_path}")
    print(f"Saved: {latex_path}  (ready-to-paste rows for main.tex's "
         f"%<<<TAB_P1_EXTENDED>>> block — not applied automatically)")
    return 0


if __name__ == "__main__":
    main()
