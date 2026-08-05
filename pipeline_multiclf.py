#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_multiclf.py — generaliza pipeline.py (RigorousEvaluator) para cinco
classificadores, respondendo ao revisor que apontou que a avaliação principal
do artigo está centrada em Random Forest.

Não modifica pipeline.py nem main.py. Correções em relação a pipeline.py:
  - bug de caminho: 'data\\relatorio_estatistico_final.csv' (\\r vira carriage
    return em string não-raw) -> pathlib.Path
  - persiste as métricas por fold (per_fold_metrics.csv) em vez de descartá-las,
    permitindo auditar o p-valor de Wilcoxon da Tabela 3
  - Wilcoxon por (dataset, classificador) calculado a partir dos valores por fold
  - uma única chamada a predict() por fold/classificador/variante (o original
    chamava duas vezes: uma para recall, outra para F1)
  - KNN e LogisticRegression recebem features padronizadas com StandardScaler
    ajustado apenas no treino

A seleção de features acontece uma vez por fold via MalwareFeatureSelector
(mesma regra target_features = min(100, max(10, n//10)) de main.py), e o
mesmo subconjunto é reaproveitado pelos cinco classificadores.

USO
---
    python pipeline_multiclf.py --data-dir ./data --out-dir results_v3/T6
    python pipeline_multiclf.py --data-dir ./data --datasets adroit,drebin215
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from main import MalwareFeatureSelector

SEED = 42
N_FOLDS = 5
SCALED_CLASSIFIERS = {"KNN", "LogisticRegression"}


def make_classifier(name: str):
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)
    if name == "DecisionTree":
        return DecisionTreeClassifier(random_state=SEED)
    if name == "KNN":
        return KNeighborsClassifier()
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=1000, random_state=SEED)
    if name == "GradientBoosting":
        return GradientBoostingClassifier(random_state=SEED)
    raise ValueError(name)


CLASSIFIER_NAMES = ("RandomForest", "DecisionTree", "KNN", "LogisticRegression",
                    "GradientBoosting")


def evaluate_variant(clf_name: str, X_train, y_train, X_test, y_test) -> dict:
    clf = make_classifier(clf_name)

    if clf_name in SCALED_CLASSIFIERS:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0

    pred = clf.predict(X_test)  # uma única chamada, reaproveitada abaixo
    return dict(
        recall=recall_score(y_test, pred, zero_division=0),
        f1=f1_score(y_test, pred, zero_division=0),
        mcc=matthews_corrcoef(y_test, pred),
        accuracy=accuracy_score(y_test, pred),
        fit_seconds=fit_s,
    )


def run_dataset(csv_path: Path, target_col: str, n_folds: int,
                skip_classifiers: dict[str, set[str]]) -> list[dict]:
    name = csv_path.stem
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    skip = skip_classifiers.get(name, set())

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    rows: list[dict] = []

    print(f"\n=== {name} === ({X.shape[0]:,} amostras x {X.shape[1]:,} features)")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        train_df = pd.concat([X_train, y_train], axis=1)
        selector = MalwareFeatureSelector(train_df, target_col=target_col)
        t0 = time.perf_counter()
        selected = selector.auto_select()
        sel_s = time.perf_counter() - t0

        for clf_name in CLASSIFIER_NAMES:
            if clf_name in skip:
                print(f"    [skip] fold={fold} {clf_name} (marcado inviável para {name})")
                continue

            m_orig = evaluate_variant(clf_name, X_train, y_train, X_test, y_test)
            rows.append(dict(dataset=name, fold=fold, classifier=clf_name, variant="original",
                             n_features=X_train.shape[1], selection_seconds=0.0, **m_orig))

            m_red = evaluate_variant(clf_name, X_train[selected], y_train,
                                     X_test[selected], y_test)
            rows.append(dict(dataset=name, fold=fold, classifier=clf_name, variant="reduced",
                             n_features=len(selected), selection_seconds=sel_s, **m_red))

        print(f"    fold={fold}  selected={len(selected)} features  sel_s={sel_s:.2f}s")

    return rows


def compute_wilcoxon(per_fold: pd.DataFrame) -> pd.DataFrame:
    out = []
    for metric in ("recall", "f1"):
        for (dataset, clf), g in per_fold.groupby(["dataset", "classifier"]):
            orig = g[g["variant"] == "original"].sort_values("fold")[metric].to_numpy()
            red = g[g["variant"] == "reduced"].sort_values("fold")[metric].to_numpy()
            n = min(len(orig), len(red))
            if n < 1:
                continue
            orig, red = orig[:n], red[:n]
            if np.allclose(orig, red):
                stat, p = np.nan, 1.0
            else:
                try:
                    stat, p = wilcoxon(orig, red)
                except ValueError:
                    stat, p = np.nan, np.nan
            out.append(dict(dataset=dataset, classifier=clf, metric=metric,
                            statistic=stat, pvalue=p, n_pairs=n))
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=Path("results_v3/T6"))
    ap.add_argument("--datasets", type=str, default="",
                    help="lista separada por vírgula; vazio = todos os CSVs em --data-dir")
    ap.add_argument("--label-col", type=str, default="class")
    ap.add_argument("--n-folds", type=int, default=N_FOLDS)
    ap.add_argument("--skip-classifier", action="append", default=[],
                    help="'dataset:Classifier', pode repetir; pula essa combinação e registra")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    skip_classifiers: dict[str, set[str]] = {}
    for item in args.skip_classifier:
        ds, _, clf = item.partition(":")
        skip_classifiers.setdefault(ds, set()).add(clf)

    wanted = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if wanted:
        csv_files = []
        for name in wanted:
            matches = sorted(args.data_dir.glob(f"{name}*.csv"))
            if not matches:
                print(f"[aviso] CSV não encontrado para '{name}'", file=sys.stderr)
                continue
            csv_files.append(matches[0])
    else:
        csv_files = sorted(args.data_dir.glob("*.csv"))

    if not csv_files:
        print("nenhum dataset encontrado", file=sys.stderr)
        return 1

    all_rows: list[dict] = []
    for csv_path in csv_files:
        try:
            all_rows += run_dataset(csv_path, args.label_col, args.n_folds, skip_classifiers)
        except Exception as exc:                      # noqa: BLE001
            print(f"[erro] {csv_path.stem}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

    if not all_rows:
        print("nenhum resultado produzido", file=sys.stderr)
        return 1

    per_fold = pd.DataFrame(all_rows)
    per_fold = per_fold[["dataset", "fold", "classifier", "variant", "recall", "f1", "mcc",
                         "accuracy", "n_features", "fit_seconds", "selection_seconds"]]
    per_fold.to_csv(args.out_dir / "per_fold_metrics.csv", index=False)

    wilcoxon_df = compute_wilcoxon(per_fold)
    wilcoxon_df.to_csv(args.out_dir / "wilcoxon_by_classifier.csv", index=False)

    summary = (per_fold.groupby(["dataset", "classifier", "variant"], as_index=False)
                        .agg(recall=("recall", "mean"), f1=("f1", "mean"),
                             mcc=("mcc", "mean"), accuracy=("accuracy", "mean"),
                             n_features=("n_features", "mean"),
                             fit_seconds=("fit_seconds", "mean"),
                             selection_seconds=("selection_seconds", "mean")))
    summary.to_csv(args.out_dir / "summary_by_classifier.csv", index=False)

    print(f"\nArquivos gravados em {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
