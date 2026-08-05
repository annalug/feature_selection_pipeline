#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_cost.py — custo computacional do pipeline de selecao de features (§6.2).

AUTOCONTIDO: mede o custo por estagio (S0-S3) DIRETO dos datasets e depois analisa.
Nao depende mais de um results_raw.csv pre-existente (aquele arquivo era orfao — foi
produzido por uma versao anterior, nao versionada, do pipeline de estagios, que ja nao
existe). Para compatibilidade, se --raw apontar para um results_raw.csv ja existente
(experiment=='stage'), o script analisa esse arquivo em vez de regerar.

ESTAGIOS
--------
  S0 baseline (all features)  — sem limpeza; RF sobre todas as features
  S1 variance only            — filtro de variancia (0.01); RF sobre as sobreviventes
  S2 correlation only         — filtro de correlacao (0.95); RF sobre as sobreviventes
  S3 variance + correlation   — variancia e depois correlacao (ordem v1); RF sobre as sobreviventes

As regras de limpeza espelham ablation_study.clean() e o RF usa RF_PARAMS — os mesmos
valores de main.py — para que os numeros sejam consistentes com o resto do repositorio.

O que mede/analisa
------------------
  - custo medio (segundos), por dataset e por estagio: variancia, correlacao, selecao
    total e treino do RF, alem do total (selecao + treino)
  - reducao de tempo do pipeline completo (S3) frente ao baseline sem selecao (S0)
  - speedup do TREINO isoladamente (fit S0 sobre todas as features vs fit S3 sobre as
    reduzidas) — a alegacao de "~8,1x mais rapido" do artigo
  - custo de correlation_seconds normalizado por p^2, para checar se a etapa de correlacao
    escala quadraticamente com o numero de features (assinatura da matriz densa de corr)

USO
---
    python analyze_cost.py --data-dir ./data/Originais --out-dir ./cost
    python analyze_cost.py --raw ./cost/results_raw.csv --out-dir ./cost   # analisa arquivo existente

ATENCAO (memoria): S2 (correlacao sobre TODAS as features) materializa uma matriz densa
p x p (float64). Para o MH100K (p=24.833) sao ~4,9 GB. Use --datasets para excluir o MH100K
em maquinas com pouca RAM; e exatamente esse custo que a ordem v1 (variancia primeiro) evita.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

# torna ablation_study importavel independente do cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablation_study import (  # noqa: E402  (mesmas constantes de main.py)
    CORRELATION_THRESHOLD,
    N_FOLDS,
    RF_PARAMS,
    SEED,
    VARIANCE_THRESHOLD,
)

STAGE_ORDER = [
    "S0 baseline (all features)",
    "S1 variance only",
    "S2 correlation only",
    "S3 variance + correlation",
]


# ---------------------------------------------------------------------------
# Geracao dos dados de estagio (substitui o results_raw.csv orfao)
# ---------------------------------------------------------------------------

def load(path: Path, label_col: str) -> tuple[np.ndarray, np.ndarray]:
    """Mesma limpeza de ablation_study.load(): NaN/inf -> 0, apenas colunas numericas."""
    df = pd.read_csv(path, low_memory=False)
    if label_col not in df.columns:
        raise KeyError(f"coluna '{label_col}' ausente. Colunas: {list(df.columns)[:8]}...")
    y = df[label_col].to_numpy()
    X = df.drop(columns=[label_col]).select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X.to_numpy(dtype=np.float32), (y > 0).astype(np.int8)


def variance_filter(X: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Regra de ablation_study.clean(): var(axis=0) > VARIANCE_THRESHOLD."""
    return idx[X[:, idx].var(axis=0) > VARIANCE_THRESHOLD]


def correlation_filter(X: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Regra de ablation_study.clean(): matriz de correlacao densa (np.corrcoef),
    subamostrando linhas acima de 20000, removendo colunas cuja maior correlacao
    absoluta com uma coluna anterior excede CORRELATION_THRESHOLD."""
    if idx.size < 2:
        return idx
    sub = X[:, idx]
    if sub.shape[0] > 20000:
        rows = np.random.default_rng(SEED).choice(sub.shape[0], 20000, replace=False)
        sub = sub[rows]
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.abs(np.corrcoef(sub, rowvar=False))
    corr = np.nan_to_num(corr, nan=0.0)
    upper = np.triu(corr, k=1)
    drop = np.any(upper > CORRELATION_THRESHOLD, axis=0)
    return idx[~drop]


def _fit_seconds(X: np.ndarray, y: np.ndarray, idx: np.ndarray) -> float:
    t0 = time.perf_counter()
    RandomForestClassifier(**RF_PARAMS).fit(X[:, idx], y)
    return time.perf_counter() - t0


def stage_experiment(name: str, path: Path, label_col: str, n_folds: int) -> list[dict]:
    """Roda S0-S3 por fold, medindo tempo de cada estagio + treino do RF."""
    X, y = load(path, label_col)
    n_total = X.shape[1]
    all_idx = np.arange(n_total)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    print(f"\n=== {name} | {X.shape[0]:,} amostras x {n_total:,} features")
    rows: list[dict] = []
    for fold, (tr, _) in enumerate(skf.split(X, y)):
        Xtr, ytr = X[tr], y[tr]

        # tempos de limpeza por estagio
        t0 = time.perf_counter(); s1_idx = variance_filter(Xtr, all_idx); var_s1 = time.perf_counter() - t0
        t0 = time.perf_counter(); s2_idx = correlation_filter(Xtr, all_idx); cor_s2 = time.perf_counter() - t0
        t0 = time.perf_counter(); s3_idx = variance_filter(Xtr, all_idx); var_s3 = time.perf_counter() - t0
        t0 = time.perf_counter(); s3_idx = correlation_filter(Xtr, s3_idx); cor_s3 = time.perf_counter() - t0

        stages = [
            ("S0 baseline (all features)", all_idx, 0.0,    0.0),
            ("S1 variance only",           s1_idx,  var_s1, 0.0),
            ("S2 correlation only",        s2_idx,  0.0,    cor_s2),
            ("S3 variance + correlation",  s3_idx,  var_s3, cor_s3),
        ]
        for cfg, idx, vs, cs in stages:
            if idx.size == 0:
                idx = all_idx[:1]
            fit_s = _fit_seconds(Xtr, ytr, idx)
            rows.append(dict(
                experiment="stage", dataset=name, config=cfg, fold=fold,
                n_total=n_total, n_features=int(idx.size),
                variance_seconds=vs, correlation_seconds=cs,
                selection_seconds=vs + cs, fit_seconds=fit_s,
            ))
        print(f"    fold={fold}  S1={s1_idx.size}  S2={s2_idx.size}  S3={s3_idx.size} feats")
    return rows


# ---------------------------------------------------------------------------
# Analise (identica a versao original que consumia results_raw.csv)
# ---------------------------------------------------------------------------

def load_stage(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[raw["experiment"] == "stage"].copy()
    if df.empty:
        raise ValueError("nenhuma linha experiment=='stage'")
    df["total_seconds"] = df["selection_seconds"] + df["fit_seconds"]
    return df


def per_dataset_stage(df: pd.DataFrame) -> pd.DataFrame:
    g = (df.groupby(["dataset", "config"])
           .agg(n_total=("n_total", "first"),
                n_features=("n_features", "mean"),
                variance_seconds=("variance_seconds", "mean"),
                correlation_seconds=("correlation_seconds", "mean"),
                selection_seconds=("selection_seconds", "mean"),
                fit_seconds=("fit_seconds", "mean"),
                total_seconds=("total_seconds", "mean"))
           .reset_index())
    order = {c: i for i, c in enumerate(STAGE_ORDER)}
    g = g[g["config"].isin(order)].copy()
    g["_ord"] = g["config"].map(order)
    return g.sort_values(["dataset", "_ord"]).drop(columns="_ord")


def speedup_table(g: pd.DataFrame) -> pd.DataFrame:
    base = g[g.config == "S0 baseline (all features)"].set_index("dataset")["total_seconds"]
    full = g[g.config == "S3 variance + correlation"].set_index("dataset")["total_seconds"]
    out = pd.DataFrame({"baseline_s": base, "full_pipeline_s": full})
    out["reducao_pct"] = 100 * (1 - out.full_pipeline_s / out.baseline_s)
    out["speedup_x"] = out.baseline_s / out.full_pipeline_s
    return out.sort_values("reducao_pct", ascending=False)


def training_speedup(g: pd.DataFrame) -> pd.DataFrame:
    """Speedup do TREINO isolado: fit sobre todas as features (S0) vs fit sobre as
    reduzidas (S3) — a alegacao de ~8,1x do artigo, separando o custo one-off da selecao."""
    base = g[g.config == "S0 baseline (all features)"].set_index("dataset")["fit_seconds"]
    red = g[g.config == "S3 variance + correlation"].set_index("dataset")["fit_seconds"]
    out = pd.DataFrame({"fit_all_s": base, "fit_reduced_s": red})
    out["train_speedup_x"] = out.fit_all_s / out.fit_reduced_s
    return out.sort_values("train_speedup_x", ascending=False)


def corr_cost_by_p(g: pd.DataFrame) -> pd.DataFrame:
    s2 = g[g.config == "S2 correlation only"].copy()
    s2["p2"] = s2["n_total"].astype(float) ** 2
    s2["seg_por_p2"] = s2["correlation_seconds"] / s2["p2"]
    return s2[["dataset", "n_total", "correlation_seconds", "seg_por_p2"]].sort_values("n_total")


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="diretorio com os CSVs (gera os estagios do zero)")
    ap.add_argument("--raw", type=Path, default=None,
                    help="results_raw.csv ja existente (experiment=='stage'); ignora --data-dir")
    ap.add_argument("--out-dir", type=Path, default=Path("cost"))
    ap.add_argument("--datasets", type=str, default="",
                    help="lista separada por virgula; vazio = todos os CSVs em --data-dir")
    ap.add_argument("--label-col", type=str, default="class")
    ap.add_argument("--n-folds", type=int, default=N_FOLDS)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1) obter os dados brutos: de um arquivo existente OU gerando do zero
    if args.raw is not None and args.raw.exists():
        print(f"Analisando results_raw.csv existente: {args.raw}")
        raw = pd.read_csv(args.raw)
    else:
        if args.data_dir is None:
            raise SystemExit("informe --data-dir (para gerar) ou --raw (arquivo existente)")
        wanted = [d.strip() for d in args.datasets.split(",") if d.strip()]
        paths = ([p for n in wanted for p in sorted(args.data_dir.glob(f"{n}*.csv"))]
                 if wanted else sorted(args.data_dir.glob("*.csv")))
        if not paths:
            raise SystemExit(f"nenhum CSV em {args.data_dir}")
        rows: list[dict] = []
        for p in paths:
            try:
                rows += stage_experiment(p.stem, p, args.label_col, args.n_folds)
            except Exception as exc:                       # noqa: BLE001
                print(f"    [ERRO] {p.stem}: {type(exc).__name__}: {exc}")
        if not rows:
            raise SystemExit("nenhum resultado gerado")
        raw = pd.DataFrame(rows)
        raw.to_csv(args.out_dir / "results_raw.csv", index=False)
        print(f"\nresults_raw.csv gerado em {args.out_dir.resolve()}")

    # 2) analise
    df = load_stage(raw)
    g = per_dataset_stage(df)
    speed = speedup_table(g)
    train = training_speedup(g)
    corr_p = corr_cost_by_p(g)

    g.to_csv(args.out_dir / "custo_por_estagio.csv", index=False)
    speed.to_csv(args.out_dir / "speedup_s0_vs_s3.csv")
    train.to_csv(args.out_dir / "training_speedup.csv")
    corr_p.to_csv(args.out_dir / "correlation_cost_por_p2.csv", index=False)

    pd.set_option("display.width", 140)
    print("\n" + "=" * 78)
    print("CUSTO MEDIO POR ESTAGIO (segundos, media sobre os folds)")
    print("=" * 78)
    print(g.round(4).to_string(index=False))

    print("\n" + "=" * 78)
    print("REDUCAO DE TEMPO: BASELINE (S0) -> PIPELINE COMPLETO (S3)")
    print("=" * 78)
    print(speed.round(4).to_string())

    print("\n" + "=" * 78)
    print("SPEEDUP DO TREINO ISOLADO: fit(todas) S0 vs fit(reduzidas) S3")
    print("=" * 78)
    print(train.round(4).to_string())

    print("\n" + "=" * 78)
    print("CUSTO DE corr() NORMALIZADO POR p^2 (checagem de escala quadratica)")
    print("=" * 78)
    print(corr_p.round(10).to_string(index=False))

    print(f"\nArquivos em {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())