#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_cost.py — custo computacional do pipeline de selecao de features (§6.2).

AUTOCONTIDO: mede o custo por estagio (S0-S3) DIRETO dos datasets e depois analisa.
Nao depende mais de um results_raw.csv pre-existente (aquele arquivo era orfao — foi
produzido por uma versao anterior, nao versionada, do pipeline de estagios, que ja nao
existe). Para compatibilidade, se --raw apontar para um results_raw.csv ja existente
(experiment=='stage'), o script analisa esse arquivo em vez de regerar.

ESTAGIOS (custo de LIMPEZA apenas — variancia + correlacao)
-------------------------------------------------------------
  S0 baseline (all features)  — sem limpeza; RF sobre todas as features
  S1 variance only            — filtro de variancia (0.01); RF sobre as sobreviventes
  S2 correlation only         — filtro de correlacao (0.95); RF sobre as sobreviventes
  S3 variance + correlation   — variancia e depois correlacao (ordem v1); RF sobre as sobreviventes

As regras de limpeza espelham ablation_study.clean() e o RF usa RF_PARAMS — os mesmos
valores de main.py — para que os numeros sejam consistentes com o resto do repositorio.

AMORTIZACAO (custo COMPLETO da selecao — variancia + correlacao + chi2 + MI + RF + voto)
------------------------------------------------------------------------------------------
O artigo (§6.2) mede o custo de selecao (selection_seconds) a partir do pipeline INTEIRO,
nao so variancia+correlacao: "The cleaning filters account for only 2.4% of the selection
time (0.33s), the ranking stage for the remaining 97.6% (13.28s)". As estagios S0-S3 acima
NAO incluem chi2/MI/RF/voto, entao NAO sao o selection_seconds do break-even. A funcao
amortization_experiment() reaproveita main.MalwareFeatureSelector.auto_select() (o pipeline
completo, identico ao que main.py roda) para medir esse custo corretamente, e calcula:
  - break-even: numero de ciclos de retreino em que a economia de treino paga o custo
    one-off da selecao: selection_s / (fit_full_s - fit_reduced_s)   (artigo: mediana 7,7)
  - economia agregada em N atualizacoes: 1 - (selection_s + N*fit_reduced_s)/(N*fit_full_s)
    (artigo: -6,2x custo extra em 1 rodada, 27,1% de economia em 10, 75,6% em 50)

O que mede/analisa
------------------
  - custo medio (segundos), por dataset e por estagio: variancia, correlacao, selecao
    total e treino do RF, alem do total (selecao + treino)
  - reducao de tempo do pipeline completo (S3) frente ao baseline sem selecao (S0)
  - speedup do TREINO isoladamente (fit S0 sobre todas as features vs fit S3 sobre as
    reduzidas) — a alegacao de "~8,1x mais rapido" do artigo
  - custo de correlation_seconds normalizado por p^2, para checar se a etapa de correlacao
    escala quadraticamente com o numero de features (assinatura da matriz densa de corr)
  - break-even e economia agregada da amortizacao (§6.2), a partir do custo COMPLETO de
    selecao (ve acima) — so quando os dados sao gerados do zero (--data-dir); um --raw
    antigo sem linhas experiment=='amortization' pula essa secao com um aviso

USO
---
    python analyze_cost.py --data-dir ./data/Originais --out-dir ./cost
    python analyze_cost.py --raw ./cost/results_raw.csv --out-dir ./cost   # analisa arquivo existente

ATENCAO (memoria): S2 (correlacao sobre TODAS as features) materializa uma matriz densa
p x p (float64). Para o MH100K (p=24.833) sao ~4,9 GB. Use --datasets para excluir o MH100K
em maquinas com pouca RAM; e exatamente esse custo que a ordem v1 (variancia primeiro) evita.

ATENCAO (tempo): amortization_experiment() roda o pipeline de selecao completo (chi2+MI+RF)
por fold — mais lento que os estagios S0-S3. Em datasets grandes (ex. MH100K) pode demorar
bastante; use --datasets para escopar um subconjunto se necessario.
"""

from __future__ import annotations

import argparse
import contextlib
import os
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
    CORR_SUBSAMPLE_ROWS,
    CORRELATION_THRESHOLD,
    N_ESTIMATORS,
    N_FOLDS,
    RF_PARAMS,
    SEED,
    VARIANCE_THRESHOLD,
    warn_if_nondefault,
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
    subamostrando linhas acima de CORR_SUBSAMPLE_ROWS, removendo colunas cuja maior
    correlacao absoluta com uma coluna anterior excede CORRELATION_THRESHOLD."""
    if idx.size < 2:
        return idx
    sub = X[:, idx]
    if sub.shape[0] > CORR_SUBSAMPLE_ROWS:
        rows = np.random.default_rng(SEED).choice(sub.shape[0], CORR_SUBSAMPLE_ROWS,
                                                    replace=False)
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


def amortization_experiment(name: str, path: Path, label_col: str, n_folds: int,
                            correlation_threshold: float) -> list[dict]:
    """Custo COMPLETO de selecao (variancia + correlacao + chi2 + MI + RF + voto), via
    main.MalwareFeatureSelector.auto_select() — o mesmo pipeline que main.py roda —, e o
    fit_seconds do RF sobre o full set vs. o subconjunto selecionado. Essas duas medidas sao
    o selection_seconds/fit_full_s/fit_reduced_s de que amortization_table()/
    amortization_savings() precisam para o break-even do §6.2 (distinto de stage_experiment,
    que so mede variancia+correlacao — 2,4% do custo one-off segundo o proprio artigo)."""
    from main import MalwareFeatureSelector  # repo root already on sys.path (see top of file)

    df = pd.read_csv(path, low_memory=False)
    if label_col not in df.columns:
        raise KeyError(f"coluna '{label_col}' ausente. Colunas: {list(df.columns)[:8]}...")
    X_df = df.drop(columns=[label_col]).select_dtypes(include=[np.number])
    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = (df[label_col].to_numpy() > 0).astype(np.int8)
    n_total = X_df.shape[1]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    rows: list[dict] = []
    for fold, (tr, _te) in enumerate(skf.split(X_df.to_numpy(), y)):
        Xtr_df = X_df.iloc[tr].reset_index(drop=True)
        ytr = y[tr]
        train_df = Xtr_df.copy()
        train_df["__label__"] = ytr

        t0 = time.perf_counter()
        selector = MalwareFeatureSelector(train_df, target_col="__label__",
                                          variance_threshold=VARIANCE_THRESHOLD)
        with open(os.devnull, "w") as fnull, contextlib.redirect_stdout(fnull):
            selected = selector.auto_select(target_features=None,
                                            correlation_threshold=correlation_threshold)
        selection_s = time.perf_counter() - t0

        selected = [c for c in selected if c in Xtr_df.columns]
        if not selected:
            selected = list(Xtr_df.columns[:1])

        Xtr_arr = Xtr_df.to_numpy(dtype=np.float32)
        fit_full_s = _fit_seconds(Xtr_arr, ytr, np.arange(n_total))
        sel_idx = np.array([Xtr_df.columns.get_loc(c) for c in selected])
        fit_reduced_s = _fit_seconds(Xtr_arr, ytr, sel_idx)

        rows.append(dict(
            experiment="amortization", dataset=name, fold=fold,
            n_total=n_total, n_selected=len(selected),
            selection_seconds=selection_s,
            fit_full_seconds=fit_full_s, fit_reduced_seconds=fit_reduced_s,
        ))
        print(f"    [amortization] fold={fold}  selected={len(selected)}  "
             f"selection_s={selection_s:.2f}  fit_full={fit_full_s:.3f}  "
             f"fit_reduced={fit_reduced_s:.3f}")
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
# Amortizacao / break-even (§6.2) — custo COMPLETO de selecao (nao so S0-S3)
# ---------------------------------------------------------------------------

def load_amortization(raw: pd.DataFrame) -> pd.DataFrame | None:
    df = raw[raw["experiment"] == "amortization"].copy()
    return df if not df.empty else None


def amortization_table(df: pd.DataFrame) -> pd.DataFrame:
    """Por dataset: custo one-off completo de selecao (selection_s), custo de treino no
    full set vs. no reduzido, e o break-even (numero de ciclos de retreino em que a economia
    de treino paga a selecao) — artigo: mediana 7,7, de 1,4 (DefenseDroid PRS) a 72,6 (Drebin-215)."""
    g = (df.groupby("dataset")
         .agg(n_total=("n_total", "first"), n_selected=("n_selected", "mean"),
              selection_s=("selection_seconds", "mean"),
              fit_full_s=("fit_full_seconds", "mean"),
              fit_reduced_s=("fit_reduced_seconds", "mean"))
         .reset_index())
    g["train_speedup_x"] = g.fit_full_s / g.fit_reduced_s
    g["breakeven_cycles"] = g.selection_s / (g.fit_full_s - g.fit_reduced_s)
    return g.sort_values("breakeven_cycles")


def amortization_savings(g: pd.DataFrame, n_updates: tuple[int, ...] = (1, 10, 50)) -> pd.DataFrame:
    """Custo agregado (soma sobre todos os datasets de g) com vs. sem o pipeline, para N
    atualizacoes de modelo — artigo (agregado sobre os dez datasets do benchmark): uma unica
    rodada custa 6,2x mais com o pipeline; 10 atualizacoes custam 27,1% menos; 50 custam
    75,6% menos."""
    total_selection = g.selection_s.sum()
    total_fit_full = g.fit_full_s.sum()
    total_fit_reduced = g.fit_reduced_s.sum()
    rows = []
    for n in n_updates:
        cost_with = total_selection + n * total_fit_reduced
        cost_without = n * total_fit_full
        rows.append(dict(
            n_updates=n,
            cost_with_pipeline_s=cost_with,
            cost_without_pipeline_s=cost_without,
            ratio_x=cost_with / cost_without,
            savings_pct=100 * (1 - cost_with / cost_without),
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------

def main() -> int:
    global SEED, N_FOLDS, VARIANCE_THRESHOLD, CORRELATION_THRESHOLD, RF_PARAMS, \
        CORR_SUBSAMPLE_ROWS
    default_seed, default_n_folds = SEED, N_FOLDS
    default_variance, default_correlation = VARIANCE_THRESHOLD, CORRELATION_THRESHOLD
    default_n_estimators, default_subsample = N_ESTIMATORS, CORR_SUBSAMPLE_ROWS

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
    ap.add_argument("--seed", type=int, default=default_seed)
    ap.add_argument("--n-folds", type=int, default=default_n_folds)
    ap.add_argument("--variance-threshold", type=float, default=default_variance)
    ap.add_argument("--correlation-threshold", type=float, default=default_correlation)
    ap.add_argument("--n-estimators", type=int, default=default_n_estimators)
    ap.add_argument("--corr-subsample-rows", type=int, default=default_subsample,
                    help="linhas usadas para ESTIMAR a matriz de correlacao em datasets "
                         "grandes (default: 20000); so tem efeito ao gerar do zero "
                         "(--data-dir), nao ao analisar --raw ja existente")
    ap.add_argument("--skip-amortization", action="store_true",
                    help="pula a medicao do custo COMPLETO de selecao (chi2+MI+RF) usada "
                         "no break-even/amortizacao do §6.2 — roda so os estagios S0-S3 "
                         "(mais rapido, mas sem a analise de break-even)")
    args = ap.parse_args()

    warn_if_nondefault(
        seed=(args.seed, default_seed), n_folds=(args.n_folds, default_n_folds),
        variance_threshold=(args.variance_threshold, default_variance),
        correlation_threshold=(args.correlation_threshold, default_correlation),
        n_estimators=(args.n_estimators, default_n_estimators),
        corr_subsample_rows=(args.corr_subsample_rows, default_subsample),
    )
    SEED = args.seed
    N_FOLDS = args.n_folds
    VARIANCE_THRESHOLD = args.variance_threshold
    CORRELATION_THRESHOLD = args.correlation_threshold
    CORR_SUBSAMPLE_ROWS = args.corr_subsample_rows
    RF_PARAMS = dict(n_estimators=args.n_estimators, n_jobs=-1, random_state=SEED)

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
            if not args.skip_amortization:
                try:
                    rows += amortization_experiment(p.stem, p, args.label_col, args.n_folds,
                                                     args.correlation_threshold)
                except Exception as exc:                   # noqa: BLE001
                    print(f"    [ERRO amortization] {p.stem}: {type(exc).__name__}: {exc}")
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

    # 3) amortizacao / break-even (§6.2) — so quando ha dados (gerados com --data-dir e
    #    sem --skip-amortization; um --raw antigo sem essas linhas so pula esta secao)
    amort_df = load_amortization(raw)
    if amort_df is not None:
        amort = amortization_table(amort_df)
        savings = amortization_savings(amort)

        amort.to_csv(args.out_dir / "amortization_por_dataset.csv", index=False)
        savings.to_csv(args.out_dir / "amortization_economia.csv", index=False)

        print("\n" + "=" * 78)
        print("AMORTIZACAO / BREAK-EVEN (§6.2) — custo COMPLETO de selecao (chi2+MI+RF)")
        print("=" * 78)
        print(amort.round(4).to_string(index=False))
        print(f"\nbreak-even (mediana entre datasets): {amort.breakeven_cycles.median():.1f} "
             f"ciclos de retreino  (artigo: 7,7)")

        print("\n" + "-" * 78)
        print("ECONOMIA AGREGADA (soma sobre todos os datasets acima)")
        print("-" * 78)
        print(savings.round(4).to_string(index=False))
    else:
        print("\n⚠  Sem linhas experiment=='amortization' no --raw usado — pulei a analise de "
             "break-even/amortizacao do §6.2. Gere de novo com --data-dir (sem "
             "--skip-amortization) para incluir essa secao.")

    print(f"\nArquivos em {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())