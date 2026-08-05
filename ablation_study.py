#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ablation_simple.py — versão enxuta do ablation study.

POR QUE ESTE SCRIPT EXISTE
--------------------------
O ablation_study.py tem 820 linhas, três blocos de experimento e flags que
precisam ser combinadas corretamente (--k-mode adaptive --mi-estimator
continuous) para reproduzir o método publicado. Errar uma delas invalida a
rodada inteira, e foi o que aconteceu.

Aqui não há flags de configuração: os valores que reproduzem main.py estão
HARDCODED. Se você não passa nada, sai o certo.

A OTIMIZAÇÃO QUE DEIXA ISSO RÁPIDO
----------------------------------
Os três rankings (chi2, MI, RF) são calculados UMA VEZ por fold. Todas as 16
configurações do estudo saem depois por operação de conjunto sobre essas três
listas — nenhum ranker é recalculado. Isso reduz o trabalho pesado de
~16 x 5 x 10 execuções de ranker para 3 x 5 x 10.

Custo dominante passa a ser o RF de avaliação, que roda sobre conjuntos de
12 a 100 features. Estimativa: 10 a 20 minutos para os 10 datasets.

O QUE ELE MEDE
--------------
  Bloco A - contribuição de cada critério (7 configurações, leave-one-out)
  Bloco B - efeito do limiar theta e do top-k (9 configurações)
  Extra   - sobreposição Jaccard entre os rankers e estabilidade (Kuncheva)

O bloco de estágios do pipeline (variância / correlação) NÃO está aqui de
propósito: aquelas configurações não dependem de k, então os resultados que
você já tem continuam válidos e não precisam ser refeitos.

USO
---
    python ablation_simple.py
    python ablation_simple.py --data-dir ./data --out-dir ./ablation_v2
    python ablation_simple.py --datasets adroit,drebin215     # teste rápido
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
# CONSTANTES — espelham main.py. Não há flag para mudá-las, de propósito.
# ---------------------------------------------------------------------------
VARIANCE_THRESHOLD = 0.01     # main.py: variance_filter(threshold=0.01)
CORRELATION_THRESHOLD = 0.95  # main.py: correlation_filter(threshold=0.95)
N_FOLDS = 5
SEED = 42
RF_PARAMS = dict(n_estimators=100, n_jobs=-1, random_state=SEED)

# Contagens publicadas em summary.csv, usadas como auto-verificação.
EXPECTED = {
    "adroit": 15, "android_permissions": 13, "androcrawl": 12, "drebin215": 22,
    "kronodroid_real_device": 29, "kronodroid_emulator": 25, "defensedroid_prs": 92,
    "defensedroid_apicalls_closeness": 81, "defensedroid_apicalls_degree": 75,
    "defensedroid_apicalls_katz": 75,
}


def adaptive_k(n_original: int) -> int:
    """main.py: target_features = min(100, max(10, len(original_features) // 10))"""
    return min(100, max(10, n_original // 10))


# ---------------------------------------------------------------------------
# Estágio 1 — limpeza (espelha main.py)
# ---------------------------------------------------------------------------

def clean(X: np.ndarray) -> np.ndarray:
    """Variância e depois correlação. Retorna índices das colunas mantidas."""
    idx = np.arange(X.shape[1])

    idx = idx[X[:, idx].var(axis=0) > VARIANCE_THRESHOLD]
    if idx.size < 2:
        return idx

    # matriz de correlação densa, como o DataFrame.corr() de main.py
    sub = X[:, idx]
    if sub.shape[0] > 20000:                      # subamostra só para estimar r
        rows = np.random.default_rng(SEED).choice(sub.shape[0], 20000, replace=False)
        sub = sub[rows]
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.abs(np.corrcoef(sub, rowvar=False))
    corr = np.nan_to_num(corr, nan=0.0)
    upper = np.triu(corr, k=1)
    drop = np.any(upper > CORRELATION_THRESHOLD, axis=0)
    return idx[~drop]


# ---------------------------------------------------------------------------
# Estágio 2 — os três rankings, calculados UMA VEZ por fold
# ---------------------------------------------------------------------------

def rank_all(X: np.ndarray, y: np.ndarray, idx: np.ndarray) -> dict[str, np.ndarray]:
    """Retorna, para cada critério, os índices ordenados do melhor para o pior."""
    sub = X[:, idx]

    c, _ = chi2(np.clip(sub, 0, None), y)
    c = np.nan_to_num(c, nan=-np.inf)

    # sem discrete_features, exatamente como main.py o chama
    m = mutual_info_classif(sub, y, random_state=SEED)

    rf = RandomForestClassifier(**RF_PARAMS).fit(sub, y)

    return {
        "chi2": idx[np.argsort(c)[::-1]],
        "mi": idx[np.argsort(m)[::-1]],
        "rf": idx[np.argsort(rf.feature_importances_)[::-1]],
    }


def vote(rankings: dict[str, np.ndarray], criteria: tuple[str, ...],
         k: int, theta: int) -> np.ndarray:
    """Features que aparecem no top-k de pelo menos theta dos critérios."""
    counts: dict[int, int] = {}
    for name in criteria:
        for f in rankings[name][:k]:
            counts[int(f)] = counts.get(int(f), 0) + 1
    sel = sorted(f for f, n in counts.items() if n >= min(theta, len(criteria)))
    return np.asarray(sel, dtype=int)


# ---------------------------------------------------------------------------
# As 16 configurações do estudo
# ---------------------------------------------------------------------------

def build_configs(k_ad: int) -> list[dict]:
    C = ("chi2", "mi", "rf")
    cfgs = [
        # Bloco A — contribuição de cada critério (k adaptativo, theta=2)
        dict(bloco="A_criterio", nome="C1 chi2",            crit=("chi2",),      k=k_ad, theta=1),
        dict(bloco="A_criterio", nome="C2 MI",              crit=("mi",),        k=k_ad, theta=1),
        dict(bloco="A_criterio", nome="C3 RF",              crit=("rf",),        k=k_ad, theta=1),
        dict(bloco="A_criterio", nome="C4 chi2+MI  (-RF)",  crit=("chi2", "mi"), k=k_ad, theta=2),
        dict(bloco="A_criterio", nome="C5 chi2+RF  (-MI)",  crit=("chi2", "rf"), k=k_ad, theta=2),
        dict(bloco="A_criterio", nome="C6 MI+RF  (-chi2)",  crit=("mi", "rf"),   k=k_ad, theta=2),
        dict(bloco="A_criterio", nome="C7 os tres (metodo)", crit=C,             k=k_ad, theta=2),
    ]
    # Bloco B — limiar theta x top-k
    for k, lab in [(k_ad, "adapt"), (50, "50"), (100, "100")]:
        for theta in (1, 2, 3):
            cfgs.append(dict(bloco="B_limiar", nome=f"k={lab}, theta={theta}",
                             crit=C, k=k, theta=theta))
    return cfgs


# ---------------------------------------------------------------------------
# Métricas auxiliares
# ---------------------------------------------------------------------------

def jaccard(a: set, b: set) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def kuncheva(sets: list[set], total: int) -> float:
    """Estabilidade entre os subconjuntos dos folds, corrigida pelo acaso."""
    vals = []
    for A, B in itertools.combinations(sets, 2):
        k = (len(A) + len(B)) / 2
        if k <= 0 or k >= total:
            continue
        vals.append((len(A & B) * total - k * k) / (k * (total - k)))
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------

def load(path: Path, label_col: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, low_memory=False)
    if label_col not in df.columns:
        raise KeyError(f"coluna '{label_col}' ausente. Colunas: {list(df.columns)[:8]}...")
    y = df[label_col].to_numpy()
    X = df.drop(columns=[label_col]).select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X.to_numpy(dtype=np.float32), (y > 0).astype(np.int8)


def run_dataset(name: str, path: Path, label_col: str) -> tuple[list[dict], list[dict]]:
    X, y = load(path, label_col)
    n_total = X.shape[1]
    k_ad = adaptive_k(n_total)
    print(f"\n=== {name} | {X.shape[0]:,} amostras x {n_total:,} features | k adaptativo = {k_ad}")

    configs = build_configs(k_ad)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    rows, ovl = [], []
    picked: dict[str, list[set]] = {c["nome"]: [] for c in configs}

    for fold, (tr, te) in enumerate(skf.split(X, y)):
        Xtr, ytr = X[tr], y[tr]

        t0 = time.perf_counter()
        idx = clean(Xtr)
        t_clean = time.perf_counter() - t0

        t0 = time.perf_counter()
        rankings = rank_all(Xtr, ytr, idx)          # <<< uma vez por fold
        t_rank = time.perf_counter() - t0

        # sobreposição entre os rankers, no k adaptativo
        kk = min(k_ad, idx.size)
        tops = {c: set(map(int, r[:kk])) for c, r in rankings.items()}
        for a, b in itertools.combinations(("chi2", "mi", "rf"), 2):
            ovl.append(dict(dataset=name, fold=fold, par=f"{a}-{b}",
                            jaccard=jaccard(tops[a], tops[b]), k=kk))

        for cfg in configs:
            sel = vote(rankings, cfg["crit"], min(cfg["k"], idx.size), cfg["theta"])
            if sel.size == 0:
                sel = idx[:1]
            picked[cfg["nome"]].append(set(map(int, sel)))

            t0 = time.perf_counter()
            clf = RandomForestClassifier(**RF_PARAMS).fit(Xtr[:, sel], ytr)
            t_fit = time.perf_counter() - t0
            pred = clf.predict(X[te][:, sel])

            rows.append(dict(
                dataset=name, bloco=cfg["bloco"], config=cfg["nome"], fold=fold,
                n_features=int(sel.size), n_total=n_total,
                reducao_pct=100 * (1 - sel.size / n_total),
                recall=recall_score(y[te], pred, zero_division=0),
                f1=f1_score(y[te], pred, zero_division=0),
                seg_limpeza=t_clean, seg_ranking=t_rank, seg_treino=t_fit,
            ))

    # estabilidade entre folds
    for cfg in configs:
        ki = kuncheva(picked[cfg["nome"]], n_total)
        for r in rows:
            if r["dataset"] == name and r["config"] == cfg["nome"]:
                r["kuncheva"] = ki

    # auto-verificação contra summary.csv
    metodo = [r["n_features"] for r in rows
              if r["dataset"] == name and r["config"].startswith("C7")]
    obtido = float(np.mean(metodo))
    esperado = EXPECTED.get(name)
    if esperado is not None:
        ok = abs(obtido - esperado) <= 3
        print(f"    verificacao: C7 selecionou {obtido:.1f} features | "
              f"summary.csv diz {esperado} | {'OK' if ok else 'DIVERGE >3'}")
    else:
        print(f"    verificacao: C7 selecionou {obtido:.1f} features (sem referencia)")

    return rows, ovl


def resumo(df: pd.DataFrame, ovl: pd.DataFrame, out: Path) -> None:
    g = (df.groupby(["bloco", "config"])
           .agg(feat=("n_features", "mean"), red=("reducao_pct", "mean"),
                recall=("recall", "mean"), f1=("f1", "mean"),
                kuncheva=("kuncheva", "mean"))
           .round(4))
    g["tradeoff"] = (g.recall * g.red / 100).round(4)
    g = g[["feat", "red", "recall", "f1", "tradeoff", "kuncheva"]]

    print("\n" + "=" * 78)
    print("BLOCO A — contribuicao de cada criterio")
    print("=" * 78)
    print(g.loc["A_criterio"].sort_values("tradeoff", ascending=False).to_string())

    print("\n" + "=" * 78)
    print("BLOCO B — limiar theta x top-k")
    print("=" * 78)
    print(g.loc["B_limiar"].sort_values("tradeoff", ascending=False).to_string())

    print("\n" + "=" * 78)
    print("SOBREPOSICAO ENTRE OS RANKERS (Jaccard)")
    print("=" * 78)
    o = ovl.copy()
    o["tipo"] = np.where(o.dataset.str.contains("apicalls|prs"), "API/continua", "permissoes")
    # adroit e android_permissions saturam quando k >= features pos-limpeza
    print(o.pivot_table(index="tipo", columns="par", values="jaccard").round(3).to_string())
    print("\npor dataset:")
    print(o.pivot_table(index="dataset", columns="par", values="jaccard").round(3).to_string())

    g.to_csv(out / "resumo.csv")
    print(f"\nArquivos em {out.resolve()}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=Path("ablation_v2"))
    ap.add_argument("--datasets", type=str, default="")
    ap.add_argument("--label-col", type=str, default="class")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    wanted = [d.strip() for d in args.datasets.split(",") if d.strip()]
    paths = ([p for n in wanted for p in sorted(args.data_dir.glob(f"{n}*.csv"))]
             if wanted else sorted(args.data_dir.glob("*.csv")))
    if not paths:
        print(f"nenhum CSV em {args.data_dir}")
        return 1

    t_start = time.perf_counter()
    rows, ovl = [], []
    for p in paths:
        try:
            r, o = run_dataset(p.stem, p, args.label_col)
            rows += r
            ovl += o
        except Exception as exc:                    # noqa: BLE001
            print(f"    [ERRO] {p.stem}: {type(exc).__name__}: {exc}")

    if not rows:
        print("nenhum resultado")
        return 1

    df = pd.DataFrame(rows)
    ov = pd.DataFrame(ovl)
    df.to_csv(args.out_dir / "resultados.csv", index=False)
    ov.to_csv(args.out_dir / "sobreposicao.csv", index=False)
    resumo(df, ov, args.out_dir)
    print(f"tempo total: {(time.perf_counter() - t_start) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())