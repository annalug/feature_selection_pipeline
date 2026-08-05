#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_filter_order.py — mede tempo de parede e pico de memória das duas
ordens possíveis para os filtros de limpeza do pipeline, respondendo ao
revisor que apontou que nenhum script do repositório mede a alegação de
"variância primeiro reduz o tempo de 1.957 s para 400 s".

  v1: filtro de variância -> filtro de correlação   (ordem usada no paper)
  v2: filtro de correlação -> filtro de variância    (ordem alternativa)

Para cada (dataset, ordem), roda um subprocesso isolado (para que o pico de
memória de uma medição não vaze para a próxima) que carrega os dados, aplica
os dois estágios na ordem pedida e reporta, por estágio: tempo de parede,
pico de RSS do processo (resource.getrusage, KB -> registrado em MB) e número
de features após o estágio.

DataFrame.corr()/a matriz de correlação densa é o estágio caro em memória:
para p features em float64 são p^2 * 8 bytes. Se v2 estourar memória em algum
dataset, isso é um resultado válido — o subprocesso roda com um teto de
memória virtual (--mem-limit-gb) para falhar de forma controlada em vez de
travar a máquina, e a falha é registrada em vez de contornada.

Reaproveita CORRELATION_THRESHOLD/VARIANCE_THRESHOLD/SEED de ablation_study.py
(sem modificá-lo). A versão atual de ablation_study.py funde variância e
correlação numa única função (clean()), sem expor as duas etapas separadas
nem nomes de colunas — o que este benchmark precisa para poder testar as duas
ordens. Por isso variance_filter/correlation_filter/load_dataset são
reimplementadas aqui, decompostas a partir da mesma lógica de clean(), em vez
de importadas.

Saídas: filter_order_bench.csv e uma tabela LaTeX.

USO
---
    python bench_filter_order.py --data-dir ./data --out-dir results_v3/T8
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablation_study import CORRELATION_THRESHOLD, SEED, VARIANCE_THRESHOLD


def load_dataset(path: Path, label_col: str | None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Mesma limpeza de ablation_study.load() (NaN/inf -> 0, apenas colunas
    numéricas), mas também devolve os nomes das colunas."""
    label_col = label_col or "class"
    df = pd.read_csv(path, low_memory=False)
    if label_col not in df.columns:
        raise KeyError(f"coluna '{label_col}' ausente. Colunas: {list(df.columns)[:8]}...")
    y = df[label_col].to_numpy()
    Xdf = df.drop(columns=[label_col]).select_dtypes(include=[np.number])
    Xdf = Xdf.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = Xdf.to_numpy(dtype=np.float32)
    return X, (y > 0).astype(np.int8), list(Xdf.columns)


def variance_filter(X: np.ndarray, idx: np.ndarray, thr: float) -> np.ndarray:
    """Mesma regra usada dentro de ablation_study.clean(): var(axis=0) > thr."""
    sub = X[:, idx]
    return idx[sub.var(axis=0) > thr]


def correlation_filter(X: np.ndarray, idx: np.ndarray, thr: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Mesma regra usada dentro de ablation_study.clean(): matriz de
    correlação densa via np.corrcoef (equivalente ao DataFrame.corr() denso
    do main.py), com subamostragem de linhas acima de 20000, removendo
    colunas cuja maior correlação absoluta com uma coluna anterior excede thr."""
    if idx.size < 2:
        return idx
    sub = X[:, idx]
    if sub.shape[0] > 20000:
        rows = rng.choice(sub.shape[0], 20000, replace=False)
        sub = sub[rows]
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.abs(np.corrcoef(sub, rowvar=False))
    corr = np.nan_to_num(corr, nan=0.0)
    upper = np.triu(corr, k=1)
    drop = np.any(upper > thr, axis=0)
    return idx[~drop]


# ---------------------------------------------------------------------------
# modo worker: roda dentro de um subprocesso isolado
# ---------------------------------------------------------------------------

def _run_worker(data_path: str, label_col: str | None, order: str, mem_limit_gb: float) -> None:
    if mem_limit_gb > 0:
        limit_bytes = int(mem_limit_gb * (1024 ** 3))
        try:
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        except (ValueError, OSError) as exc:
            print(json.dumps({"error": f"não foi possível aplicar RLIMIT_AS: {exc}"}))
            return

    t_load0 = time.perf_counter()
    X, y, feat_names = load_dataset(Path(data_path), label_col or None)
    load_s = time.perf_counter() - t_load0
    n_total = X.shape[1]

    rng = np.random.default_rng(SEED)
    stages = ["variance", "correlation"] if order == "v1" else ["correlation", "variance"]

    cur_idx = np.arange(n_total)
    stage_results = []
    for stage in stages:
        t0 = time.perf_counter()
        if stage == "variance":
            cur_idx = variance_filter(X, cur_idx, thr=VARIANCE_THRESHOLD)
        else:
            cur_idx = correlation_filter(X, cur_idx, thr=CORRELATION_THRESHOLD, rng=rng)
        seconds = time.perf_counter() - t0
        peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        stage_results.append(dict(stage=stage, seconds=seconds,
                                  n_features_after=int(cur_idx.size),
                                  peak_rss_mb=peak_rss_kb / 1024.0))

    print(json.dumps(dict(
        order=order, n_total=n_total, load_seconds=load_s,
        stages=stage_results, n_features_final=int(cur_idx.size),
    )))


# ---------------------------------------------------------------------------
# processo pai: orquestra um subprocesso por (dataset, ordem)
# ---------------------------------------------------------------------------

def run_one(script: Path, data_path: Path, label_col: str | None, order: str,
           mem_limit_gb: float, timeout_s: float) -> dict:
    cmd = [sys.executable, str(script), "--worker",
          "--data-path", str(data_path), "--order", order,
          "--mem-limit-gb", str(mem_limit_gb)]
    if label_col:
        cmd += ["--label-col", label_col]

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return dict(order=order, status="timeout", wall_seconds=timeout_s)
    wall_s = time.perf_counter() - t0

    if proc.returncode != 0:
        return dict(order=order, status="failed", wall_seconds=wall_s,
                    returncode=proc.returncode,
                    error=(proc.stderr or "").strip()[-2000:])

    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        return dict(order=order, status="unparseable_output", wall_seconds=wall_s,
                    error=f"{exc}; stdout={proc.stdout[-2000:]}")

    if "error" in payload:
        return dict(order=order, status="worker_error", wall_seconds=wall_s,
                    error=payload["error"])

    payload["status"] = "ok"
    payload["wall_seconds"] = wall_s
    return payload


def flatten_rows(dataset: str, result: dict) -> list[dict]:
    rows = []
    if result.get("status") != "ok":
        rows.append(dict(dataset=dataset, order=result.get("order"), stage="ALL",
                         status=result.get("status"), seconds=result.get("wall_seconds"),
                         n_features_after=None, peak_rss_mb=None,
                         error=result.get("error")))
        return rows

    for s in result["stages"]:
        rows.append(dict(dataset=dataset, order=result["order"], stage=s["stage"],
                         status="ok", seconds=s["seconds"],
                         n_features_after=s["n_features_after"],
                         peak_rss_mb=s["peak_rss_mb"], error=None))
    rows.append(dict(dataset=dataset, order=result["order"], stage="TOTAL",
                     status="ok", seconds=sum(s["seconds"] for s in result["stages"]),
                     n_features_after=result["n_features_final"],
                     peak_rss_mb=max(s["peak_rss_mb"] for s in result["stages"]),
                     error=None))
    return rows


def latex_table(bench: pd.DataFrame) -> str:
    def esc(s):
        return str(s).replace("_", "\\_")

    totals = bench[bench["stage"] == "TOTAL"]
    rows = "\n".join(
        "{:<28} & {:<12} & {} & {:8.3f} & {} & {} \\\\".format(
            esc(r.dataset), r.order,
            r.status,
            r.seconds if pd.notna(r.seconds) else float("nan"),
            f"{r.peak_rss_mb:.1f}" if pd.notna(r.peak_rss_mb) else "--",
            str(int(r.n_features_after)) if pd.notna(r.n_features_after) else "--")
        for r in totals.itertuples()
    )
    return r"""
\begin{table}[ht]
\centering
\caption{Wall-clock time and peak RSS memory for the two cleaning-filter orderings:
v1 (variance $\to$ correlation, used in the paper) vs.\ v2 (correlation $\to$ variance).}
\label{tab:filter-order-bench}
\small
\begin{tabular}{llllrl}
\hline
\textbf{Dataset} & \textbf{Order} & \textbf{Status} & \textbf{Time (s)} & \textbf{Peak RSS (MB)}
 & \textbf{\#Feat.\ final} \\
\hline
""" + rows + r"""
\hline
\end{tabular}
\end{table}
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--data-path", type=str, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--order", type=str, choices=["v1", "v2"], default=None, help=argparse.SUPPRESS)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=Path("results_v3/T8"))
    ap.add_argument("--datasets", type=str, default="")
    ap.add_argument("--label-col", type=str, default=None)
    ap.add_argument("--mem-limit-gb", type=float, default=18.0,
                    help="teto de memória virtual do subprocesso worker; 0 desativa")
    ap.add_argument("--timeout-s", type=float, default=1800.0)
    args = ap.parse_args()

    if args.worker:
        _run_worker(args.data_path, args.label_col, args.order, args.mem_limit_gb)
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()

    wanted = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if wanted:
        csv_files = []
        for name in wanted:
            matches = sorted(args.data_dir.glob(f"{name}*.csv"))
            if matches:
                csv_files.append(matches[0])
            else:
                print(f"[aviso] CSV não encontrado para '{name}'", file=sys.stderr)
    else:
        csv_files = sorted(args.data_dir.glob("*.csv"))

    all_rows: list[dict] = []
    for csv_path in csv_files:
        name = csv_path.stem
        print(f"=== {name} ===")
        for order in ("v1", "v2"):
            result = run_one(script, csv_path, args.label_col, order,
                             args.mem_limit_gb, args.timeout_s)
            status = result.get("status")
            print(f"    {order}: status={status} wall={result.get('wall_seconds'):.2f}s"
                 if status else f"    {order}: sem resultado")
            all_rows += flatten_rows(name, result)

    bench = pd.DataFrame(all_rows)
    bench.to_csv(args.out_dir / "filter_order_bench.csv", index=False)
    (args.out_dir / "filter_order_bench.tex").write_text(latex_table(bench), encoding="utf-8")

    print(f"\nArquivos gravados em {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
