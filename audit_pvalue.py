#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_pvalue.py — auditoria da grade de p-valores do teste de Wilcoxon pareado.

POR QUE ESTE SCRIPT EXISTE
--------------------------
pipeline.py e pipeline_multiclf.py comparam modelo original x modelo reduzido
com um teste de Wilcoxon signed-rank pareado sobre os folds de
StratifiedKFold(n_splits=5) (ver pipeline.py:13, pipeline_multiclf.py:50/102).

Com n=5 pares e nenhum empate/diferenca zero, o teste exato tem apenas 2^5=32
atribuicoes de sinal igualmente prováveis sob H0. O p-valor bicaudal mais
extremo possível é 2 * (1/2^n) = 2/32 = 0.0625 — MAIOR que o limiar usual de
0.05. Ou seja: com este desenho experimental (5 folds), nenhum resultado pode
ser relatado como significativo a p<0.05, independentemente do tamanho do
efeito. Empates/diferencas zero (comuns quando dois modelos acertam o mesmo
fold) reduzem ainda mais o n efetivo, pioram o problema.

Este script NAO precisa do per_fold_metrics.csv para provar o problema — a
enumeracao da grade teorica (n x p-valor minimo) ja e suficiente. Se o arquivo
existir (gerado por pipeline_multiclf.py), o script tambem confere os
p-valores relatados contra o minimo teorico para o n_pairs de cada linha.

USO
---
    python audit_pvalue.py --out-dir ./pvalue
    python audit_pvalue.py --out-dir ./pvalue --per-fold ./multiclf/per_fold_metrics.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

N_FOLDS_USADO_NO_REPO = 5
ALPHA = 0.05


def min_pvalue_grid(n_max: int = 15) -> pd.DataFrame:
    """Para cada n de pares (sem empates/zeros), o menor p-valor bicaudal
    exato possível no teste de Wilcoxon, obtido empiricamente via scipy
    (caso maximamente extremo: todas as diferencas com o mesmo sinal)."""
    rows = []
    for n in range(2, n_max + 1):
        a = np.arange(1, n + 1, dtype=float)
        b = np.zeros(n)
        res = wilcoxon(a, b, zero_method="wilcox", method="exact")
        teorico = 2 * (0.5 ** n)
        rows.append(dict(n_pares=n, p_minimo_exato=res.pvalue,
                          p_minimo_teorico=teorico,
                          pode_atingir_0_05=res.pvalue < ALPHA))
    return pd.DataFrame(rows)


def audit_reported(per_fold_path: Path) -> pd.DataFrame | None:
    """Confere p-valores ja relatados (wilcoxon_by_classifier.csv ou
    equivalente) contra o minimo teorico para o n_pairs de cada linha."""
    if not per_fold_path.exists():
        return None
    df = pd.read_csv(per_fold_path)
    if "pvalue" not in df.columns or "n_pairs" not in df.columns:
        print(f"[aviso] {per_fold_path} nao tem colunas 'pvalue'/'n_pairs'; pulando auditoria de valores relatados.")
        return None
    df = df.copy()
    df["p_minimo_possivel"] = 2 * (0.5 ** df["n_pairs"])
    df["abaixo_do_minimo_teorico"] = df["pvalue"] < df["p_minimo_possivel"] - 1e-12
    df["significativo_mas_n_insuficiente"] = (df["pvalue"] < ALPHA) & (df["p_minimo_possivel"] >= ALPHA)
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path("pvalue"))
    ap.add_argument("--per-fold", type=Path, default=None,
                     help="wilcoxon_by_classifier.csv ou similar, com colunas pvalue e n_pairs (opcional)")
    ap.add_argument("--n-max", type=int, default=15)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    grid = min_pvalue_grid(args.n_max)
    grid.to_csv(args.out_dir / "grade_pvalor_minimo.csv", index=False)

    print("=" * 78)
    print("GRADE DE P-VALOR MINIMO EXATO (Wilcoxon pareado, sem empates/zeros)")
    print("=" * 78)
    print(grid.round(6).to_string(index=False))

    linha_repo = grid[grid.n_pares == N_FOLDS_USADO_NO_REPO].iloc[0]
    print("\n" + "=" * 78)
    print(f"DESENHO USADO NO REPO: n_splits={N_FOLDS_USADO_NO_REPO} "
          f"(pipeline.py, pipeline_multiclf.py, ablation_study.py)")
    print("=" * 78)
    print(f"p-valor minimo possivel com {N_FOLDS_USADO_NO_REPO} folds: "
          f"{linha_repo.p_minimo_exato:.4f}")
    if linha_repo.p_minimo_exato >= ALPHA:
        minimo_n = int(grid.loc[grid.pode_atingir_0_05, "n_pares"].min())
        print(f"=> NENHUM resultado pode ser p<{ALPHA} com {N_FOLDS_USADO_NO_REPO} folds, "
              f"seja qual for o efeito observado.")
        print(f"=> Seriam necessarios pelo menos {minimo_n} folds para que p<{ALPHA} "
              f"seja sequer matematicamente atingivel.")
        print(f"=> Empates/diferencas zero entre os folds reduzem o n efetivo e pioram o problema.")
    else:
        print(f"=> p<{ALPHA} e atingivel com {N_FOLDS_USADO_NO_REPO} folds no caso mais extremo.")

    if args.per_fold is not None:
        print("\n" + "=" * 78)
        print(f"AUDITORIA DOS P-VALORES RELATADOS EM {args.per_fold}")
        print("=" * 78)
        reported = audit_reported(args.per_fold)
        if reported is None:
            print("arquivo ausente ou sem colunas esperadas — pulado.")
        else:
            reported.to_csv(args.out_dir / "auditoria_pvalores_relatados.csv", index=False)
            print(reported.round(6).to_string(index=False))
            n_suspeitos = int(reported["significativo_mas_n_insuficiente"].sum())
            n_impossiveis = int(reported["abaixo_do_minimo_teorico"].sum())
            print(f"\n{n_suspeitos} linha(s): p<{ALPHA} relatado mas n_pairs insuficiente para isso ser possivel.")
            print(f"{n_impossiveis} linha(s): p relatado abaixo do minimo teorico exato (erro de calculo?).")
    else:
        print(f"\n(nenhum --per-fold informado; auditoria da grade teorica e suficiente para a questao — ver docstring)")

    print(f"\nArquivos em {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
