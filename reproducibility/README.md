# Reproducibility

This folder is the single entry point for reproducing every empirical result in the paper
*Statistical Ranking: A Voting-Based Ensemble Approach to Feature Selection in Android
Malware Detection*. For each table and figure it lists the exact script and command.

## 1. Environment

The paper's measurements (§5.2) were taken on an Intel Core i7-1165G7 (8 logical cores,
23.2 GB RAM), Ubuntu 24.04.4, Python 3.12.3, with **no GPU** for Protocol P1. Pinned library
versions are in the repo-root [`requirements.txt`](../requirements.txt):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The Random Forest used for ranking (Stage 2) and for P1 classification is fixed:
`n_estimators=100`, `criterion=gini`, `max_depth=None`, `min_samples_split=2`,
`min_samples_leaf=1`, `max_features=sqrt`, `bootstrap=True`, `random_state=42`, `n_jobs=-1`.

### Scientific parameters — flags and defaults

`main.py`, `ablation_study.py`, `analyze_cost.py`, and `bench_filter_order.py` expose the
scientific parameters below as CLI flags. **Every default reproduces the paper's published
configuration exactly** — running with no flags behaves exactly as before this was added. Any
run using a non-default value prints a `⚠ Non-default parameter(s) in use` warning, so a
deviation from the published configuration is always visible in the run's own output, never
silent.

| Flag | Default | Meaning | Scripts |
|---|---|---|---|
| `--seed` | `42` | RNG seed (`random_state`, correlation subsampling) | ablation, cost, bench |
| `--n-folds` | `5` | `StratifiedKFold` splits | ablation, cost |
| `--variance-threshold` | `0.01` | min. variance to keep a feature (Stage 1) | main, ablation, cost, bench |
| `--correlation-threshold` | `0.95` | max. pairwise correlation before dropping a feature (Stage 1) | main, ablation, cost, bench |
| `--n-estimators` | `100` | Random Forest trees (ranking + evaluation) | ablation, cost |
| `--min-votes` | `2` | criteria (of chi², MI, RF) that must agree for a feature to survive the ensemble vote — the ≥2-of-3 consensus rule | main |
| `--corr-subsample-rows` | `20000` | see below | ablation, cost, bench |

**The 20,000-row correlation subsampling.** Estimating the dense p×p correlation matrix
(Stage 1) over the full row count is what makes the memory cost roughly O(p²) regardless of
*n* — tractable for the feature counts in this benchmark, but the row count still matters for
wall-clock time on datasets with tens of thousands of samples. `clean()` (`ablation_study.py`)
and its equivalents in `analyze_cost.py`/`bench_filter_order.py` therefore *estimate* the
correlation matrix from a random subsample of at most 20,000 rows (`np.random.default_rng(SEED)`)
when a dataset has more rows than that — the **variance filter and the chi²/MI/RF rankings
still run over every row**; only the correlation-matrix estimate is subsampled. This affects 5
of the 11 benchmark datasets (those with >20,000 samples): `android_permissions` (26,864),
`kronodroid_emulator` (63,991), `kronodroid_real_device` (78,137), `androcrawl` (96,744), and
`mh100k` (101,934). Because the subsample is randomized, it can in principle change which
features cross the correlation threshold on those five datasets between runs with different
`--seed` values — use `--corr-subsample-rows` (e.g. a value ≥ the dataset's row count) to
disable subsampling entirely and confirm whether a given result is sensitive to it.

## 2. Datasets

Raw datasets come from the group's public repository:
**https://github.com/AILabs4All/datasets-mh30plus** (11 datasets, 141–24,833 features,
three modalities; CSVs versioned with Git LFS). Each CSV has numeric feature columns and a
binary `class` label. Use the full datasets (`Originais/`) for P1 and the ablation.

Protocol P2 compares four reduction methods, so its inputs are the **reduced** CSVs of each:

| Prefix         | Method                       | Produced by                         |
|----------------|------------------------------|-------------------------------------|
| `statistical_` | Statistical Ranking (ours)   | `main.py` in this repo              |
| `rfe_`         | RFE baseline                 | [MH-FSF](https://github.com/SBSegSF24/MH-FSF) [Rocha et al. 2026a] |
| `semidroid_`   | SemiDroid baseline           | [MH-FSF](https://github.com/SBSegSF24/MH-FSF) [Rocha et al. 2026a] |
| `lasso_`       | Lasso baseline               | [MH-FSF](https://github.com/SBSegSF24/MH-FSF) [Rocha et al. 2026a] |

All three baselines were generated with **MH-FSF** (Malware-Hunter Feature Selection
Framework), the evaluation framework cited in the paper's related work (Table 1). Its
`METHODS.md` confirms it implements RFE, LASSO, and SemiDroid among its 17 supported
methods. See MH-FSF's own documentation for how to run it against the `datasets-mh30plus`
originals and produce the `rfe_*`/`semidroid_*`/`lasso_*` reduced CSVs.

## 3. What reproduces what

The `reproduce.sh` driver at the repo root automates the in-repo claims (see below). Individual
commands:

| Paper artifact | Protocol / Section | Script | Command |
|---|---|---|---|
| **Table 3** (recall/F1, orig vs reduced) | P1, §6 | `reproducibility/reproduce_p1.py` | `python reproducibility/reproduce_p1.py --data-dir ./data/Originais --out-dir ./p1_out` |
| **Table 4** + **Figure 2** (TSTR, AE & VAE) | P2, §6.1 | orchestrated by `reproduce.sh` | `./reproduce.sh --p2-only --pin-commit <SHA>` — see [`REPRODUCING_MALDATAGEN.md`](./REPRODUCING_MALDATAGEN.md) |
| **§6.2** computational cost | §6.2 | `analyze_cost.py`, `bench_filter_order.py` | `python bench_filter_order.py --data-dir ./data/Originais --out-dir ./filter_order_bench` |
| **Table 5** (4 classifier families, MH100K) | P1-extended, §6.3 | `reproducibility/reproduce_table5.py` | `python reproducibility/reproduce_table5.py --data-dir ./data/Originais --dataset mh100k --out-dir ./table5` |
| **Tables 6 & 7** (ablation, budget/θ/criteria) | §6.4 | `ablation_study.py` | `python ablation_study.py --data-dir ./data/Originais --out-dir ./ablation_out` |
| Wilcoxon floor note (min p = 0.0625 at n=5) | §5.3, §7 | `audit_pvalue.py` | `python audit_pvalue.py --out-dir ./pvalue` |

> **Note on Table 5.** `pipeline_multiclf.py` is a **complementary robustness check**, not the
> script that produced the published Table 5 numbers: it reselects features independently
> inside every fold and evaluates **five** classifiers (RandomForest, DecisionTree, KNN,
> LogisticRegression, GradientBoosting) — see its own docstring, which frames it explicitly as
> an extra experiment added in response to a reviewer concern about Random-Forest-centrism, not
> as a Table 5 reproduction. `reproducibility/reproduce_table5.py` is the exact published
> protocol instead: a subset selected **once** over the whole MH100K dataset (93 features), not
> per-fold, evaluated across the four classifiers Table 5 actually reports (Decision Tree, k-NN,
> Random Forest, Logistic Regression — confirmed against the paper, §6.3). This is deliberately
> the same selected-once design the paper itself flags as optimistically biased relative to
> Table 3's in-fold numbers (0.7502 vs. 0.8587 recall for Random Forest on MH100K, a 0.109
> gap) — reproducing that bias is the point, not a defect. Pass `--reduced-csv <path>` instead
> of `--data-dir`/`--dataset` to run against an already-reduced CSV (e.g.
> `Maldatagen_additional_metrics/Datasets/statistical_mh100k.csv`) rather than selecting from
> scratch. For the broader (5-classifier, per-fold) robustness check, run
> `python pipeline_multiclf.py --data-dir ./data/Originais --out-dir ./multiclf` instead.

Quick sanity check on two datasets (~2 min):
`python ablation_study.py --datasets adroit,drebin215 --out-dir ./ablation_test`
Each run self-checks against the published feature counts (`... | OK`).

## 4. Protocol P2 (external dependency)

Table 4 and Figure 2 are produced inside **MalDataGen** [Paim et al. 2025], which trains the
**autoencoder** and **variational** generators (`--model_type autoencoder` /
`--model_type variational`) on the reduced sets and runs TSTR. Because MalDataGen is a
separate, actively-developed repository, its version must be pinned. Full instructions,
including the pinned commit and the AE/VAE configs, are in
[`REPRODUCING_MALDATAGEN.md`](./REPRODUCING_MALDATAGEN.md).

**Determinism.** MalDataGen exposes no seed flag, so P2 is not bit-for-bit reproducible;
report hardware and library versions and expect small numerical variation in Table 4 /
Figure 2. Protocols P1 and the ablation are deterministic (`random_state=42`, in-fold
selection).

## 5. Note on statistical testing

Per-dataset significance in P1 uses the paired Wilcoxon signed-rank test over 5 folds, whose
exact two-sided p-value floor at n=5 is 0.0625 > 0.05 — it cannot reject the null regardless
of effect size. `audit_pvalue.py` enumerates this. The paper's inferential claims rest on the
Friedman test across datasets (§6.4), not on the per-dataset Wilcoxon values.
