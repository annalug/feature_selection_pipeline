# 📊 Statistical Ranking — Voting-Based Ensemble Feature Selection for Android Malware Detection

This artifact provides the reference implementation of **Statistical Ranking**, a
multi-perspective consensus feature-selection pipeline for high-dimensional Android malware
datasets, together with the scripts that reproduce the paper's empirical claims (Tables 3–7
and Figure 2). A feature is retained only when it ranks in the top-*k* by **at least two of
three** independent criteria (χ², Mutual Information, and Random Forest importance), under a
per-criterion budget that scales with the original dimensionality.

> **Associated paper:** _Statistical Ranking: A Voting-Based Ensemble Approach to Feature
> Selection in Android Malware Detection_
>
> **Abstract:** High-dimensional Android malware datasets inflate training time and damage
> model generalization. We propose Statistical Ranking, an ensemble feature selection
> framework that enforces multi-perspective consensus: a feature is retained only when ranked
> in the top-*k* by at least two of three independent criteria (χ², Mutual Information, and
> Random Forest importance), under a per-criterion budget that scales with the original
> dimensionality. Across 11 public datasets under stratified 5-fold cross-validation, the
> method removes 89.8%–99.6% of the feature space with a median recall drop of 0.04, retaining
> recall ≥ 0.85 in 9 of 11 environments. An ablation over ranking criteria and voting
> thresholds shows that the ≥ 2-of-3 consensus maximizes the reduction/recall trade-off. On
> MH100K, 24,833 features are reduced to 93 (99.6%) at a cost of 6.5% in F1-score, and the
> one-off cost of selection is recovered after roughly eight retraining cycles.

---

# 📑 README structure

| Section | Description |
| ------- | ----------- |
| **Considered Badges** | Badges submitted for evaluation |
| **Basic information** | Execution environment, hardware, and software |
| **Dependencies** | Libraries, versions, and datasets used |
| **Security concerns** | Risks of running the artifact |
| **Installation** | Step-by-step environment setup |
| **Minimal test** | Quick run to validate the installation |
| **Feature selection on your own data** | Run the selector standalone, outside the paper's claims |
| **`reproduce.sh` flag reference** | Every flag the driver script accepts, one by one |
| **Experiments** | Reproduction of the paper's claims |
| **LICENSE** | Artifact license |

Repository layout:

```
feature_selection_pipeline/
├── main.py                       # Statistical Ranking (MalwareFeatureSelector / BatchFeatureSelector)
│                                  #   CLI: --data-dir --out-dir --target-col --target-features --correlation-threshold
├── ablation_study.py             # ablation: criteria, adaptive budget, vote threshold θ (Tables 6–7)
├── analyze_cost.py               # per-stage computational cost (§6.2)
├── bench_filter_order.py         # time/memory of filter orderings variance→correlation (§6.2)
├── pipeline_multiclf.py          # generalization across classifiers (Table 5)
├── audit_pvalue.py               # Wilcoxon floor audit (n=5 → min p = 0.0625)
├── reproduce.sh                  # unified driver: in-repo claims + standalone feature selection
│                                  #   (--feature-selection --data-folder) + optional P2 (--with-p2)
├── requirements.txt              # Python dependencies (pinned)
├── README.md                     # this file
└── reproducibility/
    ├── README.md                 # detailed index: paper artifact → script → command
    ├── reproduce_p1.py           # Protocol P1, original vs. reduced (Table 3)
    └── REPRODUCING_MALDATAGEN.md # Protocol P2 (TSTR) guide via MalDataGen
```

---

# 🏅 Considered Badges

The badges considered for this artifact are: **Available (SeloD)**, **Reproducible (SeloR)**,
**Functional (SeloF)**, and **Sustainable (SeloS)**.

---

# ⚙️ Basic information

The artifact consists of:

1. **`main.py`** — the Statistical Ranking pipeline (variance filter → correlation filter →
   ensemble vote), applied per dataset via `BatchFeatureSelector`. Runnable standalone over any
   CSV folder (see "Feature selection on your own data" below).
2. **`reproduce.sh`** — a driver that automates the in-repo claims (Tables 3, 5, 6, 7 and §6.2)
   and also exposes `main.py` as a standalone step via `--feature-selection --data-folder`.
3. **`reproducibility/`** — Protocol P1 reproduction (`reproduce_p1.py`) and the Protocol P2
   (TSTR) path via the external **MalDataGen** framework.

### Tested execution environment

- **Operating system:** Ubuntu 24.04.4 (Linux x86-64)
- **Python:** 3.12.3
- **CPU:** Intel Core i7-1165G7 (8 logical cores); **no GPU** for Protocol P1

### Hardware requirements

- **CPU:** 4 cores or more (recommended)
- **RAM:** minimum **8 GB**; **16 GB+** recommended for the larger datasets (`mh100k`, `androcrawl`)
- **Disk:** approximately **5–10 GB** free for the datasets and outputs
- **GPU (optional):** only for the AE/VAE experiments of Protocol P2 (MalDataGen)

> ⏱️ **Expected runtimes:**
> - Minimal test (`reproduce.sh --minimal`): ~2 minutes.
> - In-repo claims (P1, ablation, cost, multi-classifier): minutes to a few hours; the
>   full-dimensional model on `mh100k` (24,833 features) dominates and can take hours.
> - Protocol P2 (`reproduce.sh --with-p2`, external): **several hours per experiment** (2 models
>   × N datasets). A GPU and background execution are recommended.

---

# 📦 Dependencies

Dependencies are listed in `requirements.txt`, pinned to the versions reported in the paper (§5.2):

| Library        | Version   |
| -------------- | --------- |
| `numpy`        | == 2.4.4  |
| `scipy`        | == 1.17.1 |
| `pandas`       | == 3.0.2  |
| `scikit-learn` | == 1.8.0  |
| `matplotlib`   | == 3.10.8 |
| `seaborn`      | == 0.13.2 |

**Datasets** are public and shared by our group at
**https://github.com/AILabs4All/datasets-mh30plus** (original, reduced, and balanced
versions; CSVs versioned with **Git LFS**). The benchmark spans 11 datasets across three
modalities:

| Dataset | Samples | Features (Full) | Benign | Malware |
| ------- | ------- | --------------- | ------ | ------- |
| `adroit` | 11,476 | 166 | 8,058 | 3,418 |
| `android_permissions` | 26,864 | 151 | 9,077 | 17,787 |
| `androcrawl` | 96,744 | 141 | 86,574 | 10,170 |
| `defensedroid_prs` | 11,975 | 2,877 | 5,975 | 6,000 |
| `drebin215` | 15,031 | 215 | 9,476 | 5,555 |
| `kronodroid_real_device` | 78,137 | 286 | 36,755 | 41,382 |
| `kronodroid_emulator` | 63,991 | 276 | 35,246 | 28,745 |
| `defensedroid_apicalls_closeness` | 10,476 | 4,274 | 5,222 | 5,254 |
| `defensedroid_apicalls_degree` | 10,476 | 6,002 | 5,222 | 5,254 |
| `defensedroid_apicalls_katz` | 10,476 | 6,002 | 5,222 | 5,254 |
| `mh100k` | 101,934 | 24,833 | 92,134 | 9,800 |

**Third-party resource access (Protocol P2 only):** the **MalDataGen** framework
[Paim et al. 2025], repository
https://github.com/MalwareDataLab/Maldatagen_additional_metrics (Python 3.10+, TensorFlow).
No keys or credentials are required.

> **Fill in:** the pinned MalDataGen commit (`git rev-parse HEAD` in the clone you used) and
> the provenance of the baseline reductions (RFE, SemiDroid, Lasso) compared under Protocol P2.

---

# 🔒 Security concerns

Running this artifact **poses no risk** to reviewers.

The datasets contain only numerical feature vectors (binary attributes / graph centralities
and the class label), **with no malware binaries, executables, or real malicious samples**.
The in-repo scripts only read CSV files, select features, train models, and compute metrics;
they run locally, require **no root/sudo**, and need **no network access**.

> ⚠️ **Note:** the `--with-p2` path **clones an external GitHub repository** (MalDataGen)
> and **installs dependencies via `pip`**. Run it inside a virtual environment (`venv`) or
> container to isolate installations. MalDataGen's optional Docker execution requires sudo
> access to the Docker engine; local execution does not.

---

# 💾 Installation

### 1. Clone this repository

```bash
git clone https://github.com/annalug/feature_selection_pipeline.git
cd feature_selection_pipeline
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Get the datasets (Git LFS)

```bash
sudo apt-get update && sudo apt-get install -y git-lfs git
git lfs install
git clone https://github.com/AILabs4All/datasets-mh30plus.git ./data
cd ./data && git lfs pull && cd ..
```

> ⚠️ **5 of the 11 datasets ship as `.zip`, not `.csv`.** In `data/Originais/`,
> `defensedroid_apicalls_closeness`, `defensedroid_apicalls_degree`, `defensedroid_apicalls_katz`,
> `defensedroid_prs`, and `mh100k` are `.zip` archives; the other 6 (`adroit`, `androcrawl`,
> `android_permissions`, `drebin215`, `kronodroid_emulator`, `kronodroid_real_device`) are
> already plain `.csv`. Every script here reads `*.csv` directly (`glob("*.csv")`), so extract
> those 5 before pointing `--data-dir` at the folder:
> ```bash
> cd data/Originais
> for z in *.zip; do unzip -o "$z"; done
> cd ../..
> ```
> Verify each extracted file matches its dataset's stem name (e.g. `mh100k.csv`) — rename it if
> the archive unpacks to something else.

Use the original (full) datasets for Protocol P1 and the ablation. For Protocol P2, follow the
additional MalDataGen setup in `reproducibility/REPRODUCING_MALDATAGEN.md`.

---

# ✅ Minimal test

A quick run (~2 minutes) over two datasets that validates the installation and exercises the
core functionality (selection inside folds, ablation self-check, statistical audit):

```bash
chmod +x reproduce.sh
./reproduce.sh --minimal --data-dir ./data/Originais
```

**Expected result:** a reduced Table 3 for `adroit` and `drebin215`; the ablation prints its
self-check against the published feature counts (lines ending in `... | OK`); and the Wilcoxon
audit confirms the 0.0625 floor at n=5. A final summary lists the status of each step.

**Real output** (captured directly from this repo, `adroit` only — `drebin215` is the same
shape):

```
=== adroit | 11,476 amostras x 166 features | k adaptativo = 16
    verificacao: C7 selecionou 15.2 features | summary.csv diz 15 | OK
```

```
=== Protocol P1 — Table 3 ===
dataset  features  orig_recall  delta_recall  red_recall  orig_f1  pvalue_recall  red_f1  reduction_pct  n_selected
 adroit       166       0.7876       -0.0506       0.737   0.8437         0.0625   0.816          90.84          15
```

> 📝 Adjust `--data-dir` to the folder that holds the CSVs (e.g. `Originais/` inside the
> datasets repository) and `--target-col` if the label column is not named `class`.

---

# 🔍 Feature selection on your own data

Independently of reproducing the paper's claims, the ensemble selector itself (χ² + Mutual
Information + Random Forest, ≥ 2-of-3 vote) can be run over **any folder of your own CSV
datasets** via `reproduce.sh --feature-selection`:

```bash
./reproduce.sh --feature-selection --data-folder /path/to/your/datasets
```

- `--data-folder <path>` is **required** with `--feature-selection` — this is the folder
  containing your raw `*.csv` files, and is independent of `--data-dir` (which points to
  `./data/Originais` for the paper's reproduction claims below).
- Each CSV must have a binary target column (default `class`, every other column is treated
  as a binary feature). Use `--out-dir` to change where results go (default
  `reproduction_output/`).
- Without `--feature-selection`, running the script proceeds to the paper's reproduction
  claims instead (see "Experiments" below); the two modes don't mix in one call.
- `--out-dir` changes where results are written, e.g.
  `./reproduce.sh --feature-selection --data-folder ./meus_datasets --out-dir ./my_results`.

**Output**, per dataset, under `<out-dir>/feature_selection/<dataset_name>/`:
`<dataset_name>_reduced.csv` (selected features only), `_selected_features.csv`,
per-criterion `_chi2_scores.csv` / `_mutual_info_scores.csv` / `_rf_importance_scores.csv` /
`_ensemble_votes_scores.csv`, and a `_feature_importance.png` plot — plus a global
`summary.csv` / `summary.txt` across all processed datasets.

**Real output** (captured directly from this repo, running on `adroit`):

```
============================================================
FINAL: 166 -> 15 features
Reduction: 91.0%
============================================================
Plot saved to .../adroit/adroit_feature_importance.png

✓ Results saved to .../adroit/
✓ Global summary saved to .../summary.txt

================================================================================
BATCH PROCESSING COMPLETE
================================================================================
Processed: 1/1 datasets
Duration: 2.68 seconds
```

To call `main.py` directly (e.g. to change `--target-col`, `--target-features`, or
`--correlation-threshold`) instead of going through `reproduce.sh`:

```bash
python3 main.py --data-dir /path/to/your/datasets --out-dir ./results \
    --target-col class --correlation-threshold 0.95
```

> ℹ️ This is separate from Protocol P2 (`--with-p2` / `--p2-only`), which already reads from
> its own configured dataset location (`<P2_REPO_DIR>/Datasets`, or `--p2-data-dir` /
> `--p2-import`) and needs no `--data-folder` — see "Claim #4" under Experiments below.

---

# 🎛️ `reproduce.sh` flag reference

Every flag `reproduce.sh` accepts, in one place (also available at any time via
`./reproduce.sh --help`, which prints the script's own header comments):

| Flag | Default | Meaning |
|---|---|---|
| `--data-dir <path>` | `./data/Originais` | Folder of full CSV datasets for the in-repo claims (Claims #1–#3 below). |
| `--out-dir <path>` | `./reproduction_output` | Where every step writes its results and logs. |
| `--only <keys>` | *(all)* | Run a subset of the in-repo claims. Comma-separated keys: `p1`, `ablation`, `cost`, `table5`, `multiclf`, `pvalue`. |
| `--minimal` | off | ~2 min smoke test over 2 datasets (`adroit`, `drebin215`) instead of the full run — see "Minimal test" above. |
| `--with-p2` | off | After the in-repo claims finish, also run Protocol P2 (external, MalDataGen). Still requires `--data-dir` for the in-repo part. |
| `--p2-only` | off | Run **only** Protocol P2, skipping the in-repo claims entirely — no `--data-dir`/`./data/Originais` needed. Implies `--with-p2`. |
| `--p2-data-dir <path>` | *(unset)* | Point P2 at reduced CSVs already sitting in `<path>`, instead of `Maldatagen_additional_metrics/Datasets/`. |
| `--p2-import <dir>` | *(unset)* | Copy the reduced CSVs from `<dir>` into P2's `Datasets/` before running (non-destructive `cp`; skips Git-LFS pointers). |
| `--p2-no-venv` | off | Use the current `python3` for P2 instead of creating an isolated venv inside `Maldatagen_additional_metrics/`. |
| `--pin-commit <SHA>` | *(unset — warns)* | Pin the MalDataGen clone to a specific commit, for reproducibility. Without it, P2 runs whatever the clone's current checkout is. |
| `--feature-selection` | off | Run `main.py`'s ensemble selector standalone over `--data-folder`, independent of every other mode — see "Feature selection on your own data" above. |
| `--data-folder <path>` | *(required with `--feature-selection`)* | The folder of your own raw CSVs to run feature selection on. |
| `--yes` / `-y` | off | Non-interactive: auto-confirm every prompt (venv creation, P2 start, etc.). Use for unattended/background runs. |
| `-h` / `--help` | — | Print usage and exit. |

Modes are mutually exclusive in the sense that only one drives the main run:
`--feature-selection` and `--p2-only` each take over the whole invocation (in-repo claims are
skipped); plain `--with-p2` runs the in-repo claims **and then** P2; everything else (`--only`,
`--minimal`) shapes the in-repo claims themselves.

---

# 🧪 Experiments

The `reproduce.sh` driver automates the in-repo claims. Each can be run in isolation with
`--only <key>` (see the flag reference above). Full in-repo reproduction:

```bash
./reproduce.sh --data-dir ./data/Originais
```

Outputs are written under `reproduction_output/` with per-step logs.

**At a glance:**

| Claim | Paper artifact | `--only` key | Needs |
|---|---|---|---|
| #1 | Table 3 (P1) | `p1` | `--data-dir` |
| #2 | Tables 6 & 7 (ablation) | `ablation` | `--data-dir` |
| #3 | §6.2 cost + Table 5 (+ robustness check) | `cost,table5,multiclf` | `--data-dir` |
| #4 | Table 4 / Figure 2 (P2) | *(not an `--only` key — use `--with-p2`/`--p2-only`)* | external MalDataGen |
| — | Wilcoxon floor note | `pvalue` | nothing |

## Claim #1 — Reduction preserving detection (Table 3, Protocol P1)

The method removes 89.8%–99.6% of the features while retaining recall ≥ 0.85 in 9 of 11
datasets (median recall drop 0.04), with selection performed **inside each training fold**.

```bash
./reproduce.sh --only p1 --data-dir ./data/Originais
# equivalent: python reproducibility/reproduce_p1.py --data-dir ./data/Originais --out-dir ./p1_out
```

- **Flags:** `--folds 5` (default), `--seed 42` (default), `--datasets` to scope a subset.
- **Expected resources/time:** ≥ 8 GB RAM; minutes for the small datasets, up to hours for
  `mh100k` (full-dimensional model on 24,833 features).
- **Expected result:** `table3_p1.csv` (original vs. reduced recall/F1, paired Wilcoxon
  p-value) and `per_fold_p1.csv`. Nine of ten benchmark datasets favour the original model;
  the exception is `android_permissions`, where reduction raises recall from 0.8962 to 0.9559.

## Claim #2 — Ablation: the ≥ 2-of-3 consensus (Tables 6 and 7)

The θ = 2 consensus jointly maximizes reduction and recall; Random Forest is the indispensable
ranker, and the value of a third ranker is dataset-dependent.

```bash
./reproduce.sh --only ablation --data-dir ./data/Originais
# equivalent: python ablation_study.py --data-dir ./data/Originais --out-dir ./ablation_out
```

- **Expected resources/time:** ~20–40 min for 10 datasets (`mh100k` is excluded from the
  ablation, §6.4).
- **Expected result:** adaptive-budget × θ table (Table 6) and criterion ablation with the
  Kuncheva stability index (Table 7); each dataset prints its `... | OK` self-check.
- **Scientific parameters** (seed, thresholds, folds, RF size, the ensemble's min-votes, and
  the 20,000-row correlation-matrix subsampling used on the 5 largest datasets) are exposed as
  CLI flags with defaults that reproduce the paper exactly — see the "Scientific parameters"
  section in [`reproducibility/README.md`](reproducibility/README.md) for the full table and
  what each one does.

## Claim #3 — Cost and generalization (§6.2 and Table 5)

Training on the reduced subsets is ~8.1× faster than on the full feature set, with a median
break-even of ~7.7 retraining cycles; the reduced subsets generalize across classifier
families.

```bash
./reproduce.sh --only cost,table5 --data-dir ./data/Originais
# equivalents:
#   python bench_filter_order.py --data-dir ./data/Originais --out-dir ./cost
#   python analyze_cost.py       --data-dir ./data/Originais --out-dir ./cost
#   python reproducibility/reproduce_table5.py --data-dir ./data/Originais --dataset mh100k --out-dir ./table5
#   python pipeline_multiclf.py  --data-dir ./data/Originais --out-dir ./multiclf   # broader robustness check
```

- **Expected resources/time:** tens of minutes for `analyze_cost.py`'s S0–S3 stages; the
  amortization measurement (`--data-dir`, unless `--skip-amortization`) additionally runs the
  full χ²/MI/RF selection per fold, which is slower — see its own runtime notes.
- **Expected result:** per-stage cost, the S0-vs-S3 training speedup (§6.2, the ~8.1× figure),
  and — from `analyze_cost.py`'s amortization section — the break-even point (median ~7.7
  retraining cycles) and aggregate savings (27.1% at 10 updates, 75.6% at 50), computed from
  the **full** one-off selection cost (variance + correlation + χ² + Mutual Information +
  Random Forest ranking + voting), not just the cleaning filters.
- **`reproducibility/reproduce_table5.py`** is Table 5's exact published protocol: a subset
  selected **once** over the whole MH100K dataset (93 features, not per-fold), evaluated
  across the four classifiers Table 5 actually reports (Decision Tree, k-NN, Random Forest,
  Logistic Regression).
- **`pipeline_multiclf.py`** is a separate, **complementary robustness check** (five
  classifiers — the four above plus GradientBoosting — with feature reselection inside every
  fold), not a Table 5 reproduction. See the "Note on Table 5" in
  [`reproducibility/README.md`](reproducibility/README.md) for the full explanation.

## Claim #4 — TSTR comparison across methods (Table 4, Figure 2)

Under Protocol P2 (Train-on-Synthetic / Test-on-Real), Statistical Ranking attains the highest
Trade-off (Recall × Reduction / 100) against Lasso, RFE and SemiDroid, using the autoencoder
and variational generators.

This experiment depends on the external **MalDataGen** framework and is **not** reproduced by
the in-repo scripts. Step by step:

1. **Get the reduced CSVs.** Table 4 compares 4 methods (`statistical_`, `rfe_`, `semidroid_`,
   `lasso_` prefixes) across the 11 datasets. `statistical_*` comes from this repo — run
   `reproduce.sh --feature-selection --data-folder ./data/Originais` (see "Feature selection
   on your own data" above) and rename the `*_reduced.csv` outputs with the `statistical_`
   prefix. The `rfe_`/`semidroid_`/`lasso_` baselines come from elsewhere — see
   `reproducibility/README.md` for what's currently documented about their provenance.
2. **Place them where MalDataGen reads from.** Either put the CSVs directly in
   `Maldatagen_additional_metrics/Datasets/`, or use `--p2-import <dir>` to copy them there, or
   `--p2-data-dir <dir>` to point elsewhere entirely (see the flag reference above).
3. **Run.** `--with-p2` runs the in-repo claims above it first (needs `--data-dir`/
   `./data/Originais`) and then P2; `--p2-only` skips straight to P2 (no local dataset needed,
   since P2 clones/reads its own):

   ```bash
   # in-repo claims + P2:
   ./reproduce.sh --with-p2 --pin-commit <MALDATAGEN_COMMIT_SHA>

   # P2 only:
   ./reproduce.sh --p2-only --pin-commit <MALDATAGEN_COMMIT_SHA>
   ```

   Either command prints a **coverage matrix** (which of the 4 methods × 11 datasets are
   actually present in the Datasets folder) before starting — check that output before waiting
   hours for a run that's missing baselines you need.

- **Model configuration (as in the paper):**
  - **Autoencoder (AE):** 300 epochs · latent 128 · `leakyrelu` · dropout 0.10 · layers
    `128 128` · batch 8 · loss `mean_squared_error` · Adam · 5 folds.
  - **Variational (VAE):** latent 32 · `leakyrelu` · dropout 0.25 · encoder `128 128` /
    decoder `160 320` · batch 64 · loss `cross_entropy_kl` · Adam · 5 folds.
- **Expected resources/time:** **several hours per experiment**; GPU recommended.
- **Reproducibility:** MalDataGen exposes no seed flag, so P2 is not bit-for-bit reproducible;
  small numerical variation is expected. See `reproducibility/REPRODUCING_MALDATAGEN.md`.

> 🎲 **Note on statistical testing.** The paired Wilcoxon test per dataset (n = 5 folds) has a
> minimum two-sided p-value of 0.0625 > 0.05 and is reported for completeness only
> (`python audit_pvalue.py --out-dir ./pvalue`). The paper's inferential claims rest on the
> Friedman test across datasets (§6.4).
>
> **Real output** (needs no data — runs anywhere):
> ```
> DESENHO USADO NO REPO: n_splits=5 (pipeline.py, pipeline_multiclf.py, ablation_study.py)
> p-valor minimo possivel com 5 folds: 0.0625
> => NENHUM resultado pode ser p<0.05 com 5 folds, seja qual for o efeito observado.
> => Seriam necessarios pelo menos 6 folds para que p<0.05 seja sequer matematicamente atingivel.
> ```

---

# 📄 LICENSE

This artifact is released under the **MIT License**. See the [`LICENSE`](LICENSE) file at the
repository root for the full text.

```
MIT License — Copyright (c) 2026 Anna Luiza Gomes da Silva, Lucas Ferreira, Angelo Nogueira, Diego Kreutz
```

---

# 👥 Maintainers

| Name |
| ---- |
| Anna Luiza Gomes da Silva |
| Lucas Ferreira |
| Angelo Nogueira |