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

| Library        | Version |
| -------------- | ------- |
| `numpy`        | 2.4.4   |
| `scipy`        | 1.17.1  |
| `pandas`       | 3.0.2   |
| `scikit-learn` | 1.8.0   |
| `matplotlib`   | ≥ 3.4.0 |
| `seaborn`      | ≥ 0.11.0 |

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

# 🧪 Experiments

The `reproduce.sh` driver automates the in-repo claims. Each can be run in isolation with
`--only <key>`. Full in-repo reproduction:

```bash
./reproduce.sh --data-dir ./data/Originais
```

Outputs are written under `reproduction_output/` with per-step logs.

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

## Claim #3 — Cost and generalization (§6.2 and Table 5)

Training on the reduced subsets is ~8.1× faster with amortization after ≈ 8 retraining cycles;
the reduced subsets generalize across classifier families.

```bash
./reproduce.sh --only cost,multiclf --data-dir ./data/Originais
# equivalents:
#   python bench_filter_order.py --data-dir ./data/Originais --out-dir ./cost
#   python pipeline_multiclf.py  --data-dir ./data/Originais --out-dir ./multiclf
```

- **Expected resources/time:** tens of minutes.
- **Expected result:** per-stage cost and speedup (§6.2); Table 5 with recall/F1/MCC of the
  reduced `mh100k` subset across four classifier families (recall span of only 0.022).

## Claim #4 — TSTR comparison across methods (Table 4, Figure 2)

Under Protocol P2 (Train-on-Synthetic / Test-on-Real), Statistical Ranking attains the highest
Trade-off (Recall × Reduction / 100) against Lasso, RFE and SemiDroid, using the autoencoder
and variational generators.

This experiment depends on the external **MalDataGen** framework and is **not** reproduced by
the in-repo scripts:

```bash
# from the repo root; clones/pins MalDataGen and runs AE+VAE.
# place the reduced CSVs in Maldatagen_additional_metrics/Datasets/ (or pass --p2-data-dir):
./reproduce.sh --with-p2 --pin-commit <MALDATAGEN_COMMIT_SHA>

# --with-p2 also runs the in-repo claims above it (needs ./data/Originais). To run ONLY
# Protocol P2 — no local dataset required, since P2 clones/reads its own — use --p2-only:
./reproduce.sh --p2-only --pin-commit <MALDATAGEN_COMMIT_SHA>
```

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