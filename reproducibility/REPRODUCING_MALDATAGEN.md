# Reproducing Protocol P2 (TSTR) — Table 4 & Figure 2

The feature-selection pipeline in this repository reproduces Protocol **P1** (direct
classification, Table 3) and the analyses of Sections 6.2–6.4 via the in-repo scripts (see the
main README and `reproduce.sh`).

The one experiment those scripts do **not** cover is Protocol **P2 (Train-on-Synthetic /
Test-on-Real)**, which produces **Table 4** and **Figure 2**. P2 is run inside an external
framework, **MalDataGen** [Paim et al. 2025], which trains **autoencoder** and **variational**
generators on the reduced feature sets and evaluates classifiers trained on the synthetic data
against real data. This document is the reproduction path for that half.

> ⚠️ **Generators.** Per Section 5.3 and Table 4, P2 uses the **autoencoder** (AE) and
> **variational** (VAE) generators — `--model_type autoencoder` and `--model_type variational`
> in MalDataGen. These are *not* the `adversarial` (GAN) or `wasserstein` (WGAN) model types,
> which are separate models in MalDataGen and do not reproduce Table 4.

## Pipeline overview

```
raw datasets  (github.com/AILabs4All/datasets-mh30plus)
    │
    ├── [THIS REPO] main.py  → statistical_*.csv   (Statistical Ranking, ours)
    ├── [baselines] RFE       → rfe_*.csv
    ├── [baselines] SemiDroid → semidroid_*.csv
    └── [baselines] Lasso     → lasso_*.csv
    │
    ▼
[MalDataGen]  autoencoder + variational generators, TSTR
    │
    ▼
per-method recall / F1 / reduction  →  Table 4 (aggregate)  &  Figure 2 (per dataset)
```

Table 4 is a **comparison across four reduction methods**, so P2 consumes the reduced CSVs of
*each* method (Statistical, RFE, SemiDroid, Lasso) over the 11 datasets, and aggregates.

> **Fill in:** the provenance of the baseline reductions (`rfe_*`, `semidroid_*`, `lasso_*`) —
> e.g. MH-FSF or your own scripts — so the comparison is reproducible end to end.

---

## Step 0 — Pin the MalDataGen version

Results depend on the exact MalDataGen commit; the repo is actively developed.

```bash
git clone https://github.com/MalwareDataLab/Maldatagen_additional_metrics.git
cd Maldatagen_additional_metrics
git checkout <PIN_COMMIT_SHA_HERE>   # ← the commit you used for the paper
git rev-parse HEAD                    # confirm
```

> Find the SHA with `git rev-parse HEAD` in the clone you actually ran. Do not pin to "main"
> (it moves).

Environment (per MalDataGen): Python 3.10+, TensorFlow. Install with either:

```bash
pip install -r requirements.txt      # or: pip install .
# the repo also ships a Pipfile (pipenv) — its test scripts use `pipenv run python3`
```

---

## Step 1 — Produce the reduced datasets (four methods)

- `statistical_*.csv` — this repo's `main.py` (Statistical Ranking).
- `rfe_*.csv`, `semidroid_*.csv`, `lasso_*.csv` — the baseline reducers compared in Table 4.

Place all reduced CSVs in MalDataGen's dataset directory (numeric feature columns, binary
`class` label).

---

## Step 2 — Run the generative experiments (MalDataGen)

From the repo root, the unified driver clones/pins MalDataGen and runs both generators.
Place the reduced CSVs in `Maldatagen_additional_metrics/Datasets/` (or pass `--p2-data-dir`):

```bash
./reproduce.sh --with-p2 --pin-commit <MALDATAGEN_COMMIT_SHA>
```

The AE/VAE hyperparameter blocks are embedded in `reproduce.sh` (values below).

The runner exercises the two generators the paper reports:

| `--model_type`  | Architecture | Config flag prefix               |
|-----------------|--------------|----------------------------------|
| `autoencoder`   | AE           | `--autoencoder_*`                |
| `variational`   | VAE          | `--variational_autoencoder_*`    |

### Model configuration (as used for the paper)

**Autoencoder (AE):**
`--autoencoder_number_epochs 300` · `--autoencoder_latent_dimension 128` ·
`--autoencoder_activation_function leakyrelu` ·
`--autoencoder_dropout_decay_rate_encoder 0.10` · `--autoencoder_dropout_decay_rate_decoder 0.10` ·
`--autoencoder_dense_layer_sizes_encoder 128 128` · `--autoencoder_dense_layer_sizes_decoder 128 128` ·
`--autoencoder_batch_size 8` · `--autoencoder_loss_function mean_squared_error` ·
`--autoencoder_momentum 0.8` · `--autoencoder_latent_mean_distribution 0.5` ·
`--autoencoder_latent_stander_deviation 0.125` · `--autoencoder_initializer_mean 0.0` ·
`--autoencoder_initializer_deviation 0.125` · `--autoencoder_training_algorithm Adam` ·
`--number_k_folds 5`.

**Variational (VAE):**
`--variational_autoencoder_number_epochs 300` · `--variational_autoencoder_latent_dimension 32` ·
`--variational_autoencoder_activation_function leakyrelu` ·
`--variational_autoencoder_dropout_decay_rate_encoder 0.25` · `--variational_autoencoder_dropout_decay_rate_decoder 0.25` ·
`--variational_autoencoder_dense_layer_sizes_encoder 128 128` · `--variational_autoencoder_dense_layer_sizes_decoder 160 320` ·
`--variational_autoencoder_batch_size 64` · `--variational_autoencoder_loss_function cross_entropy_kl` ·
`--variational_autoencoder_momentum 0.8` · `--variational_autoencoder_mean_distribution 0.5` ·
`--variational_autoencoder_stander_deviation 0.125` · `--variational_autoencoder_initializer_mean 0.0` ·
`--variational_autoencoder_initializer_deviation 0.125` · `--variational_autoencoder_training_algorithm Adam` ·
`--number_k_folds 5`.

These are the exact blocks embedded in `reproduce.sh` (the `--with-p2` path).

### P2 aggregation (as in the paper)

- Metrics averaged over **three classifiers** (Decision Tree, Random Forest, k-NN) **and both
  generators**.
- **Trade-off = Recall × Reduction / 100.**
- Per-dataset method selection (Figure 2): rank lexicographically by **recall → reduction →
  F1 preservation** (recall leads: a missed malicious sample costs more than a false alarm).
- **MH100K** (24,833 → ~93 features) is the large-scale case, generated/reduced via MalDataGen.

### Determinism

MalDataGen exposes **no `--seed` CLI flag** (verified in its argument definitions), so runs are
not bit-for-bit reproducible and GPU training adds further nondeterminism. Report the hardware,
driver/CUDA, and library versions, and expect small numerical variation in the Table 4 /
Figure 2 values.

---

## Expected outputs → paper artifacts

MalDataGen writes, per model, `outputs/.../<model>/<dataset>/` with generated data, per-fold
evaluation (confusion matrices, heatmaps, TSTR/TRTS bar charts), logs, and (optionally) saved
models, plus a top-level SVM comparison PDF.

| Paper artifact | Produced from |
|----------------|---------------|
| Table 4 (aggregate, AE & VAE columns) | mean over 11 datasets, 3 classifiers, both generators, per method |
| Figure 2 (per-dataset reduction vs recall) | best method per dataset under the P2 ranking rule |

---

## Versions used (fill in and report)

- MalDataGen commit: `<PIN_COMMIT_SHA_HERE>`
- Python / TensorFlow: `<x.y.z>` / `<x.y.z>`
- OS / GPU / CUDA: `<...>`