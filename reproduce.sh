#!/bin/bash
#
# ==========================================================================
#  Unified reproduction script — artifact:
#  "Statistical Ranking: A Voting-Based Ensemble Approach to Feature Selection
#   in Android Malware Detection" (SBSeg)
# ==========================================================================
#
# Single entry point for the whole reproduction:
#   * in-repo claims (run locally, no root, no network):
#       [C2] Table 3       — P1, original vs. reduced          (reproduce_p1.py)
#       [C3] Tables 6 & 7  — ablation (criteria / budget / θ)  (ablation_study.py)
#       [C4] §6.2          — computational cost                (bench_filter_order.py, analyze_cost.py)
#       [C4] Table 5       — generalisation across classifiers (pipeline_multiclf.py)
#       stat. note         — Wilcoxon floor at n=5             (audit_pvalue.py)
#   * external claim (optional, opt-in with --with-p2):
#       [P2] Table 4 / Fig 2 — TSTR via MalDataGen (AE + VAE)
#
# USAGE
#   ./reproduce.sh --minimal                         # ~2 min smoke test (Teste minimo)
#   ./reproduce.sh                                   # in-repo claims (default data ./data/Originais)
#   ./reproduce.sh --only p1,ablation                # subset of in-repo claims
#   ./reproduce.sh --with-p2 --pin-commit <SHA>      # also run the external MalDataGen P2
#   ./reproduce.sh --yes                             # non-interactive
#
# KEYS for --only: p1, ablation, cost, multiclf, pvalue
# ==========================================================================

set -uo pipefail

# ---- Paths & defaults ------------------------------------------------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$HERE"
REPRO_DIR="$REPO_ROOT/reproducibility"
REQUIREMENTS="$REPO_ROOT/requirements.txt"

DATA_DIR="./data/Originais"                 # full datasets for the in-repo claims
OUT_DIR="./reproduction_output"

# Protocol P2 (external MalDataGen) — only used with --with-p2
P2_REPO_URL="https://github.com/MalwareDataLab/Maldatagen_additional_metrics.git"
P2_REPO_DIR="Maldatagen_additional_metrics"
P2_DATA_DIR=""                              # reduced CSVs for P2 (default: <P2_REPO_DIR>/Datasets)
PIN_COMMIT="PUT_PINNED_COMMIT_SHA_HERE"     # MalDataGen commit used for the paper

MINIMAL=0; WITH_P2=0; ASSUME_YES=0; ONLY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir)     DATA_DIR="$2"; shift 2 ;;
        --out-dir)      OUT_DIR="$2";  shift 2 ;;
        --only)         ONLY="$2";     shift 2 ;;
        --minimal)      MINIMAL=1;     shift ;;
        --with-p2)      WITH_P2=1;     shift ;;
        --p2-data-dir)  P2_DATA_DIR="$2"; shift 2 ;;
        --pin-commit)   PIN_COMMIT="$2";  shift 2 ;;
        --yes|-y)       ASSUME_YES=1;  shift ;;
        -h|--help)      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: $1"; exit 2 ;;
    esac
done

mkdir -p "$OUT_DIR"
LOG_DIR="$OUT_DIR/logs_$(date '+%Y%m%d_%H%M%S')"
mkdir -p "$LOG_DIR"
STATUS=()

# ---- Helpers ---------------------------------------------------------------
confirm() {   # confirm "<question>" -> 0 if yes (auto-yes with --yes)
    [[ "$ASSUME_YES" -eq 1 ]] && return 0
    read -p "$1 (y/N): " -n 1 -r; echo
    [[ $REPLY =~ ^[Yy]$ ]]
}

run_step() {  # run_step <name> <logfile> <command...>
    local name="$1"; local log="$2"; shift 2
    echo "----------------------------------------------------"
    echo "▶  $name"
    echo "   log: $log"
    local t0; t0=$(date +%s)
    if "$@" > "$log" 2>&1; then
        echo "   ✅ ok ($(( $(date +%s) - t0 ))s)"; STATUS+=("✅ $name")
    else
        echo "   ❌ failed (see log)"; STATUS+=("❌ $name")
    fi
}

want() { [[ -z "$ONLY" ]] && return 0; [[ ",$ONLY," == *",$1,"* ]] && return 0 || return 1; }

check_python() { command -v python3 >/dev/null 2>&1 || { echo "❌ python3 not found"; exit 1; }; }
check_git()    { command -v git     >/dev/null 2>&1 || { echo "❌ git not found (required for --with-p2)"; exit 1; }; }

install_dependencies() {
    if python3 -c "import numpy, pandas, sklearn, scipy" 2>/dev/null; then return; fi
    echo "⚠  Missing Python dependencies."
    if [[ -f "$REQUIREMENTS" ]] && confirm "   Install from requirements.txt?"; then
        pip3 install -r "$REQUIREMENTS"
    fi
}

# ==========================================================================
#  In-repo claims
# ==========================================================================
run_minimal() {
    echo ""; echo "### Teste minimo (2 datasets, ~2 min) ###"
    run_step "P1 quick (adroit, drebin215)" "$LOG_DIR/p1_min.log" \
        python3 "$REPRO_DIR/reproduce_p1.py" --data-dir "$DATA_DIR" \
            --out-dir "$OUT_DIR/p1_min" --datasets adroit,drebin215
    run_step "Ablation self-check (adroit, drebin215)" "$LOG_DIR/ablation_min.log" \
        python3 "$REPO_ROOT/ablation_study.py" --data-dir "$DATA_DIR" \
            --datasets adroit,drebin215 --out-dir "$OUT_DIR/ablation_min"
    run_step "Wilcoxon floor audit" "$LOG_DIR/pvalue_min.log" \
        python3 "$REPO_ROOT/audit_pvalue.py" --out-dir "$OUT_DIR/pvalue"
}

run_claims() {
    if want p1;       then run_step "[C2] Table 3 — P1" "$LOG_DIR/p1.log" \
        python3 "$REPRO_DIR/reproduce_p1.py" --data-dir "$DATA_DIR" --out-dir "$OUT_DIR/p1"; fi
    if want ablation; then run_step "[C3] Tables 6 & 7 — ablation" "$LOG_DIR/ablation.log" \
        python3 "$REPO_ROOT/ablation_study.py" --data-dir "$DATA_DIR" --out-dir "$OUT_DIR/ablation"; fi
    if want cost; then
        run_step "[C4] §6.2 — filter-order cost" "$LOG_DIR/cost_filter.log" \
            python3 "$REPO_ROOT/bench_filter_order.py" --data-dir "$DATA_DIR" --out-dir "$OUT_DIR/cost"
        run_step "[C4] §6.2 — per-stage cost & speedup" "$LOG_DIR/cost_stages.log" \
            python3 "$REPO_ROOT/analyze_cost.py" --data-dir "$DATA_DIR" --out-dir "$OUT_DIR/cost"
    fi
    if want multiclf; then run_step "[C4] Table 5 — classifier families" "$LOG_DIR/multiclf.log" \
        python3 "$REPO_ROOT/pipeline_multiclf.py" --data-dir "$DATA_DIR" --out-dir "$OUT_DIR/multiclf"; fi
    if want pvalue;   then run_step "stat. note — Wilcoxon floor" "$LOG_DIR/pvalue.log" \
        python3 "$REPO_ROOT/audit_pvalue.py" --out-dir "$OUT_DIR/pvalue"; fi
}

# ==========================================================================
#  Protocol P2 (external MalDataGen) — optional
# ==========================================================================
AE_CONFIG="
    --number_k_folds 5 \
    --save_data True \
    --autoencoder_number_epochs 300 \
    --autoencoder_latent_dimension 128 \
    --autoencoder_activation_function leakyrelu \
    --autoencoder_dropout_decay_rate_encoder 0.10 \
    --autoencoder_dropout_decay_rate_decoder 0.10 \
    --autoencoder_dense_layer_sizes_encoder 128 128 \
    --autoencoder_dense_layer_sizes_decoder 128 128 \
    --autoencoder_batch_size 8 \
    --autoencoder_loss_function mean_squared_error \
    --autoencoder_momentum 0.8 \
    --autoencoder_latent_mean_distribution 0.5 \
    --autoencoder_latent_stander_deviation 0.125 \
    --autoencoder_initializer_mean 0.0 \
    --autoencoder_initializer_deviation 0.125 \
    --autoencoder_training_algorithm Adam \
    --verbosity 20
"
VAE_CONFIG="
    --number_k_folds 5 \
    --save_data True \
    --variational_autoencoder_number_epochs 300 \
    --variational_autoencoder_latent_dimension 32 \
    --variational_autoencoder_activation_function leakyrelu \
    --variational_autoencoder_dropout_decay_rate_encoder 0.25 \
    --variational_autoencoder_dropout_decay_rate_decoder 0.25 \
    --variational_autoencoder_dense_layer_sizes_encoder 128 128 \
    --variational_autoencoder_dense_layer_sizes_decoder 160 320 \
    --variational_autoencoder_batch_size 64 \
    --variational_autoencoder_loss_function cross_entropy_kl \
    --variational_autoencoder_momentum 0.8 \
    --variational_autoencoder_mean_distribution 0.5 \
    --variational_autoencoder_stander_deviation 0.125 \
    --variational_autoencoder_initializer_mean 0.0 \
    --variational_autoencoder_initializer_deviation 0.125 \
    --variational_autoencoder_training_algorithm Adam \
    --verbosity 20
"

run_p2() {
    echo ""
    echo "=========================================================="
    echo "  PROTOCOL P2 (external) — MalDataGen AE + VAE (Table 4/Fig 2)"
    echo "=========================================================="
    echo "⚠  This clones an external repo, installs its dependencies, and can take MANY HOURS."
    echo "⚠  MalDataGen exposes no --seed flag: P2 is not bit-for-bit reproducible."
    confirm "Continue with Protocol P2?" || { echo "P2 skipped."; STATUS+=("⏭  [P2] skipped by user"); return; }

    check_git

    # 1) clone / pin MalDataGen
    if [[ ! -d "$P2_REPO_DIR" ]]; then
        echo "📦 Cloning MalDataGen..."
        git clone "$P2_REPO_URL" "$P2_REPO_DIR" || { STATUS+=("❌ [P2] clone failed"); return; }
    fi
    if [[ "$PIN_COMMIT" != "PUT_PINNED_COMMIT_SHA_HERE" ]]; then
        ( cd "$P2_REPO_DIR" && git checkout "$PIN_COMMIT" ) || echo "⚠  could not checkout $PIN_COMMIT"
    else
        echo "⚠  No pinned commit set (use --pin-commit <SHA>); running the current checkout."
    fi

    # 2) install MalDataGen deps
    if [[ -f "$P2_REPO_DIR/requirements.txt" ]] && confirm "Install MalDataGen dependencies?"; then
        pip3 install -r "$P2_REPO_DIR/requirements.txt"
    fi

    # 3) datasets: reduced CSVs must be in the MalDataGen Datasets dir
    local ds_dir="${P2_DATA_DIR:-$P2_REPO_DIR/Datasets}"
    mkdir -p "$ds_dir"
    mapfile -t DATASETS < <(cd "$ds_dir" 2>/dev/null && ls -1 *.csv 2>/dev/null)
    if [[ ${#DATASETS[@]} -eq 0 ]]; then
        echo "❌ No CSVs in $ds_dir — place the reduced datasets (statistical_/rfe_/semidroid_/lasso_*) there."
        STATUS+=("❌ [P2] no datasets in $ds_dir"); return
    fi

    local p2_log_dir="$OUT_DIR/p2_logs_$(date '+%Y%m%d_%H%M%S')"; mkdir -p "$p2_log_dir"
    local total=$(( ${#DATASETS[@]} * 2 )) n=0 fails=0
    echo "📊 P2 experiments: $total  (${#DATASETS[@]} datasets × AE,VAE)"
    confirm "Start P2 training now?" || { STATUS+=("⏭  [P2] training skipped"); return; }

    ( cd "$P2_REPO_DIR" && {
        for MODEL in autoencoder variational; do
            [[ "$MODEL" == autoencoder ]] && CONFIG="$AE_CONFIG" || CONFIG="$VAE_CONFIG"
            for DATASET in "${DATASETS[@]}"; do
                n=$((n+1)); DNAME="${DATASET%.csv}"
                echo "── [$n/$total] $MODEL / $DNAME"
                DPATH="${ds_dir#$P2_REPO_DIR/}"; DPATH="Datasets/${DATASET}"
                [[ -n "$P2_DATA_DIR" ]] && DPATH="$P2_DATA_DIR/$DATASET"
                LOG="$OLDPWD/$p2_log_dir/${MODEL}_${DNAME}.log"
                python3 main.py --model_type "$MODEL" \
                    --data_load_path_file_input "$DPATH" \
                    --output_dir "outputs/p2_experiments/${MODEL}/${DNAME}" \
                    $CONFIG 2>&1 | tee "$LOG"
                [[ ${PIPESTATUS[0]} -ne 0 ]] && fails=$((fails+1))
            done
        done
    } )
    if [[ $fails -eq 0 ]]; then STATUS+=("✅ [P2] AE+VAE ($total experiments)")
    else STATUS+=("❌ [P2] AE+VAE ($fails/$total failed)"); fi
    echo "📋 P2 logs: $p2_log_dir"
}

# ==========================================================================
#  Main
# ==========================================================================
echo "=========================================================="
echo "Statistical Ranking — reproduction"
echo "=========================================================="
echo "Repo root : $REPO_ROOT"
echo "Data dir  : $DATA_DIR"
echo "Output    : $OUT_DIR"
echo "Mode      : $([[ $MINIMAL -eq 1 ]] && echo MINIMAL || echo FULL)$([[ $WITH_P2 -eq 1 ]] && echo ' +P2')"
echo ""
echo "Security  : in-repo claims run locally, need no root/sudo and no network."
echo "            --with-p2 clones an external repo and installs deps (run in a venv/container)."
echo ""

check_python
install_dependencies

if [[ ! -d "$DATA_DIR" ]] && ! want pvalue; then
    echo "❌ Data directory not found: $DATA_DIR"
    echo "   Get the datasets from https://github.com/AILabs4All/datasets-mh30plus (Git LFS),"
    echo "   or pass --data-dir. (The Wilcoxon audit 'pvalue' needs no data.)"
    exit 1
fi

confirm "Proceed?" || { echo "Cancelled."; exit 1; }

START=$(date +%s)
if [[ $MINIMAL -eq 1 ]]; then run_minimal; else run_claims; fi
[[ $WITH_P2 -eq 1 ]] && run_p2
DUR=$(( $(date +%s) - START ))

echo ""
echo "=========================================================="
echo "SUMMARY  (total $((DUR/3600))h $((DUR%3600/60))m $((DUR%60))s)"
echo "=========================================================="
for s in "${STATUS[@]}"; do echo "  $s"; done
echo ""
echo "Outputs: $OUT_DIR"
echo "Logs   : $LOG_DIR"

for s in "${STATUS[@]}"; do [[ "$s" == ❌* ]] && exit 1; done
exit 0