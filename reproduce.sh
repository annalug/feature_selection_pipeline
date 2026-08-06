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
#       [P2] Table 4 / Fig 2 — TSTR via MalDataGen (AE + VAE) on the reduced
#                              datasets produced by each feature-selection method
#                              (statistical_ / rfe_ / semidroid_ / lasso_*)
#
# The P2 orchestration mirrors the battle-tested run_experiments_2.sh:
#   dedicated venv, Git-LFS pointer detection, pre-creation of the MalDataGen
#   output tree (avoids FileNotFoundError races), per-experiment logging,
#   and a summary broken down by feature-selection method.
#
# USAGE
#   ./reproduce.sh --minimal                         # ~2 min smoke test (Teste minimo)
#   ./reproduce.sh                                   # in-repo claims (default data ./data/Originais)
#   ./reproduce.sh --only p1,ablation                # subset of in-repo claims
#   ./reproduce.sh --with-p2 --pin-commit <SHA>      # in-repo claims (needs ./data/Originais) + P2
#   ./reproduce.sh --p2-only --pin-commit <SHA>      # ONLY Protocol P2 — no local data needed,
#                                                     # P2 fetches/clones its own datasets/repo
#   ./reproduce.sh --with-p2 --p2-import ./reduzidos # copy reduced CSVs into Datasets/, then run P2
#   ./reproduce.sh --with-p2 --p2-no-venv            # P2 using the current python3 (no venv)
#   ./reproduce.sh --feature-selection --data-folder ./my_datasets  # run main.py's ensemble
#                                                     # feature selection over a user-chosen folder
#   ./reproduce.sh --yes                             # non-interactive
#
# P2 data:  the reduced CSVs (statistical_/rfe_/semidroid_/lasso_*) must end up in
#           <P2_REPO_DIR>/Datasets/. Either place them there yourself, point to them
#           with --p2-data-dir <ABS_DIR>, or let --p2-import <DIR> copy them in for you
#           (non-destructive cp; Git-LFS pointers are skipped). This replaces move_datasets.sh.
#           This is the "already fetching from configured datasets" path: it needs no
#           --data-folder because it reads from <P2_REPO_DIR>/Datasets (or --p2-data-dir).
#           --with-p2 ADDS P2 on top of the in-repo claims (needs ./data/Originais too);
#           --p2-only RUNS ONLY P2 and skips the in-repo claims, so ./data/Originais is not required.
#
# Feature selection (main.py):  --feature-selection runs the chi2 + mutual-information +
#           Random Forest ensemble selector (main.py) over every *.csv in the folder given
#           with --data-folder (required, no default — this is the user's own raw-datasets
#           folder, distinct from --data-dir/./data/Originais used by the in-repo claims).
#           Reduced datasets + scores are written to <out-dir>/feature_selection/.
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
P2_IMPORT=""                                # optional source dir: copy reduced CSVs INTO Datasets/
P2_OUT_SUBDIR="outputs/p2_experiments"      # (relative to P2_REPO_DIR) where MalDataGen writes
PIN_COMMIT="PUT_PINNED_COMMIT_SHA_HERE"     # MalDataGen commit used for the paper
P2_USE_VENV=1                               # 1 = isolated venv inside P2_REPO_DIR, 0 = system python3
P2_PYTHON=""                                # resolved after venv setup

MINIMAL=0; WITH_P2=0; ASSUME_YES=0; ONLY=""
FEATURE_SELECTION=0; DATA_FOLDER=""
SKIP_CLAIMS=0                                # 1 = --p2-only: skip in-repo claims, no local data needed

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir)           DATA_DIR="$2"; shift 2 ;;
        --out-dir)            OUT_DIR="$2";  shift 2 ;;
        --only)               ONLY="$2";     shift 2 ;;
        --minimal)             MINIMAL=1;     shift ;;
        --with-p2)             WITH_P2=1;     shift ;;
        --p2-only)             WITH_P2=1; SKIP_CLAIMS=1; shift ;;
        --p2-data-dir)         P2_DATA_DIR="$2"; shift 2 ;;
        --p2-import)           P2_IMPORT="$2";   shift 2 ;;
        --p2-no-venv)          P2_USE_VENV=0; shift ;;
        --pin-commit)          PIN_COMMIT="$2";  shift 2 ;;
        --feature-selection)   FEATURE_SELECTION=1; shift ;;
        --data-folder)         DATA_FOLDER="$2"; shift 2 ;;
        --yes|-y)              ASSUME_YES=1;  shift ;;
        -h|--help)             grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: $1"; exit 2 ;;
    esac
done

if [[ "$FEATURE_SELECTION" -eq 1 && -z "$DATA_FOLDER" ]]; then
    echo "❌ --feature-selection requires --data-folder <path-to-your-datasets>"
    exit 2
fi

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

# ---- feature-selection method inferred from the reduced-dataset filename ---
fs_method_of() {   # fs_method_of <dataset.csv> -> statistical|rfe|semidroid|lasso|other
    case "$1" in
        statistical_*) echo "statistical" ;;
        rfe_*)         echo "rfe" ;;
        semidroid_*)   echo "semidroid" ;;
        lasso_*)       echo "lasso" ;;
        *)             echo "other" ;;
    esac
}

# ==========================================================================
#  Feature selection (main.py) — user-chosen folder, opt-in with --feature-selection
# ==========================================================================
run_feature_selection() {
    echo ""
    echo "=========================================================="
    echo "  FEATURE SELECTION (main.py) — $DATA_FOLDER"
    echo "=========================================================="
    if [[ ! -d "$DATA_FOLDER" ]]; then
        echo "❌ --data-folder not found: $DATA_FOLDER"
        STATUS+=("❌ [feature-selection] folder not found")
        return 1
    fi
    run_step "Feature selection ensemble (chi2 + MI + RF)" "$LOG_DIR/feature_selection.log" \
        python3 "$REPO_ROOT/main.py" --data-dir "$DATA_FOLDER" --out-dir "$OUT_DIR/feature_selection"
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
#  Orchestration ported from run_experiments_2.sh
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

# Git-LFS pointer detection (real CSVs must NOT start with the LFS header)
is_lfs_pointer() {
    local file="$1"
    [[ -f "$file" ]] && head -n 1 "$file" 2>/dev/null | grep -q "^version https://git-lfs"
}

# Isolated venv inside the MalDataGen repo (mirrors run_experiments_2.sh)
setup_p2_venv() {   # sets P2_PYTHON to an ABSOLUTE path — the training loop later does
                     # `cd "$P2_REPO_DIR"`, so a path relative to $repo would resolve
                     # against the wrong cwd once inside that subshell.
    local repo="$1"
    local venv="$repo/venv"

    if [[ "$P2_USE_VENV" -eq 0 ]]; then
        P2_PYTHON="$(command -v python3)"
        echo "🐍 P2 using system python3: $P2_PYTHON (no venv)"
        return
    fi

    echo "=========================================================="
    echo "  P2 — VIRTUAL ENVIRONMENT"
    echo "=========================================================="
    local pyver; pyver=$(python3 --version 2>&1 | grep -oP '(?<=Python )\d+\.\d+')
    echo "🐍 Python version: ${pyver:-unknown}"

    if [[ -d "$venv" ]]; then
        echo "📦 venv already exists: $venv"
        if confirm "   Recreate the P2 virtual environment?"; then
            rm -rf "$venv"; python3 -m venv "$venv"; echo "   ✅ recreated"
        else
            echo "   Using existing venv"; P2_PYTHON="$(realpath "$venv/bin/python")"; return
        fi
    else
        echo "📦 Creating venv..."; python3 -m venv "$venv"; echo "   ✅ created at $venv"
    fi

    echo "🔧 Installing MalDataGen dependencies (TensorFlow stack — may take several minutes)..."
    "$venv/bin/pip" install --upgrade pip setuptools wheel
    "$venv/bin/pip" install numpy pandas matplotlib seaborn scikit-learn scipy
    "$venv/bin/pip" install tensorflow
    # honour the repo's own requirements if present, then a few known extras
    [[ -f "$repo/requirements.txt" ]] && "$venv/bin/pip" install -r "$repo/requirements.txt"
    "$venv/bin/pip" install xgboost gputil psutil plotly tenacity h5py joblib || true

    if "$venv/bin/python" -c "import numpy, pandas, tensorflow, sklearn" 2>/dev/null; then
        echo "   ✅ Core packages OK"
    else
        echo "   ⚠️  Verifying/fixing numpy..."; "$venv/bin/pip" install --force-reinstall numpy
    fi
    P2_PYTHON="$(realpath "$venv/bin/python")"
}

# Pre-create the MalDataGen output tree so it never races on mkdir / logging.log
precreate_p2_dirs() {   # precreate_p2_dirs <repo> <out_subdir> <model> <dataset_name>
    local base="$1/$2/$3/$4"
    mkdir -p "$base"/{Logs,ModelsSaved,GraphsSaved} \
             "$base"/DataGenerated/{Training,Validation,Test}
    touch "$base/Logs/logging.log" 2>/dev/null || true
}

# Copy reduced CSVs from a source dir INTO the MalDataGen Datasets/ (replaces
# move_datasets.sh). Non-destructive (cp, keeps the source) and skips Git-LFS
# pointers. No .gitattributes/LFS toggling: a plain cp into Datasets/ does not
# pass through git's LFS filters, and this pipeline never commits.
p2_import_datasets() {   # p2_import_datasets <src_dir> <dest_datasets_dir>
    local src="$1" dest="$2"
    [[ -d "$src" ]] || { echo "❌ --p2-import source not found: $src"; return 1; }
    mkdir -p "$dest"

    mapfile -t SRC_CSVS < <(cd "$src" 2>/dev/null && ls -1 *.csv 2>/dev/null | sort)
    [[ ${#SRC_CSVS[@]} -gt 0 ]] || { echo "❌ no CSVs to import from $src"; return 1; }

    local existing; existing=$(cd "$dest" 2>/dev/null && ls -1 *.csv 2>/dev/null | wc -l)
    if [[ "${existing:-0}" -gt 0 ]]; then
        echo "⚠  $dest already has $existing CSV(s); import overwrites matching names."
        confirm "   Continue importing into $dest?" || { echo "   import skipped by user"; return 1; }
    fi

    echo "📥 Importing reduced CSVs:  $src  →  $dest"
    local copied=0 skipped=0
    for f in "${SRC_CSVS[@]}"; do
        if is_lfs_pointer "$src/$f"; then
            echo "   ⚠️  skip $f (Git-LFS pointer, not real data)"; skipped=$((skipped+1)); continue
        fi
        if cp -f "$src/$f" "$dest/$f"; then echo "   ✅ $f"; copied=$((copied+1))
        else                                echo "   ❌ failed: $f"; fi
    done
    echo "   → imported $copied, skipped $skipped (LFS pointers)"
    [[ $copied -gt 0 ]]
}

run_p2() {
    echo ""
    echo "=========================================================="
    echo "  PROTOCOL P2 (external) — MalDataGen AE + VAE (Table 4/Fig 2)"
    echo "=========================================================="
    echo "⚠  This clones an external repo, installs its dependencies, and can take MANY HOURS."
    echo "⚠  MalDataGen exposes no --seed flag: P2 is not bit-for-bit reproducible."
    confirm "Continue with Protocol P2?" || { echo "P2 skipped."; STATUS+=("⏭  [P2] skipped by user"); return; }

    check_git

    # 1) clone / pin MalDataGen ------------------------------------------------
    if [[ ! -d "$P2_REPO_DIR" ]]; then
        echo "📦 Cloning MalDataGen..."
        git clone "$P2_REPO_URL" "$P2_REPO_DIR" || { STATUS+=("❌ [P2] clone failed"); return; }
    fi
    if [[ "$PIN_COMMIT" != "PUT_PINNED_COMMIT_SHA_HERE" ]]; then
        ( cd "$P2_REPO_DIR" && git checkout "$PIN_COMMIT" ) || echo "⚠  could not checkout $PIN_COMMIT"
    else
        echo "⚠  No pinned commit set (use --pin-commit <SHA>); running the current checkout."
    fi

    # 1b) optional: import reduced CSVs into Datasets/ (replaces move_datasets.sh)
    if [[ -n "$P2_IMPORT" ]]; then
        p2_import_datasets "$P2_IMPORT" "$P2_REPO_DIR/Datasets" \
            || { STATUS+=("❌ [P2] dataset import failed"); return; }
        P2_DATA_DIR=""   # after importing, always read from the repo's Datasets/
    fi

    # 1c) normalise --p2-data-dir to an ABSOLUTE path: the training loop runs from
    #     inside the repo (cd), so a relative dir would resolve against the wrong cwd.
    if [[ -n "$P2_DATA_DIR" ]]; then
        if [[ -d "$P2_DATA_DIR" ]]; then P2_DATA_DIR="$(realpath "$P2_DATA_DIR")"
        else echo "❌ --p2-data-dir not found: $P2_DATA_DIR"; STATUS+=("❌ [P2] bad --p2-data-dir"); return; fi
    fi

    # 2) datasets: reduced CSVs must be in the MalDataGen Datasets dir ---------
    local ds_dir="${P2_DATA_DIR:-$P2_REPO_DIR/Datasets}"
    mkdir -p "$ds_dir"
    mapfile -t DATASETS < <(cd "$ds_dir" 2>/dev/null && ls -1 *.csv 2>/dev/null | sort)
    if [[ ${#DATASETS[@]} -eq 0 ]]; then
        echo "❌ No CSVs in $ds_dir — place the reduced datasets (statistical_/rfe_/semidroid_/lasso_*) there."
        STATUS+=("❌ [P2] no datasets in $ds_dir"); return
    fi

    # 2b) reject Git-LFS pointers (they look like CSVs but hold no data) -------
    echo "🔍 Checking $ds_dir for Git-LFS pointers..."
    local has_lfs=0
    for f in "${DATASETS[@]}"; do
        if is_lfs_pointer "$ds_dir/$f"; then
            echo "   ⚠️  $f is an LFS pointer, not real data!"; has_lfs=1
        fi
    done
    if [[ $has_lfs -eq 1 ]]; then
        echo "❌ Some P2 datasets are LFS pointers. Run 'git lfs pull' where they were fetched."
        STATUS+=("❌ [P2] LFS pointers in $ds_dir"); return
    fi
    echo "   ✅ ${#DATASETS[@]} real CSV(s) found"

    # print inventory grouped by feature-selection method ---------------------
    echo "📊 P2 datasets (by feature-selection method):"
    for f in "${DATASETS[@]}"; do
        local sz ln
        sz=$(du -h "$ds_dir/$f" 2>/dev/null | cut -f1 || echo "?")
        ln=$(wc -l < "$ds_dir/$f" 2>/dev/null || echo "?")
        printf "   [%-11s] %-40s %6s  %s lines\n" "$(fs_method_of "$f")" "$f" "$sz" "$ln"
    done

    # 3) venv + deps ----------------------------------------------------------
    setup_p2_venv "$P2_REPO_DIR"

    # 4) plan + confirm -------------------------------------------------------
    local total=$(( ${#DATASETS[@]} * 2 )) n=0 fails=0
    echo ""
    echo "📊 P2 experiments: $total  (${#DATASETS[@]} datasets × {AE, VAE})"
    echo "   Estimated time: several hours per experiment."
    confirm "Start P2 training now?" || { STATUS+=("⏭  [P2] training skipped"); return; }

    local p2_log_dir="$OUT_DIR/p2_logs_$(date '+%Y%m%d_%H%M%S')"; mkdir -p "$p2_log_dir"
    local p2_summary="$p2_log_dir/summary.txt"
    {
        echo "P2 (MalDataGen TSTR) SUMMARY"
        echo "============================"
        echo "Date        : $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Datasets    : ${#DATASETS[@]}"
        echo "Experiments : $total (AE + VAE)"
        echo "Pinned SHA  : $PIN_COMMIT"
        echo ""
    } > "$p2_summary"

    # 5) pre-create every output dir (defensive, before any training) ---------
    echo "🔧 Pre-creating MalDataGen output tree..."
    for MODEL in autoencoder variational; do
        for DATASET in "${DATASETS[@]}"; do
            precreate_p2_dirs "$P2_REPO_DIR" "$P2_OUT_SUBDIR" "$MODEL" "${DATASET%.csv}"
        done
    done
    echo "   ✅ directories ready"

    # 6) run — from inside the repo, using the venv python --------------------
    export TF_ENABLE_ONEDNN_OPTS=0
    local START; START=$(date +%s)

    ( cd "$P2_REPO_DIR" && {
        for MODEL in autoencoder variational; do
            [[ "$MODEL" == autoencoder ]] && CONFIG="$AE_CONFIG" || CONFIG="$VAE_CONFIG"
            echo ""
            echo "╔════════════════════════════════════════════════════╗"
            [[ "$MODEL" == autoencoder ]] \
                && echo "║  MODEL: AUTOENCODER (AE)                            ║" \
                || echo "║  MODEL: VARIATIONAL AUTOENCODER (VAE)               ║"
            echo "╚════════════════════════════════════════════════════╝"

            for DATASET in "${DATASETS[@]}"; do
                n=$((n+1)); DNAME="${DATASET%.csv}"; FS="$(fs_method_of "$DATASET")"

                # resolve the dataset path MalDataGen should read
                if [[ -n "$P2_DATA_DIR" ]]; then DPATH="$P2_DATA_DIR/$DATASET"
                else                             DPATH="Datasets/$DATASET"; fi

                local OUTPUT_DIR="$P2_OUT_SUBDIR/$MODEL/$DNAME"
                local LOG="$OLDPWD/$p2_log_dir/${MODEL}_${DNAME}.log"

                echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                echo "📊 [$n/$total]  $MODEL / $DNAME   (FS: $FS)"
                echo "   log: $LOG"
                echo "   ⚠️  may take several HOURS"

                if [[ ! -f "$DPATH" ]]; then
                    echo "   ⚠️  dataset not found: $DPATH — skipping"; fails=$((fails+1)); continue
                fi

                local ES; ES=$(date +%s)
                "$P2_PYTHON" main.py \
                    --model_type "$MODEL" \
                    --data_load_path_file_input "$DPATH" \
                    --output_dir "$OUTPUT_DIR" \
                    $CONFIG 2>&1 | tee "$LOG"
                local rc=${PIPESTATUS[0]}
                local dur=$(( $(date +%s) - ES ))

                if [[ $rc -eq 0 ]]; then
                    echo "   ✅ done in $((dur/3600))h $((dur%3600/60))m $((dur%60))s"
                    printf "✅ %-13s %-11s %-30s %dh %dm\n" \
                        "$MODEL" "$FS" "$DNAME" "$((dur/3600))" "$((dur%3600/60))" >> "$OLDPWD/$p2_summary"
                else
                    echo "   ❌ exit $rc after $((dur/60))m $((dur%60))s"
                    printf "❌ %-13s %-11s %-30s (exit %d)\n" \
                        "$MODEL" "$FS" "$DNAME" "$rc" >> "$OLDPWD/$p2_summary"
                    fails=$((fails+1))
                fi
            done
        done
    } )

    # the training loop ran in a subshell, so recover the failure count
    # from the summary file (the subshell's `fails` does not propagate back)
    local fails; fails=$(grep -c '^❌' "$p2_summary" 2>/dev/null || echo 0)

    local DUR=$(( $(date +%s) - START ))
    {
        echo ""
        echo "Total P2 time : $((DUR/86400))d $((DUR%86400/3600))h $((DUR%3600/60))m"
        echo "Failures      : $fails/$total"
    } >> "$p2_summary"

    echo ""; echo "📄 P2 summary:"; cat "$p2_summary"
    echo "📋 P2 logs   : $p2_log_dir"
    echo "📁 P2 outputs: $P2_REPO_DIR/$P2_OUT_SUBDIR"

    if [[ $fails -eq 0 ]]; then STATUS+=("✅ [P2] AE+VAE ($total experiments)")
    else                        STATUS+=("❌ [P2] AE+VAE ($fails/$total failed)"); fi
}

# ==========================================================================
#  Main
# ==========================================================================
echo "=========================================================="
echo "Statistical Ranking — reproduction"
echo "=========================================================="
echo "Repo root : $REPO_ROOT"
if [[ $FEATURE_SELECTION -eq 1 ]]; then
    echo "Data folder : $DATA_FOLDER  (--feature-selection)"
elif [[ $SKIP_CLAIMS -eq 1 ]]; then
    echo "Data dir  : (not needed — --p2-only)"
else
    echo "Data dir  : $DATA_DIR"
fi
echo "Output    : $OUT_DIR"
echo "Mode      : $(
    if   [[ $FEATURE_SELECTION -eq 1 ]]; then echo FEATURE-SELECTION
    elif [[ $SKIP_CLAIMS -eq 1 ]];       then echo P2-ONLY
    elif [[ $MINIMAL -eq 1 ]];           then echo MINIMAL
    else                                       echo FULL
    fi
)$([[ $WITH_P2 -eq 1 && $SKIP_CLAIMS -eq 0 ]] && echo ' +P2')"
echo ""
echo "Security  : in-repo claims run locally, need no root/sudo and no network."
echo "            --with-p2/--p2-only clones an external repo and installs deps (isolated venv by default)."
echo ""

check_python
install_dependencies

if [[ $FEATURE_SELECTION -eq 0 ]] && [[ $SKIP_CLAIMS -eq 0 ]] && [[ ! -d "$DATA_DIR" ]] && [[ "$ONLY" != "pvalue" ]]; then
    echo "❌ Data directory not found: $DATA_DIR"
    echo "   Get the datasets from https://github.com/AILabs4All/datasets-mh30plus (Git LFS),"
    echo "   or pass --data-dir. (The Wilcoxon audit 'pvalue' needs no data; --p2-only needs none either.)"
    exit 1
fi

confirm "Proceed?" || { echo "Cancelled."; exit 1; }

START=$(date +%s)
if [[ $FEATURE_SELECTION -eq 1 ]]; then run_feature_selection
elif [[ $SKIP_CLAIMS -eq 1 ]]; then :   # --p2-only: in-repo claims intentionally skipped
elif [[ $MINIMAL -eq 1 ]]; then run_minimal
else run_claims; fi
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