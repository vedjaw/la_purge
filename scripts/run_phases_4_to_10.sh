#!/usr/bin/env bash
###############################################################################
# PURGE — Resume Phases 4–10 (full settings)
#
# Use this if run_all_new_experiments.sh stopped after Phase 3.
# Run from 17_purge/ root:
#
#   bash scripts/run_phases_4_to_10.sh
#   bash scripts/run_phases_4_to_10.sh --force   # re-run even if outputs exist
#
# Prerequisites (must already exist from Phases 0–3):
#   checkpoints/base_cifar10_resnet18.pth
#   results/multiseed/cifar10_c0_s42/purge_*.pth   (from Phase 1)
###############################################################################
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="./data"
CKPT_DIR="./checkpoints"
RESULTS_DIR="./results"
LOG_DIR="$RESULTS_DIR/phase_logs"
FORCE=false

for arg in "$@"; do
    case $arg in
        --force) FORCE=true ;;
    esac
done

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

require_file() {
    local path="$1"
    local msg="$2"
    if [ ! -f "$path" ]; then
        log "ERROR: $msg"
        log "  Missing: $path"
        exit 1
    fi
}

skip_if_exists() {
    local path="$1"
    if [ "$FORCE" = false ] && [ -e "$path" ]; then
        log "[SKIP] Output exists: $path"
        return 0
    fi
    return 1
}

find_purge_ckpt() {
    find "$RESULTS_DIR/multiseed/cifar10_c0_s42" -name "purge_*.pth" 2>/dev/null | head -1
}

log "============================================================"
log " PURGE Phases 4–10 (full resume)"
log "  Data:        $DATA_DIR"
log "  Checkpoints: $CKPT_DIR"
log "  Results:     $RESULTS_DIR"
log "  Force:       $FORCE"
log "============================================================"

# --- prerequisites ---
require_file "$CKPT_DIR/base_cifar10_resnet18.pth" \
    "CIFAR-10 base checkpoint required (Phase 0A)."

PURGE_CKPT="$(find_purge_ckpt || true)"
if [ -z "$PURGE_CKPT" ]; then
    log "WARNING: No PURGE checkpoint in results/multiseed/cifar10_c0_s42/"
    log "  Phases 5, 9, 10 will be skipped."
    log "  Re-run Phase 1 for CIFAR-10 class 0 seed 42 first."
else
    log "PURGE checkpoint: $PURGE_CKPT"
fi

###############################################################################
# PHASE 4: Per-step retain-loss trajectory (full: 5 epochs, all steps logged)
###############################################################################
log ""
log "============================================================"
log " PHASE 4: Per-Step Retain-Loss Trajectory"
log "============================================================"
OUT4="$RESULTS_DIR/perstep/perstep_cifar10_c0_s42.csv"
if ! skip_if_exists "$OUT4"; then
    python experiments/exp_perstep_trajectory.py \
        --dataset cifar10 --forget_class 0 \
        --checkpoint "$CKPT_DIR/base_cifar10_resnet18.pth" \
        --data_dir "$DATA_DIR" \
        --out_dir "$RESULTS_DIR/perstep" \
        --epochs 5 --seed 42 \
        2>&1 | tee "$LOG_DIR/phase4_perstep.log"
fi

###############################################################################
# PHASE 5: Fine-tuning recovery attack (full: 50 samples, 10 FT epochs)
###############################################################################
log ""
log "============================================================"
log " PHASE 5: Fine-Tuning Recovery Attack"
log "============================================================"
OUT5="$RESULTS_DIR/finetune_recovery/recovery_cifar10_c0.csv"
if [ -n "$PURGE_CKPT" ]; then
    if ! skip_if_exists "$OUT5"; then
        python experiments/exp_finetune_recovery.py \
            --dataset cifar10 --forget_class 0 \
            --checkpoint "$PURGE_CKPT" \
            --data_dir "$DATA_DIR" \
            --out_dir "$RESULTS_DIR/finetune_recovery" \
            --n_samples 50 --ft_epochs 10 --seed 42 \
            2>&1 | tee "$LOG_DIR/phase5_finetune_recovery.log"
    fi
else
    log "[SKIP] Phase 5 — no PURGE checkpoint."
fi

###############################################################################
# PHASE 6: FID (PURGE at minimum; add baselines manually if you have ckpts)
###############################################################################
log ""
log "============================================================"
log " PHASE 6: Feature-Space FID"
log "============================================================"
OUT6="$LOG_DIR/phase6_fid.log"
if [ -n "$PURGE_CKPT" ]; then
    if ! skip_if_exists "$OUT6"; then
        python experiments/fid_baselines.py \
            --reference "$CKPT_DIR/base_cifar10_resnet18.pth" \
            --data_dir "$DATA_DIR" \
            --forget_class 0 \
            --checkpoints "purge:$PURGE_CKPT" \
            2>&1 | tee "$LOG_DIR/phase6_fid.log" || \
            log "WARNING: Phase 6 FID failed (non-fatal). Add baseline ckpts and re-run."
    fi
else
    log "[SKIP] Phase 6 — no PURGE checkpoint."
    log "  Manual: python experiments/fid_baselines.py \\"
    log "    --reference $CKPT_DIR/base_cifar10_resnet18.pth \\"
    log "    --checkpoints purge:<ckpt> salun:<ckpt> scrub:<ckpt>"
fi

###############################################################################
# PHASE 7: Sequential unlearning (full: classes 0 -> 1 -> 2)
###############################################################################
log ""
log "============================================================"
log " PHASE 7: Sequential Unlearning (class 0 -> 1 -> 2)"
log "============================================================"
OUT7="$RESULTS_DIR/sequential/sequential_summary.csv"
if ! skip_if_exists "$OUT7"; then
    python experiments/exp_sequential_unlearn.py \
        --dataset cifar10 \
        --forget_classes 0 1 2 \
        --base_ckpt "$CKPT_DIR/base_cifar10_resnet18.pth" \
        --config configs/cifar10_kl_retain.yaml \
        --data_dir "$DATA_DIR" \
        --save_dir "$RESULTS_DIR/sequential" \
        --seed 42 \
        2>&1 | tee "$LOG_DIR/phase7_sequential.log"
fi

###############################################################################
# PHASE 8: Inner-product histogram (full: 500 steps)
###############################################################################
log ""
log "============================================================"
log " PHASE 8: Inner-Product Histogram (Theorem 2 support)"
log "============================================================"
OUT8="$RESULTS_DIR/innerproduct/innerproducts_cifar10_c0.csv"
if ! skip_if_exists "$OUT8"; then
    python experiments/exp_innerproduct_histogram.py \
        --dataset cifar10 --forget_class 0 \
        --checkpoint "$CKPT_DIR/base_cifar10_resnet18.pth" \
        --data_dir "$DATA_DIR" \
        --out_dir "$RESULTS_DIR/innerproduct" \
        --max_steps 500 --seed 42 \
        2>&1 | tee "$LOG_DIR/phase8_innerproduct.log"
fi

###############################################################################
# PHASE 9: BN recalibration (corrected normalisation)
###############################################################################
log ""
log "============================================================"
log " PHASE 9: BN Recalibration"
log "============================================================"
OUT9="$RESULTS_DIR/bn_recalibration_corrected.csv"
if [ -n "$PURGE_CKPT" ]; then
    if ! skip_if_exists "$OUT9"; then
        python experiments/bn_recalibration.py \
            --checkpoints "purge:$PURGE_CKPT" \
            --data_dir "$DATA_DIR" \
            --forget_class 0 \
            --out_csv "$OUT9" \
            2>&1 | tee "$LOG_DIR/phase9_bn_recalibration.log"
    fi
else
    log "[SKIP] Phase 9 — no PURGE checkpoint."
fi

###############################################################################
# PHASE 10: Linear probe (representation leakage)
###############################################################################
log ""
log "============================================================"
log " PHASE 10: Linear Probe"
log "============================================================"
if [ -n "$PURGE_CKPT" ]; then
    if [ "$FORCE" = true ] || [ ! -f "$LOG_DIR/phase10_linear_probe.log" ]; then
        python experiments/linear_probe.py \
            --checkpoints \
                "base:$CKPT_DIR/base_cifar10_resnet18.pth" \
                "purge:$PURGE_CKPT" \
            --data_dir "$DATA_DIR" --forget_class 0 \
            2>&1 | tee "$LOG_DIR/phase10_linear_probe.log"
    else
        log "[SKIP] Phase 10 log exists: $LOG_DIR/phase10_linear_probe.log"
    fi
else
    log "[SKIP] Phase 10 — no PURGE checkpoint."
fi

###############################################################################
# SUMMARY
###############################################################################
log ""
log "============================================================"
log " PHASES 4–10 COMPLETE"
log "============================================================"
log " Logs:    $LOG_DIR/"
log " Results: $RESULTS_DIR/"
log ""
log " Key outputs:"
log "   $RESULTS_DIR/perstep/perstep_cifar10_c0_s42.csv"
log "   $RESULTS_DIR/finetune_recovery/recovery_cifar10_c0.csv"
log "   $RESULTS_DIR/sequential/sequential_summary.csv"
log "   $RESULTS_DIR/innerproduct/innerproducts_cifar10_c0.csv"
log "   $RESULTS_DIR/bn_recalibration_corrected.csv"
log "============================================================"
