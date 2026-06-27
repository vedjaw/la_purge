#!/usr/bin/env bash
###############################################################################
# PURGE — Master Experiment Runner
#
# Runs all experiments needed for the paper revision in the correct order.
# Transfer the entire la_purge/ directory to the GPU server, then run:
#
#   bash scripts/run_all_new_experiments.sh [--skip-training] [--force-training] [--quick]
#
# Flags:
#   --skip-training    Skip Phase 0 entirely (no base-model training)
#   --force-training   Retrain all base models even if checkpoints exist
#   --quick            Run minimal configs (1 seed, 1 class per dataset)
#
# Phase 0 training policy (via experiments/train_base_all.py):
#   - Skip datasets whose checkpoint already exists (unless --force-training)
#   - Early-stop: per-dataset test-acc threshold OR plateau (no improvement)
#   - Per-dataset LR (ImageNet-pretrained fine-tuning):
#       CIFAR-10 : lr=0.1
#       CIFAR-100 / MNIST / SVHN / STL-10 / PathMNIST : lr=0.01
#   - CIFAR-100: 150 epochs, threshold 68%, patience 25
#
# Estimated total GPU time (A100, with early-stop + existing ckpts):
#   Full run (incl. Phase 0):  ~25-40 GPU-hours
#   Full run (--skip-training): ~35-45 GPU-hours
#   Quick run:                 ~8-12 GPU-hours
#
# Prerequisites:
#   pip install -r requirements.txt
###############################################################################
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="./data"
CKPT_DIR="./checkpoints"
RESULTS_DIR="./results"

SKIP_TRAINING=false
FORCE_TRAINING=false
QUICK=false
for arg in "$@"; do
    case $arg in
        --skip-training)  SKIP_TRAINING=true ;;
        --force-training) FORCE_TRAINING=true ;;
        --quick)          QUICK=true ;;
    esac
done

TRAIN_FORCE_FLAG=()
if [ "$FORCE_TRAINING" = true ]; then
    TRAIN_FORCE_FLAG=(--force)
fi

echo "============================================================"
echo " PURGE Experiment Suite"
echo "  Data:        $DATA_DIR"
echo "  Checkpoints: $CKPT_DIR"
echo "  Results:     $RESULTS_DIR"
echo "  Skip train:  $SKIP_TRAINING"
echo "  Force train: $FORCE_TRAINING"
echo "  Quick mode:  $QUICK"
echo "============================================================"

mkdir -p "$CKPT_DIR" "$RESULTS_DIR"

# Report which base checkpoints already exist.
report_base_ckpts() {
    local tag="$1"   # "" or "_groupnorm"
    local label="$2"
    echo ""
    echo "  $label:"
    for ds in cifar10 mnist svhn stl10 pathmnist cifar100; do
        local ckpt="$CKPT_DIR/base_${ds}_resnet18${tag}.pth"
        if [ -f "$ckpt" ]; then
            echo "    [exists]  $ckpt"
        else
            echo "    [missing] $ckpt"
        fi
    done
}

# Train base models: skips existing checkpoints, per-dataset LR + early stop.
# Extra args are forwarded to train_base_all.py (e.g. --datasets, --use_groupnorm).
run_base_training() {
    local title="$1"
    shift
    echo ""
    echo "============================================================"
    echo " $title"
    echo "============================================================"
    echo "  skip existing | threshold + plateau stop | lr 0.1 (CIFAR-10) / 0.01 (others)"
    python experiments/train_base_all.py \
        --data_dir "$DATA_DIR" --save_dir "$CKPT_DIR" \
        --seed 42 \
        "$@" \
        "${TRAIN_FORCE_FLAG[@]}"
}

###############################################################################
# PHASE 0: Train base models (BN + GN) — only missing checkpoints
###############################################################################
if [ "$SKIP_TRAINING" = false ]; then
    echo ""
    echo "============================================================"
    echo " PHASE 0: Base model checkpoint status"
    echo "============================================================"
    report_base_ckpts "" "BN base models"
    report_base_ckpts "_groupnorm" "GroupNorm base models"

    run_base_training "PHASE 0A: Train BN base models (skip existing)" \
        --datasets cifar10 mnist svhn stl10 pathmnist \
        --epochs 200

    run_base_training "PHASE 0B: Train GroupNorm base models (skip existing)" \
        --datasets cifar10 mnist svhn stl10 pathmnist \
        --epochs 200 --use_groupnorm

    run_base_training "PHASE 0C: Train CIFAR-100 base model (skip if exists)" \
        --datasets cifar100 \
        --epochs 150 --min_epochs 10 --patience 25
else
    echo ""
    echo "[SKIP] Phase 0 base-model training (--skip-training)"
fi

###############################################################################
# PHASE 1: 3-seed runs on all 5 datasets  [CRITICAL — Expt 2.1]
###############################################################################
echo ""
echo "============================================================"
echo " PHASE 1: 3-Seed Multi-Dataset PURGE Runs"
echo "============================================================"
if [ "$QUICK" = true ]; then
    python experiments/exp_multiseed_all.py \
        --datasets cifar10 mnist svhn stl10 pathmnist \
        --seeds 42 123 456 \
        --ckpt_dir "$CKPT_DIR" --data_dir "$DATA_DIR" \
        --save_dir "$RESULTS_DIR/multiseed" \
        --classes_per_dataset 1
else
    python experiments/exp_multiseed_all.py \
        --datasets cifar10 mnist svhn stl10 pathmnist \
        --seeds 42 123 456 \
        --ckpt_dir "$CKPT_DIR" --data_dir "$DATA_DIR" \
        --save_dir "$RESULTS_DIR/multiseed"
fi

###############################################################################
# PHASE 2: GroupNorm multi-dataset + 3 seeds  [CRITICAL — Expt 2.2]
###############################################################################
echo ""
echo "============================================================"
echo " PHASE 2: GroupNorm Multi-Dataset (3 seeds)"
echo "============================================================"
if [ "$QUICK" = true ]; then
    python experiments/exp_groupnorm_multi.py \
        --datasets cifar10 mnist svhn \
        --seeds 42 123 456 \
        --ckpt_dir "$CKPT_DIR" --data_dir "$DATA_DIR" \
        --save_dir "$RESULTS_DIR/groupnorm_multi"
else
    python experiments/exp_groupnorm_multi.py \
        --datasets cifar10 mnist svhn stl10 pathmnist \
        --seeds 42 123 456 \
        --ckpt_dir "$CKPT_DIR" --data_dir "$DATA_DIR" \
        --save_dir "$RESULTS_DIR/groupnorm_multi"
fi

###############################################################################
# PHASE 3: CIFAR-100 evaluation  [CRITICAL — Expt 2.4]
###############################################################################
echo ""
echo "============================================================"
echo " PHASE 3: CIFAR-100 Unlearning (10 classes)"
echo "============================================================"
if [ -f "$CKPT_DIR/base_cifar100_resnet18.pth" ]; then
    for FC in 0 10 20 30 40 50 60 70 80 90; do
        echo "  Forgetting class $FC..."
        python run.py \
            --config configs/cifar100_kl_retain.yaml \
            --checkpoint "$CKPT_DIR/base_cifar100_resnet18.pth" \
            --data_dir "$DATA_DIR" \
            --save_dir "$RESULTS_DIR/cifar100/class_${FC}" \
            --forget_class "$FC" --seed 42 \
            2>&1 | tee "$RESULTS_DIR/cifar100/class_${FC}_log.txt"
    done
else
    echo "  [SKIP] CIFAR-100 base checkpoint not found. Run Phase 0 first."
fi

###############################################################################
# PHASE 4: Per-step retain-loss trajectory  [HIGH — Expt 2.5]
###############################################################################
echo ""
echo "============================================================"
echo " PHASE 4: Per-Step Retain-Loss Trajectory"
echo "============================================================"
python experiments/exp_perstep_trajectory.py \
    --dataset cifar10 --forget_class 0 \
    --checkpoint "$CKPT_DIR/base_cifar10_resnet18.pth" \
    --data_dir "$DATA_DIR" \
    --out_dir "$RESULTS_DIR/perstep" \
    --epochs 3 --seed 42

###############################################################################
# PHASE 5: Fine-tuning recovery attack  [HIGH — Expt 2.6]
###############################################################################
echo ""
echo "============================================================"
echo " PHASE 5: Fine-Tuning Recovery Attack"
echo "============================================================"
# Find the PURGE checkpoint from Phase 1 (seed 42)
PURGE_CKPT=$(find "$RESULTS_DIR/multiseed/cifar10_c0_s42" -name "purge_*.pth" 2>/dev/null | head -1)
if [ -n "$PURGE_CKPT" ]; then
    python experiments/exp_finetune_recovery.py \
        --dataset cifar10 --forget_class 0 \
        --checkpoint "$PURGE_CKPT" \
        --data_dir "$DATA_DIR" \
        --out_dir "$RESULTS_DIR/finetune_recovery" \
        --n_samples 50 --ft_epochs 10 --seed 42
else
    echo "  [SKIP] No PURGE checkpoint found. Run Phase 1 first."
fi

###############################################################################
# PHASE 6: FID for all baselines  [HIGH — Expt 2.7]
###############################################################################
echo ""
echo "============================================================"
echo " PHASE 6: FID for All Baselines"
echo "============================================================"
echo "  NOTE: This requires baseline checkpoints (SalUn, SCRUB, etc.)"
echo "  Run manually with:"
echo "    python experiments/fid_baselines.py \\"
echo "      --reference $CKPT_DIR/base_cifar10_resnet18.pth \\"
echo "      --checkpoints purge:<ckpt> salun:<ckpt> scrub:<ckpt> \\"
echo "      --data_dir $DATA_DIR"

###############################################################################
# PHASE 7: Sequential unlearning  [HIGH — Expt 2.8]
###############################################################################
echo ""
echo "============================================================"
echo " PHASE 7: Sequential Unlearning (class 0 -> 1 -> 2)"
echo "============================================================"
python experiments/exp_sequential_unlearn.py \
    --dataset cifar10 \
    --forget_classes 0 1 2 \
    --base_ckpt "$CKPT_DIR/base_cifar10_resnet18.pth" \
    --config configs/cifar10_kl_retain.yaml \
    --data_dir "$DATA_DIR" \
    --save_dir "$RESULTS_DIR/sequential" \
    --seed 42

###############################################################################
# PHASE 8: Inner-product histogram  [MEDIUM — Expt 2.14]
###############################################################################
echo ""
echo "============================================================"
echo " PHASE 8: Inner-Product Histogram (Theorem 2 support)"
echo "============================================================"
python experiments/exp_innerproduct_histogram.py \
    --dataset cifar10 --forget_class 0 \
    --checkpoint "$CKPT_DIR/base_cifar10_resnet18.pth" \
    --data_dir "$DATA_DIR" \
    --out_dir "$RESULTS_DIR/innerproduct" \
    --max_steps 200 --seed 42

###############################################################################
# PHASE 9: BN recalibration on corrected preprocessing  [CRITICAL — Expt 2.3]
###############################################################################
echo ""
echo "============================================================"
echo " PHASE 9: BN Recalibration (corrected normalisation)"
echo "============================================================"
if [ -n "$PURGE_CKPT" ]; then
    python experiments/bn_recalibration.py \
        --checkpoints "purge:$PURGE_CKPT" \
        --data_dir "$DATA_DIR" \
        --forget_class 0 \
        --out_csv "$RESULTS_DIR/bn_recalibration_corrected.csv"
else
    echo "  [SKIP] No PURGE checkpoint found. Run Phase 1 first."
fi

###############################################################################
# PHASE 10: Linear probe (with fixed script)
###############################################################################
echo ""
echo "============================================================"
echo " PHASE 10: Linear Probe (representation leakage)"
echo "============================================================"
if [ -n "$PURGE_CKPT" ]; then
    python experiments/linear_probe.py \
        --checkpoints \
            "base:$CKPT_DIR/base_cifar10_resnet18.pth" \
            "purge:$PURGE_CKPT" \
        --data_dir "$DATA_DIR" --forget_class 0
else
    echo "  [SKIP] No PURGE checkpoint. Run Phase 1 first."
fi

###############################################################################
# SUMMARY
###############################################################################
echo ""
echo "============================================================"
echo " ALL PHASES COMPLETE"
echo "============================================================"
echo " Results saved under: $RESULTS_DIR/"
echo ""
echo " Key output files:"
echo "   $RESULTS_DIR/multiseed/multiseed_all.csv"
echo "   $RESULTS_DIR/groupnorm_multi/groupnorm_multi_summary.csv"
echo "   $RESULTS_DIR/perstep/perstep_cifar10_c0_s42.csv"
echo "   $RESULTS_DIR/sequential/sequential_summary.csv"
echo "   $RESULTS_DIR/finetune_recovery/recovery_cifar10_c0.csv"
echo "   $RESULTS_DIR/innerproduct/innerproducts_cifar10_c0.csv"
echo ""
echo " Manual steps remaining:"
echo "   1. Run FID with baseline checkpoints (Phase 6)"
echo "   2. Implement SSD/SISA baselines (external code needed)"
echo "   3. Add ViT-S/16 support to run.py and re-run"
echo "   4. Run LiRA/RMIA stronger MIA (requires shadow model training)"
echo "============================================================"
