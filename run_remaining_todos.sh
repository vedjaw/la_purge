#!/usr/bin/env bash
# =============================================================================
# Close the runnable TODOs left in paper/purge_tmlr.tex.
#
# Every result the paper still marks as pending was measured under the
# SUPERSEDED optimiser (before the projection was moved to the post-momentum
# update). This script regenerates them with the current implementation.
#
#   A  CIFAR-10 unlearned checkpoints (BN + GroupNorm)   [prerequisite]
#   B  BN-recalibration diagnostic  (Delta F)            -> todo 1
#   C  Component ablation, fixed-budget + stopping modes -> todo 2
#   D  Epsilon sweeps A and B                            -> todo 3
#   E  Sequential unlearning                             -> todo 5
#   F  Fine-tuning recovery                              -> todo 5
#   G  Forget-objective comparison (GA/kl_retain/kl_unif)-> todo 8
#   H  Direct projected-gradient baseline                -> reviewer ask #3
#
# NOT covered (cannot be a shell command):
#   todo 4  Patel & Qiu empirical row -- needs their method reimplemented.
#   todo 6  trajectory sweep -- ALREADY DONE.
#   todo 7  anonymised repo URL -- paste the 4open link into the paper.
#
# Usage:  bash run_remaining_todos.sh
#         PYTHON=python3 bash run_remaining_todos.sh
# =============================================================================

set -u
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="results/todo_logs"; mkdir -p "$LOG_DIR"
MASTER="$LOG_DIR/todos_${STAMP}.log"

CFG=configs/cifar10_kl_retain.yaml
BASE=checkpoints/base_cifar10_resnet18.pth
BASE_GN=checkpoints/base_cifar10_resnet18_groupnorm.pth
DATA=data

NAMES=(); CODES=()

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER"; }

phase(){
  local name="$1"; shift
  local lg="$LOG_DIR/${name}_${STAMP}.log"
  log "===================================================================="
  log "START $name"
  log "CMD   $*"
  local t0=$SECONDS
  ( "$@" ) 2>&1 | tee -a "$lg" | tee -a "$MASTER"
  local rc=${PIPESTATUS[0]}
  NAMES+=("$name"); CODES+=("$rc")
  if [ "$rc" -eq 0 ]; then log "DONE  $name ($((SECONDS-t0))s)"
  else
    log "FAIL  $name (exit=$rc) -- last 15 lines:"
    tail -15 "$lg" | sed 's/^/      | /' | tee -a "$MASTER"
  fi
}

# newest .pth in a directory (run.py's filename encodes its hyper-parameters)
newest_ckpt(){ ls -t "$1"/*.pth 2>/dev/null | head -1; }

for f in "$CFG" "$BASE"; do
  [ -e "$f" ] || { log "FATAL missing $f"; exit 1; }
done
[ -e "$BASE_GN" ] || log "WARN  $BASE_GN missing -- GroupNorm steps will be skipped"

log "######## remaining TODOs | stamp=$STAMP ########"
T0=$SECONDS

# ---- A. CIFAR-10 unlearned checkpoints (needed by B, F) --------------------
phase A1_purge_cifar10_bn \
  "$PYTHON" -u run.py --config "$CFG" --checkpoint "$BASE" \
    --data_dir "$DATA" --save_dir results/todo/purge_cifar10_bn --seed 42
CK_BN="$(newest_ckpt results/todo/purge_cifar10_bn)"
log "  BN checkpoint: ${CK_BN:-NONE}"

if [ -e "$BASE_GN" ]; then
  phase A2_purge_cifar10_gn \
    "$PYTHON" -u run.py --config "$CFG" --checkpoint "$BASE_GN" \
      --data_dir "$DATA" --save_dir results/todo/purge_cifar10_gn \
      --use_groupnorm --seed 42
  CK_GN="$(newest_ckpt results/todo/purge_cifar10_gn)"
  log "  GN checkpoint: ${CK_GN:-NONE}"
else
  CK_GN=""
fi

# ---- B. BN recalibration (Delta F) ----------------------------------------
# Delta F distinguishes weight-level forgetting (~0) from BN-mediated
# forgetting (>>0). Run BN and GroupNorm separately: the flag changes the
# architecture, so they cannot share one invocation.
if [ -n "${CK_BN:-}" ]; then
  phase B1_bn_recal_bn \
    "$PYTHON" -u experiments/bn_recalibration.py \
      --data_dir "$DATA" --forget_class 0 \
      --checkpoints "purge_bn:$CK_BN" \
      --out_csv results/todo/bn_recal_bn.csv
fi
if [ -n "${CK_GN:-}" ]; then
  phase B2_bn_recal_gn \
    "$PYTHON" -u experiments/bn_recalibration.py \
      --data_dir "$DATA" --forget_class 0 --use_groupnorm \
      --checkpoints "purge_gn:$CK_GN" \
      --out_csv results/todo/bn_recal_gn.csv
fi

# ---- C. Component ablation (both modes) ------------------------------------
# 'fixed' mode disables stopping for every variant so the components are
# compared at equal compute; without it, dual stopping truncates all variants
# early and masks their effects.
phase C_component_ablation \
  "$PYTHON" -u experiments/component_ablation.py --mode both

# ---- D. Epsilon sweeps A and B ---------------------------------------------
phase D_epsilon_sweep \
  "$PYTHON" -u experiments/epsilon_sweep.py --base_ckpt "$BASE" --data_dir "$DATA"

# ---- E. Sequential unlearning ----------------------------------------------
phase E_sequential \
  "$PYTHON" -u experiments/exp_sequential_unlearn.py \
    --dataset cifar10 --forget_classes 0 1 2 --base_ckpt "$BASE" \
    --config "$CFG" --data_dir "$DATA" --save_dir results/todo/sequential

# ---- F. Fine-tuning recovery attack ----------------------------------------
if [ -n "${CK_BN:-}" ]; then
  phase F_finetune_recovery \
    "$PYTHON" -u experiments/exp_finetune_recovery.py \
      --dataset cifar10 --forget_class 0 --checkpoint "$CK_BN" \
      --data_dir "$DATA" --out_dir results/todo/finetune_recovery
fi

# ---- G. Forget-objective comparison ----------------------------------------
# Re-measures the GA / kl_retain / kl_uniform comparison. The figures currently
# in the paper pre-date the optimiser fix and are backed by no result file.
for OBJ in ga kl_retain kl_uniform; do
  phase "G_objective_$OBJ" \
    "$PYTHON" -u run.py --config "$CFG" --checkpoint "$BASE" \
      --data_dir "$DATA" --save_dir "results/todo/objective_$OBJ" \
      --forget_objective "$OBJ" --seed 42
done

# ---- H. Direct projected-gradient baseline ---------------------------------
# The reviewer asked for a direct projected-gradient baseline: A-GEM projection
# on plain gradient ascent, WITHOUT PURGE's KD anchor or retain-confusion
# target. Isolates what the projection alone contributes.
phase H_projection_only \
  "$PYTHON" -u run.py --config "$CFG" --checkpoint "$BASE" \
    --data_dir "$DATA" --save_dir results/todo/projection_only \
    --forget_objective ga --kd_weight 0.0 --seed 42

# ---- summary ---------------------------------------------------------------
log ""
log "############################ SUMMARY ############################"
overall=0
for i in "${!NAMES[@]}"; do
  if [ "${CODES[$i]}" -eq 0 ]; then st=OK; else st="FAIL(${CODES[$i]})"; overall=1; fi
  log "  ${NAMES[$i]}: $st"
done
log "  wall-clock: $((SECONDS-T0))s"
log "  master log: $MASTER"
log ""
log "  Send back:"
log "    results/todo/                       (all CSVs)"
log "    results/ablation/component_ablation_summary.csv"
log "    results/epsilon_sweep/  or  results_sweep/"
log "    results/todo/sequential/, results/todo/finetune_recovery/"
log "  Plus, from each run.py log above, the final TA/FA/RA/MIA block."
log "#################################################################"
exit "$overall"
