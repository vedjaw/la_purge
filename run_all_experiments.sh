#!/usr/bin/env bash
# =============================================================================
# Run all PURGE experiments strictly one-at-a-time (no overlap -> no OOM).
#
# Usage:
#   bash run_all_experiments.sh
#   PYTHON=python3 bash run_all_experiments.sh          # override interpreter
#
# For a long unattended run, launch inside tmux/screen or with nohup:
#   nohup bash run_all_experiments.sh > run_all.out 2>&1 &
#
# Each phase logs to results/run_all_logs/<phase>_<stamp>.log and to a master
# log. A phase failure is recorded but does NOT stop the remaining phases
# (they are independent), and a summary is printed at the end.
# =============================================================================

set -u
cd "$(dirname "$0")"                      # run from repo root

PYTHON="${PYTHON:-python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="results/run_all_logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/run_all_${STAMP}.log"

NAMES=()
CODES=()

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER_LOG"; }

run_phase() {
  local name="$1"; shift
  local phase_log="$LOG_DIR/${name}_${STAMP}.log"
  log "===================================================================="
  log "START  $name"
  log "CMD    $*"
  log "===================================================================="
  local start=$SECONDS
  ( "$@" ) 2>&1 | tee -a "$phase_log" | tee -a "$MASTER_LOG"
  local code=${PIPESTATUS[0]}
  local dur=$(( SECONDS - start ))
  NAMES+=("$name"); CODES+=("$code")
  if [ "$code" -eq 0 ]; then
    log "DONE   $name  (exit=0, ${dur}s)"
  else
    log "FAIL   $name  (exit=$code, ${dur}s) -- continuing to next phase"
  fi
}

# --- lightweight preflight: warn (do not block) on missing hard-required ckpts
for ck in checkpoints/base_cifar10_resnet18.pth checkpoints/base_cifar100_resnet18.pth; do
  [ -f "$ck" ] || log "WARN   missing checkpoint: $ck (phases that need it will fail)"
done

log "######## PURGE full experiment sweep | stamp=$STAMP | python=$PYTHON ########"
SWEEP_START=$SECONDS

# 1. ESSENTIAL -- headline + cross-dataset (5 datasets x 3 seeds)  -> results/multiseed/
run_phase "1_multiseed" \
  "$PYTHON" -u experiments/exp_multiseed_all.py \
    --datasets cifar10 mnist svhn stl10 pathmnist --seeds 42 123 456

# 2. Component ablation (CIFAR-10, seed 42)                        -> results/ablation/
run_phase "2_component_ablation" \
  "$PYTHON" -u experiments/component_ablation.py

# 3. GroupNorm multi-dataset                                       -> results/groupnorm_multi/
run_phase "3_groupnorm_multi" \
  "$PYTHON" -u experiments/exp_groupnorm_multi.py

# 4. Per-step trajectory at mu=0                                   -> results/perstep/
run_phase "4_perstep_trajectory" \
  "$PYTHON" -u experiments/exp_perstep_trajectory.py \
    --checkpoint checkpoints/base_cifar10_resnet18.pth

# 5. CIFAR-100 (9 classes) -- run.py per class                     -> results/cifar100/class_$c
log "===================================================================="
log "START  5_cifar100 (9 classes: 10 20 30 40 50 60 70 80 90)"
log "===================================================================="
c100_start=$SECONDS
c100_fail=0
c100_log="$LOG_DIR/5_cifar100_${STAMP}.log"
for c in 10 20 30 40 50 60 70 80 90; do
  log "  -- cifar100 forget_class=$c"
  ( "$PYTHON" -u run.py \
      --config configs/cifar100_kl_retain.yaml \
      --checkpoint checkpoints/base_cifar100_resnet18.pth \
      --forget_class "$c" --data_dir data \
      --save_dir "results/cifar100/class_$c" ) 2>&1 | tee -a "$c100_log" | tee -a "$MASTER_LOG"
  code=${PIPESTATUS[0]}
  [ "$code" -ne 0 ] && { c100_fail="$code"; log "  FAIL cifar100 class=$c (exit=$code) -- continuing"; }
done
c100_dur=$(( SECONDS - c100_start ))
NAMES+=("5_cifar100"); CODES+=("$c100_fail")
if [ "$c100_fail" -eq 0 ]; then
  log "DONE   5_cifar100  (exit=0, ${c100_dur}s)"
else
  log "FAIL   5_cifar100  (some classes failed, ${c100_dur}s)"
fi

# --- summary ----------------------------------------------------------------
TOTAL_DUR=$(( SECONDS - SWEEP_START ))
log ""
log "############################ SUMMARY ############################"
overall=0
for i in "${!NAMES[@]}"; do
  if [ "${CODES[$i]}" -eq 0 ]; then st="OK"; else st="FAIL(${CODES[$i]})"; overall=1; fi
  log "  ${NAMES[$i]}: $st"
done
log "  total wall-clock: ${TOTAL_DUR}s"
log "  master log: $MASTER_LOG"
log "#################################################################"
exit "$overall"
