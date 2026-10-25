#!/usr/bin/env bash
# =============================================================================
# Closes the two remaining evidence gaps that do not need new method code:
#
#   A  retrained CIFAR-10 reference           (~40 min)  -> reviewer ask #3
#   B  BN-recalibration on the baselines      (~5 min)   -> last data TODO
#   C  MIA suite: 4 attacks x TPR@low-FPR     (~10 min)  -> reviewer ask #4
#   D  distance-to-retrained on CIFAR-10      (~10 min)  -> reviewer asks #3/#4
#
# C and D are the point of A: until now the retrained gold standard and every
# distance-to-retrained metric existed only for three PathMNIST classes, not
# for the dataset the paper leads with.
#
# Usage:  bash run_mia_and_reference.sh
# =============================================================================
set -u
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG=results/mia_logs; mkdir -p "$LOG" results/todo
MASTER="$LOG/run_${STAMP}.log"

DATA=data
BASE=checkpoints/base_cifar10_resnet18.pth
REF=checkpoints/retrain_cifar10_c0.pth
SALUN=../01_salun/checkpoints/salun_cifar10_resnet18_class.pth
BADT=../02_bad_teacher/checkpoints/bad_teacher_cifar10_class.pth
SCRUB=../03_scrub/checkpoints/scrub_cifar10_class.pth

NAMES=(); CODES=()
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER"; }
phase(){
  local n="$1"; shift; local lg="$LOG/${n}_${STAMP}.log"
  log "=========================================================="
  log "START $n"; log "CMD   $*"
  local t0=$SECONDS
  ( "$@" ) 2>&1 | tee -a "$lg" | tee -a "$MASTER"
  local rc=${PIPESTATUS[0]}; NAMES+=("$n"); CODES+=("$rc")
  if [ "$rc" -eq 0 ]; then log "DONE  $n ($((SECONDS-t0))s)"
  else log "FAIL  $n (exit=$rc) — last 15 lines:"; tail -15 "$lg" | sed 's/^/      | /' | tee -a "$MASTER"; fi
}

[ -e "$BASE" ] || { log "FATAL missing $BASE"; exit 1; }

# locate the PURGE CIFAR-10 checkpoint produced by run_remaining_todos.sh
PURGE_CKPT="$(ls -t results/todo/purge_cifar10_bn/*.pth 2>/dev/null | head -1)"
if [ -z "${PURGE_CKPT:-}" ]; then
  log "WARN no PURGE CIFAR-10 checkpoint under results/todo/purge_cifar10_bn/;"
  log "     producing one now so C and D have something to evaluate."
  phase A0_purge_cifar10 "$PYTHON" -u run.py --config configs/cifar10_kl_retain.yaml \
      --checkpoint "$BASE" --data_dir "$DATA" \
      --save_dir results/todo/purge_cifar10_bn --seed 42
  PURGE_CKPT="$(ls -t results/todo/purge_cifar10_bn/*.pth 2>/dev/null | head -1)"
fi
log "PURGE checkpoint: ${PURGE_CKPT:-NONE}"

# ---- A. retrained CIFAR-10 reference ---------------------------------------
# Trains on the retain set only, keeping all K heads. No absolute accuracy
# threshold: patience is disabled so the cosine schedule anneals fully and this
# reference is comparable to the PathMNIST ones.
phase A_retrain_cifar10 "$PYTHON" -u experiments/retrain_reference.py \
    --dataset cifar10 --forget_class 0 --data_dir "$DATA" \
    --save_dir checkpoints --epochs 30 --patience 0 --seed 42

# ---- B. BN recalibration on the baselines ----------------------------------
BN_ARGS=()
[ -e "$SALUN" ] && BN_ARGS+=("salun:$SALUN")
[ -e "$BADT" ]  && BN_ARGS+=("bad_teacher:$BADT")
[ -e "$SCRUB" ] && BN_ARGS+=("scrub:$SCRUB")
if [ ${#BN_ARGS[@]} -gt 0 ]; then
  phase B_bn_baselines "$PYTHON" -u experiments/bn_recalibration.py \
      --data_dir "$DATA" --forget_class 0 \
      --checkpoints "${BN_ARGS[@]}" \
      --out_csv results/todo/bn_recal_baselines.csv
else
  log "SKIP B — no baseline checkpoints found"
fi

# ---- C. MIA suite ----------------------------------------------------------
MIA_ARGS=()
[ -n "${PURGE_CKPT:-}" ] && MIA_ARGS+=("purge:$PURGE_CKPT")
[ -e "$SALUN" ] && MIA_ARGS+=("salun:$SALUN")
[ -e "$BADT" ]  && MIA_ARGS+=("bad_teacher:$BADT")
[ -e "$SCRUB" ] && MIA_ARGS+=("scrub:$SCRUB")
MIA_ARGS+=("base(no unlearning):$BASE")     # upper anchor: nothing forgotten
if [ -e "$REF" ]; then
  phase C_mia_suite "$PYTHON" -u experiments/mia_suite.py \
      --dataset cifar10 --forget_class 0 --data_dir "$DATA" \
      --retrained "$REF" --checkpoints "${MIA_ARGS[@]}" \
      --out_csv results/todo/mia_suite_cifar10.csv
else
  log "WARN reference missing; running the suite WITHOUT calibration"
  phase C_mia_suite "$PYTHON" -u experiments/mia_suite.py \
      --dataset cifar10 --forget_class 0 --data_dir "$DATA" \
      --checkpoints "${MIA_ARGS[@]}" \
      --out_csv results/todo/mia_suite_cifar10.csv
fi

# ---- D. distance to the retrained reference --------------------------------
if [ -e "$REF" ] && [ -n "${PURGE_CKPT:-}" ]; then
  D_ARGS=("purge:$PURGE_CKPT")
  [ -e "$SALUN" ] && D_ARGS+=("salun:$SALUN")
  [ -e "$BADT" ]  && D_ARGS+=("bad_teacher:$BADT")
  [ -e "$SCRUB" ] && D_ARGS+=("scrub:$SCRUB")
  phase D_distance_cifar10 "$PYTHON" -u experiments/distance_to_retrained.py \
      --dataset cifar10 --forget_class 0 --data_dir "$DATA" \
      --retrained "$REF" --original "$BASE" \
      --checkpoints "${D_ARGS[@]}" \
      --out_csv results/todo/distance_cifar10.csv
else
  log "SKIP D — need both the reference and a PURGE checkpoint"
fi

# ---- summary ---------------------------------------------------------------
log ""; log "########################## SUMMARY ##########################"
ov=0
for i in "${!NAMES[@]}"; do
  if [ "${CODES[$i]}" -eq 0 ]; then s=OK; else s="FAIL(${CODES[$i]})"; ov=1; fi
  log "  ${NAMES[$i]}: $s"
done
log ""
log "  Send back:"
log "    results/todo/bn_recal_baselines.csv"
log "    results/todo/mia_suite_cifar10.csv"
log "    results/todo/distance_cifar10.csv"
log "  (or: tar -czf mia_results.tgz results/todo/*.csv)"
log "#############################################################"
exit "$ov"
