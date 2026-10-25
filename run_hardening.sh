#!/usr/bin/env bash
# =============================================================================
# PURGE — reviewer-hardening experiments
#
# Everything here targets a WEAKNESS A TMLR REVIEWER WILL FIND, not new claims.
# Phases are ordered by (value / cost). Runs strictly sequentially: the earlier
# OOM was caused by concurrent runs, so nothing here backgrounds anything.
#
#   Phase 1  epsilon_F knob         ~25 min   un-retracts a retracted claim
#   Phase 2  STL-10 / CIFAR-100     ~40 min   fixes or hardens the one failure
#   Phase 3  random-subset forget   ~20 min   answers "class-only?" (exploratory)
#   Phase 4  second architecture    ~60 min   answers "ResNet-18-only?"
#   Phase 5  MIA ref, 2nd dataset   ~50 min   generalises the headline finding
#   Phase 6  PURGE vs that ref       ~35 min   completes the protocol argument
#   Phase 7  sample-level, fixed     ~20 min   retries a misconfigured phase
#   Phase 8  table-grade reruns      ~2 h      lets Table 1 rows be replaced
#   Phase 9  BN attack, canonical    ~15 min   prof's attack, exact operator
#
# Phases are independent. To run a subset:   bash run_hardening.sh 1 3 5
# Default is all five.
#
# Assumes the repo layout on the GPU box and that base checkpoints already
# exist in ./checkpoints (same ones the previous phases used).
# =============================================================================
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1
PY=python                                   # GPU box uses `python`, not python3
CKPT=./checkpoints
OUT=results/hardening
LOG=logs/hardening
mkdir -p "$OUT" "$LOG" "$CKPT"

PHASES="${*:-1 2 3 4 5}"
want() { [[ " $PHASES " == *" $1 "* ]]; }

banner() { echo; echo "==============================================================";
           echo "  $*"; echo "=============================================================="; }

# run <logname> <args...>   — never aborts the script on a single failure
run() { local n="$1"; shift
  echo "--- $n"
  if "$@" > "$LOG/$n.log" 2>&1; then echo "    ok"
  else echo "    FAILED (see $LOG/$n.log)"; tail -n 12 "$LOG/$n.log" | sed 's/^/      /'; fi
}

START=$(date +%s)

# -----------------------------------------------------------------------------
# PHASE 1 — does epsilon_F become a knob once the FA check is adaptive?
#
# The paper currently RETRACTS the epsilon_F claim: targets 1..30% all stopped at
# FA 1.88 because FA falls from ~99% to ~0% inside one 50-batch check interval,
# so the run always overshoots. --fa_check_adaptive (new) tightens the interval
# to 1 step once FA is within 3x the target. If the sweep now separates, the
# retraction is replaced by a working knob AND the over-forgetting section gets
# its remedy. If it does NOT separate, that is a much stronger negative result
# than the current "we did not investigate": it says the collapse is genuinely
# sub-step and no stopping rule can catch it.
# -----------------------------------------------------------------------------
if want 1; then
banner "PHASE 1 — epsilon_F with adaptive FA checking"
for T in 1 5 10 20 30; do
  run "eps_adaptive_t${T}" $PY run.py --config configs/cifar10_kl_retain.yaml \
      --checkpoint $CKPT/base_cifar10_resnet18.pth \
      --data_dir ./data --save_dir $CKPT \
      --fa_target "$T" --fa_check_adaptive
done
# control: same targets, fixed interval (reproduces the flat result)
for T in 1 30; do
  run "eps_fixed_t${T}" $PY run.py --config configs/cifar10_kl_retain.yaml \
      --checkpoint $CKPT/base_cifar10_resnet18.pth \
      --data_dir ./data --save_dir $CKPT \
      --fa_target "$T"
done
fi

# -----------------------------------------------------------------------------
# PHASE 2 — is STL-10 a method failure or a compute failure?
#
# STL-10 is the paper's one honest failure (FA 75.7%). Diagnosis was that 500
# images/class gives only ~60 steps, so it never gets far enough. That has never
# been tested. Four cheap runs separate "the method fails here" from "this run
# was under-trained" -- and either answer is publishable. CIFAR-100 gets the
# same treatment (it lands at 4.07% vs 1% chance).
# -----------------------------------------------------------------------------
if want 2; then
banner "PHASE 2 — compute rescue for STL-10 and CIFAR-100"
run "stl10_e60"      $PY run.py --config configs/stl10_kl_retain.yaml \
    --checkpoint $CKPT/base_stl10_resnet18.pth --data_dir ./data --save_dir $CKPT \
    --epochs 60
run "stl10_e60_lr1e3" $PY run.py --config configs/stl10_kl_retain.yaml \
    --checkpoint $CKPT/base_stl10_resnet18.pth --data_dir ./data --save_dir $CKPT \
    --epochs 60 --lr 0.001
run "stl10_e60_fa1"   $PY run.py --config configs/stl10_kl_retain.yaml \
    --checkpoint $CKPT/base_stl10_resnet18.pth --data_dir ./data --save_dir $CKPT \
    --epochs 60 --fa_target 1.0 --fa_check_adaptive
run "cifar100_e60"    $PY run.py --config configs/cifar100_kl_retain.yaml \
    --checkpoint $CKPT/base_cifar100_resnet18.pth --data_dir ./data --save_dir $CKPT \
    --epochs 60 --fa_target 1.0 --fa_check_adaptive
fi

# -----------------------------------------------------------------------------
# PHASE 3 — random-subset forgetting  (EXPLORATORY: read the note)
#
# The paper only ever unlearns whole classes; "does this work for sample-level
# requests?" is the single most predictable reviewer question, and run.py has
# supported --forget_type sample all along.
#
# NOTE ON THE TARGET. For class unlearning the goal is FA -> chance. For sample
# unlearning it is NOT: a correctly unlearned model should score the forgotten
# samples like unseen samples of the same class, i.e. FA ~ TA (~86%), and
# driving FA to 0 is *over*-forgetting and is itself a detectable signal. So the
# stopping target is set near TA, not near chance. Expect to iterate on this
# once; the number to watch is MIA, not FA.
# -----------------------------------------------------------------------------
if want 3; then
banner "PHASE 3 — random-subset (sample-level) forgetting"
for PCT in 0.01 0.05; do
  run "sample_pct${PCT}" $PY run.py --config configs/cifar10_kl_retain.yaml \
      --checkpoint $CKPT/base_cifar10_resnet18.pth --data_dir ./data --save_dir $CKPT \
      --forget_type sample --forget_pct "$PCT" \
      --fa_target 86.0 --fa_check_adaptive --fa_check_freq 10
  # projection off, same budget: does the guarantee still buy anything here?
  run "sample_pct${PCT}_noproj" $PY run.py --config configs/cifar10_kl_retain.yaml \
      --checkpoint $CKPT/base_cifar10_resnet18.pth --data_dir ./data --save_dir $CKPT \
      --forget_type sample --forget_pct "$PCT" --disable_projection \
      --fa_target 86.0 --fa_check_adaptive --fa_check_freq 10
done
fi

# -----------------------------------------------------------------------------
# PHASE 4 — is the result ResNet-18-specific?
#
# Every number in the paper is one architecture. run.py already supports
# resnet50 and vgg16. One dataset x two architectures x {PURGE, -projection} is
# enough to answer the question; VGG-16 is the more informative of the two
# because it has no residual connections and no BN, so it also probes whether
# the BN-recalibration story is architecture-bound.
#
# Requires base checkpoints for these architectures. If they do not exist, train
# them first (see MANUAL step printed below) -- these runs will fail fast.
# -----------------------------------------------------------------------------
if want 4; then
banner "PHASE 4 — second and third architecture"
for M in resnet50 vgg16; do
  B="$CKPT/base_cifar10_${M}.pth"
  if [[ ! -f "$B" ]]; then
    echo "--- SKIP $M: no base checkpoint at $B"
    echo "    train it first, e.g.:"
    echo "    $PY run.py --dataset cifar10 --model $M --forget_type class \\"
    echo "        --forget_class 0 --data_dir ./data --save_dir $CKPT --epochs 0"
    continue
  fi
  run "arch_${M}"        $PY run.py --config configs/cifar10_kl_retain.yaml \
      --checkpoint "$B" --model "$M" --data_dir ./data --save_dir $CKPT
  run "arch_${M}_noproj" $PY run.py --config configs/cifar10_kl_retain.yaml \
      --checkpoint "$B" --model "$M" --data_dir ./data --save_dir $CKPT \
      --disable_projection --fa_target -1.0 --retain_budget_factor 9999.0
done
fi

# -----------------------------------------------------------------------------
# PHASE 5 — does the MIA-reference finding hold on a second dataset?
#
# The paper's most novel empirical claim is that the standard forget-vs-test MIA
# does not measure membership: a *retrained* CIFAR-10 model scores 0.694, not
# 0.5, so 0.5 is the wrong reference point. Right now that rests on ONE dataset,
# which is thin for a claim that criticises everyone else's protocol. PathMNIST
# is the cheapest second witness (small, 9 classes, different domain).
# -----------------------------------------------------------------------------
if want 5; then
banner "PHASE 5 — MIA reference on a second dataset (PathMNIST)"
run "retrain_pathmnist" $PY experiments/retrain_reference.py \
    --dataset pathmnist --forget_class 0 --data_dir ./data --save_dir $CKPT
PM="$CKPT/retrain_pathmnist_c0.pth"
if [[ -f "$PM" ]]; then
  run "mia_pathmnist" $PY experiments/mia_suite.py \
      --dataset pathmnist --forget_class 0 --data_dir ./data \
      --checkpoints "base:$CKPT/base_pathmnist_resnet18.pth" \
      --retrained "$PM" \
      --out_csv "$OUT/mia_suite_pathmnist.csv"
else
  echo "--- SKIP mia_pathmnist: reference was not produced"
fi
fi

# -----------------------------------------------------------------------------
banner "DONE in $(( ($(date +%s)-START)/60 )) min"
echo "Parse the logs into CSVs, then collect:"
echo "  $PY collect_hardening.py"
echo "  tar czf hardening.tgz $OUT $LOG"
echo
echo "Files produced:"
ls -la "$OUT" 2>/dev/null

# -----------------------------------------------------------------------------
# PHASE 6 — complete the MIA-reference argument on PathMNIST
#
# Phase 5 established the REFERENCE point on a second dataset: a retrained
# PathMNIST model scores AUROC 0.705 on the loss attack, not 0.5. What it did
# not do is show where PURGE lands relative to that reference, because no
# unlearned PathMNIST checkpoint existed -- only base and retrained.
#
# That comparison is the actual argument. On CIFAR-10, PURGE scores 0.693 vs a
# reference of 0.694 (a near-exact match) while SalUn scores 0.500 -- which
# looks perfect against the conventional "0.5 is ideal" reading but is in fact
# 0.19 further from the retrained model than PURGE is. Replicating that on a
# second dataset is what turns a CIFAR-10 curiosity into a claim about the
# protocol. References for c0/c2/c3/c4 already exist, so this is cheap.
# -----------------------------------------------------------------------------
if want 6; then
banner "PHASE 6 — PURGE vs retrained reference on PathMNIST"
for C in 0 2 3 4; do
  REF="$CKPT/retrain_pathmnist_c${C}.pth"
  [[ -f "$REF" ]] || { echo "--- SKIP c$C: no reference at $REF"; continue; }
  # reuse an existing unlearned checkpoint so re-running this phase costs only
  # the MIA step; run.py writes ..._class_c<N>_ep<E>_lr<L>.pth
  PU=$(ls -t "$CKPT"/purge_pathmnist_resnet18_class_c${C}_*.pth 2>/dev/null | head -1)
  if [[ -z "$PU" ]]; then
    run "purge_pathmnist_c${C}" $PY run.py --config configs/pathmnist_kl_retain.yaml \
        --checkpoint $CKPT/base_pathmnist_resnet18.pth \
        --forget_class "$C" --data_dir ./data --save_dir $CKPT
    PU=$(ls -t "$CKPT"/purge_pathmnist_resnet18_class_c${C}_*.pth 2>/dev/null | head -1)
  else
    echo "--- reusing $(basename "$PU")"
  fi
  if [[ -n "$PU" ]]; then
    run "mia_pathmnist_c${C}" $PY experiments/mia_suite.py \
        --dataset pathmnist --forget_class "$C" --data_dir ./data \
        --checkpoints "purge:$PU" --retrained "$REF" \
        --out_csv "$OUT/mia_suite_pathmnist_c${C}.csv"
  else
    echo "--- SKIP mia c$C: no unlearned checkpoint matched"
    ls "$CKPT" | grep -i "purge_pathmnist" | sed 's/^/      have: /'
  fi
done
fi

# -----------------------------------------------------------------------------
# PHASE 7 — sample-level forgetting, retried with a config that fits the regime
#
# Phase 3 did not test the method; it tested a misconfigured run. Three faults,
# all mine, all visible in the logs:
#
#   1. WRONG TARGET. --fa_target 86.0 was copied from class-level unlearning,
#      where TA ~86% because one class is gone. In sample-level forgetting no
#      class is removed, so the base model scores TA ~95.3% and the correct
#      "forgotten looks like unseen" target is ~95, not 86.
#   2. BUDGET MIS-SCALED. The retain budget is multiplicative (5x initial), and
#      the base model is essentially converged on retain data: initial retain
#      loss 0.0018 -> budget 0.0089. That is a near-zero absolute ceiling. Class
#      runs never hit it because fa_target fires first; sample runs tripped it
#      every time and halted at epoch 4.
#   3. TOO FEW STEPS. 500 forget images / batch 128 = 4 optimiser steps per
#      epoch, so the whole run was ~36 steps -- against ~40 steps per epoch for
#      class-level CIFAR-10.
#
# This phase fixes all three and adds a no-stopping fixed-budget arm, because
# for sample-level requests MIA is the metric that matters and FA-based
# stopping is the wrong instrument.
#
# NOTE: p_r is still uniform in this regime (retain covers every class), so
# kl_retain degenerates to kl_uniform here. That part of the Phase 3 finding
# stands regardless of how this run turns out, and is worth stating either way.
# -----------------------------------------------------------------------------
if want 7; then
banner "PHASE 7 — sample-level forgetting, corrected config"
for PCT in 0.01 0.05; do
  # (a) stopping arm: target matched to the ACTUAL base test accuracy
  run "s7_pct${PCT}_tgt95" $PY run.py --config configs/cifar10_kl_retain.yaml \
      --checkpoint $CKPT/base_cifar10_resnet18.pth --data_dir ./data --save_dir $CKPT \
      --forget_type sample --forget_pct "$PCT" --epochs 60 \
      --retain_budget_factor 100.0 \
      --fa_target 95.0 --fa_check_adaptive --fa_check_freq 1
  # (b) fixed-budget arm: no stopping at all, judge on MIA not FA
  run "s7_pct${PCT}_fixed" $PY run.py --config configs/cifar10_kl_retain.yaml \
      --checkpoint $CKPT/base_cifar10_resnet18.pth --data_dir ./data --save_dir $CKPT \
      --forget_type sample --forget_pct "$PCT" --epochs 60 \
      --retain_budget_factor 9999.0 --fa_target -1.0
  # (c) fixed budget, projection off -- does the guarantee buy anything here?
  run "s7_pct${PCT}_fixed_noproj" $PY run.py --config configs/cifar10_kl_retain.yaml \
      --checkpoint $CKPT/base_cifar10_resnet18.pth --data_dir ./data --save_dir $CKPT \
      --forget_type sample --forget_pct "$PCT" --epochs 60 --disable_projection \
      --retain_budget_factor 9999.0 --fa_target -1.0
done
fi

# -----------------------------------------------------------------------------
# PHASE 8 — make the STL-10 and CIFAR-100 rescues table-grade
#
# Phase 2 established the diagnosis on single runs, which is why the paper
# reports them beside Table 1 rather than inside it. To replace the rows, the
# new numbers have to be produced on the same protocol as the rest of the table:
#
#   STL-10    3 seeds {42,123,456} at 60 epochs (table is 3 seeds x 1 class)
#   CIFAR-100 the nine tens-group classes {10,20,...,90} at 60 epochs, 1 seed
#             each (table is 9 classes x 1 seed)
#
# NOTE: Phase 2's CIFAR-100 run used class 0, which is NOT one of the nine
# classes the table averages over -- the config default, not the evaluated set.
# So it was never comparable to the 4.07% mean, and this phase supersedes it.
#
# Cost is dominated by CIFAR-100: ~11 min per class at 60 epochs, so ~100 min,
# against ~22 min for the three STL-10 seeds.
# -----------------------------------------------------------------------------
if want 8; then
banner "PHASE 8 — STL-10 x 3 seeds, CIFAR-100 x 9 classes, 60 epochs"
for SEED in 42 123 456; do
  run "t8_stl10_e60_s${SEED}" $PY run.py --config configs/stl10_kl_retain.yaml \
      --checkpoint $CKPT/base_stl10_resnet18.pth --data_dir ./data --save_dir $CKPT \
      --epochs 60 --seed "$SEED" --fa_target 1.0 --fa_check_adaptive
done
for C in 10 20 30 40 50 60 70 80 90; do
  run "t8_cifar100_e60_c${C}" $PY run.py --config configs/cifar100_kl_retain.yaml \
      --checkpoint $CKPT/base_cifar100_resnet18.pth --data_dir ./data --save_dir $CKPT \
      --epochs 60 --forget_class "$C" --fa_target 1.0 --fa_check_adaptive
done
fi

# -----------------------------------------------------------------------------
# PHASE 9 — the BN-recalibration attack, canonical operator
#
# We already run this attack (S5.4: PURGE-BN dF = +92.2pp, PURGE-GN dF = 0.00),
# but with a weaker operator: BN train mode + forward passes, so the statistics
# were an EMA that still carried some of the pre-attack buffers and depended on
# batch ordering. The canonical operator is torch.optim.swa_utils.update_bn,
# which RESETS the buffers and accumulates a cumulative average, giving the exact
# retain-set statistics. That is the attack as its authors define it, and it is
# at least as strong as ours.
#
# experiments/bn_illusion_attack.py runs BOTH operators on the same checkpoint,
# so the comparison is direct. It asserts the attack leaves every learnable
# parameter bit-identical -- if that fails, the operator is doing more than
# recalibrating and the result is meaningless.
#
# The baselines are included deliberately. Our paper reports the vulnerability as
# PURGE-specific on the strength of OUR reproductions (SalUn 0.0, BadTeacher
# -0.5, SCRUB -4.4). The attack's own authors report SalUn +77.9pp and
# BadTeacher +98.7pp. Both can hold, because vulnerability depends on whether a
# method's BN buffers were updated during unlearning -- but our claim is then
# implementation-specific and the paper must say so. Re-running the baselines
# under the canonical operator is what settles which it is.
# -----------------------------------------------------------------------------
if want 9; then
banner "PHASE 9 — BN recalibration, canonical update_bn operator"
CK_ARGS=()
add() { [[ -f "$2" ]] && CK_ARGS+=("$1:$2") || echo "    (missing, skipped) $1 -> $2"; }
add base       "$CKPT/base_cifar10_resnet18.pth"
add purge_bn   "results/todo/purge_cifar10_bn/purge_cifar10_resnet18_class_c0_ep15_lr0.0005.pth"
add purge_gn   "results/todo/purge_cifar10_gn/purge_cifar10_resnet18_class_c0_ep15_lr0.0005.pth"
add salun      "../01_salun/checkpoints/salun_cifar10_resnet18_class.pth"
add bad_teacher "../02_bad_teacher/checkpoints/bad_teacher_cifar10_class.pth"
add scrub      "../03_scrub/checkpoints/scrub_cifar10_class.pth"
if [[ ${#CK_ARGS[@]} -eq 0 ]]; then
  echo "--- no checkpoints found; nothing to attack"
else
  echo "    attacking ${#CK_ARGS[@]} checkpoints"
  run "bn_illusion" $PY experiments/bn_illusion_attack.py \
      --dataset cifar10 --forget_class 0 --data_dir ./data \
      --checkpoints "${CK_ARGS[@]}" \
      --out_csv "$OUT/bn_illusion.csv"
  # the verdict block is the point of this phase, so surface it
  sed -n '/VERDICT/,$p' "$LOG/bn_illusion.log" 2>/dev/null | sed 's/^/    /'
fi
fi

# -----------------------------------------------------------------------------
# PHASE 10 — ViT-B/16 on CIFAR-10, under the CURRENT algorithm
#
# results/vit/ already holds ViT runs, but they are unusable as they stand: they
# date from 29 June, which predates the projected-momentum redesign, and the ViT
# support that produced them lived only in a server-side run.py that has since
# been overwritten (it was never committed). This phase restores that support in
# the current run.py and re-runs the CIFAR-10 arm under the deployed update rule.
#
# Two reasons this is the most valuable remaining architecture test:
#   1. It answers "is the result convolution-specific?" far better than
#      ResNet-50 would -- different family, different inductive bias.
#   2. ViT normalises with LayerNorm, which keeps NO running statistics. So it
#      is structurally outside the reach of the BN-recalibration attack, exactly
#      as the GroupNorm backbone is. Phase 9 showed PURGE-BN is fully reversible
#      (dF = +97.96pp); a second BN-free architecture that unlearns properly
#      turns "we swapped the norm layer" into a general statement about which
#      normalisation schemes admit the attack at all.
#
# rep_weight is forced to 0: the stored ViT configs set 0.03, but representation
# erasure was removed from the method, and every ResNet config now uses 0.
#
# ~21 min per run x 3 seeds, plus the attack. CIFAR-100 ViT is Phase 11.
# -----------------------------------------------------------------------------
if want 10; then
banner "PHASE 10 — ViT-B/16, CIFAR-10, current algorithm"
B="$CKPT/base_cifar10_vit_b_16.pth"
if [[ ! -f "$B" ]]; then
  echo "--- SKIP: no ViT base checkpoint at $B"
else
  for SEED in 42 123 456; do
    run "vit_cifar10_s${SEED}" $PY run.py --config configs/cifar10_vit_kl_retain.yaml \
        --checkpoint "$B" --data_dir ./data --save_dir $CKPT \
        --rep_weight 0.0 --seed "$SEED" --fa_check_adaptive
  done
  # LayerNorm has no running stats, so the attack should be undefined here --
  # the script reports that explicitly rather than printing a misleading 0.0
  VU=$(ls -t "$CKPT"/purge_cifar10_vit_b_16_class_c0_*.pth 2>/dev/null | head -1)
  if [[ -n "$VU" ]]; then
    run "bn_illusion_vit" $PY experiments/bn_illusion_attack.py \
        --dataset cifar10 --model vit_b_16 --forget_class 0 --data_dir ./data \
        --checkpoints "purge_vit:$VU" --out_csv "$OUT/bn_illusion_vit.csv"
    sed -n '/VERDICT/,$p' "$LOG/bn_illusion_vit.log" 2>/dev/null | sed 's/^/    /'
  fi
fi
fi

# -----------------------------------------------------------------------------
# PHASE 11 — ViT-B/16 on CIFAR-100 (EXPENSIVE: ~67 min x 9 classes = ~10 h)
#
# Only worth starting if the box is free overnight. Phase 10 alone already
# answers the architecture question; this extends it to the 100-class setting
# where the June run left one class (c20) failing at FA 79.6% -- a failure the
# stored sweep showed was a learning-rate artifact, not a method limit.
# -----------------------------------------------------------------------------
if want 11; then
banner "PHASE 11 — ViT-B/16, CIFAR-100, 9 classes (~10 h)"
B="$CKPT/base_cifar100_vit_b_16.pth"
if [[ ! -f "$B" ]]; then
  echo "--- SKIP: no ViT CIFAR-100 base checkpoint at $B"
else
  for C in 10 20 30 40 50 60 70 80 90; do
    run "vit_cifar100_c${C}" $PY run.py --config configs/cifar100_vit_kl_retain.yaml \
        --checkpoint "$B" --data_dir ./data --save_dir $CKPT \
        --rep_weight 0.0 --forget_class "$C" --fa_check_adaptive
  done
fi
fi

# -----------------------------------------------------------------------------
# PHASE 12 — is the momentum dependence PURGE-specific? (reviewer comment)
#
# The paper reports that removing momentum breaks PURGE's forgetting on 5 of 6
# datasets, and a reviewer asked the obvious follow-up: is that a property of
# PURGE, or of gradient-based unlearning generally? We answer it INSIDE our own
# codebase rather than across four baseline repos, because a cross-repo
# comparison would confound the momentum change with every other difference in
# their optimisers, schedules and stopping rules.
#
# Four update rules, each at mu=0 and mu=0.9, everything else identical:
#   purge         kl_retain + projection      (the deployed method)
#   kl_uniform    kl_uniform + projection     (different forget objective)
#   proj_only     gradient ascent + projection, KD removed
#   ga            gradient ascent, NO projection  <- the "other method" control
#
# The `ga` arm is the informative one: it is a standard unlearning update with
# no projection at all. If it also collapses at mu=0, the dependence is a
# property of the unlearning objective; if only the projected arms collapse, it
# is specific to operating under a hard per-step constraint.
# -----------------------------------------------------------------------------
if want 12; then
banner "PHASE 12 — momentum dependence across update rules"
for MU in 0.0 0.9; do
  TAG=$(echo "$MU" | tr -d .)
  run "mom_purge_mu${TAG}" $PY run.py --config configs/cifar10_kl_retain.yaml \
      --checkpoint $CKPT/base_cifar10_resnet18.pth --data_dir ./data --save_dir $CKPT \
      --momentum "$MU"
  run "mom_kluniform_mu${TAG}" $PY run.py --config configs/cifar10_kl_retain.yaml \
      --checkpoint $CKPT/base_cifar10_resnet18.pth --data_dir ./data --save_dir $CKPT \
      --forget_objective kl_uniform --momentum "$MU"
  run "mom_projonly_mu${TAG}" $PY run.py --config configs/cifar10_kl_retain.yaml \
      --checkpoint $CKPT/base_cifar10_resnet18.pth --data_dir ./data --save_dir $CKPT \
      --forget_objective ga --kd_weight 0 --momentum "$MU"
  run "mom_ga_mu${TAG}" $PY run.py --config configs/cifar10_kl_retain.yaml \
      --checkpoint $CKPT/base_cifar10_resnet18.pth --data_dir ./data --save_dir $CKPT \
      --forget_objective ga --kd_weight 0 --disable_projection --momentum "$MU"
done
fi

# -----------------------------------------------------------------------------
# PHASE 13 — the same question on the EXTERNAL baselines  (OPT-IN, EDITS FILES)
#
# SalUn, Bad Teacher and SCRUB hardcode momentum=0.9 in their own run.py with no
# command-line flag, so answering the reviewer's question literally requires
# editing those files. This phase makes that edit reversible: it copies each
# run.py to run.py.mombak, rewrites the hardcoded 0.9 to read an environment
# variable defaulting to 0.9, runs both settings, and restores the backup on the
# way out (including on Ctrl-C).
#
# Run this only if you are comfortable with the script touching files in
# ../01_salun, ../02_bad_teacher and ../03_scrub. Phase 12 answers the same
# question without leaving our own directory, and is the one we would report.
# -----------------------------------------------------------------------------
if want 13; then
banner "PHASE 13 — momentum on external baselines (edits + restores their run.py)"
BASE_DIRS=(../01_salun ../02_bad_teacher ../03_scrub)
restore_baselines() {
  for d in "${BASE_DIRS[@]}"; do
    [[ -f "$d/run.py.mombak" ]] && mv -f "$d/run.py.mombak" "$d/run.py" \
      && echo "    restored $d/run.py"
  done
}
trap restore_baselines EXIT INT TERM
for d in "${BASE_DIRS[@]}"; do
  if [[ ! -f "$d/run.py" ]]; then echo "--- SKIP $d (no run.py)"; continue; fi
  cp -f "$d/run.py" "$d/run.py.mombak"
  # momentum=0.9 -> momentum=float(os.environ.get("PURGE_MU","0.9"))
  perl -pi -e 's/momentum=0\.9/momentum=float(__import__("os").environ.get("PURGE_MU","0.9"))/g' "$d/run.py"
  echo "    patched $d/run.py ($(grep -c PURGE_MU "$d/run.py") sites)"
done
for MU in 0.0 0.9; do
  TAG=$(echo "$MU" | tr -d .)
  for d in "${BASE_DIRS[@]}"; do
    [[ -f "$d/run.py" ]] || continue
    nm=$(basename "$d")
    ( cd "$d" && PURGE_MU="$MU" $PY run.py --dataset cifar10 --forget_class 0 \
        --data_dir ./data 2>&1 ) > "$LOG/mombase_${nm}_mu${TAG}.log" 2>&1 \
      && echo "--- mombase_${nm}_mu${TAG}  ok" \
      || { echo "--- mombase_${nm}_mu${TAG}  FAILED"; tail -n 6 "$LOG/mombase_${nm}_mu${TAG}.log" | sed 's/^/      /'; }
  done
done
restore_baselines
trap - EXIT INT TERM
fi

# -----------------------------------------------------------------------------
# PHASE 14 — momentum across update rules, on the datasets where it MATTERS
#
# Phase 12 asked the reviewer's question ("does removing momentum break other
# methods, or only PURGE?") but answered it on CIFAR-10 -- which Table 5 shows is
# the single dataset where mu=0 does NOT break PURGE (FA 5.8% against a 10%
# chance level). The runs were correct and reproduced that row (2.14 -> 5.54),
# but they cannot separate the hypotheses, because there is no PURGE failure on
# CIFAR-10 to compare against.
#
# Two design faults, both fixed here:
#   1. WRONG DATASETS. This phase uses MNIST (mu=0: FA 95.9%) and SVHN (61.9%),
#      where the effect Table 5 reports is large.
#   2. UNEQUAL COMPUTE. Phase 12 left dual stopping on, so variants halted after
#      1-12 epochs and the momentum change was confounded with step count.
#      Stopping is disabled here, exactly as in the equal-compute arm of the
#      component ablation.
#
# The `ga` arm carries the argument: gradient ascent with NO projection
# (proj_rate 0). If it degrades as much as the projected arms, the momentum
# dependence belongs to gradient-based unlearning in general and not to
# operating under a hard per-step constraint -- which is what the reviewer asked.
# -----------------------------------------------------------------------------
if want 14; then
banner "PHASE 14 — momentum x update rule x {MNIST, SVHN}, equal compute"
NOSTOP=(--fa_target -1.0 --retain_budget_factor 9999.0)
for DS in mnist svhn; do
  B="$CKPT/base_${DS}_resnet18.pth"
  if [[ ! -f "$B" ]]; then echo "--- SKIP $DS: no base checkpoint at $B"; continue; fi
  for MU in 0.0 0.9; do
    TAG=$(echo "$MU" | tr -d .)
    run "m14_${DS}_purge_mu${TAG}" $PY run.py --config configs/${DS}_kl_retain.yaml \
        --checkpoint "$B" --data_dir ./data --save_dir $CKPT \
        --momentum "$MU" "${NOSTOP[@]}"
    run "m14_${DS}_kluniform_mu${TAG}" $PY run.py --config configs/${DS}_kl_retain.yaml \
        --checkpoint "$B" --data_dir ./data --save_dir $CKPT \
        --forget_objective kl_uniform --momentum "$MU" "${NOSTOP[@]}"
    run "m14_${DS}_projonly_mu${TAG}" $PY run.py --config configs/${DS}_kl_retain.yaml \
        --checkpoint "$B" --data_dir ./data --save_dir $CKPT \
        --forget_objective ga --kd_weight 0 --momentum "$MU" "${NOSTOP[@]}"
    run "m14_${DS}_ga_mu${TAG}" $PY run.py --config configs/${DS}_kl_retain.yaml \
        --checkpoint "$B" --data_dir ./data --save_dir $CKPT \
        --forget_objective ga --kd_weight 0 --disable_projection \
        --momentum "$MU" "${NOSTOP[@]}"
  done
done
fi

# -----------------------------------------------------------------------------
# PHASES 15/16 — where WE fail, how does everyone else do?
#
# Reviewer principle: a setting in which every method fails is a property of the
# setting, not a weakness of one method. Our two remaining negative results are
# reported against nothing, which makes them read as our defects:
#
#   sample-level forgetting  PURGE reaches FA 58.4% only by wrecking the model
#                            (RA 100 -> 73.2). Untested for any baseline.
#   CIFAR-100                PURGE needs 60 epochs to reach 1.27%. No baseline
#                            comparison exists at all -- Table 3 is CIFAR-10 only.
#
# IMPORTANT SCOPE NOTE. Of the 16 baseline directories on the box, only four are
# functioning implementations with checkpoints (SalUn, Bad Teacher, SCRUB,
# Fast-Unlearn); ten are placeholder files whose main() never calls their own
# unlearning routine, and none is a clone of the original authors' repository.
# These are OUR reimplementations, which is what Table 3 already says, and it is
# the most likely explanation for the BN-recalibration disagreement with
# concurrent work (S5.4). Results here inherit that caveat.
#
# Each baseline runs at its own default configuration, as in Table 3. This is a
# "does the setting defeat everyone?" test, not an equal-compute comparison.
# -----------------------------------------------------------------------------
BASE_ROOT=..
BL_DIRS=(01_salun 02_bad_teacher 03_scrub 04_fast_unlearn)

run_baseline() {   # <tag> <dir> <extra args...>
  local tag="$1" d="$2"; shift 2
  local dir="$BASE_ROOT/$d"
  if [[ ! -f "$dir/run.py" ]]; then echo "--- SKIP $tag (no $dir/run.py)"; return; fi
  echo "--- $tag"
  if ( cd "$dir" && $PY run.py "$@" ) > "$LOG/$tag.log" 2>&1; then echo "    ok"
  else echo "    FAILED"; tail -n 8 "$LOG/$tag.log" | sed 's/^/      /'; fi
}

if want 15; then
banner "PHASE 15 — sample-level forgetting: do the baselines fare any better?"
for d in "${BL_DIRS[@]}"; do
  m="${d#*_}"
  run_baseline "x15_${m}" "$d" --dataset cifar10 --model resnet18 \
      --forget_type sample --forget_pct 0.01 \
      --checkpoint ../17_purge/checkpoints/base_cifar10_resnet18.pth \
      --data_dir ./data --seed 42
done
fi

if want 16; then
banner "PHASE 16 — CIFAR-100 class-level: the first baseline comparison there"
for d in "${BL_DIRS[@]}"; do
  m="${d#*_}"
  run_baseline "x16_${m}" "$d" --dataset cifar100 --model resnet18 \
      --forget_type class --forget_class 10 \
      --checkpoint ../17_purge/checkpoints/base_cifar100_resnet18.pth \
      --data_dir ./data --seed 42
done
# our own matching run, same class, for the comparison row
run "x16_purge" $PY run.py --config configs/cifar100_kl_retain.yaml \
    --checkpoint $CKPT/base_cifar100_resnet18.pth --data_dir ./data --save_dir $CKPT \
    --forget_class 10 --epochs 60 --fa_target 1.0 --fa_check_adaptive
fi

# -----------------------------------------------------------------------------
# PHASE 17 — BN-robust PURGE: does keeping phi tracking theta survive the attack?
#
# PURGE freezes BN statistics at phi_0 while theta adapts against
# them, so the final weights are valid only for statistics an attacker can
# replace: phi <- R(theta_T, D_r) recovers FA from 1.4% to 99.7% (S5.4). The fix
# is to stop letting phi go stale.
#
#   off    current behaviour. phi frozen at phi_0.               dF = +98.4
#   exact  the proposed rule: phi <- R(theta, D_r) before every step,
#          computed with the attacker's own operator.
#   ema    cheap analogue: the RETAIN forward pass updates the
#          running stats; forget batches never touch them. This
#          is what GA+FT gets for free from its fine-tuning stage
#          (GA+FT shows dF ~ -0.2pp under the attack).
#   gn     GroupNorm fallback. No statistics to recalibrate.
#
# READ THE VERDICT CAREFULLY. Both robust arms end with a full recalibration, so
# phi IS R(theta_T, D_r) and -- because R is idempotent -- the attack cannot move
# anything. dF = 0 is therefore TRUE BY CONSTRUCTION and proves nothing on its
# own. The question is the POST-attack FA: if it stays near chance, forgetting
# now lives in the weights and the method is genuinely robust; if it rises toward
# 99%, the method never forgot without stale statistics and the proposal has
# surfaced that rather than fixed it. That is the result to look at.
# -----------------------------------------------------------------------------
if want 17; then
banner "PHASE 17 — BN-robust PURGE (off / exact / ema / groupnorm)"
B10=$CKPT/base_cifar10_resnet18.pth
for MODE in off exact ema; do
  SD="$CKPT/bnrobust_$MODE"; mkdir -p "$SD"
  EXTRA=(); [[ "$MODE" != off ]] && EXTRA=(--bn_robust "$MODE")
  [[ "$MODE" == exact ]] && EXTRA+=(--bn_recal_every "${BN_EVERY:-1}" --bn_recal_samples "${BN_SAMPLES:-2048}")
  run "bnr_${MODE}" $PY run.py --config configs/cifar10_kl_retain.yaml \
      --checkpoint "$B10" --data_dir ./data --save_dir "$SD" "${EXTRA[@]}"
done
# GroupNorm fallback arm needs its own base model
BGN=$CKPT/base_cifar10_resnet18_groupnorm.pth
if [[ -f "$BGN" ]]; then
  SD="$CKPT/bnrobust_gn"; mkdir -p "$SD"
  run "bnr_gn" $PY run.py --config configs/cifar10_kl_retain.yaml \
      --checkpoint "$BGN" --use_groupnorm --data_dir ./data --save_dir "$SD"
else
  echo "--- SKIP gn arm: no $BGN"
fi

# ---- attack every arm with the canonical operator ----------------------------
CK=()
for MODE in off exact ema gn; do
  f=$(ls -t "$CKPT/bnrobust_$MODE"/purge_cifar10_resnet18_class_c0_*.pth 2>/dev/null | head -1)
  [[ -n "$f" ]] && CK+=("$MODE:$f") || echo "    (no checkpoint for arm $MODE)"
done
if [[ ${#CK[@]} -gt 0 ]]; then
  run "bnr_attack" $PY experiments/bn_illusion_attack.py \
      --dataset cifar10 --forget_class 0 --data_dir ./data \
      --checkpoints "${CK[@]}" --out_csv "$OUT/bn_robust.csv"
  echo
  echo "    pre-F is what the method achieved; post-F is what survives the attack."
  sed -n '/VERDICT/,$p' "$LOG/bnr_attack.log" 2>/dev/null | sed 's/^/    /'
fi
fi

# -----------------------------------------------------------------------------
# PHASE 18 — BN-robust PURGE, second pass
#
# Phase 17 established that the idea works: in-loop recalibration cuts the
# attack's recovery from +98.4pp to +6.7pp while holding test accuracy
# (86.07 vs 86.10). Two things need fixing before the result is quotable.
#
# 1. BOTH robust arms halted on the RETAIN BUDGET, not the forget target
#    ('exact' at epoch 4, 'ema' at epoch 2). This is the predicted theoretical
#    gap made visible: the projection controls the theta-step, but recalibrating
#    phi is a second move on the retain loss that nothing constrains, so the
#    budget trips early. 'ema' never got far enough to forget at all (FA 95.6%),
#    which is why its delta_F ~ 0 was meaningless. Raising the budget lets both
#    arms run to the forget target and be judged on equal terms.
#
# 2. The residual +6.7pp is partly an artefact we control: run.py's final
#    recalibration used the TRAINING loader (RandomCrop + flip), so the shipped
#    phi was R_augmented(theta) while the attacker computes R_clean(theta). That
#    mismatch is itself recoverable signal. run.py now swaps in the deterministic
#    transform for the final call, so both sides compute the same operator.
#
# If 'exact' now reaches chance AND holds post-attack FA near chance, the
# proposal is confirmed and the theorem is worth formulating over (theta, phi).
# -----------------------------------------------------------------------------
if want 18; then
banner "PHASE 18 — BN-robust with a budget that lets it finish"
B10=$CKPT/base_cifar10_resnet18.pth
for MODE in exact ema; do
  SD="$CKPT/bnr2_$MODE"; mkdir -p "$SD"
  EXTRA=(--bn_robust "$MODE" --retain_budget_factor 100.0 --fa_check_adaptive)
  [[ "$MODE" == exact ]] && EXTRA+=(--bn_recal_every "${BN_EVERY:-1}" --bn_recal_samples "${BN_SAMPLES:-2048}")
  run "bnr2_${MODE}" $PY run.py --config configs/cifar10_kl_retain.yaml \
      --checkpoint "$B10" --data_dir ./data --save_dir "$SD" "${EXTRA[@]}"
done
CK=()
for MODE in exact ema; do
  f=$(ls -t "$CKPT/bnr2_$MODE"/purge_cifar10_resnet18_class_c0_*.pth 2>/dev/null | head -1)
  [[ -n "$f" ]] && CK+=("$MODE:$f") || echo "    (no checkpoint for arm $MODE)"
done
# carry the Phase-17 reference arms into the same table
for M in off gn; do
  f=$(ls -t "$CKPT/bnrobust_$M"/purge_cifar10_resnet18_class_c0_*.pth 2>/dev/null | head -1)
  [[ -n "$f" ]] && CK+=("${M}_ref:$f")
done
if [[ ${#CK[@]} -gt 0 ]]; then
  run "bnr2_attack" $PY experiments/bn_illusion_attack.py \
      --dataset cifar10 --forget_class 0 --data_dir ./data \
      --checkpoints "${CK[@]}" --out_csv "$OUT/bn_robust2.csv"
  sed -n '/VERDICT/,$p' "$LOG/bnr2_attack.log" 2>/dev/null | sed 's/^/    /'
fi
fi

# -----------------------------------------------------------------------------
# PHASE 19 — BN-robust PURGE, three seeds, plus a cheaper recalibration schedule
#
# Phases 17/18 established on one seed that recalibrating BN inside the loop
# makes forgetting survive the attack: forget accuracy under the attacker's own
# statistics R(theta, D_r) was 99.2% for current PURGE and 10.3-13.0% for the
# exact variant (chance 10%). This phase makes that table-grade and tests cost:
#
#   off      current PURGE (phi frozen)                        the control
#   exact    phi <- R(theta, D_r) before EVERY step            the proposal
#   exact5   the same, but only every 5th step                 ~5x fewer recals
#
# All three use the deployed config (5x retain budget, FA checked every 50
# batches) -- the Phase-17 protocol, under which exact landed on chance -- so
# bn_robust and its schedule are the only things that change.
#
# Seeds change the unlearning trajectory (data order, augmentation), not the
# base model: all nine runs start from the same checkpoint, as in the paper's
# multiseed protocol. NOTE that checkpoint is the 93.97% base, not the paper's
# lost 95.50% one, so compare these rows with each other, not with paper tables.
#
# The number to read is post_F (FA after the attack), not delta_F: the exact
# variants end with a full recalibration, which makes delta_F ~ 0 by
# construction.
# -----------------------------------------------------------------------------
if want 19; then
banner "PHASE 19 — BN-robust PURGE: 3 seeds x {off, exact, exact every 5th step}"
B10=$CKPT/base_cifar10_resnet18.pth
for SEED in 42 123 456; do
  for MODE in off exact exact5; do
    SD="$CKPT/bnr3_${MODE}_s${SEED}"; mkdir -p "$SD"
    EXTRA=(--seed "$SEED")
    case "$MODE" in
      exact)  EXTRA+=(--bn_robust exact --bn_recal_every 1 --bn_recal_samples 2048) ;;
      exact5) EXTRA+=(--bn_robust exact --bn_recal_every 5 --bn_recal_samples 2048) ;;
    esac
    run "bnr3_${MODE}_s${SEED}" $PY run.py --config configs/cifar10_kl_retain.yaml \
        --checkpoint "$B10" --data_dir ./data --save_dir "$SD" "${EXTRA[@]}"
  done
done
CK=()
for SEED in 42 123 456; do
  for MODE in off exact exact5; do
    f=$(ls -t "$CKPT/bnr3_${MODE}_s${SEED}"/purge_cifar10_resnet18_class_c0_*.pth 2>/dev/null | head -1)
    [[ -n "$f" ]] && CK+=("${MODE}_s${SEED}:$f") || echo "    (no checkpoint for ${MODE}_s${SEED})"
  done
done
if [[ ${#CK[@]} -gt 0 ]]; then
  run "bnr3_attack" $PY experiments/bn_illusion_attack.py \
      --dataset cifar10 --forget_class 0 --data_dir ./data \
      --checkpoints "${CK[@]}" --out_csv "$OUT/bn_robust3.csv"
  sed -n '/VERDICT/,$p' "$LOG/bnr3_attack.log" 2>/dev/null | sed 's/^/    /'
fi
fi

# -----------------------------------------------------------------------------
# PHASE 20 — BN-robust PURGE, every step, on fewer recalibration images
#
# Phase 19 (3 seeds) showed the cost lever matters: recalibrating only every
# 5th step lets phi drift between recalibrations, and that staleness is exactly
# what costs forgetting (post-attack FA 17.05% vs 11.25% for every step). So keep
# recalibrating EVERY step but estimate phi from 512 retain images instead of
# 2048. phi stays fresh; only its estimate gets noisier. Even at 512 images the
# last ResNet stage sees ~25k values per channel, so the moments should hold.
#
# Expected saving is ~40%, not most of the cost: per-step recalibration accounts
# for ~2.2x of exact's ~12.8x over off, and the other ~5.8x is that genuine
# forgetting needs more epochs -- which no recalibration schedule changes.
#
# Same base (93.97%), config and seeds as Phase 19, so its exact (2048) and off
# arms are the references and are not rerun.
# -----------------------------------------------------------------------------
if want 20; then
banner "PHASE 20 — BN-robust PURGE, every step, 512 recalibration images, 3 seeds"
B10=$CKPT/base_cifar10_resnet18.pth
CK=()
for SEED in 42 123 456; do
  SD="$CKPT/bnr4_exact512_s${SEED}"; mkdir -p "$SD"
  run "bnr4_exact512_s${SEED}" $PY run.py --config configs/cifar10_kl_retain.yaml \
      --checkpoint "$B10" --data_dir ./data --save_dir "$SD" --seed "$SEED" \
      --bn_robust exact --bn_recal_every 1 --bn_recal_samples 512
  f=$(ls -t "$SD"/purge_cifar10_resnet18_class_c0_*.pth 2>/dev/null | head -1)
  [[ -n "$f" ]] && CK+=("exact512_s${SEED}:$f") || echo "    (no checkpoint for exact512_s${SEED})"
done
if [[ ${#CK[@]} -gt 0 ]]; then
  run "bnr4_attack" $PY experiments/bn_illusion_attack.py \
      --dataset cifar10 --forget_class 0 --data_dir ./data \
      --checkpoints "${CK[@]}" --out_csv "$OUT/bn_robust4.csv"
  sed -n '/VERDICT/,$p' "$LOG/bnr4_attack.log" 2>/dev/null | sed 's/^/    /'
fi
fi

# -----------------------------------------------------------------------------
# PHASE 21 — BN-robust PURGE with the persistent recalibration loader
#
# Identical to Phase 20 (exact, every step, 512 images, same seeds, same base)
# except --bn_recal_persistent. Phase 20 measured ~0.67 s of FIXED cost per
# recalibration -- 72% of the overhead at 512 images -- from re-spawning the
# retain loader's 4 workers on every call. The flag draws each step's batches
# from one persistent loader instead, keeping each draw inside a single shuffle
# (512 distinct images, as before), so it should change runtime and nothing
# else. Phase 20 is the exact reference for both claims:
#   - post-attack FA should match Phase 20 within seed noise
#   - runtime should fall from ~18 min toward ~13 min
# -----------------------------------------------------------------------------
if want 21; then
banner "PHASE 21 — BN-robust PURGE, 512 images, persistent recalibration loader, 3 seeds"
B10=$CKPT/base_cifar10_resnet18.pth
CK=()
for SEED in 42 123 456; do
  SD="$CKPT/bnr5_exact512p_s${SEED}"; mkdir -p "$SD"
  run "bnr5_exact512p_s${SEED}" $PY run.py --config configs/cifar10_kl_retain.yaml \
      --checkpoint "$B10" --data_dir ./data --save_dir "$SD" --seed "$SEED" \
      --bn_robust exact --bn_recal_every 1 --bn_recal_samples 512 --bn_recal_persistent
  f=$(ls -t "$SD"/purge_cifar10_resnet18_class_c0_*.pth 2>/dev/null | head -1)
  [[ -n "$f" ]] && CK+=("exact512p_s${SEED}:$f") || echo "    (no checkpoint for exact512p_s${SEED})"
done
if [[ ${#CK[@]} -gt 0 ]]; then
  run "bnr5_attack" $PY experiments/bn_illusion_attack.py \
      --dataset cifar10 --forget_class 0 --data_dir ./data \
      --checkpoints "${CK[@]}" --out_csv "$OUT/bn_robust5.csv"
  sed -n '/VERDICT/,$p' "$LOG/bnr5_attack.log" 2>/dev/null | sed 's/^/    /'
fi
fi

# -----------------------------------------------------------------------------
# PHASE 22 — the GroupNorm arm of the three-way comparison
#
# The comparison is PURGE-BN (current) -> PURGE-GN (fallback) -> BN-robust PURGE.
# The first and third exist on this cluster; GN does not, because its base
# model was lost with the old server. This phase rebuilds it and runs PURGE-GN
# on the same three seeds as Phases 19-21.
#
# Base training follows experiments/train_groupnorm_base.py from scratch (GN
# cannot reuse ImageNet weights, which carry BatchNorm), but for 100 epochs
# instead of 200: the cosine schedule still anneals fully, it halves GPU time on
# a shared quota, and the base only needs to land near the BN base (93.97%) for
# the comparison to be fair. The old 200-epoch GN base reached 93.39%.
# Caveat inherited from that script: it keeps the best-TEST-accuracy epoch.
#
# The attack is undefined on GN (no BN buffers); the attack script reports it
# as such rather than as robustness.
# -----------------------------------------------------------------------------
if want 22; then
banner "PHASE 22 — GroupNorm base (100 epochs) + PURGE-GN x 3 seeds"
BGN=$CKPT/base_cifar10_resnet18_groupnorm.pth
if [[ ! -f "$BGN" ]]; then
  run "gn_base_train" $PY experiments/train_groupnorm_base.py \
      --data_dir ./data --save_path "$BGN" --epochs "${GN_EPOCHS:-100}"
  grep -E "Best test accuracy|Epoch +(1|5|10|50|100)/" "$LOG/gn_base_train.log" | sed 's/^/    /'
fi
if [[ -f "$BGN" ]]; then
  CK=()
  for SEED in 42 123 456; do
    SD="$CKPT/bnr6_gn_s${SEED}"; mkdir -p "$SD"
    run "bnr6_gn_s${SEED}" $PY run.py --config configs/cifar10_kl_retain.yaml \
        --checkpoint "$BGN" --use_groupnorm --data_dir ./data --save_dir "$SD" --seed "$SEED"
    f=$(ls -t "$SD"/purge_cifar10_resnet18_class_c0_*.pth 2>/dev/null | head -1)
    [[ -n "$f" ]] && CK+=("gn_s${SEED}:$f") || echo "    (no checkpoint for gn_s${SEED})"
  done
  [[ ${#CK[@]} -gt 0 ]] && run "bnr6_attack" $PY experiments/bn_illusion_attack.py \
      --dataset cifar10 --forget_class 0 --data_dir ./data \
      --checkpoints "${CK[@]}" --out_csv "$OUT/bn_robust6_gn.csv"
else
  echo "--- GN base was not produced; skipping PURGE-GN"
fi
fi

# -----------------------------------------------------------------------------
# PHASE 23 — does BN-robust PURGE generalise across classes?
#
# Every BN-robust result so far forgets CIFAR-10 class 0. The paper already
# makes all-class claims for MNIST and PathMNIST, so "only class 0?" is the
# obvious next question. This needs no new base model -- the same checkpoint
# serves all ten classes -- so it covers classes 1-9 (class 0 has 3 seeds from
# Phases 19-21), one seed each, with the current method as the per-class
# control: the vulnerability itself has to be shown on each class too.
#
#   off        current PURGE                         expected: attack recovers
#   exact512   BN-robust, every step, 512 images     expected: stays forgotten
#
# BN_PERSIST=1 (default) adds --bn_recal_persistent; set BN_PERSIST=0 to fall
# back to the per-call loader if the persistent one is ever in doubt.
# Each class is attacked separately because the forget set differs per class.
# -----------------------------------------------------------------------------
if want 23; then
banner "PHASE 23 — BN-robust across CIFAR-10 classes 1-9 (seed 42)"
B10=$CKPT/base_cifar10_resnet18.pth
PERSIST=(); [[ "${BN_PERSIST:-1}" == 1 ]] && PERSIST=(--bn_recal_persistent)
for C in 1 2 3 4 5 6 7 8 9; do
  CK=()
  for MODE in off exact512; do
    SD="$CKPT/bnr7_${MODE}_c${C}"; mkdir -p "$SD"
    EXTRA=(--forget_class "$C" --seed 42)
    [[ "$MODE" == exact512 ]] && EXTRA+=(--bn_robust exact --bn_recal_every 1 --bn_recal_samples 512 "${PERSIST[@]}")
    run "bnr7_${MODE}_c${C}" $PY run.py --config configs/cifar10_kl_retain.yaml \
        --checkpoint "$B10" --data_dir ./data --save_dir "$SD" "${EXTRA[@]}"
    f=$(ls -t "$SD"/purge_cifar10_resnet18_class_c${C}_*.pth 2>/dev/null | head -1)
    [[ -n "$f" ]] && CK+=("${MODE}_c${C}:$f") || echo "    (no checkpoint for ${MODE}_c${C})"
  done
  [[ ${#CK[@]} -gt 0 ]] && run "bnr7_attack_c${C}" $PY experiments/bn_illusion_attack.py \
      --dataset cifar10 --forget_class "$C" --data_dir ./data \
      --checkpoints "${CK[@]}" --out_csv "$OUT/bn_robust7_c${C}.csv"
done
fi

# -----------------------------------------------------------------------------
# PHASE 24 — the attacker recalibrates on data of its own choosing
#
# BN-robust PURGE recalibrates on the retain set while it unlearns, and every
# attack so far (Phases 17-21) recalibrated on that same set. An attacker is not
# bound to it. This re-attacks existing checkpoints with statistics from the full
# training set (forget class included) and from the held-out test set (every
# class). No training -- only the attack, so it is cheap.
#   off        plain PURGE                            (Phase 19, 3 seeds)
#   exact      BN-robust, every step, 2048 images     (Phase 19, 3 seeds)
#   exact512p  BN-robust, every step, 512, persistent (Phase 21, 3 seeds)
# -----------------------------------------------------------------------------
if want 24; then
banner "PHASE 24 — BN attack with the attacker's own recalibration data"
CK=()
for SEED in 42 123 456; do
  for ARM in bnr3_off bnr3_exact bnr5_exact512p; do
    f=$(ls -t "$CKPT/${ARM}_s${SEED}"/purge_cifar10_resnet18_class_c0_*.pth 2>/dev/null | head -1)
    [[ -n "$f" ]] && CK+=("${ARM#bnr?_}_s${SEED}:$f") || echo "    (no checkpoint for ${ARM}_s${SEED})"
  done
done
if [[ ${#CK[@]} -gt 0 ]]; then
  for SET in train test; do
    run "bnr8_attack_${SET}" $PY experiments/bn_illusion_attack.py \
        --dataset cifar10 --forget_class 0 --data_dir ./data --recal_on "$SET" \
        --checkpoints "${CK[@]}" --out_csv "$OUT/bn_robust8_recal_${SET}.csv"
    sed -n '/VERDICT/,$p' "$LOG/bnr8_attack_${SET}.log" 2>/dev/null | sed 's/^/    /'
  done
fi
fi

# -----------------------------------------------------------------------------
# PHASE 25 — the control row: the base model under the same attacks
#
# The paper's BN table (tab:bnrecal) anchors everything on the un-unlearned base
# model: recalibrating a normally trained network should change nothing, because
# its statistics already agree with its weights. The cluster runs use a
# different base (93.97%) from the paper's, so this re-measures that control on
# it, with each of the three recalibration sets used in Phases 19-24.
# The verdict printer calls the base "never forgot" -- correct, and expected for
# a control; read its delta_F and delta_TA, which should both be ~0.
# -----------------------------------------------------------------------------
if want 25; then
banner "PHASE 25 — base model (no unlearning) under the BN attack"
B10=$CKPT/base_cifar10_resnet18.pth
for SET in retain train test; do
  run "bnr9_base_${SET}" $PY experiments/bn_illusion_attack.py \
      --dataset cifar10 --forget_class 0 --data_dir ./data --recal_on "$SET" \
      --checkpoints "base:$B10" --out_csv "$OUT/bn_robust9_base_${SET}.csv"
  grep -E "pre  :|post:" "$LOG/bnr9_base_${SET}.log" 2>/dev/null | sed 's/^/    /'
done
fi

# -----------------------------------------------------------------------------
# PHASE 26 — BN-robust with a calibrated stopping rule
#
# In Phases 19-21 BN-robust stopped when FA <= 10%, but that FA was measured
# under the last in-loop statistics (512 augmented retain images, one step
# stale) on augmented forget images. The model ships with phi <- R(theta_T, D_r)
# on deterministic transforms, and there FA reads 1.6-3.6pp higher: 9.2-12.6%
# instead of <= 10%. Plain PURGE has no such gap (its phi never changes).
# --bn_calibrated_stop measures the stopping FA under the shipped statistics,
# so the shipped FA meets the target. Same setting as Phase 21 otherwise,
# except the epoch cap: 25 instead of the config's 15. Near the target
# BN-robust forgetting slows to a crawl (Phase 23, class 4 sat at 10.4-11.6%
# for its last epochs and hit 15), and a stricter rule needs more of them; the
# question here is what the rule costs, so the rule -- not the cap -- should end
# the run. Phase 21 never went past 13 epochs, so the cap did not touch it.
# -----------------------------------------------------------------------------
if want 26; then
banner "PHASE 26 — BN-robust (512, persistent) + calibrated stopping, 3 seeds"
B10=$CKPT/base_cifar10_resnet18.pth
CK=()
for SEED in 42 123 456; do
  SD="$CKPT/bnr10_cal_s${SEED}"; mkdir -p "$SD"
  run "bnr10_cal_s${SEED}" $PY run.py --config configs/cifar10_kl_retain.yaml \
      --checkpoint "$B10" --data_dir ./data --save_dir "$SD" --seed "$SEED" \
      --bn_robust exact --bn_recal_every 1 --bn_recal_samples 512 --bn_recal_persistent \
      --bn_calibrated_stop --epochs 25
  grep -E "under the shipped phi|stopping at epoch|Wall-clock" "$LOG/bnr10_cal_s${SEED}.log" | tail -4 | sed 's/^/    /'
  f=$(ls -t "$SD"/purge_cifar10_resnet18_class_c0_*.pth 2>/dev/null | head -1)
  [[ -n "$f" ]] && CK+=("cal_s${SEED}:$f") || echo "    (no checkpoint for cal_s${SEED})"
done
if [[ ${#CK[@]} -gt 0 ]]; then
  run "bnr10_attack" $PY experiments/bn_illusion_attack.py \
      --dataset cifar10 --forget_class 0 --data_dir ./data \
      --checkpoints "${CK[@]}" --out_csv "$OUT/bn_robust10_calstop.csv"
  sed -n '/VERDICT/,$p' "$LOG/bnr10_attack.log" 2>/dev/null | sed 's/^/    /'
fi
fi

# -----------------------------------------------------------------------------
# PHASE 27 — calibrated stopping across CIFAR-10 classes 1-9
#
# Phase 23 showed the in-loop vs shipped FA gap is class-dependent (-1.0 to
# +9.0pp; classes 8 and 9 shipped at 16.7% and 15.4% after stopping at an
# in-loop 10.0% and 6.4%). This reruns Phase 23's BN-robust arm with
# --bn_calibrated_stop and the 25-epoch cap of Phase 26. The plain-PURGE arm is
# unchanged by the flag, so Phase 23's off_c<k> rows remain the control.
# Largest-gap classes first, so the informative rows land early.
# -----------------------------------------------------------------------------
if want 27; then
banner "PHASE 27 — calibrated stopping across classes 1-9 (seed 42)"
B10=$CKPT/base_cifar10_resnet18.pth
for C in 9 8 1 7 3 4 2 6 5; do
  SD="$CKPT/bnr11_cal_c${C}"; mkdir -p "$SD"
  run "bnr11_cal_c${C}" $PY run.py --config configs/cifar10_kl_retain.yaml \
      --checkpoint "$B10" --data_dir ./data --save_dir "$SD" --forget_class "$C" --seed 42 \
      --bn_robust exact --bn_recal_every 1 --bn_recal_samples 512 --bn_recal_persistent \
      --bn_calibrated_stop --epochs 25
  f=$(ls -t "$SD"/purge_cifar10_resnet18_class_c${C}_*.pth 2>/dev/null | head -1)
  if [[ -n "$f" ]]; then
    run "bnr11_attack_c${C}" $PY experiments/bn_illusion_attack.py \
        --dataset cifar10 --forget_class "$C" --data_dir ./data \
        --checkpoints "cal_c${C}:$f" --out_csv "$OUT/bn_robust11_cal_c${C}.csv"
    grep -E "canonical_update_bn +post" "$LOG/bnr11_attack_c${C}.log" | sed 's/^/    /'
  else
    echo "    (no checkpoint for cal_c${C})"
  fi
done
fi

# -----------------------------------------------------------------------------
# PHASE 28 — class 4 with --restore_best_epoch
#
# Phase 27's one miss: class 4 never got its shipped FA under 10% (it sat at
# 10.2-10.9% from epoch 19 to 24), ran into the 25-epoch cap, and shipped its
# worse last epoch (11.94% FA, RA 93.34%). Same run plus --restore_best_epoch,
# which ships the lowest-FA epoch within the retain budget whenever the target
# is not reached (and changes nothing when it is).
# -----------------------------------------------------------------------------
if want 28; then
banner "PHASE 28 — class 4, calibrated stop + restore best epoch (seed 42)"
B10=$CKPT/base_cifar10_resnet18.pth
SD="$CKPT/bnr12_calbest_c4"; mkdir -p "$SD"
run "bnr12_calbest_c4" $PY run.py --config configs/cifar10_kl_retain.yaml \
    --checkpoint "$B10" --data_dir ./data --save_dir "$SD" --forget_class 4 --seed 42 \
    --bn_robust exact --bn_recal_every 1 --bn_recal_samples 512 --bn_recal_persistent \
    --bn_calibrated_stop --epochs 25 --restore_best_epoch
grep -E "best-epoch|stopping at epoch|Wall-clock" "$LOG/bnr12_calbest_c4.log" | sed 's/^/    /'
f=$(ls -t "$SD"/purge_cifar10_resnet18_class_c4_*.pth 2>/dev/null | head -1)
if [[ -n "$f" ]]; then
  run "bnr12_attack_c4" $PY experiments/bn_illusion_attack.py \
      --dataset cifar10 --forget_class 4 --data_dir ./data \
      --checkpoints "calbest_c4:$f" --out_csv "$OUT/bn_robust12_calbest_c4.csv"
  sed -n '/VERDICT/,$p' "$LOG/bnr12_attack_c4.log" 2>/dev/null | sed 's/^/    /'
else
  echo "    (no checkpoint for calbest_c4)"
fi
fi

# -----------------------------------------------------------------------------
# PHASE 29 — a second dataset: MNIST
#
# Every BN-robust result so far is on CIFAR-10. MNIST is the cheapest second
# dataset the paper already covers. Its base model was lost with the old server,
# so it is retrained with the repo's own recipe (experiments/train_base_all.py:
# ImageNet-pretrained ResNet-18, lr 0.01, early stop at 99% test accuracy) --
# self-contained, like everything on this cluster. Forget class 0, 3 seeds:
#   off      plain PURGE (configs/mnist_kl_retain.yaml)
#   robust   BN-robust as finally configured: exact every step on 512 images,
#            persistent loader, calibrated stop, restore best epoch, cap 25
# then the attack on all six checkpoints.
# -----------------------------------------------------------------------------
if want 29; then
banner "PHASE 29 — MNIST: plain PURGE vs BN-robust (calibrated), 3 seeds"
BM=$CKPT/base_mnist_resnet18.pth
if [[ ! -f "$BM" ]]; then
  run "mnist_base_train" $PY experiments/train_base_all.py --datasets mnist \
      --data_dir ./data --save_dir "$CKPT"
  grep -iE "best test|early stop|saved" "$LOG/mnist_base_train.log" | tail -4 | sed 's/^/    /'
fi
if [[ -f "$BM" ]]; then
  ROBUST=(--bn_robust exact --bn_recal_every 1 --bn_recal_samples 512 --bn_recal_persistent)
  ROBUST+=(--bn_calibrated_stop --restore_best_epoch --epochs 25)
  CK=()
  for SEED in 42 123 456; do
    for MODE in off robust; do
      SD="$CKPT/bnr13_mnist_${MODE}_s${SEED}"; mkdir -p "$SD"
      EXTRA=(--seed "$SEED")
      [[ "$MODE" == robust ]] && EXTRA+=("${ROBUST[@]}")
      run "bnr13_mnist_${MODE}_s${SEED}" $PY run.py --config configs/mnist_kl_retain.yaml \
          --checkpoint "$BM" --data_dir ./data --save_dir "$SD" "${EXTRA[@]}"
      f=$(ls -t "$SD"/purge_mnist_resnet18_class_c0_*.pth 2>/dev/null | head -1)
      [[ -n "$f" ]] && CK+=("${MODE}_s${SEED}:$f") || echo "    (no checkpoint for ${MODE}_s${SEED})"
    done
  done
  if [[ ${#CK[@]} -gt 0 ]]; then
    run "bnr13_mnist_attack" $PY experiments/bn_illusion_attack.py \
        --dataset mnist --forget_class 0 --data_dir ./data \
        --checkpoints "${CK[@]}" --out_csv "$OUT/bn_robust13_mnist.csv"
    sed -n '/VERDICT/,$p' "$LOG/bnr13_mnist_attack.log" 2>/dev/null | sed 's/^/    /'
  fi
else
  echo "--- MNIST base was not produced; skipping"
fi
fi

# -----------------------------------------------------------------------------
# PHASE 30 — a cheaper calibrated stopping check
#
# --bn_calibrated_stop recalibrates on the full retain set (45,000 images) at
# every epoch-end check, ~22 s an epoch -- most of why BN-robust went from 9.8
# to 15.7 min a run (Phase 21 -> 26). This caps the check's recalibration at
# 10,240 images (--bn_stop_recal_samples); the shipped statistics still come
# from the full set. Class 0, Phase 26's seeds, plus --restore_best_epoch (a
# no-op whenever the target is reached, as it was in every Phase 26 run).
# Question: same stopping epochs and shipped FA as Phase 26, in less time?
# -----------------------------------------------------------------------------
if want 30; then
banner "PHASE 30 — calibrated stop with a 10,240-image check, class 0, 3 seeds"
B10=$CKPT/base_cifar10_resnet18.pth
CK=()
for SEED in 42 123 456; do
  SD="$CKPT/bnr14_calcap_s${SEED}"; mkdir -p "$SD"
  run "bnr14_calcap_s${SEED}" $PY run.py --config configs/cifar10_kl_retain.yaml \
      --checkpoint "$B10" --data_dir ./data --save_dir "$SD" --seed "$SEED" \
      --bn_robust exact --bn_recal_every 1 --bn_recal_samples 512 --bn_recal_persistent \
      --bn_calibrated_stop --bn_stop_recal_samples 10240 --restore_best_epoch --epochs 25
  grep -E "best-epoch|stopping at epoch|Wall-clock" "$LOG/bnr14_calcap_s${SEED}.log" | sed 's/^/    /'
  f=$(ls -t "$SD"/purge_cifar10_resnet18_class_c0_*.pth 2>/dev/null | head -1)
  [[ -n "$f" ]] && CK+=("calcap_s${SEED}:$f") || echo "    (no checkpoint for calcap_s${SEED})"
done
if [[ ${#CK[@]} -gt 0 ]]; then
  run "bnr14_attack" $PY experiments/bn_illusion_attack.py \
      --dataset cifar10 --forget_class 0 --data_dir ./data \
      --checkpoints "${CK[@]}" --out_csv "$OUT/bn_robust14_calcap.csv"
  sed -n '/VERDICT/,$p' "$LOG/bnr14_attack.log" 2>/dev/null | sed 's/^/    /'
fi
fi
