# Experiments

Reproduce paper results from the repo root. All scripts write CSVs to `results/`.

**Prerequisite:** train the base model first.

```bash
bash scripts/train_base_cifar10.sh
# → checkpoints/base_cifar10_resnet18.pth
```

---

## 1. Main unlearning run

```bash
bash scripts/unlearn_cifar10.sh
```

| Flag | Value | Maps to |
|------|-------|---------|
| `--forget_objective` | `kl_retain` | Retain-confusion target |
| `--retain_budget_factor` | `5.0` | ε_R |
| `--fa_target` | `10.0` | ε_F (%) |
| `--fa_check_freq` | `50` | Intra-epoch FA check |

**Expected (seed 42):** TA ≈ 85.4% · FA ≈ 9.5% · RA ≈ 97.3% · MIA ≈ 0.506

**3-seed headline (paper):** seeds `{42, 123, 456}` → TA 85.2±0.2 · FA 8.9±0.6 · RA 97.2±0.1 · MIA 0.507±0.001

---

## 2. Component ablation

```bash
bash scripts/run_component_ablation.sh
```

| Config | Disable mechanism | Paper result (TA / FA / RA / MIA) |
|--------|-------------------|----------------------------------|
| `minus_kd` | `--kd_weight 0` | 81.35 / 6.52 / 92.67 / 0.510 |
| `minus_rep` | `--rep_weight 0` | 85.38 / 9.52 / 97.30 / 0.506 |
| `minus_stopping` | budget=9999, fa_target=-1 | 81.03 / 0.00 / 92.26 / 0.532 |
| `minus_gate_cap` | `--forget_gate 0` | 85.38 / 9.52 / 97.30 / 0.506 |
| `minus_proj` | `--disable_projection` | **79.69 / 6.00 / 90.77 / 0.511** |
| `selective_layer` | `--freeze_early_layers` | 85.37 / 8.18 / 97.41 / 0.499 |

Output: `results/ablation/component_ablation_summary.csv`

---

## 3. Epsilon sweeps

```bash
bash scripts/run_epsilon_sweep.sh
bash scripts/run_epsilon_sweep.sh --only B
```

### Sweep A — retain budget (fix FA target = 10%)

Vary `--retain_budget_factor` ∈ {1.5, 2, 3, 5, 10, 9999}. **Finding:** ε_R is non-binding.

### Sweep B — FA target (fix α = 5)

Vary `--fa_target` ∈ {1, 5, 10, 20, 30}%. **Finding:** Monotonic privacy–utility frontier.

Output: `results/epsilon_sweep/` · published summaries in `results/sweep_*.csv`

---

## 4. BN recalibration attack

```bash
python experiments/bn_recalibration.py \
  --checkpoints purge:checkpoints/<purge_ckpt>.pth
```

| Method | ΔF |
|--------|-----|
| PURGE (BN) | +88.1pp |
| PURGE (GroupNorm) | 0.0pp |

---

## 5. GroupNorm fix

```bash
bash scripts/train_groupnorm_base.sh
python run.py --config configs/cifar10_kl_retain.yaml \
  --checkpoint checkpoints/base_cifar10_resnet18_groupnorm.pth \
  --use_groupnorm --data_dir ./data --save_dir ./checkpoints
```

**Paper result:** TA 84.86% · FA 7.56% · RA 99.22% · ΔF 0.00

---

## 6. Diagnostics

```bash
python experiments/base_mia.py --checkpoint checkpoints/base_cifar10_resnet18.pth
python experiments/linear_probe.py --checkpoints purge:checkpoints/<ckpt>.pth
python experiments/fid_baselines.py --reference checkpoints/base_cifar10_resnet18.pth \
  --checkpoints purge:checkpoints/<ckpt>.pth
```

---

## Compute estimates

| Experiment | GPU-hours (A100) |
|------------|------------------|
| Main CIFAR-10 | ~0.5 |
| Component ablation (6 runs) | ~3 |
| Sweep A + B (11 runs) | ~5 |
| GroupNorm base training | ~2 |
