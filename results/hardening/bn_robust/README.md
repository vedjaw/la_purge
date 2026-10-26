# BN-robust EXCISE: results behind §6.4, Appendix F.6, Tables 6 and 20, and Figures 5, 6, 9 and 13

These runs test BN-robust EXCISE (`run.py --bn_robust exact`), which keeps the
BatchNorm statistics consistent with the weights throughout unlearning, against
the BatchNorm-recalibration attack of `experiments/bn_illusion_attack.py`.
EXCISE was called PURGE when they were made, so the checkpoint paths in the CSVs
start with `purge_`.

## Setup, and what the numbers can be compared with

* **Base models.** CIFAR-10 ResNet-18 at **93.97%** test accuracy, a GroupNorm
  ResNet-18 retrained for 100 epochs (**91.89%**), and an MNIST ResNet-18
  (**99.59%**). The CIFAR-10 base behind the paper's earlier tables (95.5%) was
  not available when these runs were made, so absolute numbers here are **not**
  comparable row for row with those tables. Every comparison below is between
  arms that share a base.
* **Metrics.** FA and post-attack FA come from `bn_illusion_attack.py`: test-time
  transforms, the canonical `update_bn` operator, recalibration on the retain
  set unless stated otherwise. TA, RA, MIA and wall-clock come from `run.py`.
  ± is the sample standard deviation over seeds {42, 123, 456} (n−1), as in the
  paper's other multi-seed tables.
* **Chance.** 10% for a 10-class problem. "Within chance + 3 SE" means at most
  11.27% for a 5,000-image forget set.

## Headline comparison (CIFAR-10, class 0, three seeds) — Table 6

| variant                         | base TA | FA         | post-attack FA | TA         | RA         | MIA   | min  |
|---------------------------------|---------|------------|----------------|------------|------------|-------|------|
| base model (no unlearning)      | 93.97   | 99.80      | 99.82          | 93.97      | —          | —     | —    |
| EXCISE, frozen statistics       | 93.97   | 8.51±0.30  | 99.12±0.09     | 85.20±0.02 | 97.34±0.05 | 0.505 | 1.9  |
| EXCISE, GroupNorm               | 91.89   | 5.91±0.09  | n/a (no BN)    | 83.13±0.05 | 98.92±0.03 | 0.521 | 2.5  |
| BN-robust, naive stopping       | 93.97   | 11.43±1.90 | 11.44±1.87     | 84.42±0.12 | 95.46±0.10 | 0.491 | 9.8  |
| BN-robust, calibrated stopping  | 93.97   | 9.19±0.08  | 9.18±0.07      | 84.30±0.04 | 95.59±0.02 | 0.483 | 13.3 |

Removing one class from a base with no other loss would give TA ≈ base × 0.9:
84.6 for the BatchNorm base and 82.7 for the GroupNorm base. After the attack,
the frozen configuration's TA returns to 94.1%, i.e. the attack hands back
essentially the original model.

## Files

Run settings use `configs/cifar10_kl_retain.yaml` (or `configs/mnist_kl_retain.yaml`)
plus the flags listed. "BN-robust" means
`--bn_robust exact --bn_recal_every 1 --bn_recal_samples 512 --bn_recal_persistent`
unless stated otherwise. `run_hardening.sh` holds the exact command for every
phase.

| file | what it holds | backs |
|------|---------------|-------|
| `bn_robust_phase17.csv`, `bn_robust_phase18.csv` | first exploration: frozen / exact / EMA variants, 5× and 100× retain budgets | — |
| `bn_robust_phase19_3seeds.csv` | frozen (`off_s*`), exact on 2,048 images (`exact_s*`), exact every 5th step (`exact5_s*`) | Table 6, Fig. 6, Table 20 (class 0) |
| `bn_robust_phase20_512imgs.csv` | exact on 512 images, per-call loader: post-attack FA 11.09±0.98, same as 2,048 images | the 512-image setting (§6.4, App. F.6) |
| `bn_robust_phase21_512imgs_persistent.csv` | BN-robust with naive stopping (`exact512p_s*`) | Table 6, Fig. 6, Table 20 (class 0) |
| `bn_robust_phase22_groupnorm.csv` | EXCISE on the GroupNorm base (`gn_s*`); the attack is undefined on it | Table 6 |
| `bn_robust_phase23_classes1-9.csv` | classes 1–9, seed 42: frozen (`off_c*`) vs BN-robust, naive stopping (`exact512_c*`) | Table 20, Fig. 13 |
| `bn_robust_phase24_recal_{train,test}.csv` | the attack with the attacker's own data: the full training set (forget class included) or the test set (`--recal_on`) | Fig. 6b |
| `bn_robust_phase25_base_control_{retain,train,test}.csv` | the un-unlearned base model under the same attacks: the control | Table 6 |
| `bn_robust_phase26_calibrated_stop.csv` | `--bn_calibrated_stop`, full-set check: 9.19±0.06 | Table 20 (class 0) |
| `bn_robust_phase27_calibrated_classes1-9.csv` | calibrated stopping on classes 1–9 (`cal_c*`), cap 25 epochs | Table 20, Fig. 13 |
| `bn_robust_phase28_class4_restore_best.csv` | class 4 with `--restore_best_epoch`: ships epoch 22 at 10.18% | Table 20, Fig. 9c |
| `bn_robust_phase29_mnist.csv` | MNIST, three seeds: frozen (`off_s*`) vs BN-robust in its final form (`robust_s*`) | Fig. 6a |
| `bn_robust_phase30_capped_check.csv` | the final configuration: calibrated stopping on a 10,240-image check (`--bn_stop_recal_samples 10240`) plus `--restore_best_epoch` | Table 6 |
| `stopping_traces.csv` | per-epoch FA under the in-loop and the shipped statistics for every calibrated run, with the retain loss and the epoch that shipped | Fig. 9 |

Figures 5, 6, 9 and 13 are generated from these files by `gen_bnrobust_figs.py`.

## Findings recorded here in more detail than in the paper

* **The attacker's choice of data does not matter.** Post-attack FA with the
  statistics recomputed on the retain set / the full training set / the test
  set: frozen 99.12 / 97.42 / 97.31; BN-robust (2,048 images) 11.25 / 10.09 /
  10.24; BN-robust (512 images) 11.44 / 10.29 / 10.33. With the forget class in
  the statistics BN-robust moves by about −1pp, towards more forgetting. Under
  the test set, post-attack TA is measured on the recalibration data itself, so
  read that column for FA only.
* **Why naive stopping lands above chance.** The FA that triggers stopping is
  read under the last in-loop statistics (512 augmented images, one step
  behind) on augmented forget images. At the stop, on classes 1–9, the shipped
  FA exceeded that reading by +4.06, +1.48, +2.10, +1.72, −1.02, +0.10, +2.76,
  +6.74 and +9.04pp. Calibrated stopping reads FA under the statistics that
  ship and removes the gap: 8 of 9 classes land within chance + 3 SE, and the
  ninth (class 4, which plateaus at 10.2–10.9%) does too once the best epoch is
  shipped.
* **Cost.** About 7× the wall-clock of the frozen configuration (13.3 vs 1.9
  min), mostly because BN-robust needs about 13 epochs instead of 2. The
  persistent recalibration loader cut the in-loop overhead from 53.6 s to 10.6 s
  per epoch; capping the stopping check at 10,240 images saved a further 16%
  with identical stopping decisions.
* **MNIST.** The attack restores the forgotten class completely under the
  frozen configuration (0.00% → 99.97%, TA back to 99.56%), while BN-robust holds
  (3.78% → 3.87%). Seeds barely matter on MNIST (std ≤ 0.03pp on every metric).

Raw run logs are not released: they record absolute machine paths.
