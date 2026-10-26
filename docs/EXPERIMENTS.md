# Experiments

How every table and figure in [`paper/excise_tmlr.pdf`](../paper/excise_tmlr.pdf)
is produced. Run everything from the repository root. Each experiment writes CSVs
under `results/`, the published numbers are those CSVs, and the figures are
redrawn from them by the `gen_*.py` scripts. Every script takes `--help`.

Seeds are `{42, 123, 456}` throughout. ± is the sample standard deviation over
seeds (and over classes where a row aggregates several).

EXCISE was called PURGE when most of these runs were made, so checkpoint names
(`purge_*.pth`), some result folders and the method labels inside the CSVs use
the old name.

---

## 0. Base models (once per dataset)

```bash
# ImageNet-pretrained ResNet-18, fine-tuned -> checkpoints/base_<dataset>_resnet18.pth
python experiments/train_base_all.py --datasets cifar10 mnist svhn stl10 pathmnist cifar100 \
  --data_dir ./data --save_dir ./checkpoints

# GroupNorm ResNet-18, trained from scratch (ImageNet weights carry BatchNorm)
python experiments/train_groupnorm_base.py --data_dir ./data \
  --save_path ./checkpoints/base_cifar10_resnet18_groupnorm.pth --epochs 200

# Retrained-from-scratch references (no forget class), for Appendices F.11–F.12
python experiments/retrain_reference.py --dataset cifar10 --forget_class 0
```

## 1. One EXCISE run

```bash
python run.py --config configs/<dataset>_kl_retain.yaml \
  --checkpoint ./checkpoints/base_<dataset>_resnet18.pth \
  --data_dir ./data --save_dir ./checkpoints [--forget_class k] [--seed s]
```

The configuration files set every hyperparameter (Appendix I); the paper uses
them unmodified.

---

## 2. Map: paper → script → CSV

Rows follow the order in which results first appear in the paper.

| Paper | What | Command | Output |
|-------|------|---------|--------|
| Fig. 3, App. F.4 | step-level audit of the guarantee | `python experiments/exp_perstep_trajectory.py --checkpoint <base.pth>` | `results/perstep/perstep_cifar10_c0_s42.csv` |
| App. F.4 | inner-product histogram | `python experiments/exp_innerproduct_histogram.py --checkpoint <base.pth>` | `results/innerproduct/` |
| Table 2, Fig. 7 | cross-dataset results, 31 tasks | `python experiments/exp_multiseed_all.py` | `results/multiseed/multiseed_all.csv`, `results/cifar100/cifar100_summary_projmom.csv` |
| Table 2, lower block | STL-10 and CIFAR-100 at 60 epochs | `bash run_hardening.sh 2 8` | `results/hardening/rescue.csv`, `results/hardening/table_grade.csv` |
| Table 3 | CIFAR-10 class 0 against baselines | EXCISE row as Table 2; projection-only is `run.py --forget_objective ga --kd_weight 0` | — |
| Table 4, Fig. 8 | momentum ablation (μ = 0 vs projected momentum) | the runs of Table 2 with `--momentum 0` | `results/archive_mu0/*_MU0.csv` |
| Table 5 | BatchNorm-recalibration attack on EXCISE and baselines | `bash run_hardening.sh 9`, or `bn_illusion_attack.py` (below) | `results/hardening/bn_illusion.csv`, `bn_illusion_vit.csv` |
| Tables 6 and 20, Figs. 5, 6, 9 and 13 | BN-robust EXCISE | `bash run_hardening.sh 19 20 21 22 23 24 25 26 27 28 29 30` | `results/hardening/bn_robust/` (see its README) |
| Table 8 | CIFAR-100, nine classes | `python experiments/exp_multiseed_all.py --datasets cifar100` | `results/cifar100/cifar100_summary_projmom.csv` |
| Table 9 | where the momentum dependence comes from | `bash run_hardening.sh 12 14` | `results/hardening/momentum_datasets.csv`, `momentum_rules.csv` |
| Table 10 | GroupNorm backbone, five datasets | `python experiments/exp_groupnorm_multi.py` | `results/groupnorm_multi/groupnorm_multi_summary.csv` |
| Table 11 | ViT-B/16 on CIFAR-10 | `bash run_hardening.sh 10` (`run.py --model vit_b_16`) | `results/vit/`, `results/hardening/vit.csv` |
| Table 12 | component ablation | `python experiments/component_ablation.py` | `results/ablation/component_ablation_summary.csv` |
| Table 13, Fig. 10 | projection strength, from soft penalty to hard constraint | `run.py --projection_strength s --fa_target -1 --retain_budget_factor 9999`, s ∈ {0, 0.25, 0.5, 0.75, 1} | `results/strength_sweep.csv` |
| Table 14 | retain-budget and forget-target sweeps | `python experiments/epsilon_sweep.py` | `results/epsilon_sweep/`, `results_sweep/` |
| App. F.10 | adaptive forget-accuracy check | `bash run_hardening.sh 1` | `results/hardening/eps_adaptive.csv` |
| Table 15 | distance to a retrained model | `python experiments/distance_to_retrained.py` | `results/todo/distance_cifar10.csv`, `results/kl_stopping/distance_to_retrained.csv` |
| Tables 16–17 | membership inference against a retrained reference | `python experiments/mia_suite.py`, `bash run_hardening.sh 5 6` | `results/todo/mia_suite_cifar10.csv`, `results/hardening/mia_suite_pathmnist*.csv` |
| Fig. 11, Table 23 | over-forgetting, KL stopping, where to stop | `python experiments/trajectory_sweep.py`, `python experiments/validate_kl_stopping.py` | `results/trajectory/trajectory_sweep.csv`, `results/kl_stopping/` |
| Tables 18–19, Fig. 12 | per-class results, MNIST and PathMNIST | the runs of Table 2 | `results/multiseed/multiseed_all.csv` |
| §7, App. G | sample-level forgetting | `bash run_hardening.sh 3 7` | `results/hardening/sample_level*.csv` |
| §7, App. G | sequential unlearning, fine-tuning recovery | `experiments/exp_sequential_unlearn.py`, `experiments/exp_finetune_recovery.py` | `results/sequential/`, `results/finetune_recovery/`, `results/todo/` |

The SalUn, Bad Teacher, SCRUB and NegGrad+ rows of Tables 3 and 5 come from
separate implementations of those methods, which are not part of this
repository.

Table and figure numbers follow the current build of the paper.
`run_hardening.sh` runs numbered phases (`bash run_hardening.sh <n> [<n> ...]`);
each phase's comment block states what it tests and why, and `collect_hardening.py`
gathers phase logs into the CSVs. `run_all_experiments.sh`,
`run_mia_and_reference.sh` and `run_remaining_todos.sh` are the drivers for the
other result folders.

---

## 3. BatchNorm: attack and BN-robust EXCISE

Attack any checkpoint. The attack resets the BatchNorm statistics, recomputes
them on the retain set, checks that every weight is bit-identical, and reports
forget accuracy before and after. Use `--recal_on train` (forget class included)
or `--recal_on test` to let the attacker pick the data.

```bash
python experiments/bn_illusion_attack.py --dataset cifar10 --forget_class 0 --data_dir ./data \
  --checkpoints frozen:<unlearned.pth> robust:<bn_robust.pth> --out_csv results/bn_attack.csv
```

BN-robust EXCISE, final configuration (Table 6, last row):

```bash
python run.py --config configs/cifar10_kl_retain.yaml \
  --checkpoint ./checkpoints/base_cifar10_resnet18.pth --data_dir ./data --save_dir ./checkpoints \
  --bn_robust exact --bn_recal_every 1 --bn_recal_samples 512 --bn_recal_persistent \
  --bn_calibrated_stop --bn_stop_recal_samples 10240 --restore_best_epoch --epochs 25
```

| flag | what it does |
|------|--------------|
| `--bn_robust exact` | recompute the BatchNorm statistics from retain data before every step, then on the full retain set after the last one |
| `--bn_recal_samples 512` | retain images per in-loop recalibration |
| `--bn_recal_persistent` | draw them from one persistent loader (about 5× less overhead) |
| `--bn_calibrated_stop` | read the stopping forget accuracy under the statistics that ship |
| `--bn_stop_recal_samples 10240` | recalibrate on a 10,240-image subset for that check (same decisions, 16% faster) |
| `--restore_best_epoch` | if the target is not reached, ship the lowest-FA epoch within the retain budget |

---

## 4. Figures

| Script | Figures |
|--------|---------|
| `gen_schematic_figs.py` | `fig_pipeline.png` (Fig. 1), `fig_projection_order.png` (Fig. 2) |
| `gen_advantage_fig.py` | `fig_advantage.png` (Fig. 4) |
| `gen_paper_figs.py` | `fig_guarantee.png` (Fig. 3), `fig_crossdataset.png` (Fig. 7), `fig_momentum.png` (Fig. 8), `fig_perclass.png` (Fig. 12) |
| `gen_strength_fig.py` | `fig_strength.png` (Fig. 10) |
| `gen_trajectory_fig.py` | `fig_trajectory.png` (Fig. 11) |
| `gen_bnrobust_figs.py` | `fig_bnrobust_mechanism.png` (Fig. 5), `fig_bnrobust_attack.png` (Fig. 6), `fig_bnrobust_stopping.png` (Fig. 9), `fig_bnrobust_perclass.png` (Fig. 13) |

`gen_pareto.py` and `gen_purge_vs_salun_fig.py` draw figures used only by the
earlier AAAI version in `paper/`.

---

## Compute

All runs used one GPU. On an A100, a CIFAR-10 EXCISE run takes about 2 minutes,
BN-robust EXCISE about 13, and a ResNet-18 base model under an hour. Wall-clock
time depends on the GPU and on when dual stopping fires, so step counts are the
portable measure.
