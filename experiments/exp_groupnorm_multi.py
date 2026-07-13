"""
GroupNorm multi-dataset experiment.

For each dataset: train GN base model -> run PURGE+GN -> run BN recalibration
diagnostic -> report results. Uses 3 seeds.
"""
import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

os.environ.pop("MKL_THREADING_LAYER", None)
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

ROOT = Path(__file__).resolve().parent.parent

DATASET_CONFIGS = {
    "cifar10":   {"config": "configs/cifar10_kl_retain.yaml",   "forget_class": 0},
    "mnist":     {"config": "configs/mnist_kl_retain.yaml",     "forget_class": 0},
    "svhn":      {"config": "configs/svhn_kl_retain.yaml",      "forget_class": 0},
    "stl10":     {"config": "configs/stl10_kl_retain.yaml",     "forget_class": 0},
    "pathmnist": {"config": "configs/pathmnist_kl_retain.yaml", "forget_class": 0},
}

SEEDS = [42, 123, 456]


def parse_metrics(log_text):
    out = {}
    patterns = {
        "TA":  r"Test Accuracy\s*\(TA\)\s*:\s*([\d.]+)%",
        "FA":  r"Forget Accuracy\s*\(FA[^)]*\)\s*:\s*([\d.]+)%",
        "RA":  r"Retain Accuracy\s*\(RA[^)]*\)\s*:\s*([\d.]+)%",
        "MIA": r"MIA \(average\)\s*:\s*([\d.]+)",
    }
    for k, p in patterns.items():
        m = re.search(p, log_text)
        out[k] = float(m.group(1)) if m else None
    return out


def train_gn_base(dataset, data_dir, save_dir, seed):
    """Train GroupNorm base model for a dataset."""
    save_path = os.path.join(save_dir, f"base_{dataset}_resnet18_groupnorm.pth")
    if os.path.exists(save_path):
        print(f"  [skip] GN base model exists: {save_path}")
        return save_path

    cmd = [
        sys.executable, str(ROOT / "experiments" / "train_base_all.py"),
        "--datasets", dataset,
        "--data_dir", data_dir,
        "--save_dir", save_dir,
        "--seed", str(seed),
        "--use_groupnorm",
        "--epochs", "200",
    ]
    print(f"  Training GN base model for {dataset}...")
    subprocess.run(cmd, text=True)
    return save_path


def run_purge_gn(dataset, forget_class, seed, base_ckpt, config, data_dir,
                 save_dir):
    label = f"gn_{dataset}_c{forget_class}_s{seed}"
    cmd = [
        sys.executable, str(ROOT / "run.py"),
        "--config", config,
        "--checkpoint", base_ckpt,
        "--data_dir", data_dir,
        "--save_dir", str(Path(save_dir) / label),
        "--forget_class", str(forget_class),
        "--seed", str(seed),
        "--use_groupnorm",
    ]
    log_dir = Path(save_dir) / label
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "log.txt"

    print(f"  [{label}] PURGE+GN running...", end=" ", flush=True)
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)

    metrics = parse_metrics(log_path.read_text())
    status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
    ta = f"{metrics.get('TA','?')}" if metrics.get("TA") else "ERR"
    print(f"{status} TA={ta}")
    metrics.update({"dataset": dataset, "forget_class": forget_class,
                    "seed": seed, "label": label})
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASET_CONFIGS.keys()))
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--ckpt_dir", default=str(ROOT / "checkpoints"))
    ap.add_argument("--data_dir", default=str(ROOT / "data"))
    ap.add_argument("--save_dir", default=str(ROOT / "results/groupnorm_multi"))
    args = ap.parse_args()

    all_results = []
    for dataset in args.datasets:
        cfg = DATASET_CONFIGS[dataset]
        fc = cfg["forget_class"]
        config = str(ROOT / cfg["config"])

        print(f"\n{'='*60}")
        print(f"GroupNorm experiment: {dataset}")
        print(f"{'='*60}")

        base_ckpt = train_gn_base(dataset, args.data_dir, args.ckpt_dir,
                                  args.seeds[0])
        if not os.path.exists(base_ckpt):
            print(f"  [SKIP] GN base model not found: {base_ckpt}")
            continue

        for seed in args.seeds:
            result = run_purge_gn(dataset, fc, seed, base_ckpt, config,
                                  args.data_dir, args.save_dir)
            all_results.append(result)

    out_csv = Path(args.save_dir) / "groupnorm_multi_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "forget_class", "seed", "TA", "FA", "RA", "MIA", "label"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_results:
            w.writerow({k: r.get(k) for k in fields})

    print(f"\n{'='*60}")
    print("GroupNorm Multi-Dataset Summary (mean +/- std)")
    print(f"{'='*60}")
    for dataset in args.datasets:
        rows = [r for r in all_results if r["dataset"] == dataset
                and r.get("TA") is not None]
        if not rows:
            continue
        for metric in ["TA", "FA", "RA", "MIA"]:
            vals = [r[metric] for r in rows if r.get(metric) is not None]
            if vals:
                mean = np.mean(vals)
                std = np.std(vals, ddof=1) if len(vals) > 1 else 0
                print(f"  {dataset:<12} {metric}: {mean:.2f} +/- {std:.2f}")
    print(f"\nCSV: {out_csv}")


if __name__ == "__main__":
    main()
