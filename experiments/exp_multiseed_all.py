"""
3-seed multi-dataset runner for EXCISE.
Runs EXCISE on all datasets with seeds {42, 123, 456} and reports mean +/- std.
"""
import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

os.environ.pop("MKL_THREADING_LAYER", None)
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

ROOT = Path(__file__).resolve().parent.parent

DATASET_CONFIGS = {
    "cifar10":   {"config": "configs/cifar10_kl_retain.yaml",   "classes": [0], "num_classes": 10},
    "mnist":     {"config": "configs/mnist_kl_retain.yaml",     "classes": list(range(10)), "num_classes": 10},
    "svhn":      {"config": "configs/svhn_kl_retain.yaml",      "classes": [0], "num_classes": 10},
    "stl10":     {"config": "configs/stl10_kl_retain.yaml",     "classes": [0], "num_classes": 10},
    "pathmnist": {"config": "configs/pathmnist_kl_retain.yaml", "classes": list(range(9)), "num_classes": 9},
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


def run_one(dataset, forget_class, seed, base_ckpt, data_dir, save_dir,
            use_groupnorm=False):
    config = str(ROOT / DATASET_CONFIGS[dataset]["config"])
    label = f"{dataset}_c{forget_class}_s{seed}"
    if use_groupnorm:
        label += "_gn"

    cmd = [
        sys.executable, str(ROOT / "run.py"),
        "--config", config,
        "--checkpoint", base_ckpt,
        "--data_dir", data_dir,
        "--save_dir", str(Path(save_dir) / label),
        "--forget_class", str(forget_class),
        "--seed", str(seed),
    ]
    if use_groupnorm:
        cmd.append("--use_groupnorm")

    log_dir = Path(save_dir) / label
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "log.txt"

    print(f"  [{label}] running...", end=" ", flush=True)
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)

    metrics = parse_metrics(log_path.read_text())
    metrics.update({"dataset": dataset, "forget_class": forget_class,
                    "seed": seed, "label": label, "returncode": proc.returncode})

    status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
    ta = f"{metrics['TA']:.2f}" if metrics["TA"] is not None else "ERR"
    fa = f"{metrics['FA']:.2f}" if metrics["FA"] is not None else "ERR"
    print(f"{status}  TA={ta} FA={fa}")
    return metrics


def main():
    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASET_CONFIGS.keys()))
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--ckpt_dir", default=str(ROOT / "checkpoints"),
                    help="Directory containing base_<dataset>_resnet18.pth files")
    ap.add_argument("--data_dir", default=str(ROOT / "data"))
    ap.add_argument("--save_dir", default=str(ROOT / "results/multiseed"))
    ap.add_argument("--use_groupnorm", action="store_true")
    ap.add_argument("--classes_per_dataset", type=int, default=None,
                    help="Limit number of classes to forget per dataset (for quick runs)")
    args = ap.parse_args()

    all_results = []
    for dataset in args.datasets:
        cfg = DATASET_CONFIGS[dataset]
        classes = cfg["classes"]
        if args.classes_per_dataset is not None:
            classes = classes[:args.classes_per_dataset]

        gn_tag = "_groupnorm" if args.use_groupnorm else ""
        base_ckpt = os.path.join(args.ckpt_dir,
                                 f"base_{dataset}_resnet18{gn_tag}.pth")
        if not os.path.exists(base_ckpt):
            print(f"\n[SKIP] {dataset}: base checkpoint not found at {base_ckpt}")
            continue

        print(f"\n{'='*60}")
        print(f"Dataset: {dataset} | Classes: {classes} | Seeds: {args.seeds}")
        print(f"{'='*60}")

        for fc in classes:
            for seed in args.seeds:
                result = run_one(dataset, fc, seed, base_ckpt, args.data_dir,
                                 args.save_dir, args.use_groupnorm)
                all_results.append(result)

    out_csv = Path(args.save_dir) / "multiseed_all.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "forget_class", "seed", "TA", "FA", "RA", "MIA",
              "returncode", "label"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_results:
            w.writerow({k: r.get(k) for k in fields})

    print(f"\n{'='*60}")
    print("Multi-Seed Summary (mean +/- std)")
    print(f"{'='*60}")
    print(f"{'Dataset':<12} {'Class':>5} {'TA':>14} {'FA':>14} {'RA':>14} {'MIA':>14}")
    print("-" * 75)

    for dataset in args.datasets:
        cfg = DATASET_CONFIGS[dataset]
        classes = cfg["classes"]
        if args.classes_per_dataset is not None:
            classes = classes[:args.classes_per_dataset]
        for fc in classes:
            rows = [r for r in all_results
                    if r["dataset"] == dataset and r["forget_class"] == fc
                    and r["TA"] is not None]
            if not rows:
                continue
            for metric in ["TA", "FA", "RA", "MIA"]:
                vals = [r[metric] for r in rows if r[metric] is not None]
                if vals:
                    mean, std = np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else 0
                    rows[0][f"{metric}_str"] = f"{mean:.2f}+/-{std:.2f}"
                else:
                    rows[0][f"{metric}_str"] = "N/A"
            r = rows[0]
            print(f"{dataset:<12} {fc:>5} {r['TA_str']:>14} {r['FA_str']:>14} "
                  f"{r['RA_str']:>14} {r['MIA_str']:>14}")

    print(f"\nCSV saved: {out_csv}")


if __name__ == "__main__":
    main()
