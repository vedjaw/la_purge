"""
Sequential unlearning: forget class 0 -> class 1 -> class 2 -> ...

Measures cumulative TA/FA/RA drift after each sequential forget step.
"""
import argparse
import copy
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


def find_checkpoint(save_dir):
    """Find the .pth file saved by run.py in a directory."""
    pths = list(Path(save_dir).glob("purge_*.pth"))
    if pths:
        return str(max(pths, key=lambda p: p.stat().st_mtime))
    pths = list(Path(save_dir).glob("*.pth"))
    return str(pths[0]) if pths else None


def run_one_step(dataset, forget_class, checkpoint, config, data_dir,
                 save_dir, seed):
    label = f"seq_c{forget_class}"
    cmd = [
        sys.executable, str(ROOT / "run.py"),
        "--config", config,
        "--checkpoint", checkpoint,
        "--data_dir", data_dir,
        "--save_dir", str(Path(save_dir) / label),
        "--forget_class", str(forget_class),
        "--seed", str(seed),
    ]

    log_dir = Path(save_dir) / label
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "log.txt"

    print(f"\n  Sequential step: forget class {forget_class}")
    print(f"  Input checkpoint: {checkpoint}")
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)

    metrics = parse_metrics(log_path.read_text())
    out_ckpt = find_checkpoint(str(log_dir))

    print(f"  -> TA={metrics.get('TA','?')} FA={metrics.get('FA','?')} "
          f"RA={metrics.get('RA','?')} MIA={metrics.get('MIA','?')}")

    return metrics, out_ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--forget_classes", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs/cifar10_kl_retain.yaml"))
    ap.add_argument("--data_dir", default=str(ROOT / "data"))
    ap.add_argument("--save_dir", default=str(ROOT / "results/sequential"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"{'='*60}")
    print(f"Sequential Unlearning: {args.dataset}")
    print(f"  Forget order: {args.forget_classes}")
    print(f"{'='*60}")

    current_ckpt = args.base_ckpt
    all_results = []
    forgotten_so_far = []

    for fc in args.forget_classes:
        forgotten_so_far.append(fc)
        metrics, new_ckpt = run_one_step(
            args.dataset, fc, current_ckpt, args.config,
            args.data_dir, args.save_dir, args.seed)
        metrics["step"] = len(forgotten_so_far)
        metrics["forget_class"] = fc
        metrics["forgotten_so_far"] = str(forgotten_so_far)
        all_results.append(metrics)

        if new_ckpt is None:
            print(f"  ERROR: no output checkpoint found. Stopping.")
            break
        current_ckpt = new_ckpt

    out_csv = Path(args.save_dir) / "sequential_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["step", "forget_class", "forgotten_so_far", "TA", "FA", "RA", "MIA"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_results:
            w.writerow({k: r.get(k) for k in fields})

    print(f"\n{'='*60}")
    print("Sequential Unlearning Summary")
    print(f"{'='*60}")
    print(f"{'Step':>4} {'Forgot':>6} {'Cumulative':<16} {'TA':>8} {'FA':>8} {'RA':>8} {'MIA':>8}")
    print("-" * 70)
    for r in all_results:
        ta = f"{r['TA']:.2f}" if r["TA"] is not None else "ERR"
        fa = f"{r['FA']:.2f}" if r["FA"] is not None else "ERR"
        ra = f"{r['RA']:.2f}" if r["RA"] is not None else "ERR"
        mi = f"{r['MIA']:.4f}" if r["MIA"] is not None else "ERR"
        print(f"{r['step']:>4} {r['forget_class']:>6} {r['forgotten_so_far']:<16} "
              f"{ta:>8} {fa:>8} {ra:>8} {mi:>8}")
    print(f"\nCSV: {out_csv}")


if __name__ == "__main__":
    main()
