"""
Epsilon / tolerance sweeps for PURGE (Sweep A and Sweep B in paper).
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


SWEEP_A_BUDGET = [1.5, 2.0, 3.0, 5.0, 10.0, 9999.0]   # retain_budget_factor
SWEEP_B_FA     = [1.0, 5.0, 10.0, 20.0, 30.0]         # fa_target


def parse_metrics(log_text: str) -> dict:
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


def run_one(args, label, alpha, fa_tgt):
    cmd = [
        sys.executable, str(Path(args.purge_dir) / "run.py"),
        "--dataset", "cifar10", "--model", "resnet18",
        "--forget_type", "class", "--forget_class", "0",
        "--checkpoint", args.base_ckpt,
        "--data_dir", args.data_dir,
        "--save_dir", str(Path(args.save_dir) / label),
        "--epochs", "15", "--lr", "5e-4",
        "--rep_weight", "0.05", "--kd_weight", "2.0",
        "--max_grad_norm", "1.0",
        "--forget_objective", "kl_retain",
        "--forget_gate", "0",
        "--retain_budget_factor", str(alpha),
        "--fa_target", str(fa_tgt),
        "--seed", str(args.seed),
    ]
    log_dir = Path(args.save_dir) / label
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "log.txt"
    print(f"\n[run] {label}  (alpha={alpha}, fa_target={fa_tgt})")
    with open(log_path, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    m = parse_metrics(log_path.read_text())
    m["label"] = label
    m["alpha"] = alpha
    m["fa_target"] = fa_tgt
    return m


def write_csv(rows, path, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def plot_sweep(rows, x_key, x_label, save_path, title):
    """Two-panel: TA / RA vs x, and FA / MIA vs x."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"matplotlib not installed; skipping plot {save_path}")
        return
    xs = [r[x_key] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(xs, [r["TA"] for r in rows], "o-", label="TA")
    ax1.plot(xs, [r["RA"] for r in rows], "s-", label="RA")
    ax1.set_xlabel(x_label); ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Utility vs " + x_label); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(xs, [r["FA"] for r in rows], "^-", color="#e74c3c", label="FA")
    ax2.axhline(10.0, color="gray", ls=":", lw=1, label="random chance (10%)")
    ax2b = ax2.twinx()
    ax2b.plot(xs, [r["MIA"] for r in rows], "D-", color="#3498db", label="MIA (right)")
    ax2b.axhline(0.5, color="lightgray", ls=":", lw=1)
    ax2.set_xlabel(x_label); ax2.set_ylabel("FA (%)", color="#e74c3c")
    ax2b.set_ylabel("MIA AUROC", color="#3498db")
    ax2.set_title("Forgetting vs " + x_label)
    ax2.grid(True, alpha=0.3)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"saved {save_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--purge_dir", default=str(ROOT))
    ap.add_argument("--base_ckpt", default=str(ROOT / "checkpoints/base_cifar10_resnet18.pth"))
    ap.add_argument("--data_dir",  default=str(ROOT / "data"))
    ap.add_argument("--save_dir",  default=str(ROOT / "results/epsilon_sweep"))
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--only", choices=["A", "B"], default=None,
                    help="Run only Sweep A or Sweep B (default: both).")
    args = ap.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.only is None or args.only == "A":
        # Sweep A: vary retain budget (epsilon_R), fix FA target
        sweep_A = []
        for alpha in SWEEP_A_BUDGET:
            label = f"A_alpha_{alpha:g}"
            sweep_A.append(run_one(args, label, alpha, 10.0))

        write_csv(sweep_A, save_dir / "sweep_A_retain_budget.csv",
                  ["label", "alpha", "fa_target", "TA", "FA", "RA", "MIA"])
        plot_sweep(sweep_A, "alpha", "retain budget factor (alpha)",
                   save_dir / "sweep_A_retain_budget.png",
                   "Sweep A: retain-budget tolerance (CIFAR-10 class 0)")

    if args.only is None or args.only == "B":
        # Sweep B: vary FA target (epsilon_F), fix retain budget
        sweep_B = []
        for fa_tgt in SWEEP_B_FA:
            label = f"B_fa_{fa_tgt:g}"
            sweep_B.append(run_one(args, label, 5.0, fa_tgt))

        write_csv(sweep_B, save_dir / "sweep_B_fa_target.csv",
                  ["label", "alpha", "fa_target", "TA", "FA", "RA", "MIA"])
        plot_sweep(sweep_B, "fa_target", "FA target (%)",
                   save_dir / "sweep_B_fa_target.png",
                   "Sweep B: forget-distribution tolerance (CIFAR-10 class 0)")

    print(f"\nDone. CSVs and plots saved under {save_dir}.")


if __name__ == "__main__":
    main()
