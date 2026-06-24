"""
Component-isolation ablation for PURGE (Table: component ablation in paper).

Disables exactly one PURGE component at a time on CIFAR-10 class 0.
"""

import argparse
import csv
import os
import re
import subprocess
import sys

os.environ.pop("MKL_THREADING_LAYER", None)
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


CONFIGS = [
    # (label, [extra --flag value] pairs to ADD on top of the recommended config)
    ("minus_kd",       ["--kd_weight", "0.0"]),
    ("minus_rep",      ["--rep_weight", "0.0"]),
    ("minus_stopping", ["--retain_budget_factor", "9999.0",
                        "--fa_target", "-1.0"]),
    ("minus_gate_cap", ["--forget_gate", "0.0"]),  # no-op safety check
    ("minus_proj",     ["--disable_projection"]),
    ("selective_layer", ["--freeze_early_layers"]),
]


def parse_metrics(log_text: str) -> dict:
    """Extract TA / FA / RA / MIA-avg from a PURGE run.py log."""
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


def check_projection_flag(run_py: Path) -> bool:
    """Check whether run.py supports --disable_projection."""
    if not run_py.exists():
        return False
    return "--disable_projection" in run_py.read_text()


def run_one(args, label: str, extra_flags: list) -> dict:
    cmd = [
        sys.executable, str(Path(args.purge_dir) / "run.py"),
        "--dataset", "cifar10",
        "--model", "resnet18",
        "--forget_type", "class",
        "--forget_class", "0",
        "--checkpoint", args.base_ckpt,
        "--data_dir", args.data_dir,
        "--save_dir", str(Path(args.save_dir) / label),
        "--epochs", "15",
        "--lr", "5e-4",
        "--rep_weight", "0.05",
        "--kd_weight", "2.0",
        "--max_grad_norm", "1.0",
        "--forget_objective", "kl_retain",
        "--forget_gate", "0",
        "--retain_budget_factor", "5.0",
        "--fa_target", "10.0",
        "--seed", str(args.seed),
    ] + extra_flags

    log_dir = Path(args.save_dir) / label
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "log.txt"

    print(f"\n{'='*60}\nRunning: {label}\n{'='*60}")
    print("  " + " ".join(cmd))

    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    log_text = log_path.read_text()
    metrics = parse_metrics(log_text)
    metrics["label"] = label
    metrics["returncode"] = proc.returncode
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--purge_dir", default=str(ROOT))
    ap.add_argument("--base_ckpt", default=str(ROOT / "checkpoints/base_cifar10_resnet18.pth"))
    ap.add_argument("--data_dir",  default=str(ROOT / "data"))
    ap.add_argument("--save_dir",  default=str(ROOT / "results/ablation"))
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--skip_proj", action="store_true",
                    help="Skip the -Projection run (use this if you cannot patch run.py).")
    args = ap.parse_args()

    run_py = Path(args.purge_dir) / "run.py"
    has_proj_flag = check_projection_flag(run_py)
    if not has_proj_flag and not args.skip_proj:
        print("\n[WARNING] run.py does not currently support --disable_projection.")
        print("To run the -Projection ablation, apply the following one-line patch to run.py:\n")
        print("""    # Inside purge_unlearn(...) around the projection step:
    if getattr(args, 'disable_projection', False):
        # skip projection: use raw forget gradient
        pass
    elif inner_product < 0:
        g_f = g_f - (inner_product / (g_r_norm**2 + 1e-12)) * g_r

    # And in main()'s argparse:
    p.add_argument('--disable_projection', action='store_true',
                   help='Ablation: skip the A-GEM projection step.')
""")
        print("Re-run this script after patching, or pass --skip_proj to skip it.")
        # Continue but mark the projection row TBD
        skip_proj = True
    else:
        skip_proj = args.skip_proj

    results = []
    for label, flags in CONFIGS:
        if label == "minus_proj" and skip_proj:
            results.append({"label": label, "TA": None, "FA": None,
                            "RA": None, "MIA": None,
                            "returncode": -1, "note": "skipped"})
            continue
        results.append(run_one(args, label, flags))

    # Write CSV summary
    out_csv = Path(args.save_dir) / "component_ablation_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["label", "TA", "FA", "RA", "MIA", "returncode", "note"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            row = {k: r.get(k) for k in fields}
            w.writerow(row)

    # Pretty print
    print(f"\n\n{'='*70}")
    print(f"Component-Isolation Ablation Summary (seed={args.seed})")
    print(f"{'='*70}")
    print(f"{'Config':<18} {'TA':>8} {'FA':>8} {'RA':>8} {'MIA':>8}")
    print("-" * 70)
    for r in results:
        ta = f"{r['TA']:.2f}" if r["TA"] is not None else "TBD"
        fa = f"{r['FA']:.2f}" if r["FA"] is not None else "TBD"
        ra = f"{r['RA']:.2f}" if r["RA"] is not None else "TBD"
        mi = f"{r['MIA']:.4f}" if r["MIA"] is not None else "TBD"
        print(f"{r['label']:<18} {ta:>8} {fa:>8} {ra:>8} {mi:>8}")
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
