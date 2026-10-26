"""
Component-isolation ablation for EXCISE (component-ablation table in the paper).

Disables (or re-adds) exactly one EXCISE component at a time on CIFAR-10 class 0.

Two modes, because dual stopping confounds a naive ablation
--------------------------------------------------------------------------
With the projected-momentum optimiser, every variant forgets fast enough that
dual stopping fires almost immediately, so all variants look equally good
(retain damage never has time to develop) and the ablation loses its power.

  * mode "stopping" — dual stopping active, i.e. the deployed configuration.
    Answers "does removing X hurt the method as shipped?".  This is the only
    mode in which the `minus_stopping` row is meaningful.

  * mode "fixed" — dual stopping DISABLED for every variant
    (--fa_target -1 --retain_budget_factor 9999), so all variants run the same
    number of epochs.  Answers "what does X actually contribute per unit of
    compute?", with the stopping confound removed.  `minus_stopping` is skipped
    here because it would be identical to `full` by construction.

Both modes are run by default and written to one CSV with a `mode` column.

Hyper-parameters come from the dataset CONFIG (configs/cifar10_kl_retain.yaml)
rather than being hardcoded, so the ablation always tracks the shipped method.
Because run.py lets explicit CLI flags override config values, each variant is
exactly "the config, with one thing changed".

Note on `plus_rep` / `plus_gate`: representation erasure and the entropy gate
were REMOVED from EXCISE (ablation-inactive), so the config ships them at 0.
Disabling them would be a no-op; instead we re-ADD them to verify they still
fail to help — the up-to-date evidence for having dropped them.
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

# (label, extra flags on top of the config, description, stopping-mode-only?)
VARIANTS = [
    ("full",            [],                                    "config as shipped",            False),
    ("minus_kd",        ["--kd_weight", "0.0"],                "no KD anchor",                 False),
    ("minus_proj",      ["--disable_projection"],              "no A-GEM projection",          False),
    ("selective_layer", ["--freeze_early_layers"],             "freeze conv1-layer2",          False),
    ("plus_rep",        ["--rep_weight", "0.05"],              "re-add representation erasure", False),
    ("plus_gate",       ["--forget_gate", "0.9"],              "re-add entropy gate",          False),
    # meaningful only when stopping is active (in fixed mode it == full):
    ("minus_stopping",  ["--retain_budget_factor", "9999.0",
                         "--fa_target", "-1.0"],               "no dual stopping",             True),
]

# flags that switch dual stopping off for every variant in fixed-budget mode
FIXED_BUDGET_FLAGS = ["--fa_target", "-1.0", "--retain_budget_factor", "9999.0"]


def parse_metrics(log_text: str) -> dict:
    """Extract metrics + run diagnostics from an EXCISE run.py log."""
    out = {}
    patterns = {
        "TA":         r"Test Accuracy\s*\(TA\)\s*:\s*([\d.]+)%",
        "FA":         r"Forget Accuracy\s*\(FA[^)]*\)\s*:\s*([\d.]+)%",
        "RA":         r"Retain Accuracy\s*\(RA[^)]*\)\s*:\s*([\d.]+)%",
        "MIA":        r"MIA \(average\)\s*:\s*([\d.]+)",
        "feat_cos":   r"Feature [Cc]osine [Ss]im\w*\s*:\s*([\d.]+)",
        "proj_rate":  r"Avg Projection Rate\s*:\s*([\d.]+)%",
        "gate_rate":  r"Avg Gate Rate\s*:\s*([\d.]+)%",
        "seconds":    r"Unlearning completed in\s*([\d.]+)s",
    }
    for k, p in patterns.items():
        m = re.search(p, log_text)
        out[k] = float(m.group(1)) if m else None

    # epochs actually executed = number of per-epoch summary lines
    out["epochs_run"] = len(re.findall(r"Forget CE:", log_text))

    # Why the run ended — the key diagnostic for the stopping confound.
    # run.py emits several distinct markers; check them all, because a stop
    # triggered during the FINAL epoch never prints the loop-top
    # "stopping at epoch N" line.
    fa_markers = ("FA target reached", "forgetting complete, will stop",
                  "stopping mid-epoch")
    budget_markers = ("Retain budget exceeded", "will stop after this epoch")
    if any(s in log_text for s in fa_markers):
        out["stop_reason"] = "fa_target"
    elif any(s in log_text for s in budget_markers):
        out["stop_reason"] = "retain_budget"
    else:
        out["stop_reason"] = "all_epochs"
    out["stopped_midepoch"] = int("stopping mid-epoch" in log_text)
    m = re.search(r"stopping at epoch (\d+)", log_text)
    out["stopped_epoch"] = int(m.group(1)) if m else None
    return out


def run_one(args, mode: str, label: str, extra_flags: list) -> dict:
    tag = f"{mode}__{label}"
    cmd = [
        sys.executable, str(Path(args.excise_dir) / "run.py"),
        "--config", args.config,
        "--checkpoint", args.base_ckpt,
        "--data_dir", args.data_dir,
        "--save_dir", str(Path(args.save_dir) / tag),
        "--forget_class", str(args.forget_class),
        "--seed", str(args.seed),
    ]
    if mode == "fixed":
        cmd += FIXED_BUDGET_FLAGS
    cmd += extra_flags

    log_dir = Path(args.save_dir) / tag
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "log.txt"

    print(f"\n{'=' * 68}")
    print(f"[{mode}] {label}")
    print(f"{'=' * 68}")
    print("  " + " ".join(cmd))

    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)

    metrics = parse_metrics(log_path.read_text())
    metrics.update({"mode": mode, "label": label, "returncode": proc.returncode})

    ta = f"{metrics['TA']:.2f}" if metrics["TA"] is not None else "ERR"
    fa = f"{metrics['FA']:.2f}" if metrics["FA"] is not None else "ERR"
    ra = f"{metrics['RA']:.2f}" if metrics["RA"] is not None else "ERR"
    status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
    print(f"  -> {status}  TA={ta} FA={fa} RA={ra}  "
          f"epochs={metrics['epochs_run']} stop={metrics['stop_reason']}")
    return metrics


def print_table(results, mode):
    rows = [r for r in results if r["mode"] == mode and r["TA"] is not None]
    if not rows:
        print(f"\n[{mode}] no successful runs.")
        return
    full = next((r for r in rows if r["label"] == "full"), None)

    title = ("dual stopping ACTIVE (as deployed)" if mode == "stopping"
             else "FIXED BUDGET — stopping disabled for all variants")
    print(f"\n{'=' * 108}")
    print(f"  Component ablation [{mode}] — {title}")
    print(f"{'=' * 108}")
    print(f"  {'config':<17}{'TA':>7}{'FA':>7}{'RA':>7}{'MIA':>8}"
          f"{'dTA':>8}{'dFA':>8}{'dRA':>8}{'ep':>4}{'stop':>15}{'proj%':>7}")
    print("  " + "-" * 104)
    for r in rows:
        if full and r is not full and full["TA"] is not None:
            dta = f"{r['TA'] - full['TA']:+.2f}"
            dfa = f"{r['FA'] - full['FA']:+.2f}"
            dra = f"{r['RA'] - full['RA']:+.2f}"
        else:
            dta = dfa = dra = "  ref"
        mia = f"{r['MIA']:.4f}" if r["MIA"] is not None else "  n/a"
        pr = f"{r['proj_rate']:.0f}" if r["proj_rate"] is not None else "n/a"
        print(f"  {r['label']:<17}{r['TA']:7.2f}{r['FA']:7.2f}{r['RA']:7.2f}{mia:>8}"
              f"{dta:>8}{dfa:>8}{dra:>8}{r['epochs_run']:>4}{r['stop_reason']:>15}{pr:>7}")

    if mode == "fixed":
        eps = {r["epochs_run"] for r in rows}
        note = ("all variants ran the same number of epochs — comparison is clean"
                if len(eps) == 1 else
                f"WARNING: epoch counts differ {sorted(eps)} — check for crashes")
        print(f"\n  {note}")
    else:
        early = [r["label"] for r in rows if r["stop_reason"] != "all_epochs"]
        if early:
            print(f"\n  early-stopped: {', '.join(early)}")
            print("  -> differences here are partly explained by unequal compute; "
                  "see the fixed-budget table.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excise_dir", "--purge_dir", dest="excise_dir",
                    default=str(ROOT))
    ap.add_argument("--config", default=str(ROOT / "configs/cifar10_kl_retain.yaml"),
                    help="Dataset config; hyper-parameters are taken from here.")
    ap.add_argument("--base_ckpt", default=str(ROOT / "checkpoints/base_cifar10_resnet18.pth"))
    ap.add_argument("--data_dir", default=str(ROOT / "data"))
    ap.add_argument("--save_dir", default=str(ROOT / "results/ablation"))
    ap.add_argument("--forget_class", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["stopping", "fixed", "both"], default="both",
                    help="'stopping' = as deployed; 'fixed' = equal-compute "
                         "(stopping off for all); 'both' (default) runs each.")
    ap.add_argument("--only", nargs="+", default=None,
                    help="Run only these variant labels (e.g. --only full minus_proj)")
    args = ap.parse_args()

    if not os.path.exists(args.base_ckpt):
        print(f"[ERROR] base checkpoint not found: {args.base_ckpt}")
        return 1
    if not os.path.exists(args.config):
        print(f"[ERROR] config not found: {args.config}")
        return 1

    modes = ["stopping", "fixed"] if args.mode == "both" else [args.mode]
    results = []
    for mode in modes:
        for label, flags, _desc, stopping_only in VARIANTS:
            if args.only and label not in args.only:
                continue
            if stopping_only and mode == "fixed":
                print(f"\n[skip] {label} in fixed mode "
                      f"(stopping already disabled for all -> identical to 'full')")
                continue
            results.append(run_one(args, mode, label, flags))

    out_csv = Path(args.save_dir) / "component_ablation_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["mode", "label", "TA", "FA", "RA", "MIA", "feat_cos",
              "proj_rate", "gate_rate", "epochs_run", "stop_reason",
              "stopped_midepoch", "stopped_epoch", "seconds", "returncode"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k) for k in fields})

    for mode in modes:
        print_table(results, mode)

    print(f"\nSaved: {out_csv}")
    failed = [r["label"] for r in results if r["returncode"] != 0]
    if failed:
        print(f"[WARN] failed runs: {', '.join(failed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
