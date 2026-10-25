"""
Trajectory sweep: where along unlearning is the model closest to a retrained one?

Motivation
----------
The KL-stopping experiment (results/kl_stopping/) refuted the hypothesis that
stopping on KL(p_theta(x_f) || p_r) fixes over-forgetting, but it established two
things: (a) all FA-stopped runs OVERSHOOT their own forget-accuracy target by
~4pp because the intra-epoch check is coarse, and (b) Jensen-Shannon divergence
to a retrained reference is LOWER at the earlier stopping point in every class.
Since JS must be large at t=0 (the base model confidently predicts the forget
class) and is still falling when FA stopping fires, a MINIMUM exists in between.

This script measures the whole curve: run unlearning with stopping disabled and
weight snapshots every N steps, then evaluate every snapshot against the
retrained reference. Output is JS / MIA / agreement / FA / RA / TA vs step, and
the located minimum.

The question it answers
-----------------------
Is the JS-minimising point identified by a single, class-independent forget
accuracy? If yes, "stop at FA = x%" is a deployable rule needing no oracle. If
the optimum lands at a very different FA per class, it is not deployable, and we
report that instead.

Cost note
---------
RA over the full retain split (~80k images at 224x224) per snapshot would
dominate the runtime, so retain accuracy is computed on a FIXED random subsample
(--ra_sample, default 5000, same subset for every snapshot so the series is
comparable). Forget-side metrics -- which are the ones under study -- always use
the complete forget set.

Usage
-----
    python experiments/trajectory_sweep.py --dataset pathmnist --classes 2 3 4
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

os.environ.pop("MKL_THREADING_LAYER", None)
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for sibling imports
from run import (get_model, get_dataset, split_forget_retain,  # noqa: E402
                 evaluate, compute_mia)
from distance_to_retrained import (js_divergence, kl_divergence,  # noqa: E402
                                   frechet, collect_probs_and_features,
                                   load_ckpt, retain_confusion)


def run_trajectory(args, cls, cfg, base, out_dir):
    """Phase 1: one unlearning run with stopping disabled + snapshots."""
    sd = out_dir / f"traj_c{cls}"
    if (sd / "traj").exists() and any((sd / "traj").glob("step_*.pth")) \
            and not args.force:
        n = len(list((sd / "traj").glob("step_*.pth")))
        print(f"  [skip] {n} snapshots already in {sd/'traj'} (use --force)")
        return sd / "traj"
    cmd = [sys.executable, "-u", str(ROOT / "run.py"),
           "--config", str(cfg),
           "--checkpoint", str(base),
           "--data_dir", args.data_dir,
           "--save_dir", str(sd),
           "--forget_class", str(cls),
           "--seed", str(args.seed),
           "--epochs", str(args.epochs),
           # stopping fully disabled: we want the whole trajectory, and the
           # stopping rule is precisely what is under evaluation here.
           "--fa_target", "-1.0",
           "--retain_budget_factor", "999999.0",
           "--checkpoint_every", str(args.checkpoint_every)]
    sd.mkdir(parents=True, exist_ok=True)
    log = sd / "run.log"
    print(f"  running unlearning (class {cls}) -> {log}")
    with open(log, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        tail = log.read_text().strip().splitlines()[-20:]
        print(f"  !! exited {p.returncode}:")
        for ln in tail:
            print(f"  | {ln}")
    return sd / "traj"


def evaluate_trajectory(args, cls, traj_dir, ref_path, base_path, device):
    """Phase 2: evaluate every snapshot against the retrained reference."""
    train_set, test_set, K = get_dataset(args.dataset, args.data_dir)
    forget_set, retain_set = split_forget_retain(
        train_set, "class", forget_class=cls)
    kw = dict(batch_size=args.batch_size, num_workers=4)

    forget_loader = DataLoader(forget_set, shuffle=False, **kw)
    test_loader = DataLoader(test_set, shuffle=False, **kw)
    # fixed retain subsample -- same indices for every snapshot
    rng = np.random.RandomState(0)
    n_ra = min(args.ra_sample, len(retain_set))
    ra_idx = rng.choice(len(retain_set), n_ra, replace=False)
    retain_sub = DataLoader(Subset(retain_set, ra_idx.tolist()),
                            shuffle=False, **kw)

    tt = (np.array(test_set.targets) if hasattr(test_set, "targets")
          else np.array([y for _, y in test_set]))
    fc_test_idx = np.where(tt == cls)[0]
    fc_test_loader = (DataLoader(Subset(test_set, fc_test_idx), shuffle=False, **kw)
                      if len(fc_test_idx) else None)

    ref = load_ckpt(str(ref_path), K, args.use_groupnorm, device)
    P_ref, Z_ref = collect_probs_and_features(ref, forget_loader, device)
    pred_ref = P_ref.argmax(1)

    orig = load_ckpt(str(base_path), K, args.use_groupnorm, device)
    p_r = retain_confusion(orig, DataLoader(retain_set, shuffle=False, **kw),
                           device)
    del orig

    snaps = sorted(traj_dir.glob("step_*.pth"))
    print(f"  evaluating {len(snaps)} snapshots for class {cls}")
    rows = []
    for i, s in enumerate(snaps):
        step = int(s.stem.split("_")[1])
        m = load_ckpt(str(s), K, args.use_groupnorm, device)
        P, Z = collect_probs_and_features(m, forget_loader, device)
        row = {
            "dataset": args.dataset, "forget_class": cls, "step": step,
            "js": js_divergence(P, P_ref),
            "kl_u_r": kl_divergence(P, P_ref),
            "agree": (P.argmax(1) == pred_ref).float().mean().item() * 100,
            "frechet": frechet(Z, Z_ref),
            "kl_to_pr": kl_divergence(P, p_r.expand_as(P)),
            "FA": evaluate(m, forget_loader, device, desc=f"    s{step} FA"),
            "RA_sub": evaluate(m, retain_sub, device, desc=f"    s{step} RA*"),
            "TA": evaluate(m, test_loader, device, desc=f"    s{step} TA"),
        }
        if fc_test_loader is not None:
            mia = compute_mia(m, forget_loader, fc_test_loader, device)
            row["MIA"] = mia["avg"]
            row["MIA_dev"] = abs(mia["avg"] - 0.5)
        else:
            row["MIA"] = row["MIA_dev"] = None
        rows.append(row)
        print(f"    [{i+1}/{len(snaps)}] step {step:6d}  JS={row['js']:.4f}  "
              f"FA={row['FA']:6.2f}  MIA={row['MIA']:.4f}")
        del m
    return rows


def report(rows_by_class, out_csv):
    fields = ["dataset", "forget_class", "step", "js", "kl_u_r", "agree",
              "frechet", "kl_to_pr", "FA", "RA_sub", "TA", "MIA", "MIA_dev"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rows in rows_by_class.values():
            for r in rows:
                w.writerow({k: r.get(k) for k in fields})

    print(f"\n{'=' * 96}")
    print("TRAJECTORY MINIMA — where is the model closest to the retrained reference?")
    print(f"{'=' * 96}")
    print(f"{'cls':>4} {'argmin step':>12} {'JS@min':>9} {'FA@min':>8} "
          f"{'MIA@min':>9} {'JS@t0':>9} {'JS@end':>9} {'FA@end':>8}")
    print("-" * 96)
    fa_at_min = []
    for c, rows in sorted(rows_by_class.items()):
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: r["step"])
        best = min(rows, key=lambda r: r["js"])
        fa_at_min.append(best["FA"])
        print(f"{c:>4} {best['step']:>12} {best['js']:>9.4f} {best['FA']:>8.2f} "
              f"{best['MIA']:>9.4f} {rows[0]['js']:>9.4f} {rows[-1]['js']:>9.4f} "
              f"{rows[-1]['FA']:>8.2f}")

    print(f"\n{'=' * 96}")
    print("IS THE OPTIMUM DEPLOYABLE?")
    print(f"{'=' * 96}")
    if len(fa_at_min) > 1:
        lo, hi = min(fa_at_min), max(fa_at_min)
        print(f"  FA at the JS minimum spans {lo:.2f}% .. {hi:.2f}%  "
              f"(spread {hi - lo:.2f}pp)")
        if hi - lo <= 5.0:
            print("  -> tight spread: a single FA target could identify the optimum,")
            print("     so 'stop at FA = x%' would be deployable without an oracle.")
        else:
            print("  -> wide spread: the optimum is NOT identified by a common FA")
            print("     level, so a fixed FA target cannot locate it. Report as such;")
            print("     do not present a tuned per-class value as a deployable rule.")
    for c, rows in sorted(rows_by_class.items()):
        rows = sorted(rows, key=lambda r: r["step"])
        best = min(rows, key=lambda r: r["js"])
        if best["step"] == rows[-1]["step"]:
            print(f"  [WARN] class {c}: minimum is at the LAST step — the run may be "
                  f"too short to contain it. Increase --epochs.")
        if best["step"] == 0:
            print(f"  [WARN] class {c}: minimum is at step 0 (the base model) — "
                  f"unlearning only ever increases distance for this class.")
    print(f"\nSaved: {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pathmnist")
    ap.add_argument("--classes", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--epochs", type=int, default=4,
                    help="Unlearning epochs with stopping disabled. Must be long "
                         "enough that the JS minimum lies strictly inside the run.")
    ap.add_argument("--checkpoint_every", type=int, default=8)
    ap.add_argument("--ra_sample", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_dir", default=str(ROOT / "data"))
    ap.add_argument("--ckpt_dir", default=str(ROOT / "checkpoints"))
    ap.add_argument("--out_dir", default=str(ROOT / "results/trajectory"))
    ap.add_argument("--use_groupnorm", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--eval_only", action="store_true",
                    help="Skip unlearning; evaluate existing snapshots.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = ROOT / f"configs/{args.dataset}_kl_retain.yaml"
    base = Path(args.ckpt_dir) / f"base_{args.dataset}_resnet18.pth"
    for p in (cfg, base):
        if not p.exists():
            print(f"[ERROR] missing {p}"); return 1

    rows_by_class = {}
    for c in args.classes:
        ref = Path(args.ckpt_dir) / f"retrain_{args.dataset}_c{c}.pth"
        if not ref.exists():
            print(f"[SKIP] class {c}: no reference at {ref}")
            continue
        print(f"\n{'#' * 72}\n# class {c}\n{'#' * 72}")
        traj = (out / f"traj_c{c}" / "traj") if args.eval_only else \
            run_trajectory(args, c, cfg, base, out)
        if not traj.exists() or not any(traj.glob("step_*.pth")):
            print(f"  [SKIP] no snapshots for class {c}")
            continue
        rows_by_class[c] = evaluate_trajectory(args, c, traj, ref, base, device)

    if not rows_by_class:
        print("nothing evaluated."); return 1
    report(rows_by_class, out / "trajectory_sweep.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
