"""
Distance-to-retrained evaluation for unlearned models.

The gold standard for approximate unlearning is a model retrained from scratch
without the forget class. Forget accuracy (FA) only checks that the forget class
is no longer predicted; it cannot tell whether the unlearned model *behaves like*
the retrained one. In particular FA cannot distinguish

    (a) p_theta(x_f) has moved TO the retrained model's output distribution, from
    (b) p_theta(x_f) has moved PAST it (over-forgetting),

because both give FA ~ 0. Case (b) is still distinguishable from a retrained
model, and shows up as MIA AUROC well BELOW 0.5 (we observe 0.298 on PathMNIST
class 2). This script measures the distance directly.

Metrics (all computed on the forget set unless noted)
----------------------------------------------------
  js            Jensen-Shannon divergence between output distributions,
                unlearned vs retrained. Symmetric, bounded in [0, log 2].
                PRIMARY metric.
  kl_u_r        KL(p_unlearned || p_retrained), asymmetric companion.
  agree         Top-1 prediction agreement with the retrained model.
  frechet       Frechet distance between penultimate-feature distributions
                (the same statistic as feature-space FID).
  param_l2      ||theta_u - theta_retrain||_2 over all parameters.
                Reported for completeness but weak: permutation and scaling
                symmetries mean two functionally identical networks can be far
                apart in parameter space. Do not read it as a behavioural
                distance.
  ta / fa / ra  Accuracies, for context.

An ORACLE metric, not a stopping rule
-------------------------------------
These quantities require the retrained model, so they cannot be used to decide
when to stop unlearning in deployment -- if the retrained model were available
one would simply ship it. They are for evaluation. The deployable counterpart is
KL to the retain-confusion distribution p_r (computed from the ORIGINAL model,
no retraining needed), available in run.py as --kl_stop_tol / --kl_stop_patience.
This script reports both, so the proxy can be validated against the oracle.

Usage
-----
    python experiments/distance_to_retrained.py \
        --dataset pathmnist --forget_class 2 \
        --retrained checkpoints/retrain_pathmnist_c2.pth \
        --checkpoints purge:results/pathmnist_c2/purge_final.pth \
                      purge_klstop:results/pathmnist_c2_klstop/purge_final.pth
"""

import argparse
import csv
import os
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

# Reuse the main implementation's model/data code so this script cannot drift
# from what produced the checkpoints.
from run import (get_model, get_dataset, split_forget_retain,  # noqa: E402
                 evaluate, compute_mia)


# ----------------------------------------------------------------- utilities
def load_ckpt(path, num_classes, use_groupnorm, device):
    model = get_model("resnet18", num_classes=num_classes, pretrained=False,
                      use_groupnorm=use_groupnorm).to(device)
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def collect_probs_and_features(model, loader, device):
    """Return (probs [N,K], features [N,D]) for one model over one loader."""
    feats = {}

    def hook(_m, _i, out):
        feats["v"] = out

    handle = model.avgpool.register_forward_hook(hook)
    P, Z = [], []
    for x, _ in loader:
        x = x.to(device)
        logits = model(x)
        P.append(F.softmax(logits, dim=1).cpu())
        Z.append(torch.flatten(feats["v"], 1).cpu())
    handle.remove()
    return torch.cat(P), torch.cat(Z)


def js_divergence(p, q, eps=1e-12):
    """Mean Jensen-Shannon divergence (nats) between rows of p and q."""
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * ((a + eps).log() - (b + eps).log())).sum(dim=1)
    return (0.5 * kl(p, m) + 0.5 * kl(q, m)).mean().item()


def kl_divergence(p, q, eps=1e-12):
    return (p * ((p + eps).log() - (q + eps).log())).sum(dim=1).mean().item()


def frechet(z1, z2):
    """Frechet distance between two Gaussians fitted to feature sets."""
    from scipy.linalg import sqrtm
    a, b = z1.numpy().astype(np.float64), z2.numpy().astype(np.float64)
    mu1, mu2 = a.mean(0), b.mean(0)
    s1 = np.cov(a, rowvar=False)
    s2 = np.cov(b, rowvar=False)
    covmean = sqrtm(s1 @ s2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def param_l2(m1, m2):
    d = 0.0
    for (_, a), (_, b) in zip(m1.state_dict().items(), m2.state_dict().items()):
        if a.dtype.is_floating_point:
            d += (a.float() - b.float()).pow(2).sum().item()
    return d ** 0.5


@torch.no_grad()
def retain_confusion(model, retain_loader, device):
    """p_r: averaged softmax of the given model over the retain set."""
    s, n = None, 0
    for x, _ in retain_loader:
        x = x.to(device)
        p = F.softmax(model(x), dim=1).sum(0).cpu()
        s = p if s is None else s + p
        n += x.size(0)
    return (s / max(n, 1)).unsqueeze(0)


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--forget_class", type=int, required=True)
    ap.add_argument("--retrained", required=True,
                    help="Reference checkpoint retrained WITHOUT the forget class.")
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="One or more name:path entries to evaluate.")
    ap.add_argument("--original", default=None,
                    help="Optional pre-unlearning checkpoint; enables the "
                         "deployable KL-to-p_r proxy column.")
    ap.add_argument("--data_dir", default=str(ROOT / "data"))
    ap.add_argument("--out_csv", default=str(ROOT / "results/distance_to_retrained.csv"))
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--use_groupnorm", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_set, test_set, num_classes = get_dataset(args.dataset, args.data_dir)
    # NOTE: split_forget_retain returns torch Subset objects, NOT index arrays.
    forget_set, retain_set = split_forget_retain(
        train_set, "class", forget_class=args.forget_class)
    kw = dict(batch_size=args.batch_size, num_workers=4)
    forget_loader = DataLoader(forget_set, shuffle=False, **kw)
    retain_loader = DataLoader(retain_set, shuffle=False, **kw)
    test_loader = DataLoader(test_set, shuffle=False, **kw)
    print(f"{args.dataset}: forget class {args.forget_class} "
          f"({len(forget_set)} samples), retain {len(retain_set)}, K={num_classes}")

    # forget-class TEST split, for the MIA (same construction as run.py)
    _tt = (np.array(test_set.targets) if hasattr(test_set, "targets")
           else np.array([y for _, y in test_set]))
    fc_test_idx = np.where(_tt == args.forget_class)[0]
    fc_test_loader = (DataLoader(Subset(test_set, fc_test_idx), shuffle=False, **kw)
                      if len(fc_test_idx) > 0 else None)

    # ---- reference model
    ref = load_ckpt(args.retrained, num_classes, args.use_groupnorm, device)
    P_ref, Z_ref = collect_probs_and_features(ref, forget_loader, device)
    pred_ref = P_ref.argmax(1)
    print(f"reference (retrained): FA "
          f"{evaluate(ref, forget_loader, device, desc='ref-forget'):.2f}%")

    # ---- deployable proxy target p_r, from the ORIGINAL model if supplied
    p_r = None
    if args.original:
        orig = load_ckpt(args.original, num_classes, args.use_groupnorm, device)
        p_r = retain_confusion(orig, retain_loader, device)
        print("computed p_r from the original model (deployable proxy target)")

    rows = []
    for entry in args.checkpoints:
        if ":" not in entry:
            print(f"[skip] '{entry}' is not name:path")
            continue
        name, path = entry.split(":", 1)
        if not os.path.exists(path):
            print(f"[skip] {name}: missing {path}")
            continue

        m = load_ckpt(path, num_classes, args.use_groupnorm, device)
        P, Z = collect_probs_and_features(m, forget_loader, device)

        row = {
            "name": name,
            "dataset": args.dataset,
            "forget_class": args.forget_class,
            # --- oracle: distance to the retrained model
            "js": js_divergence(P, P_ref),
            "kl_u_r": kl_divergence(P, P_ref),
            "agree": (P.argmax(1) == pred_ref).float().mean().item() * 100,
            "frechet": frechet(Z, Z_ref),
            "param_l2": param_l2(m, ref),
            # --- deployable proxy: distance to p_r (no retrained model needed)
            "kl_to_pr": (kl_divergence(P, p_r.expand_as(P))
                         if p_r is not None else None),
            # --- context
            "TA": evaluate(m, test_loader, device, desc=f"{name}-test"),
            "FA": evaluate(m, forget_loader, device, desc=f"{name}-forget"),
            "RA": evaluate(m, retain_loader, device, desc=f"{name}-retain"),
        }
        # MIA, and its absolute deviation from the ideal 0.5 -- the quantity the
        # KL-stopping criterion is predicted to reduce. Deviation is signed-
        # agnostic on purpose: below 0.5 (over-forgetting) is as much a defect
        # as above 0.5 (under-forgetting).
        if fc_test_loader is not None:
            mia = compute_mia(m, forget_loader, fc_test_loader, device)
            row["MIA"] = mia["avg"]
            row["MIA_dev"] = abs(mia["avg"] - 0.5)
        else:
            row["MIA"] = row["MIA_dev"] = None
        rows.append(row)
        md = f"{row['MIA']:.4f}" if row["MIA"] is not None else "n/a"
        print(f"  {name:<16} JS={row['js']:.4f}  agree={row['agree']:.1f}%  "
              f"frechet={row['frechet']:.2f}  FA={row['FA']:.2f}%  MIA={md}")

    if not rows:
        print("no checkpoints evaluated.")
        return 1

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["name", "dataset", "forget_class", "js", "kl_u_r", "agree",
              "frechet", "param_l2", "kl_to_pr", "TA", "FA", "RA",
              "MIA", "MIA_dev"]
    write_header = not out.exists()
    with open(out, "a" if out.exists() else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})

    print(f"\n{'=' * 92}")
    print(f"Distance to retrained reference — {args.dataset} class {args.forget_class}")
    print(f"{'=' * 92}")
    print(f"{'model':<16}{'JS':>9}{'agree%':>9}{'frechet':>10}{'KL->p_r':>9}"
          f"{'MIA':>8}{'|MIA-.5|':>10}{'FA':>8}{'RA':>8}")
    print("-" * 92)
    for r in sorted(rows, key=lambda x: x["js"]):
        kp = f"{r['kl_to_pr']:.4f}" if r["kl_to_pr"] is not None else "n/a"
        mi = f"{r['MIA']:.4f}" if r["MIA"] is not None else "n/a"
        md = f"{r['MIA_dev']:.4f}" if r["MIA_dev"] is not None else "n/a"
        print(f"{r['name']:<16}{r['js']:9.4f}{r['agree']:9.1f}{r['frechet']:10.2f}"
              f"{kp:>9}{mi:>8}{md:>10}{r['FA']:8.2f}{r['RA']:8.2f}")
    print("\nLower JS / higher agreement = closer to the retrained gold standard.")
    print("|MIA-.5| is the defect magnitude: below 0.5 (over-forgetting) counts")
    print("as much as above 0.5 (under-forgetting).")
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
