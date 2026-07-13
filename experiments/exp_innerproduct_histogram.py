"""
Per-step inner-product histogram for Theorem 2 validation.

For PURGE (before projection), logs sign(dot(g_f, g_r)) at every step.
Also runs a "no-projection" (raw GA/kl_retain) baseline to show what
fraction of steps would have violated retain safety without projection.
"""
import argparse
import copy
import csv
import math
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from run import (get_model, get_dataset, split_forget_retain,
                 ActivationExtractor, select_monitor_layers,
                 representation_erasure_loss)


def collect_inner_products(model_init, forget_loader, retain_loader,
                           num_classes, device, args, max_steps=200):
    """Run through forget batches and collect raw <g_f, g_r> inner products."""
    model = copy.deepcopy(model_init)
    original = copy.deepcopy(model_init)
    original.eval()
    for p in original.parameters():
        p.requires_grad = False

    layer_names = select_monitor_layers(model)
    extractor = ActivationExtractor(model, layer_names)
    params = [p for p in model.parameters() if p.requires_grad]
    criterion = nn.CrossEntropyLoss()

    # Retain confusion
    _sum, _cnt = None, 0
    with torch.no_grad():
        for _xr, _ in retain_loader:
            _xr = _xr.to(device)
            _probs = F.softmax(original(_xr), dim=1)
            _sum = _probs.sum(0) if _sum is None else _sum + _probs.sum(0)
            _cnt += _xr.size(0)
    retain_confusion = (_sum / _cnt).unsqueeze(0)

    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()

    retain_iter = iter(retain_loader)
    dots = []
    step = 0

    for x_f, y_f in forget_loader:
        if step >= max_steps:
            break
        x_f, y_f = x_f.to(device), y_f.view(-1).to(device)
        try:
            x_r, y_r = next(retain_iter)
        except StopIteration:
            retain_iter = iter(retain_loader)
            x_r, y_r = next(retain_iter)
        x_r, y_r = x_r.to(device), y_r.view(-1).to(device)

        out_r = model(x_r)
        retain_acts = OrderedDict(
            (k, v.detach()) for k, v in extractor.activations.items())
        with torch.no_grad():
            orig_out_r = original(x_r)
        out_f = model(x_f)
        forget_acts = OrderedDict(extractor.activations)

        # Retain gradient
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        loss_ce_r = criterion(out_r, y_r)
        loss_kd = F.kl_div(
            F.log_softmax(out_r / 4.0, dim=1),
            F.softmax(orig_out_r / 4.0, dim=1),
            reduction="batchmean") * 16.0
        loss_retain = loss_ce_r + args.kd_weight * loss_kd
        loss_retain.backward()
        g_retain = [p.grad.clone() for p in params]

        # Forget gradient
        for p in params:
            p.grad.zero_()
        target = retain_confusion.expand_as(out_f)
        loss_dir = F.kl_div(F.log_softmax(out_f, dim=1), target,
                            reduction="batchmean")
        loss_rep = representation_erasure_loss(forget_acts, retain_acts)
        loss_forget = loss_dir + args.rep_weight * loss_rep
        loss_forget.backward()
        g_forget = [p.grad.clone() for p in params]

        dot = sum((gf * gr).sum() for gf, gr in
                  zip(g_forget, g_retain)).item()
        dots.append(dot)
        step += 1

    extractor.remove()
    return dots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--forget_class", type=int, default=0)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", default=str(ROOT / "data"))
    ap.add_argument("--out_dir", default=str(ROOT / "results/innerproduct"))
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--kd_weight", type=float, default=2.0)
    ap.add_argument("--rep_weight", type=float, default=0.05)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_groupnorm", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set, test_set, num_classes = get_dataset(args.dataset, args.data_dir)
    forget_set, retain_set = split_forget_retain(
        train_set, "class", forget_class=args.forget_class)

    kw = dict(batch_size=args.batch_size, num_workers=4)
    forget_loader = DataLoader(forget_set, shuffle=True, **kw)
    retain_loader = DataLoader(retain_set, shuffle=True, **kw)

    model = get_model("resnet18", num_classes,
                      use_groupnorm=args.use_groupnorm).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    print(f"{'='*60}")
    print(f"Inner-Product Histogram: {args.dataset} class {args.forget_class}")
    print(f"{'='*60}")

    dots = collect_inner_products(model, forget_loader, retain_loader,
                                  num_classes, device, args, args.max_steps)

    n_neg = sum(1 for d in dots if d < 0)
    n_total = len(dots)
    pct_neg = 100 * n_neg / max(n_total, 1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"innerproducts_{args.dataset}_c{args.forget_class}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "dot_gf_gr"])
        for i, d in enumerate(dots):
            w.writerow([i, d])

    print(f"  Total steps: {n_total}")
    print(f"  Negative <g_f, g_r> (would violate retain safety): "
          f"{n_neg}/{n_total} ({pct_neg:.1f}%)")
    print(f"  Mean <g_f, g_r>: {np.mean(dots):.6f}")
    print(f"  Min <g_f, g_r>: {min(dots):.6f}")
    print(f"  Max <g_f, g_r>: {max(dots):.6f}")
    print(f"  CSV: {csv_path}")

    # Histogram plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(dots, bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(0, color="red", linestyle="--", linewidth=2,
                   label=f"boundary (neg={pct_neg:.1f}%)")
        ax.set_xlabel("<g_forget, g_retain>")
        ax.set_ylabel("Count")
        ax.set_title(f"Inner Product Distribution — {args.dataset} class {args.forget_class}")
        ax.legend()
        plt.tight_layout()
        fig_path = out_dir / f"innerproducts_{args.dataset}_c{args.forget_class}.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"  Plot: {fig_path}")
    except ImportError:
        print("  (matplotlib not available, skipping plot)")


if __name__ == "__main__":
    main()
