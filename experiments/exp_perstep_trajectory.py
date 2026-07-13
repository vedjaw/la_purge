"""
Per-step retain-loss and inner-product trajectory logger.

Runs a modified PURGE loop that logs retain CE, forget CE, and the raw
inner product <g_f, g_r> BEFORE projection at every single gradient step.
Outputs a CSV for plotting (empirical validation of Theorem 1).
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
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from run import (get_model, get_dataset, split_forget_retain,
                 ActivationExtractor, select_monitor_layers,
                 representation_erasure_loss, project_gradient, evaluate)


def run_instrumented(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set, test_set, num_classes = get_dataset(args.dataset, args.data_dir)
    forget_set, retain_set = split_forget_retain(
        train_set, "class", forget_class=args.forget_class)

    kw = dict(batch_size=args.batch_size, num_workers=4)
    forget_loader = DataLoader(forget_set, shuffle=True, **kw)
    retain_loader = DataLoader(retain_set, shuffle=True, **kw)
    test_loader = DataLoader(test_set, shuffle=False, **kw)

    model = get_model("resnet18", num_classes,
                      use_groupnorm=args.use_groupnorm).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    original = copy.deepcopy(model)
    original.eval()
    for p in original.parameters():
        p.requires_grad = False

    layer_names = select_monitor_layers(model)
    extractor = ActivationExtractor(model, layer_names)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # Retain confusion distribution
    _sum, _cnt = None, 0
    with torch.no_grad():
        for _xr, _ in retain_loader:
            _xr = _xr.to(device)
            _probs = F.softmax(original(_xr), dim=1)
            _sum = _probs.sum(0) if _sum is None else _sum + _probs.sum(0)
            _cnt += _xr.size(0)
    retain_confusion = (_sum / _cnt).unsqueeze(0)

    # Initial retain loss
    model.eval()
    _rl, _rn = 0.0, 0
    with torch.no_grad():
        for _xr, _yr in retain_loader:
            _xr, _yr = _xr.to(device), _yr.view(-1).to(device)
            _rl += F.cross_entropy(model(_xr), _yr, reduction="sum").item()
            _rn += _yr.size(0)
    initial_retain_loss = _rl / max(_rn, 1)

    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()

    retain_iter = iter(retain_loader)
    max_entropy = math.log(num_classes)
    step_logs = []
    global_step = 0

    for epoch in range(args.epochs):
        for x_f, y_f in tqdm(forget_loader,
                              desc=f"Epoch {epoch+1}/{args.epochs}"):
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
            optimizer.zero_grad()
            loss_ce_r = criterion(out_r, y_r)
            loss_kd = F.kl_div(
                F.log_softmax(out_r / 4.0, dim=1),
                F.softmax(orig_out_r / 4.0, dim=1),
                reduction="batchmean") * 16.0
            loss_retain = loss_ce_r + args.kd_weight * loss_kd
            loss_retain.backward()
            g_retain = [p.grad.clone() for p in params]

            # Forget gradient
            optimizer.zero_grad()
            target = retain_confusion.expand_as(out_f)
            loss_dir = F.kl_div(F.log_softmax(out_f, dim=1), target,
                                reduction="batchmean")
            loss_rep = representation_erasure_loss(forget_acts, retain_acts)
            loss_forget = loss_dir + args.rep_weight * loss_rep
            loss_forget.backward()
            g_forget = [p.grad.clone() for p in params]

            # Raw inner product BEFORE projection
            dot_raw = sum((gf * gr).sum() for gf, gr in
                          zip(g_forget, g_retain)).item()
            g_f_norm = sum((gf**2).sum() for gf in g_forget).sqrt().item()
            g_r_norm = sum((gr**2).sum() for gr in g_retain).sqrt().item()

            # Project
            g_proj, was_proj = project_gradient(g_forget, g_retain)
            dot_post = sum((gp * gr).sum() for gp, gr in
                           zip(g_proj, g_retain)).item()

            # Apply
            optimizer.zero_grad()
            for p, g in zip(params, g_proj):
                p.grad = g
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            # Measure retain loss AFTER step
            model.eval()
            with torch.no_grad():
                post_retain_ce = F.cross_entropy(model(x_r), y_r).item()
            model.train()
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.eval()

            with torch.no_grad():
                forget_ce = criterion(out_f, y_f).item()

            step_logs.append({
                "step": global_step,
                "epoch": epoch,
                "retain_ce_pre": loss_ce_r.item(),
                "retain_ce_post": post_retain_ce,
                "retain_delta": post_retain_ce - loss_ce_r.item(),
                "forget_ce": forget_ce,
                "forget_kl": loss_dir.item(),
                "dot_raw": dot_raw,
                "dot_post": dot_post,
                "g_f_norm": g_f_norm,
                "g_r_norm": g_r_norm,
                "projected": int(was_proj),
            })
            global_step += 1

    extractor.remove()

    # Save CSV
    out_path = Path(args.out_dir) / f"perstep_{args.dataset}_c{args.forget_class}_s{args.seed}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(step_logs[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(step_logs)

    # Final eval
    ta = evaluate(model, test_loader, device, "TA")

    n_violations = sum(1 for s in step_logs if s["retain_delta"] > 0)
    print(f"\n{'='*60}")
    print(f"Per-Step Trajectory: {args.dataset} class {args.forget_class}")
    print(f"{'='*60}")
    print(f"  Total steps: {global_step}")
    print(f"  Steps projected: {sum(s['projected'] for s in step_logs)}")
    print(f"  Steps where retain CE increased: {n_violations}/{global_step} "
          f"({100*n_violations/max(global_step,1):.1f}%)")
    print(f"  Max retain CE increase: "
          f"{max(s['retain_delta'] for s in step_logs):.6f}")
    print(f"  Initial retain CE: {initial_retain_loss:.4f}")
    print(f"  Final TA: {ta:.2f}%")
    print(f"  CSV: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--forget_class", type=int, default=0)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", default=str(ROOT / "data"))
    ap.add_argument("--out_dir", default=str(ROOT / "results/perstep"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--kd_weight", type=float, default=2.0)
    ap.add_argument("--rep_weight", type=float, default=0.05)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_groupnorm", action="store_true")
    args = ap.parse_args()

    run_instrumented(args)


if __name__ == "__main__":
    main()
