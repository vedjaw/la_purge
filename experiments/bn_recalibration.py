"""BatchNorm-recalibration diagnostic (Finding: BN vulnerability)."""

import argparse
import csv
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

ROOT = Path(__file__).resolve().parent.parent


def _replace_bn_with_gn(model, num_groups=32):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            nc = module.num_features
            g = min(num_groups, nc)
            while nc % g != 0:
                g //= 2
            setattr(model, name, nn.GroupNorm(g, nc))
        else:
            _replace_bn_with_gn(module, num_groups)
    return model


def get_model(num_classes=10, use_groupnorm=False):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    if use_groupnorm:
        m = _replace_bn_with_gn(m)
    return m


def get_cifar10(data_dir):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                         download=True, transform=tf)
    test = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                        download=True, transform=tf)
    return train, test


def accuracy_class(model, loader, device, target_class):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            mask = (y == target_class)
            if mask.sum() == 0:
                continue
            preds = model(x[mask]).argmax(1)
            correct += (preds == y[mask]).sum().item()
            total += mask.sum().item()
    return 100.0 * correct / max(total, 1)


def overall_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / max(total, 1)


def set_bn_train(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.train()


def set_bn_eval(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


def bn_recalibrate(model, retain_loader, device, n_batches=None):
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    set_bn_train(model)
    seen = 0
    with torch.no_grad():
        for x, _ in retain_loader:
            x = x.to(device)
            _ = model(x)
            seen += x.size(0)
            if n_batches is not None and (seen >= n_batches * x.size(0)):
                break
    set_bn_eval(model)
    return seen


def evaluate_one(model, train_set, test_set, fc, device, batch_size=128,
                 retain_pass_n_batches=None):
    forget_test_idx = [i for i, t in enumerate(test_set.targets) if t == fc]
    retain_train_idx = [i for i, t in enumerate(train_set.targets) if t != fc]

    forget_test_loader = DataLoader(Subset(test_set, forget_test_idx),
                                    batch_size=batch_size, shuffle=False)
    retain_loader = DataLoader(Subset(train_set, retain_train_idx),
                               batch_size=batch_size, shuffle=True)
    full_test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    set_bn_eval(model)
    pre_F = accuracy_class(model, forget_test_loader, device, fc)
    pre_TA = overall_accuracy(model, full_test_loader, device)

    n_seen = bn_recalibrate(model, retain_loader, device,
                            n_batches=retain_pass_n_batches)

    post_F = accuracy_class(model, forget_test_loader, device, fc)
    post_TA = overall_accuracy(model, full_test_loader, device)
    return {
        "pre_F": pre_F, "post_F": post_F, "delta_F": post_F - pre_F,
        "pre_TA": pre_TA, "post_TA": post_TA,
        "delta_TA": post_TA - pre_TA, "n_retain_samples": n_seen,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(ROOT / "data"))
    ap.add_argument("--forget_class", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--retain_pass_n_batches", type=int, default=None)
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="name:path pairs, e.g. purge:checkpoints/purge.pth")
    ap.add_argument("--out_csv", default=str(ROOT / "results/bn_recalibration_summary.csv"))
    ap.add_argument("--use_groupnorm", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, test_set = get_cifar10(args.data_dir)

    rows = []
    for spec in args.checkpoints:
        name, path = spec.split(":", 1) if ":" in spec else (Path(spec).stem, spec)
        if not Path(path).exists():
            print(f"SKIP {name}: {path} not found")
            continue

        model = get_model(use_groupnorm=args.use_groupnorm).to(device)
        state = torch.load(path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)

        metrics = evaluate_one(model, train_set, test_set, args.forget_class, device,
                               batch_size=args.batch_size,
                               retain_pass_n_batches=args.retain_pass_n_batches)
        metrics["method"] = name
        metrics["checkpoint"] = path
        rows.append(metrics)
        print(f"{name}: Pre-F={metrics['pre_F']:.2f}% Post-F={metrics['post_F']:.2f}% "
              f"ΔF={metrics['delta_F']:+.2f}pp")

    if rows:
        fields = ["method", "pre_F", "post_F", "delta_F",
                  "pre_TA", "post_TA", "delta_TA",
                  "n_retain_samples", "checkpoint"]
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in fields})
        print(f"Saved {args.out_csv}")


if __name__ == "__main__":
    main()
