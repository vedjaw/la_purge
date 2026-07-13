"""
Fine-tuning recovery attack: after PURGE unlearning, fine-tune on a small
subset of forget-class samples and measure how quickly FA recovers.

Tests durability of forgetting (related to Finding 8.4).
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from run import get_model, get_dataset, evaluate, compute_mia


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--forget_class", type=int, default=0)
    ap.add_argument("--checkpoint", required=True,
                    help="Path to the PURGE-unlearned checkpoint")
    ap.add_argument("--data_dir", default=str(ROOT / "data"))
    ap.add_argument("--out_dir", default=str(ROOT / "results/finetune_recovery"))
    ap.add_argument("--n_samples", type=int, default=50,
                    help="Number of forget-class samples to fine-tune on")
    ap.add_argument("--ft_epochs", type=int, default=10)
    ap.add_argument("--ft_lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_groupnorm", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set, test_set, num_classes = get_dataset(args.dataset, args.data_dir)
    targets = np.array(train_set.targets) if hasattr(train_set, 'targets') \
        else np.array([t for _, t in train_set])

    forget_idx = np.where(targets == args.forget_class)[0]
    retain_idx = np.where(targets != args.forget_class)[0]

    # Small attack subset from forget class
    np.random.shuffle(forget_idx)
    attack_idx = forget_idx[:args.n_samples]
    attack_set = Subset(train_set, attack_idx)
    attack_loader = DataLoader(attack_set, batch_size=args.batch_size,
                               shuffle=True, num_workers=2)

    # Full forget set for FA eval
    forget_set = Subset(train_set, forget_idx)
    forget_loader = DataLoader(forget_set, batch_size=args.batch_size,
                               shuffle=False, num_workers=2)
    retain_set = Subset(train_set, retain_idx)
    retain_loader = DataLoader(retain_set, batch_size=args.batch_size,
                               shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=2)

    # Forget-class test loader for MIA
    test_targets = np.array(test_set.targets) if hasattr(test_set, 'targets') \
        else np.array([t for _, t in test_set])
    fc_test_idx = np.where(test_targets == args.forget_class)[0]
    fc_test_loader = DataLoader(Subset(test_set, fc_test_idx),
                                batch_size=args.batch_size, shuffle=False)

    model = get_model("resnet18", num_classes,
                      use_groupnorm=args.use_groupnorm).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.ft_lr, momentum=0.9)

    print(f"{'='*60}")
    print(f"Fine-Tuning Recovery Attack")
    print(f"  Unlearned checkpoint: {args.checkpoint}")
    print(f"  Attack samples: {args.n_samples} from class {args.forget_class}")
    print(f"  FT epochs: {args.ft_epochs} | FT LR: {args.ft_lr}")
    print(f"{'='*60}")

    results = []

    # Epoch 0 = before fine-tuning
    ta0 = evaluate(model, test_loader, device, "TA (pre-FT)")
    fa0 = evaluate(model, forget_loader, device, "FA (pre-FT)")
    ra0 = evaluate(model, retain_loader, device, "RA (pre-FT)")
    mia0 = compute_mia(model, forget_loader, fc_test_loader, device)
    results.append({"epoch": 0, "TA": ta0, "FA": fa0, "RA": ra0,
                    "MIA": mia0["avg"]})

    for ep in range(1, args.ft_epochs + 1):
        model.train()
        for imgs, labs in attack_loader:
            imgs, labs = imgs.to(device), labs.view(-1).to(device)
            optimizer.zero_grad()
            criterion(model(imgs), labs).backward()
            optimizer.step()

        ta = evaluate(model, test_loader, device, f"TA (ep {ep})")
        fa = evaluate(model, forget_loader, device, f"FA (ep {ep})")
        ra = evaluate(model, retain_loader, device, f"RA (ep {ep})")
        mia = compute_mia(model, forget_loader, fc_test_loader, device)
        results.append({"epoch": ep, "TA": ta, "FA": fa, "RA": ra,
                        "MIA": mia["avg"]})

        print(f"  Epoch {ep}: TA={ta:.2f}% FA={fa:.2f}% RA={ra:.2f}% "
              f"MIA={mia['avg']:.4f}")

    out_csv = Path(args.out_dir) / f"recovery_{args.dataset}_c{args.forget_class}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["epoch", "TA", "FA", "RA", "MIA"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    print(f"\n{'='*60}")
    print(f"Recovery Summary")
    print(f"  Pre-FT FA: {results[0]['FA']:.2f}% -> Post-FT FA: {results[-1]['FA']:.2f}%")
    print(f"  FA recovered by: {results[-1]['FA'] - results[0]['FA']:.2f}pp "
          f"in {args.ft_epochs} epochs with {args.n_samples} samples")
    print(f"  CSV: {out_csv}")


if __name__ == "__main__":
    main()
