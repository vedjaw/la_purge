"""
Retrain-from-scratch reference models (the unlearning gold standard).

Trains ResNet-18 on the RETAIN set only -- i.e. as if the forget class had never
been in the training data -- while KEEPING all K output heads. The forget-class
head therefore exists but is never trained, so the model routes forget-class
inputs to surviving classes and its forget accuracy is ~0.

Why keep K heads rather than train a (K-1)-class model
------------------------------------------------------
The reference is used to measure the DISTANCE between output distributions
(experiments/distance_to_retrained.py). That requires the reference and the
unlearned model to live in the same output space. A (K-1)-head model cannot be
compared this way. Keeping K heads is also the convention that makes
"FA ~ 0 by routing forget inputs to surviving classes" meaningful.

Training recipe
---------------
Mirrors experiments/train_base_all.py (same optimiser, schedule, and per-dataset
learning rates) so that the reference differs from the base model ONLY in the
data it saw, not in how it was trained. Any other difference would confound the
distance measurement.

Usage
-----
    python experiments/retrain_reference.py --dataset pathmnist --forget_class 2
    python experiments/retrain_reference.py --dataset pathmnist --forget_class 2 3 4
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.pop("MKL_THREADING_LAYER", None)
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from run import get_model, get_dataset, split_forget_retain, evaluate  # noqa: E402

# Same per-dataset learning rates as experiments/train_base_all.py, so the
# reference is trained like the base model.
DEFAULT_LR = {
    "cifar10": 0.1, "cifar100": 0.01, "mnist": 0.01,
    "svhn": 0.01, "stl10": 0.01, "pathmnist": 0.01,
}
# NOTE: deliberately NO default absolute accuracy threshold.
#
# train_base_all.py early-stops the base model at a target test accuracy. Reusing
# those targets here would be wrong: the reference is evaluated on the retain
# classes only, so it clears a full-data threshold almost immediately and would
# stop while still under-trained. An under-trained reference is a weak gold
# standard -- it would sit far from every unlearned model and make all of them
# look equally bad. We therefore train to convergence using patience on
# retain-test accuracy, and expose --acc_threshold only as an explicit override.


def retain_test_subset(test_set, forget_class):
    """Test split with the forget class removed (for measuring retain-side TA)."""
    t = (np.array(test_set.targets) if hasattr(test_set, "targets")
         else np.array([y for _, y in test_set]))
    keep = np.where(t != forget_class)[0]
    drop = np.where(t == forget_class)[0]
    return Subset(test_set, keep), Subset(test_set, drop)


def retrain_one(dataset, forget_class, args, device):
    out_path = Path(args.save_dir) / f"retrain_{dataset}_c{forget_class}.pth"
    if out_path.exists() and not args.force:
        print(f"  [skip] exists: {out_path}  (use --force to overwrite)")
        return out_path

    train_set, test_set, num_classes = get_dataset(dataset, args.data_dir)
    # NOTE: split_forget_retain returns torch Subset objects, NOT index arrays.
    forget_set, retain_set = split_forget_retain(
        train_set, "class", forget_class=forget_class)
    kw = dict(batch_size=args.batch_size, num_workers=4)

    # TRAIN ONLY on retain data; K heads retained.
    retain_train = DataLoader(retain_set, shuffle=True, **kw)
    forget_train = DataLoader(forget_set, shuffle=False, **kw)
    test_keep, test_forget = retain_test_subset(test_set, forget_class)
    test_keep_loader = DataLoader(test_keep, shuffle=False, **kw)
    full_test_loader = DataLoader(test_set, shuffle=False, **kw)

    lr = args.lr if args.lr is not None else DEFAULT_LR.get(dataset, 0.01)
    acc_target = args.acc_threshold  # None unless explicitly overridden

    print(f"\n{'=' * 70}")
    print(f"RETRAIN reference — {dataset}, excluding class {forget_class}")
    print(f"{'=' * 70}")
    print(f"  train on {len(retain_set)} retain samples "
          f"({len(forget_set)} forget samples withheld)")
    print(f"  K={num_classes} heads kept; lr={lr}; epochs<={args.epochs}; "
          f"stopping: patience={args.patience} on retain-test acc"
          + (f", or acc>={acc_target}%" if acc_target is not None else
             " (no absolute threshold — trains to convergence)"))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = get_model("resnet18", num_classes=num_classes,
                      pretrained=True, use_groupnorm=args.use_groupnorm).to(device)
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()

    best, best_state, stale = 0.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        tot = 0.0
        for x, y in tqdm(retain_train, desc=f"  epoch {epoch+1}/{args.epochs}",
                         leave=False):
            x, y = x.to(device), y.view(-1).to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()

        acc = evaluate(model, test_keep_loader, device,
                       desc=f"  epoch {epoch+1} retain-test")
        if acc > best:
            best, stale = acc, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            stale += 1

        if (acc_target is not None and best >= acc_target
                and epoch + 1 >= args.min_epochs):
            print(f"  reached target {acc_target}% at epoch {epoch+1} "
                  f"(best {best:.2f}%) — stopping.")
            break
        if args.patience > 0 and stale >= args.patience:
            print(f"  no improvement for {stale} epochs (best {best:.2f}%) "
                  f"— stopping.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)

    print(f"\n  --- reference model checks ---")
    fa = evaluate(model, forget_train, device, desc="  FA (forget train)")
    ra = evaluate(model, DataLoader(retain_set, shuffle=False, **kw), device,
                  desc="  RA (retain train)")
    ta_full = evaluate(model, full_test_loader, device, desc="  TA (full test)")
    ta_keep = evaluate(model, test_keep_loader, device, desc="  TA (retain test)")
    print(f"  saved -> {out_path}")
    if fa > 5.0:
        print(f"  [WARN] reference FA is {fa:.2f}%, expected ~0%. The forget "
              f"class should never have been trained on — check the split.")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--forget_class", type=int, nargs="+", required=True,
                    help="One or more classes; a reference is trained per class.")
    ap.add_argument("--data_dir", default=str(ROOT / "data"))
    ap.add_argument("--save_dir", default=str(ROOT / "checkpoints"))
    ap.add_argument("--epochs", type=int, default=30,
                    help="Epoch budget. NOTE: this also sets the cosine "
                         "schedule horizon (T_max), so set it to the budget you "
                         "actually intend. A large value combined with early "
                         "stopping halts mid-schedule while the learning rate is "
                         "still high, giving an un-annealed (weaker) reference.")
    ap.add_argument("--min_epochs", type=int, default=5)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--acc_threshold", type=float, default=None,
                    help="Optional absolute retain-test accuracy at which to "
                         "stop early. Left unset by default so the reference "
                         "trains to convergence (see note at top of file).")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_groupnorm", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    paths = [retrain_one(args.dataset, c, args, device)
             for c in args.forget_class]
    print("\nReference checkpoints:")
    for p in paths:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
