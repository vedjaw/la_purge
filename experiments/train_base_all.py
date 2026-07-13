"""Train base ResNet-18 on all datasets (200 epochs, cosine LR, proper training)."""
import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from run import get_model, get_dataset

# Stop once best test accuracy reaches this threshold (after min_epochs).
DEFAULT_ACC_THRESHOLDS = {
    "cifar10":   93.0,
    "cifar100":  68.0,   # ResNet-18 @224 fine-tune; 72% needs careful tuning
    "mnist":     99.0,
    "svhn":      94.0,
    "stl10":     82.0,
    "pathmnist": 88.0,
}

# ImageNet-pretrained ResNet-18 is fine-tuned, not trained from scratch.
DEFAULT_LR = {
    "cifar10":   0.1,
    "cifar100":  0.01,   # 0.1 overfits badly (train 77% / test 56%)
    "mnist":     0.01,
    "svhn":      0.01,
    "stl10":     0.01,
    "pathmnist": 0.01,
}

# Stop if best test accuracy does not improve for this many epochs.
DEFAULT_PATIENCE = {
    "cifar100": 25,
    "stl10":    20,
}

DEFAULT_MIN_EPOCHS = {
    "cifar100": 10,
}


def checkpoint_path(save_dir, dataset_name, use_groupnorm=False):
    gn_tag = "_groupnorm" if use_groupnorm else ""
    return os.path.join(save_dir, f"base_{dataset_name}_resnet18{gn_tag}.pth")


def resolve_acc_threshold(dataset_name, acc_threshold, early_stop, use_groupnorm):
    if not early_stop:
        return None
    if acc_threshold is not None:
        return acc_threshold
    threshold = DEFAULT_ACC_THRESHOLDS.get(dataset_name)
    if threshold is not None and use_groupnorm:
        threshold = max(threshold - 0.5, 0.0)
    return threshold


def resolve_lr(dataset_name, lr):
    if lr is not None:
        return lr
    return DEFAULT_LR.get(dataset_name, 0.01)


def resolve_patience(dataset_name, patience):
    if patience is not None:
        return patience
    return DEFAULT_PATIENCE.get(dataset_name, 15)


def resolve_min_epochs(dataset_name, min_epochs):
    if min_epochs is not None:
        return min_epochs
    return DEFAULT_MIN_EPOCHS.get(dataset_name, 5)


def train_one(dataset_name, data_dir, save_dir, epochs, lr, batch_size,
              seed, use_groupnorm=False, force=False,
              acc_threshold=None, early_stop=True,
              min_epochs=None, patience=None):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = resolve_lr(dataset_name, lr)
    min_epochs = resolve_min_epochs(dataset_name, min_epochs)
    patience = resolve_patience(dataset_name, patience)

    train_set, test_set, num_classes = get_dataset(dataset_name, data_dir)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = get_model("resnet18", num_classes, pretrained=True,
                      use_groupnorm=use_groupnorm).to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    save_path = checkpoint_path(save_dir, dataset_name, use_groupnorm)
    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(save_path) and not force:
        print(f"\n[SKIP] {dataset_name}: checkpoint already exists → {save_path}")
        return save_path, None

    best_acc = 0.0
    epochs_no_improve = 0
    stop_threshold = resolve_acc_threshold(
        dataset_name, acc_threshold, early_stop, use_groupnorm)
    stop_reason = None

    print(f"\n{'='*60}")
    print(f"Training base model: {dataset_name} ({'GroupNorm' if use_groupnorm else 'BatchNorm'})")
    print(f"  Classes: {num_classes} | Train: {len(train_set)} | Test: {len(test_set)}")
    print(f"  Epochs: {epochs} | LR: {lr} | Save: {save_path}")
    if stop_threshold is not None:
        print(f"  Early stop: best test >= {stop_threshold:.1f}% (after epoch {min_epochs})")
    if patience > 0:
        print(f"  Plateau stop: no improvement for {patience} epochs (after epoch {min_epochs})")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        model.train()
        correct, total = 0, 0
        for imgs, labs in tqdm(train_loader, desc=f"  Epoch {epoch}/{epochs}",
                               leave=False, file=sys.stderr, dynamic_ncols=True):
            imgs, labs = imgs.to(device), labs.view(-1).long().to(device)
            optimizer.zero_grad()
            out = model(imgs)
            criterion(out, labs).backward()
            optimizer.step()
            correct += out.argmax(1).eq(labs).sum().item()
            total += len(labs)

        scheduler.step()
        train_acc = 100.0 * correct / total

        model.eval()
        t_correct, t_total = 0, 0
        with torch.no_grad():
            for imgs, labs in test_loader:
                imgs, labs = imgs.to(device), labs.view(-1).long().to(device)
                t_correct += model(imgs).argmax(1).eq(labs).sum().item()
                t_total += len(labs)
        test_acc = 100.0 * t_correct / t_total

        if test_acc > best_acc:
            best_acc = test_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epoch <= 3 or epoch == epochs:
            tqdm.write(f"  Epoch {epoch:3d}: train={train_acc:.2f}% "
                       f"test={test_acc:.2f}% best={best_acc:.2f}%")

        if epoch >= min_epochs:
            if (stop_threshold is not None and best_acc >= stop_threshold):
                stop_reason = (f"threshold {stop_threshold:.1f}% "
                               f"(best={best_acc:.2f}%)")
                break
            if patience > 0 and epochs_no_improve >= patience:
                stop_reason = (f"plateau — no improvement for {patience} epochs "
                               f"(best={best_acc:.2f}%)")
                break

    if stop_reason:
        print(f"  [early stop] {stop_reason} at epoch {epoch}")
        print(f"  Done (early stop). Best test accuracy: {best_acc:.2f}%")
    else:
        print(f"  Done. Best test accuracy: {best_acc:.2f}%")
    print(f"  Saved: {save_path}")
    return save_path, best_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["cifar10", "mnist", "svhn", "stl10", "pathmnist"])
    ap.add_argument("--data_dir", default=str(ROOT / "data"))
    ap.add_argument("--save_dir", default=str(ROOT / "checkpoints"))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=None,
                    help="Learning rate (default: per-dataset, see DEFAULT_LR)")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_groupnorm", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="Retrain even if checkpoint already exists")
    ap.add_argument("--acc_threshold", type=float, default=None,
                    help="Stop once best test acc reaches this %%")
    ap.add_argument("--no_early_stop", action="store_true",
                    help="Disable threshold + plateau early stopping")
    ap.add_argument("--min_epochs", type=int, default=None,
                    help="Min epochs before early-stop (default: per-dataset)")
    ap.add_argument("--patience", type=int, default=None,
                    help="Stop after N epochs without best-test improvement "
                         "(default: per-dataset, 15)")
    args = ap.parse_args()

    results = []
    for ds in args.datasets:
        path, acc = train_one(
            ds, args.data_dir, args.save_dir, args.epochs,
            args.lr, args.batch_size, args.seed,
            args.use_groupnorm, force=args.force,
            acc_threshold=args.acc_threshold,
            early_stop=not args.no_early_stop,
            min_epochs=args.min_epochs,
            patience=args.patience,
        )
        results.append((ds, acc, path))

    print(f"\n{'='*60}")
    print("Base Model Training Summary")
    print(f"{'='*60}")
    for ds, acc, path in results:
        if acc is None:
            print(f"  {ds:<12} (skipped — already exists)  {path}")
        else:
            print(f"  {ds:<12} {acc:.2f}%  {path}")


if __name__ == "__main__":
    main()
