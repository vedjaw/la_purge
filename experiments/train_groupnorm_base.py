"""Train ResNet-18 with GroupNorm on CIFAR-10 (BN vulnerability fix)."""
import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

ROOT = Path(__file__).resolve().parent.parent

os.environ.pop("MKL_THREADING_LAYER", None)
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"


def replace_bn_with_gn(model, num_groups=32):
    """Replace all BatchNorm2d layers with GroupNorm in a model."""
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            num_channels = module.num_features
            # Use min(num_groups, num_channels) to handle small channel counts
            groups = min(num_groups, num_channels)
            # Ensure num_channels is divisible by groups
            while num_channels % groups != 0:
                groups //= 2
            setattr(model, name, nn.GroupNorm(groups, num_channels))
        else:
            replace_bn_with_gn(module, num_groups)
    return model


def get_model_groupnorm(num_classes=10, num_groups=32):
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    m = replace_bn_with_gn(m, num_groups)
    return m


def get_cifar10(data_dir, batch_size):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    train_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=28),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                              download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                             download=True, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader


def accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(ROOT / "data"))
    ap.add_argument("--save_path",
                    default=str(ROOT / "checkpoints/base_cifar10_resnet18_groupnorm.pth"))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_groups", type=int, default=32,
                    help="Number of groups for GroupNorm (default 32).")
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader = get_cifar10(args.data_dir, args.batch_size)
    model = get_model_groupnorm(num_classes=10,
                                 num_groups=args.num_groups).to(device)

    # Verify no BN layers remain
    bn_count = sum(1 for m in model.modules()
                   if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)))
    gn_count = sum(1 for m in model.modules() if isinstance(m, nn.GroupNorm))
    print(f"Model: ResNet-18 with GroupNorm")
    print(f"  BatchNorm layers: {bn_count} (should be 0)")
    print(f"  GroupNorm layers: {gn_count}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        scheduler.step()
        train_acc = 100.0 * correct / total
        test_acc = accuracy(model, test_loader, device)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), args.save_path)

        if epoch % 10 == 0 or epoch <= 5:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:3d}/{args.epochs}  "
                  f"loss={running_loss/total:.4f}  "
                  f"train_acc={train_acc:.2f}%  "
                  f"test_acc={test_acc:.2f}%  "
                  f"best={best_acc:.2f}%  lr={lr_now:.6f}")

    print(f"\nTraining complete. Best test accuracy: {best_acc:.2f}%")
    print(f"Saved: {args.save_path}")

    # Final eval
    model.load_state_dict(torch.load(args.save_path, map_location=device))
    final_acc = accuracy(model, test_loader, device)
    print(f"Final checkpoint test accuracy: {final_acc:.2f}%")


if __name__ == "__main__":
    main()
