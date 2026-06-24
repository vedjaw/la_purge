"""MIA AUROC on the pre-unlearning base model (Finding: MIA insufficiency)."""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
ROOT = Path(__file__).resolve().parent.parent
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_cifar10(data_dir):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)
    tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                         download=True, transform=tf)
    test  = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                         download=True, transform=tf)
    return train, test, 10


def compute_mia_auroc(model, member_loader, nonmember_loader, device,
                      max_n=None):
    """AUROC-based MIA: members = forget-class train, non-members = forget-class test."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')

    def get_signals(loader, limit=None):
        losses, confs = [], []
        seen = 0
        with torch.no_grad():
            for imgs, labs in loader:
                imgs, labs = imgs.to(device), labs.to(device)
                logits = model(imgs)
                losses.extend(criterion(logits, labs).cpu().numpy())
                confs.extend(F.softmax(logits, dim=1).max(1).values.cpu().numpy())
                seen += imgs.size(0)
                if limit and seen >= limit:
                    break
        return np.array(losses[:limit]), np.array(confs[:limit])

    m_loss, m_conf = get_signals(member_loader)
    n = len(m_loss) if max_n is None else min(max_n, len(m_loss))
    nm_loss, nm_conf = get_signals(nonmember_loader, limit=n)

    def auroc(member_scores, nonmember_scores):
        if len(member_scores) == 0 or len(nonmember_scores) == 0:
            return 0.5
        labels = np.concatenate([np.ones(len(member_scores)),
                                 np.zeros(len(nonmember_scores))])
        scores = np.concatenate([member_scores, nonmember_scores])
        return roc_auc_score(labels, scores)

    mia_loss = auroc(-m_loss[:n], -nm_loss)
    mia_conf = auroc(m_conf[:n], nm_conf)
    mia_avg  = (mia_loss + mia_conf) / 2.0
    return mia_loss, mia_conf, mia_avg


def accuracy(model, loader, device):
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for imgs, labs in loader:
            imgs, labs = imgs.to(device), labs.to(device)
            preds = model(imgs).argmax(1)
            correct += (preds == labs).sum().item()
            total += labs.size(0)
    return 100.0 * correct / total


def main():
    p = argparse.ArgumentParser(description="EXP-A: Base-model MIA AUROC")
    p.add_argument('--checkpoint', required=True,
                   help='Path to the base (pre-unlearning) model checkpoint')
    p.add_argument('--data_dir', default=str(ROOT / 'data'))
    p.add_argument('--forget_class', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=128)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_set, test_set, num_classes = get_cifar10(args.data_dir)
    model = get_model(num_classes)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

    fc = args.forget_class

    forget_train_idx = [i for i, t in enumerate(train_set.targets) if t == fc]
    forget_test_idx  = [i for i, t in enumerate(test_set.targets)  if t == fc]
    retain_train_idx = [i for i, t in enumerate(train_set.targets) if t != fc]

    forget_train_loader = DataLoader(Subset(train_set, forget_train_idx),
                                     batch_size=args.batch_size)
    forget_test_loader  = DataLoader(Subset(test_set, forget_test_idx),
                                     batch_size=args.batch_size)
    retain_loader = DataLoader(Subset(train_set, retain_train_idx),
                               batch_size=args.batch_size)
    full_test_loader = DataLoader(test_set, batch_size=args.batch_size)

    ta = accuracy(model, full_test_loader, device)
    fa = accuracy(model, forget_train_loader, device)
    ra = accuracy(model, retain_loader, device)

    mia_loss, mia_conf, mia_avg = compute_mia_auroc(
        model, forget_train_loader, forget_test_loader, device)

    print("\n" + "=" * 60)
    print("EXP-A: BASE MODEL (before any unlearning)")
    print("=" * 60)
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Forget class: {fc}")
    print(f"  TA  : {ta:.2f}%")
    print(f"  FA  : {fa:.2f}%   (should be ~98% — model knows class {fc})")
    print(f"  RA  : {ra:.2f}%")
    print()
    print(f"  MIA (loss AUROC) : {mia_loss:.4f}")
    print(f"  MIA (conf AUROC) : {mia_conf:.4f}")
    print(f"  MIA (average)    : {mia_avg:.4f}")
    print()
    if abs(mia_avg - 0.5) < 0.05:
        print("  >> MIA ≈ 0.5 BEFORE unlearning.")
        print("  >> This proves MIA AUROC does not measure unlearning quality")
        print("     for class-level forgetting on well-generalizing models.")
        print("     The model generalizes to the class concept; individual")
        print("     train/test samples are indistinguishable by loss/confidence.")
    else:
        print(f"  >> MIA = {mia_avg:.4f}, which is NOT near 0.5.")
        print("  >> The base model shows a measurable train-test gap.")
        print("     MIA may still be informative in this setting.")
    print("=" * 60)


if __name__ == '__main__':
    main()
