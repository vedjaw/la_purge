"""Linear probe on penultimate features (Finding: representation leakage)."""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
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


def extract_features(model, loader, device):
    """Extract penultimate-layer features (after avgpool, before fc)."""
    features_list = []
    labels_list = []

    hook_output = {}

    def hook_fn(module, inp, out):
        hook_output['feat'] = out

    handle = model.avgpool.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        for imgs, labs in loader:
            imgs = imgs.to(device)
            _ = model(imgs)
            feat = hook_output['feat'].squeeze(-1).squeeze(-1)
            features_list.append(feat.cpu().numpy())
            labels_list.append(labs.numpy())

    handle.remove()
    return np.concatenate(features_list), np.concatenate(labels_list)


def run_probe(features, binary_labels, method_name):
    """5-fold CV logistic regression probe. Reports mean ± std accuracy."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []

    for train_idx, test_idx in skf.split(features, binary_labels):
        X_tr, X_te = features[train_idx], features[test_idx]
        y_tr, y_te = binary_labels[train_idx], binary_labels[test_idx]

        clf = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        accs.append(accuracy_score(y_te, preds) * 100.0)

    mean_acc = np.mean(accs)
    std_acc  = np.std(accs, ddof=1)
    return mean_acc, std_acc


def main():
    p = argparse.ArgumentParser(description="EXP-B: Linear probe for class recovery")
    p.add_argument('--data_dir', default=str(ROOT / 'data'))
    p.add_argument('--forget_class', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--checkpoints', nargs='+', required=True,
                   help='name:path pairs, e.g. salun:path/to/ckpt.pth')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    _, test_set, num_classes = get_cifar10(args.data_dir)

    fc = args.forget_class
    forget_test_idx = [i for i, t in enumerate(test_set.targets) if t == fc]
    other_test_idx  = [i for i, t in enumerate(test_set.targets) if t != fc]

    np.random.seed(42)
    n_pos = len(forget_test_idx)
    sampled_neg = np.random.choice(other_test_idx, size=n_pos, replace=False)
    probe_indices = np.array(forget_test_idx + list(sampled_neg))
    probe_labels  = np.array([1] * n_pos + [0] * n_pos)

    probe_loader = DataLoader(Subset(test_set, probe_indices),
                              batch_size=args.batch_size, shuffle=False)

    ckpt_pairs = []
    for item in args.checkpoints:
        if ':' in item:
            name, path = item.split(':', 1)
        else:
            name = item.rsplit('/', 1)[-1].replace('.pth', '')
            path = item
        ckpt_pairs.append((name, path))

    print("\n" + "=" * 70)
    print(f"EXP-B: LINEAR PROBE — can class {fc} be recovered from features?")
    print("=" * 70)
    print(f"  Positive samples (class {fc} test): {n_pos}")
    print(f"  Negative samples (other test):      {n_pos}")
    print(f"  Evaluation: 5-fold stratified CV")
    print(f"  Ideal (perfect erasure): 50.0%  (random guessing)")
    print(f"  Worst case (no erasure):  ~100% (features fully encode class)")
    print("-" * 70)
    print(f"  {'Method':<20} {'Probe Acc':<15} {'Interpretation'}")
    print("-" * 70)

    for name, path in ckpt_pairs:
        model = get_model(num_classes)
        try:
            state = torch.load(path, map_location=device)
            model.load_state_dict(state)
        except FileNotFoundError:
            print(f"  {name:<20} SKIPPED — checkpoint not found: {path}")
            continue
        except RuntimeError as e:
            print(f"  {name:<20} SKIPPED — load error: {e}")
            continue
        model.to(device)

        feats, _ = extract_features(model, probe_loader, device)
        mean_acc, std_acc = run_probe(feats, probe_labels, name)

        if mean_acc < 55:
            interp = "ERASED (near random)"
        elif mean_acc < 70:
            interp = "partial erasure"
        elif mean_acc < 85:
            interp = "weak erasure"
        else:
            interp = "NOT ERASED (class info intact)"

        print(f"  {name:<20} {mean_acc:.1f} ± {std_acc:.1f}%    {interp}")

    print("=" * 70)
    print()
    print("Interpretation guide:")
    print("  ~50%  = representations contain NO class-0 info (ideal)")
    print("  ~100% = representations fully encode class-0 (no erasure)")
    print("  A method with FA = 0% but probe >> 50% achieves output-level")
    print("  forgetting only; class info is still recoverable from features.")


if __name__ == '__main__':
    main()
