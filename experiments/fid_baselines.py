"""Feature-space FID for unlearned checkpoints."""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
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


def extract_features(model, loader, device):
    """Extract penultimate-layer features (after avgpool, before fc)."""
    features_list = []
    hook_output = {}

    def hook_fn(module, inp, out):
        hook_output['feat'] = out

    handle = model.avgpool.register_forward_hook(hook_fn)
    model.eval()

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            _ = model(imgs)
            feat = hook_output['feat'].squeeze(-1).squeeze(-1)
            features_list.append(feat.cpu().numpy())

    handle.remove()
    return np.concatenate(features_list)


def compute_fid(feats_a, feats_b):
    """Fréchet distance between two sets of features."""
    mu_a, mu_b = feats_a.mean(axis=0), feats_b.mean(axis=0)
    sigma_a = np.cov(feats_a, rowvar=False)
    sigma_b = np.cov(feats_b, rowvar=False)

    diff = mu_a - mu_b
    covmean, _ = sqrtm(sigma_a @ sigma_b, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_a + sigma_b - 2 * covmean)
    return float(fid)


def cosine_sim(feats_a, feats_b):
    """Mean cosine similarity between corresponding feature vectors."""
    n = min(len(feats_a), len(feats_b))
    a = feats_a[:n]
    b = feats_b[:n]
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return float(np.mean(np.sum(a_norm * b_norm, axis=1)))


def l2_dist(feats_a, feats_b):
    """Mean L2 distance between corresponding feature vectors."""
    n = min(len(feats_a), len(feats_b))
    return float(np.mean(np.linalg.norm(feats_a[:n] - feats_b[:n], axis=1)))


def main():
    p = argparse.ArgumentParser(description="EXP-C: FID_feat for all methods")
    p.add_argument('--reference', required=True,
                   help='Path to reference checkpoint (retrain or base model)')
    p.add_argument('--data_dir', default=str(ROOT / 'data'))
    p.add_argument('--forget_class', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--checkpoints', nargs='+', required=True,
                   help='name:path pairs for each unlearned model')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    _, test_set, num_classes = get_cifar10(args.data_dir)

    fc = args.forget_class
    forget_test_idx = [i for i, t in enumerate(test_set.targets) if t == fc]
    forget_loader = DataLoader(Subset(test_set, forget_test_idx),
                               batch_size=args.batch_size, shuffle=False)

    ref_model = get_model(num_classes)
    ref_model.load_state_dict(torch.load(args.reference, map_location=device))
    ref_model.to(device)
    ref_feats = extract_features(ref_model, forget_loader, device)
    del ref_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    ckpt_pairs = []
    for item in args.checkpoints:
        if ':' in item:
            name, path = item.split(':', 1)
        else:
            name = item.rsplit('/', 1)[-1].replace('.pth', '')
            path = item
        ckpt_pairs.append((name, path))

    print("\n" + "=" * 75)
    print(f"EXP-C: FID_feat — representation distance from reference model")
    print(f"  Reference: {args.reference}")
    print(f"  Forget class: {fc}")
    print(f"  Samples: {len(forget_test_idx)} (class-{fc} test images)")
    print("=" * 75)
    print(f"  {'Method':<20} {'FID_feat':>10} {'Cos↓':>8} {'L2↑':>8}")
    print("-" * 75)

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

        feats = extract_features(model, forget_loader, device)
        fid  = compute_fid(feats, ref_feats)
        cos  = cosine_sim(feats, ref_feats)
        l2   = l2_dist(feats, ref_feats)

        print(f"  {name:<20} {fid:>10.1f} {cos:>8.3f} {l2:>8.3f}")

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("=" * 75)
    print()
    print("Interpretation:")
    print("  If --reference is retrain: lower FID = closer to gold-standard erasure")
    print("  If --reference is base:    higher FID = more representational change")
    print("  Cos: cosine similarity (lower = more change)")
    print("  L2:  mean L2 distance (higher = more change)")


if __name__ == '__main__':
    main()
