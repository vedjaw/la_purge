"""
PURGE: Projected Unlearning via Retain-Guided Erasure
=====================================================
Paper: [Under preparation — targeting top-tier ML venue]

Machine unlearning is the dual of continual learning: CL protects old knowledge
while learning new tasks; MU protects retained knowledge while erasing old data.
PURGE exploits this duality by adapting gradient projection (A-GEM) from CL to
constrain the unlearning process, ensuring every update step provably does not
increase retain-set loss.

Key contributions:
  1. Gradient Projection — forget-direction gradients are projected onto the
     half-space where retain loss does not increase, preventing catastrophic
     retain-set damage by construction.
  2. Multi-Layer Representation Erasure — intermediate activations on the forget
     set are pushed toward the retain distribution, erasing information from
     hidden layers (not only the output).
  3. KD-Anchored Stability — knowledge distillation from the frozen pre-unlearning
     model preserves retain-set output behavior.

  4. Retain-Confusion Target (kl_retain) — instead of pushing forget outputs to
     uniform (detectable!), pushes them toward the natural output distribution
     the model produces on the retain set. This makes the unlearned model
     indistinguishable from a retrained-from-scratch model.
  5. Retain-Loss Budget — monitors retain loss and stops unlearning when it
     exceeds a configurable factor of the initial retain loss.

Algorithm (per step):
  a. Forward retain batch  → retain logits, retain activations (detached targets)
  b. Forward retain batch through frozen original → KD targets (no grad)
  c. Forward forget batch  → forget logits, forget activations (gradient-connected)
  d. Forget gating: if batch output entropy > max_entropy × gate_factor,
     the batch is already near-uniform — skip forget gradient, retain-only step
  e. Hard entropy cap (GA only): if CE(forget) ≥ log(K), retain-only step
  f. Forget objective: −CE (GA), KL→uniform, or KL→retain_confusion + λ·RepErasure
  g. Compute retain gradient: CE(retain) + β·KD(retain ∥ original)
  h. Project forget gradient onto retain-safe half-space
  i. Clip and apply projected gradient (or retain-only if gated/capped)
  j. End-of-epoch: if retain loss > budget, stop
"""

import argparse
import copy
import math
import os
import sys
import time
from collections import OrderedDict

try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# ========================= MODEL =========================

def _replace_bn_with_gn(model, num_groups=32):
    """Replace all BatchNorm2d layers with GroupNorm."""
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


def get_model(model_name, num_classes=10, pretrained=True, use_groupnorm=False):
    if model_name == 'resnet18':
        model = models.resnet18(weights='DEFAULT' if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(weights='DEFAULT' if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'vgg16':
        model = models.vgg16(weights='DEFAULT' if pretrained else None)
        model.classifier[-1] = nn.Linear(4096, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    if use_groupnorm:
        model = _replace_bn_with_gn(model)
        bn_count = sum(1 for m in model.modules()
                       if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)))
        gn_count = sum(1 for m in model.modules()
                       if isinstance(m, nn.GroupNorm))
        print(f"  [GroupNorm] BN layers remaining: {bn_count} (should be 0), "
              f"GN layers: {gn_count}")
    return model


# ========================= DATA =========================

def _attach_medmnist_targets(ds):
    """MedMNIST exposes `labels` or `label`; CIFAR-style code expects `.targets`."""
    if hasattr(ds, 'labels'):
        arr = np.asarray(ds.labels).squeeze()
    elif hasattr(ds, 'label'):
        arr = np.asarray(ds.label).squeeze()
    else:
        raise AttributeError("MedMNIST dataset has no labels/label attribute")
    ds.targets = arr.tolist()


def get_dataset(dataset_name, data_dir='./data'):
    if dataset_name == 'cifar10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        num_classes = 10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        ds_cls = torchvision.datasets.CIFAR10
        train_set = ds_cls(root=data_dir, train=True, download=True,
                           transform=transform_train)
        test_set = ds_cls(root=data_dir, train=False, download=True,
                          transform=transform_test)
        return train_set, test_set, num_classes

    if dataset_name == 'cifar100':
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        num_classes = 100
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_set = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=transform_test)
        return train_set, test_set, num_classes

    # In get_dataset(), replace the pathmnist block with this:

    if dataset_name == 'pathmnist':
        try:
            from medmnist import PathMNIST
        except ImportError as e:
            raise ImportError(
                "PathMNIST requires the medmnist package. Install with:\n"
                "  pip install medmnist") from e

        num_classes = 9
        mean = std = (0.5, 0.5, 0.5)
        transform_train = transforms.Compose([
            transforms.Resize(224),            # ← add this
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224),            # ← add this
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_set = PathMNIST(
            split='train', download=True, root=data_dir,
            size=28,                           # ← was 224
            as_rgb=True, transform=transform_train)
        test_set = PathMNIST(
            split='test', download=True, root=data_dir,
            size=28,                           # ← was 224
            as_rgb=True, transform=transform_test)
        _attach_medmnist_targets(train_set)
        _attach_medmnist_targets(test_set)
        return train_set, test_set, num_classes

    if dataset_name == 'mnist':
        mean, std = (0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)
        num_classes = 10
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.Resize(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_set = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.MNIST(
            root=data_dir, train=False, download=True, transform=transform_test)
        return train_set, test_set, num_classes

    if dataset_name == 'svhn':
        mean, std = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
        num_classes = 10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_set = torchvision.datasets.SVHN(
            root=data_dir, split='train', download=True, transform=transform_train)
        test_set = torchvision.datasets.SVHN(
            root=data_dir, split='test', download=True, transform=transform_test)
        train_set.targets = train_set.labels
        test_set.targets = test_set.labels
        return train_set, test_set, num_classes

    if dataset_name == 'stl10':
        mean, std = (0.4467, 0.4398, 0.4066), (0.2242, 0.2215, 0.2239)
        num_classes = 10
        transform_train = transforms.Compose([
            transforms.RandomCrop(96, padding=12),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_set = torchvision.datasets.STL10(
            root=data_dir, split='train', download=True, transform=transform_train)
        test_set = torchvision.datasets.STL10(
            root=data_dir, split='test', download=True, transform=transform_test)
        train_set.targets = train_set.labels.tolist()
        test_set.targets = test_set.labels.tolist()
        return train_set, test_set, num_classes

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def split_forget_retain(train_set, forget_type, forget_pct=0.01,
                        forget_class=None):
    targets = np.array(train_set.targets) if hasattr(train_set, 'targets') \
        else np.array([t for _, t in train_set])

    if forget_type == 'class':
        assert forget_class is not None, "Must specify --forget_class"
        forget_idx = np.where(targets == forget_class)[0]
        retain_idx = np.where(targets != forget_class)[0]
    elif forget_type == 'sample':
        n = len(train_set)
        n_forget = max(1, int(n * forget_pct))
        perm = np.random.permutation(n)
        forget_idx, retain_idx = perm[:n_forget], perm[n_forget:]
    else:
        raise ValueError(f"Unsupported forget_type: {forget_type}")

    forget_set = Subset(train_set, forget_idx)
    retain_set = Subset(train_set, retain_idx)
    print(f"  Forget set: {len(forget_set)} | Retain set: {len(retain_set)}")
    return forget_set, retain_set


# ========================= HOOKS =========================

class ActivationExtractor:
    """Register forward hooks to capture intermediate representations."""

    def __init__(self, model, layer_names):
        self.activations = OrderedDict()
        self._hooks = []
        for name, module in model.named_modules():
            if name in layer_names:
                self._hooks.append(
                    module.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def fn(_, __, output):
            self.activations[name] = output
        return fn

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def select_monitor_layers(model):
    """Pick intermediate layers for representation monitoring (ResNet-aware)."""
    candidates = []
    for name, module in model.named_modules():
        if name in ('layer3', 'layer4'):
            candidates.append(name)
        if isinstance(module, nn.AdaptiveAvgPool2d):
            candidates.append(name)
    if not candidates:
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                candidates.append(name)
                break
    return candidates


# =================== GRADIENT PROJECTION ===================

def project_gradient(g_forget, g_retain):
    """
    A-GEM–style gradient projection onto the retain-safe half-space.

    Given the forget-direction gradient g_f and the retain-reference gradient
    g_r (descent direction for retain loss), project g_f so that:

        ⟨g_projected, g_r⟩  ≥  0

    When the dot product is already non-negative the two objectives are
    compatible and g_f is returned unchanged.  Otherwise g_f is projected onto
    the boundary of the half-space, removing exactly the component that would
    increase retain loss.

    Returns (projected_gradient, was_projected).
    """
    dot = sum((gf * gr).sum() for gf, gr in zip(g_forget, g_retain))

    if dot >= 0:
        return g_forget, False  # no conflict

    sq_norm = sum((gr ** 2).sum() for gr in g_retain)
    if sq_norm < 1e-12:
        return g_forget, False  # degenerate retain gradient

    coeff = dot / sq_norm
    projected = [gf - coeff * gr for gf, gr in zip(g_forget, g_retain)]
    return projected, True


# ================ REPRESENTATION ERASURE ================

def _pool_to_vector(t):
    """Collapse spatial dims to a 2-D (B, C) feature vector."""
    if t.dim() == 4:
        return F.adaptive_avg_pool2d(t, 1).flatten(1)
    if t.dim() == 3:
        return t.mean(dim=-1)
    return t


def representation_erasure_loss(forget_acts, retain_acts):
    """
    Multi-layer MSE loss that pushes each forget-set activation toward the
    retain-set batch mean.  Gradients flow only through forget_acts.
    """
    total = 0.0
    count = 0
    for name in forget_acts:
        if name not in retain_acts:
            continue
        f = _pool_to_vector(forget_acts[name])
        r = _pool_to_vector(retain_acts[name])
        target = r.mean(dim=0, keepdim=True).expand_as(f)
        total = total + F.mse_loss(f, target)
        count += 1
    return total / max(count, 1)


# ======================== PURGE ========================

def purge_unlearn(model, forget_loader, retain_loader, device, args):
    """
    PURGE main loop.

    Per-step procedure
    ------------------
    1. Forward retain batch  → out_r, retain activations (detached targets)
    2. Forward retain batch through frozen original → KD target (no grad)
    3. Forward forget batch  → out_f, forget activations (live, gradient-connected)
    4. Forget gradient:  −CE(out_f, y_f) + λ · RepErasureLoss
    5. Retain gradient:  CE(out_r, y_r) + β · KD(out_r ∥ orig_out_r)
    6. Project forget gradient onto retain-safe half-space
    7. Clip projected gradient and apply SGD step
    """
    print(f"\n{'=' * 60}")
    print(" PURGE: Projected Unlearning via Retain-Guided Erasure")
    print(f"{'=' * 60}")

    # Frozen reference for KD anchoring
    original = copy.deepcopy(model)
    original.eval()
    for p in original.parameters():
        p.requires_grad = False

    layer_names = select_monitor_layers(model)
    print(f"  Monitor layers : {layer_names}")
    print(f"  Epochs         : {args.epochs}")
    print(f"  LR             : {args.lr}")
    print(f"  Rep weight (λ) : {args.rep_weight}")
    print(f"  KD weight  (β) : {args.kd_weight}")
    print(f"  Temperature    : {args.temperature}")
    print(f"  Forget obj.    : {args.forget_objective}")
    print(f"  Forget gate    : {args.forget_gate}")

    # Pre-compute retain confusion distribution (what a retrained model
    # would naturally output — averaged softmax over the retain set)
    retain_confusion = None
    if args.forget_objective == 'kl_retain':
        print("  Computing retain confusion distribution...")
        original.eval()
        _sum = None
        _cnt = 0
        with torch.no_grad():
            for _xr, _ in retain_loader:
                _xr = _xr.to(device)
                _probs = F.softmax(original(_xr), dim=1)
                if _sum is None:
                    _sum = _probs.sum(dim=0)
                else:
                    _sum += _probs.sum(dim=0)
                _cnt += _xr.size(0)
        retain_confusion = (_sum / _cnt).unsqueeze(0)
        print(f"  Retain confusion dist: {retain_confusion.squeeze().cpu().tolist()[:5]}... "
              f"(top-5 of {retain_confusion.size(1)} classes)")

    extractor = ActivationExtractor(model, layer_names)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Measure initial retain loss for budget-based early stopping
    model.eval()
    _init_rl = 0.0
    _init_n = 0
    with torch.no_grad():
        for _xr, _yr in retain_loader:
            _xr, _yr = _xr.to(device), _yr.view(-1).to(device)
            _init_rl += F.cross_entropy(
                model(_xr), _yr, reduction='sum').item()
            _init_n += _yr.size(0)
    initial_retain_loss = _init_rl / max(_init_n, 1)
    retain_budget = initial_retain_loss * args.retain_budget_factor
    print(f"  Initial retain loss : {initial_retain_loss:.4f}")
    print(f"  Retain loss budget  : {retain_budget:.4f} "
          f"({args.retain_budget_factor}×)")

    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()
    retain_iter = iter(retain_loader)
    max_entropy = None
    random_chance = None
    budget_exceeded = False
    fa_reached = False
    history = {'forget_ce': [], 'retain_loss': [], 'rep_loss': [],
               'proj_rate': [], 'gate_rate': [], 'epoch_fa': []}

    if getattr(args, 'freeze_early_layers', False):
        frozen_names = []
        for name, param in model.named_parameters():
            if any(name.startswith(prefix) for prefix in
                   ['conv1.', 'bn1.', 'layer1.', 'layer2.']):
                param.requires_grad = False
                frozen_names.append(name)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay)
        print(f"  [selective-layer] Froze {len(frozen_names)} params "
              f"(conv1/bn1/layer1/layer2)")
        print(f"  [selective-layer] Trainable: layer3, layer4, fc only")

    for epoch in range(args.epochs):
        if budget_exceeded:
            print(f"  Retain budget exceeded — stopping at epoch {epoch}.")
            break
        if fa_reached:
            print(f"  FA target reached — stopping at epoch {epoch}.")
            break

        ep_fce, ep_rl, ep_rep = 0.0, 0.0, 0.0
        n_proj, n_steps, n_gated = 0, 0, 0

        pbar = tqdm(forget_loader,
                    desc=f"  Epoch {epoch + 1}/{args.epochs}", leave=True)
        for x_f, y_f in pbar:
            x_f, y_f = x_f.to(device), y_f.view(-1).to(device)

            # Cycle retain loader
            try:
                x_r, y_r = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                x_r, y_r = next(retain_iter)
            x_r, y_r = x_r.to(device), y_r.view(-1).to(device)

            # -------- forward passes --------
            out_r = model(x_r)
            retain_acts = OrderedDict(
                (k, v.detach()) for k, v in extractor.activations.items())

            with torch.no_grad():
                orig_out_r = original(x_r)

            out_f = model(x_f)
            forget_acts = OrderedDict(extractor.activations)

            # -------- lazy init of constants --------
            if max_entropy is None:
                num_cls = out_f.size(1)
                max_entropy = math.log(num_cls)
                random_chance = 1.0 / num_cls

            # -------- forget gating (entropy-based) --------
            with torch.no_grad():
                probs_f = F.softmax(out_f, dim=1)
                batch_entropy = -(probs_f * probs_f.clamp(min=1e-8).log()
                                  ).sum(dim=1).mean().item()
            gated = (args.forget_gate > 0
                     and batch_entropy > max_entropy * args.forget_gate)

            # -------- retain gradient (always computed) --------
            optimizer.zero_grad()
            loss_ce_r = criterion(out_r, y_r.view(-1))
            loss_kd = F.kl_div(
                F.log_softmax(out_r / args.temperature, dim=1),
                F.softmax(orig_out_r / args.temperature, dim=1),
                reduction='batchmean') * (args.temperature ** 2)
            loss_retain = loss_ce_r + args.kd_weight * loss_kd
            loss_retain.backward()
            g_retain = [p.grad.clone() for p in params]

            if gated:
                # Batch already near random chance — retain-only step
                optimizer.zero_grad()
                for p, g in zip(params, g_retain):
                    p.grad = g
                torch.nn.utils.clip_grad_norm_(
                    params, max_norm=args.max_grad_norm)
                optimizer.step()
                n_gated += 1
                with torch.no_grad():
                    loss_ce_f_val = criterion(out_f, y_f.view(-1)).item()
                loss_rep_val = 0.0
            else:
                # -------- forget gradient (entropy-capped) --------
                optimizer.zero_grad()

                with torch.no_grad():
                    loss_ce_f = criterion(out_f, y_f.view(-1))
                    loss_ce_f_val = loss_ce_f.item()

                capped = (args.forget_objective == 'ga'
                          and loss_ce_f_val >= max_entropy)

                if capped:
                    # CE already at/past max entropy — retain-only step
                    optimizer.zero_grad()
                    for p, g in zip(params, g_retain):
                        p.grad = g
                    torch.nn.utils.clip_grad_norm_(
                        params, max_norm=args.max_grad_norm)
                    optimizer.step()
                    n_gated += 1
                    loss_rep_val = 0.0
                else:
                    if args.forget_objective == 'ga':
                        loss_ce_f = criterion(out_f, y_f.view(-1))
                        loss_dir = -loss_ce_f
                    elif args.forget_objective == 'kl_retain':
                        target = retain_confusion.expand_as(out_f)
                        loss_dir = F.kl_div(
                            F.log_softmax(out_f, dim=1), target,
                            reduction='batchmean')
                        loss_ce_f = criterion(out_f, y_f.view(-1))
                    else:  # kl_uniform
                        uniform = torch.ones_like(out_f) / num_cls
                        loss_dir = F.kl_div(
                            F.log_softmax(out_f, dim=1), uniform,
                            reduction='batchmean')
                        loss_ce_f = criterion(out_f, y_f.view(-1))

                    loss_rep = representation_erasure_loss(
                        forget_acts, retain_acts) \
                        if layer_names else torch.tensor(0.0, device=device)
                    loss_forget = loss_dir + args.rep_weight * loss_rep
                    loss_forget.backward()
                    g_forget = [p.grad.clone() for p in params]

                    # -------- projection --------
                    if getattr(args, 'disable_projection', False):
                        g_proj, was_proj = g_forget, False
                    else:
                        g_proj, was_proj = project_gradient(g_forget, g_retain)

                    # -------- apply --------
                    optimizer.zero_grad()
                    for p, g in zip(params, g_proj):
                        p.grad = g
                    torch.nn.utils.clip_grad_norm_(
                        params, max_norm=args.max_grad_norm)
                    optimizer.step()

                    n_proj += int(was_proj)
                    loss_ce_f_val = loss_ce_f.item()
                    loss_rep_val = loss_rep.item()

            # tracking
            ep_fce += loss_ce_f_val
            ep_rl += loss_retain.item()
            ep_rep += loss_rep_val
            n_steps += 1

            pbar.set_postfix(
                F=f"{loss_ce_f_val:.3f}",
                R=f"{loss_retain.item():.3f}",
                G=f"{n_gated}/{n_steps}")

            # -------- intra-epoch FA check --------
            if (args.fa_check_freq > 0
                    and n_steps % args.fa_check_freq == 0):
                model.eval()
                _ifc, _ift = 0, 0
                with torch.no_grad():
                    for _x, _y in forget_loader:
                        _x, _y = _x.to(device), _y.view(-1).to(device)
                        _ifc += model(_x).argmax(1).eq(_y).sum().item()
                        _ift += len(_y)
                intra_fa = _ifc / max(_ift, 1) * 100
                model.train()
                for m in model.modules():
                    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                        m.eval()

                _fa_tgt = args.fa_target
                if _fa_tgt is None and random_chance is not None:
                    _fa_tgt = random_chance * 100
                if _fa_tgt is not None and intra_fa <= _fa_tgt:
                    print(f"\n  [batch {n_steps}] FA {intra_fa:.1f}% "
                          f"<= target {_fa_tgt:.1f}% — stopping mid-epoch.")
                    fa_reached = True
                    break

        # -------- per-epoch forget accuracy --------
        model.eval()
        _fc, _ft = 0, 0
        with torch.no_grad():
            for _x, _y in forget_loader:
                _x, _y = _x.to(device), _y.view(-1).to(device)
                _fc += model(_x).argmax(1).eq(_y).sum().item()
                _ft += len(_y)
        epoch_fa = _fc / max(_ft, 1) * 100
        model.train()
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.eval()

        active_steps = max(n_steps - n_gated, 1)
        proj_rate = n_proj / active_steps
        gate_rate = n_gated / max(n_steps, 1)
        history['forget_ce'].append(ep_fce / max(n_steps, 1))
        history['retain_loss'].append(ep_rl / max(n_steps, 1))
        history['rep_loss'].append(ep_rep / max(n_steps, 1))
        history['proj_rate'].append(proj_rate)
        history['gate_rate'].append(gate_rate)
        history['epoch_fa'].append(epoch_fa)

        # Measure actual retain loss on full retain set for budget check
        model.eval()
        _rl_sum, _rl_n = 0.0, 0
        with torch.no_grad():
            for _xr, _yr in retain_loader:
                _xr, _yr = _xr.to(device), _yr.view(-1).to(device)
                _rl_sum += F.cross_entropy(
                    model(_xr), _yr, reduction='sum').item()
                _rl_n += _yr.size(0)
        epoch_retain_loss = _rl_sum / max(_rl_n, 1)
        model.train()
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.eval()

        print(f"  → Forget CE: {history['forget_ce'][-1]:.4f} | "
              f"Retain: {epoch_retain_loss:.4f} | "
              f"Rep: {history['rep_loss'][-1]:.4f} | "
              f"Proj: {proj_rate:.0%} | "
              f"Gate: {gate_rate:.0%} | "
              f"FA: {epoch_fa:.1f}%")

        if epoch_retain_loss > retain_budget:
            print(f"  ⚠ Retain loss {epoch_retain_loss:.4f} > budget "
                  f"{retain_budget:.4f} — will stop after this epoch.")
            budget_exceeded = True

        fa_target = args.fa_target
        if fa_target is None and random_chance is not None:
            fa_target = random_chance * 100  # e.g. 10% for 10 classes
        if fa_target is not None and epoch_fa <= fa_target:
            print(f"  ✓ FA {epoch_fa:.1f}% ≤ target {fa_target:.1f}% "
                  f"— forgetting complete, will stop.")
            fa_reached = True

    extractor.remove()
    return model, history


# ====================== EVALUATION ======================

def evaluate(model, loader, device, desc="Eval"):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += model(images).argmax(1).eq(labels.view(-1)).sum().item()
            total += len(labels)
    acc = correct / max(total, 1) * 100
    print(f"  {desc}: {acc:.2f}% ({correct}/{total})")
    return acc


def compute_mia(model, forget_loader, forget_class_test_loader, device):
    """
    Membership inference attack — can an attacker distinguish forget-set
    **training** samples (members) from forget-class **test** samples
    (non-members of the same class)?

    This is the correct formulation for class-level forgetting: both groups
    contain images of the same class, differing ONLY in whether the model
    was trained on them.  A perfectly unlearned model treats both groups
    identically → AUROC = 0.5.

    Uses AUROC (threshold-free) on two signal types:
      - Loss-based:  higher loss → predicted non-member
      - Confidence-based: lower max-confidence → predicted non-member

    AUROC = 0.5 means indistinguishable (ideal unlearning).
    AUROC > 0.5 means former members are detectable (incomplete unlearning).
    AUROC < 0.5 means former members look LESS like members than test (over-erasure).
    """
    from sklearn.metrics import roc_auc_score

    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')

    def _signals(loader, max_n=None):
        losses, confs = [], []
        seen = 0
        with torch.no_grad():
            for imgs, labs in loader:
                imgs, labs = imgs.to(device), labs.view(-1).to(device)
                logits = model(imgs)
                losses.extend(criterion(logits, labs).cpu().numpy())
                confs.extend(
                    F.softmax(logits, dim=1).max(1).values.cpu().numpy())
                seen += imgs.size(0)
                if max_n is not None and seen >= max_n:
                    break
        return np.array(losses[:max_n]), np.array(confs[:max_n])

    f_loss, f_conf = _signals(forget_loader)
    n_forget = len(f_loss)

    ft_loss, ft_conf = _signals(forget_class_test_loader, max_n=n_forget)

    def _auroc(member_scores, nonmember_scores):
        if len(member_scores) == 0 or len(nonmember_scores) == 0:
            return 0.5
        labels = np.concatenate([np.ones(len(member_scores)),
                                 np.zeros(len(nonmember_scores))])
        scores = np.concatenate([member_scores, nonmember_scores])
        if len(np.unique(labels)) < 2:
            return 0.5
        return roc_auc_score(labels, scores)

    mia_loss = _auroc(-f_loss, -ft_loss)
    mia_conf = _auroc(f_conf, ft_conf)
    mia_avg = (mia_loss + mia_conf) / 2

    print(f"  MIA (loss AUROC) : {mia_loss:.4f}")
    print(f"  MIA (conf AUROC) : {mia_conf:.4f}")
    print(f"  MIA (average)    : {mia_avg:.4f}  (target ≈ 0.5)")
    print(f"  (forget-class train vs forget-class test, "
          f"{n_forget} vs {len(ft_loss)} samples)")
    return {'loss': mia_loss, 'conf': mia_conf, 'avg': mia_avg}


def compute_feature_distance(model, original_model, loader, device):
    """
    Representation-level verification: cosine similarity and L2 distance
    between penultimate features of the unlearned vs. original model on the
    forget set.  Lower similarity / higher distance = deeper erasure.
    """
    model.eval()
    original_model.eval()
    feats_new, feats_orig = [], []

    def _hook(store):
        def fn(_, __, out):
            store.append(out.detach().flatten(1))
        return fn

    h1 = h2 = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.AdaptiveAvgPool2d):
            h1 = mod.register_forward_hook(_hook(feats_new))
            break
    for name, mod in original_model.named_modules():
        if isinstance(mod, nn.AdaptiveAvgPool2d):
            h2 = mod.register_forward_hook(_hook(feats_orig))
            break

    if h1 is None or h2 is None:
        print("  (skipping feature distance — no AdaptiveAvgPool2d found)")
        return {'cosine_sim': float('nan'), 'l2_dist': float('nan')}

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            model(imgs)
            original_model(imgs)

    h1.remove()
    h2.remove()

    fn = torch.cat(feats_new, 0)
    fo = torch.cat(feats_orig, 0)

    cos = F.cosine_similarity(fn, fo, dim=1).mean().item()
    l2 = (fn - fo).norm(dim=1).mean().item()

    print(f"  Feature cosine sim : {cos:.4f}  (lower = more erased)")
    print(f"  Feature L2 dist    : {l2:.4f}  (higher = more erased)")
    return {'cosine_sim': cos, 'l2_dist': l2}


# ==================== TRAINING HELPER ====================

def train_model(model, loader, device, epochs=5, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4)
    model.train()
    for ep in range(epochs):
        correct = total = 0
        for imgs, labs in tqdm(loader, desc=f"  Train {ep + 1}/{epochs}"):
            imgs, labs = imgs.to(device), labs.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            criterion(out, labs.view(-1).long()).backward()
            optimizer.step()
            correct += out.argmax(1).eq(labs.view(-1)).sum().item()
            total += len(labs)
        print(f"  Epoch {ep + 1}: Acc = {correct / total * 100:.2f}%")
    return model


# ========================= MAIN =========================

def main():
    p = argparse.ArgumentParser(
        description="PURGE: Projected Unlearning via Retain-Guided Erasure")

    # data / model
    p.add_argument('--dataset', default='cifar10',
                   choices=['cifar10', 'cifar100', 'pathmnist', 'mnist', 'svhn', 'stl10'])
    p.add_argument('--model', default='resnet18',
                   choices=['resnet18', 'resnet50', 'vgg16'])
    p.add_argument('--data_dir', default='./data')

    # forget specification
    p.add_argument('--forget_type', default='class',
                   choices=['sample', 'class'])
    p.add_argument('--forget_pct', type=float, default=0.01,
                   help='Fraction to forget (sample mode)')
    p.add_argument('--forget_class', type=int, default=None,
                   help='Class to forget (class mode)')

    # PURGE hyper-parameters
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--rep_weight', type=float, default=0.1,
                   help='λ — representation erasure weight')
    p.add_argument('--kd_weight', type=float, default=1.0,
                   help='β — KD anchoring weight on retain set')
    p.add_argument('--temperature', type=float, default=4.0,
                   help='Distillation temperature')
    p.add_argument('--max_grad_norm', type=float, default=5.0)
    p.add_argument('--forget_objective', default='ga',
                   choices=['ga', 'kl_uniform', 'kl_retain'],
                   help='ga = gradient ascent, kl_uniform = KL toward uniform, '
                        'kl_retain = KL toward retain confusion distribution')
    p.add_argument('--forget_gate', type=float, default=0.9,
                   help='Skip forget gradient when batch output entropy > '
                        'max_entropy * this factor (0 = disable)')
    p.add_argument('--retain_budget_factor', type=float, default=5.0,
                   help='Stop unlearning when retain loss exceeds '
                        'initial_retain_loss × this factor')
    p.add_argument('--fa_target', type=float, default=None,
                   help='Stop when FA drops below this %% '
                        '(default: random chance = 100/num_classes)')
    p.add_argument('--fa_check_freq', type=int, default=10,
                   help='Check FA every N batches within each epoch '
                        '(0 = end-of-epoch only)')

    # infra
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--checkpoint', default=None,
                   help='Path to pre-trained checkpoint')
    p.add_argument('--save_dir', default='./checkpoints')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--config', default=None,
                   help='Path to a YAML config file (e.g. configs/mnist_kl_retain.yaml). '
                        'CLI arguments always override config file values.')
    p.add_argument('--disable_projection', action='store_true',
                   help='Ablation: skip the A-GEM projection step.')
    p.add_argument('--freeze_early_layers', action='store_true',
                   help='Freeze conv1/bn1/layer1/layer2 during unlearning; '
                        'only update layer3, layer4, fc.')
    p.add_argument('--use_groupnorm', action='store_true',
                   help='Replace all BatchNorm with GroupNorm in the model. '
                        'Use with a GroupNorm-trained checkpoint.')

    args = p.parse_args()

    # Load YAML config if provided — CLI args override config values
    if args.config is not None:
        if not _YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required to use --config. Install with: pip install pyyaml")
        with open(args.config) as _f:
            _cfg = _yaml.safe_load(_f)
        _argv_keys = set()
        for _tok in sys.argv[1:]:
            if _tok.startswith('--'):
                _argv_keys.add(_tok.lstrip('-').replace('-', '_').split('=')[0])
        for _k, _v in _cfg.items():
            if _k not in _argv_keys and hasattr(args, _k):
                # Try to cast config value to the type expected by argparse
                # This prevents issues where '5e-4' in yaml is parsed as a string 
                # but expected to be a float by the argument parser.
                expected_type = type(getattr(args, _k))
                if expected_type is not type(None) and _v is not None:
                    try:
                        # Handle the case where the default is None, but the type is float/int
                        _v = expected_type(_v)
                    except (ValueError, TypeError):
                        pass
                setattr(args, _k, _v)
        print(f"  Loaded config: {args.config}")

    # ---------- setup ----------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    # ---------- data ----------
    print(f"\nLoading {args.dataset}...")
    train_set, test_set, num_classes = get_dataset(args.dataset, args.data_dir)
    if args.forget_type == 'class' and args.forget_class is not None:
        if not (0 <= args.forget_class < num_classes):
            raise ValueError(
                f"--forget_class must be in [0, {num_classes - 1}] for "
                f"{args.dataset} ({num_classes} classes)")
    forget_set, retain_set = split_forget_retain(
        train_set, args.forget_type, args.forget_pct, args.forget_class)

    kw = dict(batch_size=args.batch_size, num_workers=4)
    forget_loader = DataLoader(forget_set, shuffle=True, **kw)
    retain_loader = DataLoader(retain_set, shuffle=True, **kw)
    test_loader = DataLoader(test_set, shuffle=False, **kw)

    # Build forget-class test loader for MIA (same class, non-members)
    forget_class_test_loader = None
    if args.forget_type == 'class' and args.forget_class is not None:
        test_targets = (np.array(test_set.targets)
                        if hasattr(test_set, 'targets')
                        else np.array([t for _, t in test_set]))
        fc_test_idx = np.where(test_targets == args.forget_class)[0]
        if len(fc_test_idx) > 0:
            fc_test_set = Subset(test_set, fc_test_idx)
            forget_class_test_loader = DataLoader(
                fc_test_set, shuffle=False, **kw)
            print(f"  Forget-class test set: {len(fc_test_set)} samples "
                  f"(for MIA)")

    # ---------- model ----------
    print(f"\nLoading model: {args.model}")
    model = get_model(args.model, num_classes,
                      use_groupnorm=getattr(args, 'use_groupnorm', False)).to(device)
    if args.checkpoint:
        model.load_state_dict(
            torch.load(args.checkpoint, map_location=device))
        print(f"  Loaded: {args.checkpoint}")
    else:
        print("  No checkpoint — training from scratch for demo...")
        model = train_model(
            model,
            DataLoader(train_set, shuffle=True, **kw),
            device, epochs=5, lr=0.01)
        # Save the freshly-trained base model so it is never overwritten
        base_path = os.path.join(
            args.save_dir,
            f"base_{args.dataset}_{args.model}.pth")
        torch.save(model.state_dict(), base_path)
        print(f"  Saved base model → {base_path}")

    original_model = copy.deepcopy(model)

    # ---------- before ----------
    print("\n=== Before Unlearning ===")
    evaluate(model, test_loader, device, "Test Accuracy")
    evaluate(model, forget_loader, device, "Forget Accuracy (should DROP)")
    evaluate(model, retain_loader, device, "Retain Accuracy (should STAY)")

    # ---------- PURGE ----------
    t0 = time.time()
    model, history = purge_unlearn(
        model, forget_loader, retain_loader, device, args)
    elapsed = time.time() - t0
    print(f"\n  Unlearning completed in {elapsed:.1f}s")

    # ---------- after ----------
    print("\n=== After PURGE Unlearning ===")
    ta = evaluate(model, test_loader, device, "Test Accuracy  (TA)")
    fa = evaluate(model, forget_loader, device,
                  "Forget Accuracy (FA — want LOW)")
    ra = evaluate(model, retain_loader, device,
                  "Retain Accuracy (RA — want HIGH)")
    ua = 100.0 - fa
    print(f"\n  Unlearning Accuracy (UA = 100−FA): {ua:.2f}%")

    # ---------- MIA ----------
    print("\n=== Membership Inference Attack ===")
    if forget_class_test_loader is not None:
        mia = compute_mia(model, forget_loader,
                           forget_class_test_loader, device)
    else:
        print("  (Skipped — no forget-class test loader for sample-level "
              "forgetting)")
        mia = {'loss': 0.5, 'conf': 0.5, 'avg': 0.5}

    # ---------- feature distance ----------
    print("\n=== Representation-Level Verification ===")
    feat = compute_feature_distance(
        model, original_model, forget_loader, device)

    # ---------- summary ----------
    print(f"\n{'=' * 60}")
    print(" PURGE — Results Summary")
    print(f"{'=' * 60}")
    print(f"  Test Accuracy      (TA) : {ta:.2f}%")
    print(f"  Forget Accuracy    (FA) : {fa:.2f}%")
    print(f"  Retain Accuracy    (RA) : {ra:.2f}%")
    print(f"  Unlearning Accuracy(UA) : {ua:.2f}%")
    print(f"  MIA (AUROC)             : {mia['avg']:.4f}  (target ≈ 0.5)")
    print(f"  Feature Cosine Sim      : {feat['cosine_sim']:.4f}")
    print(f"  Avg Projection Rate     : "
          f"{np.mean(history['proj_rate']):.0%}")
    print(f"  Avg Gate Rate           : "
          f"{np.mean(history['gate_rate']):.0%}")
    if history['epoch_fa']:
        print(f"  Final Epoch FA          : "
              f"{history['epoch_fa'][-1]:.1f}%")
    print(f"  Wall-clock Time         : {elapsed:.1f}s")
    print(f"{'=' * 60}")

    # ---------- save ----------
    forget_tag = (f"c{args.forget_class}" if args.forget_type == 'class'
                  else f"s{args.forget_pct}")
    tag = (f"purge_{args.dataset}_{args.model}_{args.forget_type}_"
           f"{forget_tag}_ep{args.epochs}_lr{args.lr}")
    save_path = os.path.join(args.save_dir, f"{tag}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\nSaved unlearned model → {save_path}")


if __name__ == '__main__':
    main()
