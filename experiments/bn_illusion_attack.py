"""
BN-recalibration attack, canonical operator.

WHY THIS EXISTS ALONGSIDE experiments/bn_recalibration.py
---------------------------------------------------------
Our original attack put BN modules in train mode and ran forward passes, so the
running statistics were updated by PyTorch's EMA with the module's own momentum
(0.1) and the result depended on batch count and ordering, retaining some
influence from the pre-attack buffers.

The canonical operator in the BN-recalibration literature is instead
`torch.optim.swa_utils.update_bn`, which

  * resets running_mean to 0, running_var to 1, num_batches_tracked to 0,
  * sets momentum = None, making PyTorch accumulate a CUMULATIVE moving average,

so the resulting statistics are the exact retain-set statistics: independent of
the prior buffer values, and idempotent. That is a strictly cleaner and at least
as strong an attack, and it is the one an author of that attack would run.

This script applies that operator to our checkpoints inside our own data
pipeline. It deliberately does not import or vendor any third-party code: the
operator is a stdlib torch function, so matching the attack exactly requires
nothing more than calling it.

Both operators are run so the two can be compared directly on the same
checkpoint. A GroupNorm checkpoint has no BN buffers at all, so the attack is
undefined on it and reports delta_F = 0 by construction -- that is a property of
the architecture, not evidence of robustness, and is labelled as such.

--recal_on picks the data the attacker recalibrates on. The default, the retain
set, is the canonical attack. "train" (the full training set, forget class
included) and "test" (held-out data from every class) are adaptive variants: a
defence that recalibrates on the retain set during unlearning could resist only
the attack that uses the same data, and these runs check that it does not.
"""
import argparse
import math
import copy
import csv
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim.swa_utils import update_bn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run import get_model, get_dataset, split_forget_retain   # noqa: E402

MIN_RECAL_SAMPLES = 5000       # the threshold the canonical implementation enforces


# ----------------------------- helpers -------------------------------------
def bn_modules(model):
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]


def bn_state(model):
    return {n: (m.running_mean.detach().cpu().clone(),
                m.running_var.detach().cpu().clone())
            for n, m in bn_modules(model)}


@torch.no_grad()
def class_accuracy(model, loader, device, target_class):
    """Accuracy restricted to samples whose label is target_class."""
    model.eval()
    c = t = 0
    for x, y in loader:
        x, y = x.to(device), y.view(-1).to(device)
        keep = y == target_class
        if keep.any():
            c += model(x[keep]).argmax(1).eq(y[keep]).sum().item()
            t += int(keep.sum())
    return c / max(t, 1) * 100


@torch.no_grad()
def overall_accuracy(model, loader, device):
    model.eval()
    c = t = 0
    for x, y in loader:
        x, y = x.to(device), y.view(-1).to(device)
        c += model(x).argmax(1).eq(y).sum().item()
        t += len(y)
    return c / max(t, 1) * 100


def bn_drift(a, b, eps=1e-8):
    """Per-layer normalised drift; b is the reference (retain-optimal) state."""
    out = {}
    for n in a:
        mu_a, var_a = a[n]
        mu_b, var_b = b[n]
        sig_b = torch.sqrt(var_b.clamp_min(0) + eps)
        out[n] = {
            "dmu_l2_over_sigma": float((mu_a - mu_b).norm() / (sig_b.norm() + eps)),
            "dvar_l2_rel": float((var_a - var_b).norm() / (var_b.norm() + eps)),
        }
    return out


def weights_bit_identical(a, b):
    sa, sb = dict(a.named_parameters()), dict(b.named_parameters())
    return sa.keys() == sb.keys() and all(
        torch.equal(sa[k].detach().cpu(), sb[k].detach().cpu()) for k in sa)


# ----------------------------- the two operators ---------------------------
def attack_canonical(model, recal_loader, device):
    """R: reset buffers, then cumulative-average over the recalibration data (update_bn)."""
    update_bn(recal_loader, model, device=device)
    return model


def attack_ema(model, recal_loader, device):
    """Our original operator: BN train mode + forward passes, EMA momentum 0.1."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.train()
    with torch.no_grad():
        for x, _ in recal_loader:
            model(x.to(device))
    model.eval()
    return model


# ----------------------------- driver --------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--forget_class", type=int, default=0)
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="label:path ... (GroupNorm auto-detected from the state dict)")
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--out_csv", default="results/hardening/bn_illusion.csv")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--recal_on", choices=("retain", "train", "test"), default="retain",
                    help="data the attacker recalibrates on (see module docstring)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_set, test_set, num_classes = get_dataset(args.dataset, args.data_dir)

    # get_dataset returns the TRAINING transform (RandomCrop + HorizontalFlip).
    # Both uses here need it deterministic:
    #   * evaluation -- augmented forget/retain accuracy is a noisy estimate of
    #     the quantity we report, and shifts pre-F by several points;
    #   * recalibration -- the canonical operator is specified to be idempotent
    #     (applying it twice on the same data must reproduce the buffers), which
    #     a stochastic transform silently breaks.
    # Swapping in the test transform fixes both. Subsets read through to this
    # object, so it must be set before the split.
    if getattr(train_set, "transform", None) is not None and \
            getattr(test_set, "transform", None) is not None:
        train_set.transform = test_set.transform
        print("  using deterministic (test) transforms for evaluation and recalibration")

    forget_set, retain_set = split_forget_retain(
        train_set, "class", forget_class=args.forget_class)

    recal_set = {"retain": retain_set, "train": train_set, "test": test_set}[args.recal_on]
    n_recal = len(recal_set)
    if n_recal < MIN_RECAL_SAMPLES:
        print(f"  WARNING: only {n_recal} recalibration samples "
              f"(canonical implementation requires >= {MIN_RECAL_SAMPLES})")

    mk = lambda ds, sh=False: DataLoader(ds, batch_size=args.batch_size, shuffle=sh,
                                         num_workers=args.num_workers, pin_memory=True)
    recal_loader, forget_loader, test_loader = mk(recal_set), mk(forget_set), mk(test_set)

    print(f"  dataset={args.dataset} forget_class={args.forget_class} "
          f"| retain={len(retain_set)} forget={len(forget_set)} test={len(test_set)} "
          f"| attacker recalibrates on: {args.recal_on} ({n_recal})")

    rows = []
    for spec in args.checkpoints:
        label, path = spec.split(":", 1)
        sd = torch.load(path, map_location="cpu")
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        # "no running_mean" means only that the model keeps no running
        # statistics -- which is true of GroupNorm AND of LayerNorm (ViT). Name
        # the actual scheme so the CSV does not call a transformer "groupnorm".
        no_stats = not any("running_mean" in k for k in sd.keys())
        norm_name = ("layernorm" if args.model.startswith("vit")
                     else "groupnorm") if no_stats else "batchnorm"
        is_gn = no_stats
        print(f"\n=== {label}  ({norm_name})  {path}")

        base = get_model(args.model, num_classes, pretrained=False, use_groupnorm=is_gn)
        base.load_state_dict(sd)
        base.to(device).eval()

        pre_f = class_accuracy(base, forget_loader, device, args.forget_class)
        pre_ta = overall_accuracy(base, test_loader, device)
        print(f"    pre  : F={pre_f:.2f}%  TA={pre_ta:.2f}%")

        if no_stats:
            print(f"    {norm_name}: no BN buffers exist, so the attack is undefined "
                  "(delta_F = 0 by construction, NOT measured robustness)")
            rows.append(dict(label=label, norm=norm_name, operator="n/a",
                             pre_F=pre_f, post_F=pre_f, delta_F=0.0,
                             pre_TA=pre_ta, post_TA=pre_ta, delta_TA=0.0,
                             n_bn_layers=0, weights_intact=True,
                             mean_dmu_over_sigma="", n_recal=n_recal,
                             checkpoint=path, recal_on=args.recal_on))
            continue

        before = bn_state(base)
        for opname, fn in (("canonical_update_bn", attack_canonical),
                           ("ema_forward_passes", attack_ema)):
            m = copy.deepcopy(base)
            fn(m, recal_loader, device)
            post_f = class_accuracy(m, forget_loader, device, args.forget_class)
            post_ta = overall_accuracy(m, test_loader, device)
            drift = bn_drift(before, bn_state(m))
            mean_dmu = sum(d["dmu_l2_over_sigma"] for d in drift.values()) / len(drift)
            intact = weights_bit_identical(base, m)
            print(f"    {opname:<20} post: F={post_f:.2f}%  TA={post_ta:.2f}%  "
                  f"dF={post_f-pre_f:+.2f}pp  dTA={post_ta-pre_ta:+.2f}pp  "
                  f"| mean BN drift={mean_dmu:.4f}  weights untouched={intact}")
            if not intact:
                print("      ERROR: the attack modified learnable parameters — "
                      "it must only touch BN buffers")
            rows.append(dict(label=label, norm="batchnorm", operator=opname,
                             pre_F=pre_f, post_F=post_f, delta_F=post_f - pre_f,
                             pre_TA=pre_ta, post_TA=post_ta, delta_TA=post_ta - pre_ta,
                             n_bn_layers=len(before), weights_intact=intact,
                             mean_dmu_over_sigma=round(mean_dmu, 6),
                             n_recal=n_recal, checkpoint=path, recal_on=args.recal_on))

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {args.out_csv}")

    print("\n" + "=" * 66)
    print("VERDICT (delta_F > ~5pp means the attack recovers the forgotten class)")
    # delta_F alone is not a verdict. A model that never forgot has nothing for
    # the attack to recover, so it scores delta_F ~ 0 and looks robust. Judge on
    # pre_F (did it forget at all?) BEFORE reading delta_F.
    # A hard cut at exactly the chance level mislabels results sitting on it:
    # 10.34% against a 10% chance level, measured on 5,000 images, is within one
    # standard error of chance, yet a strict "<=" calls it a failure to forget.
    # Judge against the sampling error of the forget set instead.
    chance = 100.0 / num_classes
    p_ch = chance / 100.0
    n_f = len(forget_set)
    se = 100.0 * math.sqrt(p_ch * (1.0 - p_ch) / max(n_f, 1))
    at_chance = chance + 3.0 * se          # within 3 SE of chance
    for r in rows:
        forgot = r["pre_F"] <= 2.0 * chance
        if not forgot:
            v = (f"NOT ROBUST — it never forgot (pre-F {r['pre_F']:.1f}% > "
                 f"2x chance); dF is meaningless here")
        elif r["norm"] in ("groupnorm", "layernorm"):
            v = f"{r['norm']}: attack undefined (no BN buffers) — design change, not robustness"
        elif r["delta_F"] > 5:
            v = f"VULNERABLE — recovers to {r['post_F']:.1f}%"
        elif r["post_F"] > at_chance:
            v = (f"resists, but post-F {r['post_F']:.1f}% is above chance "
                 f"(> {at_chance:.1f}% = chance + 3 SE)")
        else:
            v = f"ROBUST — forgot ({r['pre_F']:.1f}%) and stays forgotten ({r['post_F']:.1f}%)"
        print(f"  {r['label']:<14} {r['operator']:<20} dF={r['delta_F']:+7.2f}pp  {v}")
    print("=" * 66)


if __name__ == "__main__":
    main()
