"""
Membership-inference suite for class-level unlearning.

Replaces the single averaged AUROC used so far. Two things are added, and both
address a specific weakness of that number.

1. FOUR attack signals instead of two. All are threshold-free scores on which a
   member is predicted when the score is high:

     loss    -CE(x,y)                     confident-and-correct -> member
     conf    max_i p_i                    peaked posterior      -> member
     entropy -H(p)                        low entropy           -> member
     mentr   -Mentr(p,y), where
             Mentr = -(1-p_y) log p_y - sum_{i != y} p_i log(1 - p_i)
             (Song & Mittal). Unlike plain entropy it uses the true label, so
             it separates "confidently right" from "confidently wrong" --
             exactly the distinction that matters when a model has been pushed
             PAST a retrained one.

2. TPR AT LOW FPR, not just AUROC. An attack that is right about a handful of
   samples with high confidence is a real privacy failure, and average AUROC
   hides it: a score of 0.5 is consistent with an attack that identifies a few
   members perfectly. We report TPR at 1% FPR alongside AUROC.

CALIBRATION AGAINST A RETRAINED MODEL
------------------------------------
"Ideal = 0.5" is an assumption, not a measurement, and our trajectory analysis
showed MIA sweeps THROUGH 0.5 rather than settling there. If --retrained is
given, the same suite is run on the retrained reference, which is
indistinguishable by construction. Its scores are what "indistinguishable"
actually looks like on this dataset, and unlearned models should be compared to
those rather than to 0.5.

Members are forget-class TRAINING samples; non-members are forget-class TEST
samples -- same class, differing only in whether the model trained on them.

Usage
-----
    python experiments/mia_suite.py --dataset cifar10 --forget_class 0 \
        --retrained checkpoints/retrain_cifar10_c0.pth \
        --checkpoints purge:results/todo/purge_cifar10_bn/<ckpt>.pth \
                      salun:../01_salun/checkpoints/salun_cifar10_resnet18_class.pth
"""

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.pop("MKL_THREADING_LAYER", None)
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import get_model, get_dataset, split_forget_retain, evaluate  # noqa: E402

# Signals are computed in float64. In float32, 1 - EPS is NOT representable for
# small EPS -- the mantissa is 24 bits, so any EPS below 2^-24 (~6e-8) rounds the
# upper clamp bound to exactly 1.0, and log(1 - p) in the modified-entropy term
# becomes -inf. Casting to float64 keeps the clamp effective and the entropy
# sums accurate.
EPS = 1e-12


@torch.no_grad()
def signals(model, loader, device):
    """Per-sample attack signals. Higher => more member-like, for every one."""
    out = {k: [] for k in ("loss", "conf", "entropy", "mentr")}
    for x, y in loader:
        x, y = x.to(device), y.view(-1).to(device)
        p = F.softmax(model(x), dim=1).double().clamp(EPS, 1 - EPS)
        py = p.gather(1, y[:, None]).squeeze(1)

        out["loss"].append((-(-py.log())).cpu().numpy())          # = log p_y
        out["conf"].append(p.max(1).values.cpu().numpy())
        out["entropy"].append((p * p.log()).sum(1).cpu().numpy())  # = -H(p)

        # Modified entropy (Song & Mittal), low for members:
        #   Mentr = -(1-p_y) log p_y - sum_{i!=y} p_i log(1-p_i)
        # The second sum excludes i=y, so subtract that term from the full sum.
        # Sign here is easy to get wrong: full-sum correction is +p_y log(1-p_y).
        term_y = -(1 - py) * py.log()
        full = -(p * (1 - p).log()).sum(1)
        term_o = full + py * (1 - py).log()
        out["mentr"].append((-(term_y + term_o)).cpu().numpy())
    sig = {k: np.concatenate(v) for k, v in out.items()}
    # Fail loudly and specifically: a non-finite signal otherwise surfaces as an
    # opaque sklearn "input contains infinity" several frames away.
    for k, v in sig.items():
        n_bad = int((~np.isfinite(v)).sum())
        if n_bad:
            raise FloatingPointError(
                f"signal '{k}' has {n_bad}/{len(v)} non-finite values "
                f"(min={np.nanmin(v)}, max={np.nanmax(v)}). This usually means "
                f"the probability clamp underflowed in the working dtype.")
    return sig


def tpr_at_fpr(member, nonmember, target_fpr):
    """TPR at the largest achievable FPR <= target_fpr."""
    from sklearn.metrics import roc_curve
    y = np.concatenate([np.ones(len(member)), np.zeros(len(nonmember))])
    s = np.concatenate([member, nonmember])
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= target_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def auroc(member, nonmember):
    from sklearn.metrics import roc_auc_score
    y = np.concatenate([np.ones(len(member)), np.zeros(len(nonmember))])
    s = np.concatenate([member, nonmember])
    return float(roc_auc_score(y, s))


def load(path, K, gn, device):
    m = get_model("resnet18", num_classes=K, pretrained=False,
                  use_groupnorm=gn).to(device)
    st = torch.load(path, map_location=device)
    if isinstance(st, dict) and "state_dict" in st:
        st = st["state_dict"]
    m.load_state_dict(st)
    m.eval()
    return m


def run_one(name, model, f_loader, t_loader, device, fpr_targets):
    sm = signals(model, f_loader, device)
    sn = signals(model, t_loader, device)
    rows = []
    for atk in ("loss", "conf", "entropy", "mentr"):
        r = {"model": name, "attack": atk,
             "auroc": auroc(sm[atk], sn[atk]),
             "n_member": len(sm[atk]), "n_nonmember": len(sn[atk])}
        for tf in fpr_targets:
            r[f"tpr@{tf:g}fpr"] = tpr_at_fpr(sm[atk], sn[atk], tf / 100.0)
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--forget_class", type=int, required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="name:path entries to evaluate.")
    ap.add_argument("--retrained", default=None,
                    help="Retrained reference; its scores calibrate what "
                         "'indistinguishable' looks like on this dataset.")
    ap.add_argument("--data_dir", default=str(ROOT / "data"))
    ap.add_argument("--out_csv", default=str(ROOT / "results/mia_suite.csv"))
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--fpr", type=float, nargs="+", default=[1.0, 0.1])
    ap.add_argument("--use_groupnorm", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_set, test_set, K = get_dataset(args.dataset, args.data_dir)
    forget_set, _ = split_forget_retain(train_set, "class",
                                        forget_class=args.forget_class)
    kw = dict(batch_size=args.batch_size, num_workers=4)
    f_loader = DataLoader(forget_set, shuffle=False, **kw)

    tt = (np.array(test_set.targets) if hasattr(test_set, "targets")
          else np.array([y for _, y in test_set]))
    idx = np.where(tt == args.forget_class)[0]
    if len(idx) == 0:
        print("[ERROR] no forget-class test samples"); return 1
    t_loader = DataLoader(Subset(test_set, idx.tolist()), shuffle=False, **kw)
    print(f"members (forget train): {len(forget_set)} | "
          f"non-members (forget test): {len(idx)}")
    for tf in args.fpr:
        if len(idx) * tf / 100.0 < 5:
            print(f"  [warn] {tf}% FPR is {len(idx)*tf/100:.1f} non-member "
                  f"samples — too few to resolve; treat that column as noisy.")

    entries = list(args.checkpoints)
    if args.retrained:
        entries = [f"retrained(reference):{args.retrained}"] + entries

    rows = []
    for e in entries:
        if ":" not in e:
            print(f"[skip] '{e}' is not name:path"); continue
        name, path = e.split(":", 1)
        if not os.path.exists(path):
            print(f"[skip] {name}: missing {path}"); continue
        m = load(path, K, args.use_groupnorm, device)
        fa = evaluate(m, f_loader, device, desc=f"  {name} FA")
        rs = run_one(name, m, f_loader, t_loader, device, args.fpr)
        for r in rs:
            r["FA"] = fa
        rows += rs
        del m

    if not rows:
        print("nothing evaluated."); return 1
    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    fields = (["model", "attack", "FA", "auroc"]
              + [f"tpr@{t:g}fpr" for t in args.fpr]
              + ["n_member", "n_nonmember"])
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})

    print(f"\n{'=' * 92}")
    print(f"MIA SUITE — {args.dataset}, forget class {args.forget_class}")
    print(f"{'=' * 92}")
    hdr = f"{'model':<22}{'attack':<9}{'FA':>7}{'AUROC':>8}"
    for t in args.fpr:
        hdr += f"{'TPR@' + format(t, 'g') + '%':>11}"
    print(hdr); print("-" * 92)
    for r in rows:
        line = (f"{r['model']:<22}{r['attack']:<9}{r['FA']:>7.2f}"
                f"{r['auroc']:>8.4f}")
        for t in args.fpr:
            line += f"{r[f'tpr@{t:g}fpr']:>11.4f}"
        print(line)
    if args.retrained:
        print("\nCompare each model against the retrained reference rows, not")
        print("against 0.5: the reference is indistinguishable by construction,")
        print("so its scores are the empirical target for this dataset.")
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
