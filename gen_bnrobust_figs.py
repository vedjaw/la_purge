"""
Figures for the BN-robust subsection (Sec. 5.6) and its appendix table.

  fig_bnrobust_mechanism.png   why frozen statistics fail, and what BN-robust changes
  fig_bnrobust_attack.png      forget accuracy before and after the recalibration attack
  fig_bnrobust_stopping.png    in-loop vs shipped FA per epoch: why stopping must be calibrated
  fig_bnrobust_perclass.png    all ten CIFAR-10 classes

Every number is read from results/hardening/bn_robust/: the CSVs written by
experiments/bn_illusion_attack.py, and stopping_traces.csv, which holds the
per-epoch FA readings of the calibrated runs (the run logs themselves are not
released, as they carry machine paths). Nothing is typed in by hand.

Usage:  python gen_bnrobust_figs.py
"""
import csv
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import blended_transform_factory

ROOT = Path(__file__).resolve().parent
IMG = ROOT / "images"
RES = ROOT / "results/hardening/bn_robust"
IMG.mkdir(exist_ok=True)

BLUE, RED, GREY, AMBER = "#2980b9", "#c0392b", "#7f8c8d", "#e67e22"
CHANCE = 10.0
# chance + 3 binomial standard errors for a 5,000-image forget set (11.27%)
CHANCE_3SE = CHANCE + 3 * 100 * (0.1 * 0.9 / 5000) ** 0.5
CANON = "canonical_update_bn"


def attack_rows(name, prefix, cls=None):
    """(pre_F, post_F, post_TA) for every canonical-operator row whose label
    starts with `prefix` (and, for the merged per-class CSVs, matches `cls`)."""
    out = []
    for r in csv.DictReader(open(RES / name)):
        if r["operator"] != CANON or not r["label"].startswith(prefix):
            continue
        if cls is not None and int(r.get("forget_class", -1)) != cls:
            continue
        out.append((float(r["pre_F"]), float(r["post_F"]), float(r["post_TA"])))
    assert out, (name, prefix, cls)
    return out


def mean(rows, i):
    return st.mean(r[i] for r in rows)


def traces():
    t = {}
    for r in csv.DictReader(open(RES / "stopping_traces.csv")):
        key = (r["phase"], r["dataset"], int(r["forget_class"]), int(r["seed"]))
        t.setdefault(key, []).append(r)
    return t


def style(ax):
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------
def mechanism():
    """Two lanes: frozen statistics (the deployed PURGE) and BN-robust."""
    frozen = attack_rows("bn_robust_phase19_3seeds.csv", "off_")
    robust = attack_rows("bn_robust_phase30_capped_check.csv", "calcap_")

    fig, ax = plt.subplots(figsize=(8.6, 4.1))
    ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=8.5, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                    fc=fc, ec=ec, lw=1.3, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, zorder=3, weight=weight)

    def link(x0, y0, x1, y1, color="0.35", ls="-"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), color=color, lw=1.4,
                                     ls=ls, arrowstyle="-|>", mutation_scale=14,
                                     shrinkA=1, shrinkB=1, zorder=1))

    for y, name, ec, fc, step, ships, after, verdict in [
        (29, "PURGE, frozen statistics", RED, "#fdeeea",
         "every step: update $\\theta$;\n$\\phi$ stays at $\\phi_0$",
         f"ships $\\theta_T$ with $\\phi_0$\nFA {mean(frozen, 0):.1f}%",
         f"FA {mean(frozen, 1):.1f}%", "the forgetting\nwas in the mismatch"),
        (6, "BN-robust PURGE", BLUE, "#e8f0f8",
         "every step: $\\phi\\leftarrow R(\\theta_t, B_r)$,\nthen update $\\theta$",
         f"ships $\\theta_T$ with $R(\\theta_T,\\mathcal{{D}}_r)$\nFA {mean(robust, 0):.1f}%",
         f"FA {mean(robust, 1):.1f}%", "nothing left\nto recover"),
    ]:
        ax.text(1, y + 13.2, name, fontsize=10, weight="bold", color=ec)
        box(1, y, 15, 9, "start\n$\\theta_0$, $\\phi_0$", "#f2f2f2", "0.45")
        box(22, y, 21, 9, step, fc, ec)
        box(49, y, 20, 9, ships, fc, ec)
        box(76, y, 23, 9, f"attack: $\\phi\\leftarrow R(\\theta_T,\\mathcal{{D}}_r)$\n{after}",
            fc, ec, weight="bold")
        ax.text(87.5, y - 3.2, verdict, ha="center", va="top", fontsize=8, color="0.25")
        link(17.2, y + 4.5, 21.8, y + 4.5)
        link(44.2, y + 4.5, 48.8, y + 4.5)
        link(70.2, y + 4.5, 75.8, y + 4.5, color="0.2", ls="--")

    ax.text(59, 3.2, "stopping reads FA under $R(\\theta,\\mathcal{D}_r)$: the FA that ships",
            ha="center", va="top", fontsize=8, color=BLUE)
    ax.text(50, 49.5, "$R$ = reset the BatchNorm statistics and recompute them on retain data "
            "(no weight is touched)", ha="center", va="top", fontsize=8.5, color="0.3")
    fig.tight_layout()
    fig.savefig(IMG / "fig_bnrobust_mechanism.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_bnrobust_mechanism.png")


# --------------------------------------------------------------------------
def attack():
    """(a) before -> after dumbbells; (b) the attacker's choice of data."""
    rows = [  # (group, label, colour, pre/post rows)
        ("CIFAR-10", "base model (no unlearning)", GREY,
         attack_rows("bn_robust_phase25_base_control_retain.csv", "base")),
        ("CIFAR-10", "PURGE, frozen statistics", RED,
         attack_rows("bn_robust_phase19_3seeds.csv", "off_")),
        ("CIFAR-10", "BN-robust, naive stopping", AMBER,
         attack_rows("bn_robust_phase21_512imgs_persistent.csv", "exact512p_")),
        ("CIFAR-10", "BN-robust, calibrated stopping", BLUE,
         attack_rows("bn_robust_phase30_capped_check.csv", "calcap_")),
        ("MNIST", "PURGE, frozen statistics", RED,
         attack_rows("bn_robust_phase29_mnist.csv", "off_")),
        ("MNIST", "BN-robust, calibrated stopping", BLUE,
         attack_rows("bn_robust_phase29_mnist.csv", "robust_")),
    ]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.8, 3.9),
                                 gridspec_kw=dict(width_ratios=[1.45, 1.0], wspace=0.32))

    ys, labels = [], []
    y = 0
    for i, (group, label, c, rs) in enumerate(rows):
        if not i or rows[i - 1][0] != group:
            if i:
                y -= 0.7
                a1.axhline(y + 0.85, color="0.8", lw=0.8)
            a1.text(-0.03, y + 0.15, f"{group}, class 0", ha="right", va="center",
                    fontsize=9, weight="bold", color="0.25",
                    transform=blended_transform_factory(a1.transAxes, a1.transData))
            y -= 0.65
        pre, post = mean(rs, 0), mean(rs, 1)
        a1.plot([pre, post], [y, y], color=c, lw=2, zorder=2, solid_capstyle="round")
        a1.scatter(pre, y, s=58, facecolor="white", edgecolor=c, lw=1.8, zorder=3)
        a1.scatter(post, y, s=58, color=c, edgecolor="white", lw=1.2, zorder=4)
        # 99.97 would round to "100.0", which overstates the recovery
        a1.annotate(f"{post:.1f}" if post < 99.95 else f"{post:.2f}", (post, y),
                    xytext=(8, 0), textcoords="offset points", va="center", fontsize=8.5,
                    bbox=dict(fc="white", ec="none", pad=0.4), zorder=5)
        ys.append(y); labels.append(label)
        y -= 1
    a1.axvline(CHANCE, color="k", lw=1.1, zorder=1)
    a1.text(CHANCE + 1.2, ys[0] + 0.55, "chance", fontsize=8.5, va="center")
    a1.set_yticks(ys); a1.set_yticklabels(labels, fontsize=9)
    a1.set_xlim(-2, 117); a1.set_ylim(ys[-1] - 0.6, ys[0] + 0.95)
    a1.set_xticks([0, 20, 40, 60, 80, 100])
    a1.set_xlabel("forget accuracy (%)")
    a1.grid(axis="x", alpha=0.25); a1.set_axisbelow(True)
    a1.set_title("(a) before $\\circ$ and after $\\bullet$ the attack", fontsize=10.5)

    sets = [("retain\nset", None), ("train set,\nforget class in", "train"),
            ("test\nset", "test")]
    frozen, robust = [], []
    for _, s in sets:
        if s is None:
            f = attack_rows("bn_robust_phase19_3seeds.csv", "off_")
            r = attack_rows("bn_robust_phase21_512imgs_persistent.csv", "exact512p_")
        else:
            f = attack_rows(f"bn_robust_phase24_recal_{s}.csv", "off_")
            r = attack_rows(f"bn_robust_phase24_recal_{s}.csv", "exact512p_")
        frozen.append(mean(f, 1)); robust.append(mean(r, 1))
    xs = range(len(sets)); w = 0.36
    for dx, vals, c, lab in [(-w / 2, frozen, RED, "PURGE, frozen statistics"),
                             (w / 2, robust, BLUE, "BN-robust (naive stop)")]:
        bars = a2.bar([x + dx for x in xs], vals, w, color=c, edgecolor="white", label=lab)
        for b, v in zip(bars, vals):
            a2.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8.5)
    a2.axhline(CHANCE, color="k", lw=1.1)
    a2.set_xticks(list(xs)); a2.set_xticklabels([s for s, _ in sets], fontsize=9)
    a2.set_ylim(0, 112); a2.set_ylabel("FA after the attack (%)")
    a2.set_title("(b) attacker's choice of data", fontsize=10.5)
    style(a2)
    a2.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=1)
    fig.savefig(IMG / "fig_bnrobust_attack.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_bnrobust_attack.png")


# --------------------------------------------------------------------------
def stopping():
    """In-loop vs shipped FA, epoch by epoch, for three calibrated runs."""
    t = traces()
    panels = [(("phase26", "cifar10", 0, 123), "(a) class 0, seed 123"),
              (("phase27", "cifar10", 9, 42), "(b) class 9"),
              (("phase28", "cifar10", 4, 42), "(c) class 4")]
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.3), sharey=True)
    arrow = dict(arrowstyle="-|>", color="0.35", lw=1)
    for ax, (key, title) in zip(axes, panels):
        rs = t[key]
        ep = [int(r["epoch"]) for r in rs]
        fin = [float(r["fa_inloop"]) for r in rs]
        fsh = [float(r["fa_shipped"]) for r in rs]
        ax.plot(ep, fin, color=AMBER, lw=2, marker="o", ms=4.5, mfc="white", mew=1.4,
                label="FA under the in-loop statistics (naive rule)")
        ax.plot(ep, fsh, color=BLUE, lw=2, marker="o", ms=4.5,
                label="FA under the shipped statistics (calibrated rule)")
        ax.axhline(CHANCE, color="k", lw=1.1, label="target (10%)")
        # By the late epochs both curves are low, so the upper right is empty:
        # the notes go there and point down at the epochs they describe.
        right = ep[-1] + 0.3
        naive = next((e for e, v in zip(ep, fin) if v <= CHANCE), None)
        if naive:
            ax.annotate(f"naive stop:\nships {fsh[naive - 1]:.1f}%",
                        (naive, fsh[naive - 1]), xytext=(right, 62), ha="right",
                        fontsize=8, color="0.2", arrowprops=arrow)
        cal = next((e for e, v in zip(ep, fsh) if v <= CHANCE), None)
        if cal:
            ax.annotate(f"calibrated\nstop: ships {fsh[cal - 1]:.1f}%",
                        (cal, fsh[cal - 1]), xytext=(right, 28), ha="right",
                        fontsize=8, color="0.2", arrowprops=arrow)
        else:
            best = int(rs[0]["shipped_epoch"])
            ax.scatter([best], [fsh[best - 1]], s=120, facecolor="none", edgecolor=BLUE,
                       lw=1.6, zorder=5)
            ax.annotate(f"cap: ships best\nepoch, {fsh[best - 1]:.1f}%",
                        (best, fsh[best - 1]), xytext=(right, 28), ha="right",
                        fontsize=8, color="0.2", arrowprops=arrow)
        ax.set_yscale("log")
        ax.set_yticks([2, 5, 10, 20, 50, 100])
        ax.set_yticklabels(["2", "5", "10", "20", "50", "100"])
        ax.set_ylim(2, 110)
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(title, fontsize=10)
        style(ax)
    axes[0].set_ylabel("forget accuracy (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8.5, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout()
    fig.savefig(IMG / "fig_bnrobust_stopping.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_bnrobust_stopping.png")


# --------------------------------------------------------------------------
def perclass():
    """(a) frozen vs BN-robust under the attack; (b) where BN-robust lands."""
    classes = list(range(10))
    frozen, naive, cal = [], [], []
    for c in classes:
        if c == 0:
            frozen.append(attack_rows("bn_robust_phase19_3seeds.csv", "off_s42")[0])
            naive.append(attack_rows("bn_robust_phase21_512imgs_persistent.csv",
                                     "exact512p_s42")[0])
            cal.append(attack_rows("bn_robust_phase26_calibrated_stop.csv", "cal_s42")[0])
        else:
            frozen.append(attack_rows("bn_robust_phase23_classes1-9.csv", "off_", c)[0])
            naive.append(attack_rows("bn_robust_phase23_classes1-9.csv", "exact512_", c)[0])
            cal.append(attack_rows("bn_robust_phase28_class4_restore_best.csv", "calbest_")[0]
                       if c == 4 else
                       attack_rows("bn_robust_phase27_calibrated_classes1-9.csv", "cal_", c)[0])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.8, 3.5), gridspec_kw=dict(wspace=0.28))
    w = 0.38
    a1.bar([c - w / 2 for c in classes], [r[1] for r in frozen], w, color=RED,
           edgecolor="white", label="PURGE, frozen statistics")
    a1.bar([c + w / 2 for c in classes], [r[1] for r in cal], w, color=BLUE,
           edgecolor="white", label="BN-robust, calibrated stopping")
    a1.axhline(CHANCE, color="k", lw=1.1)
    a1.set_xticks(classes); a1.set_xlabel("forgotten class (CIFAR-10)")
    a1.set_ylim(0, 105); a1.set_ylabel("FA after the attack (%)")
    a1.set_title("(a) under the attack, every class", fontsize=10.5)
    style(a1)
    a1.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2),
              ncol=1)

    a2.axhspan(0, CHANCE_3SE, color=BLUE, alpha=0.07, lw=0)
    a2.axhline(CHANCE, color="k", lw=1.1)
    for c, n, k in zip(classes, naive, cal):
        a2.plot([c, c], [n[0], k[0]], color="0.6", lw=1.5, zorder=1)
        a2.scatter(c, n[0], s=46, facecolor="white", edgecolor=AMBER, lw=1.6, zorder=3)
        a2.scatter(c, k[0], s=46, color=BLUE, edgecolor="white", lw=1, zorder=4)
    a2.set_xticks(classes); a2.set_xlabel("forgotten class (CIFAR-10)")
    a2.set_ylim(4, 18); a2.set_ylabel("shipped FA (%)")
    a2.set_title("(b) where BN-robust lands", fontsize=10.5)
    style(a2)
    a2.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec=AMBER, mew=1.6,
                              ms=7, label="naive stopping"),
                       Line2D([], [], marker="o", ls="", color=BLUE, ms=7,
                              label="calibrated stopping"),
                       Patch(color=BLUE, alpha=0.15, label="within chance + 3 SE")],
              frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2),
              ncol=2)
    fig.savefig(IMG / "fig_bnrobust_perclass.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_bnrobust_perclass.png")


if __name__ == "__main__":
    mechanism()
    attack()
    stopping()
    perclass()
