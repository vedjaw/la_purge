"""
Figure 1 (page 2): the paper in one picture.

-> images/fig_teaser.png

(a) Mechanism, with the measured audit. The accumulated update v_t points into
    the retain-harming half-space at every logged step; EXCISE projects it onto
    the boundary and restores its length. A soft penalty only tilts the step,
    which stays in the harming half-space (Theorem 2(iv)). The two numbers in
    the box come from results/perstep/perstep_cifar10_c0_s42.csv (Figure 3).
(b) Outcome. CIFAR-10 class 0, Table 3: forget accuracy against retain
    accuracy for every method in that table.
(c) BatchNorm. Forget accuracy before and after update_bn on the retain set,
    all ten CIFAR-10 classes (Table 20; seed 42, 93.97% base): mean bars,
    whiskers spanning the ten classes.

Every number is read from the CSVs below or copied from the table named next
to it, so the figure cannot drift from the paper.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch
import matplotlib.patheffects as pe

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "images/fig_teaser.png"
PERSTEP = ROOT / "results/perstep/perstep_cifar10_c0_s42.csv"

# Same hues as Figures 5 and 6: blue = EXCISE / retain-safe / BN-robust,
# red = retain-harming / frozen statistics, grey = other methods.
BLUE, RED, GREY = "#2980b9", "#c0392b", "#7f8c8d"
INK, MUTED = "#222222", "#555555"
BLUE_T, RED_T = "#eaf2fa", "#fbeceb"
HALO = [pe.withStroke(linewidth=2.2, foreground="white")]

plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8.2, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
})


def audit_numbers():
    rows = list(csv.DictReader(open(PERSTEP)))
    cos = lambda r, k: float(r[k]) / (float(r["g_f_norm"]) * float(r["g_r_norm"]))
    pre = [cos(r, "dot_raw") for r in rows]
    post = [cos(r, "dot_post") for r in rows]
    return (len(rows), sum(c < 0 for c in pre), sum(pre) / len(pre),
            max(abs(c) for c in post))


def arrow(ax, p, q, color, lw=1.6, ls="-", alpha=1.0, z=4, ms=9):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, ls=ls, alpha=alpha,
                                 zorder=z, shrinkA=0, shrinkB=0))


def panel_mechanism(ax):
    n, n_conf, mean_pre, max_post = audit_numbers()
    ax.set_xlim(-1.25, 1.45); ax.set_ylim(-1.0, 1.32)
    ax.set_aspect("equal"); ax.axis("off")

    ax.add_patch(plt.Rectangle((-1.25, 0), 2.7, 1.32, color=RED_T, zorder=0, lw=0))
    ax.add_patch(plt.Rectangle((-1.25, -1.0), 2.7, 1.0, color=BLUE_T, zorder=0, lw=0))
    ax.plot([-1.25, 1.45], [0, 0], color=INK, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.text(-1.18, 1.22, r"retain-harming: $\langle u, g_r\rangle < 0$",
            color=RED, fontsize=7, va="top")
    ax.text(-1.18, -0.92, r"retain-safe: $\langle u, g_r\rangle \geq 0$",
            color=BLUE, fontsize=7, va="bottom")

    o = (-0.80, 0.0)
    # retain gradient: the constraint, pointing into the safe side
    arrow(ax, o, (o[0], -0.58), INK, lw=1.3)
    ax.text(o[0] - 0.06, -0.50, r"$g_r$", fontsize=8, color=INK, va="center", ha="right")

    # accumulated update v_t, in the harming half-space
    v = (0.70, 0.80)
    arrow(ax, o, (o[0] + v[0], o[1] + v[1]), RED, lw=1.7)
    ax.text(o[0] + v[0] - 0.02, o[1] + v[1] + 0.04,
            r"$v_t = \mu v_{t-1} + d_t$", color=RED, fontsize=7.6,
            ha="center", va="bottom")

    # soft penalty: the step is tilted toward safety but stays harmful
    s = (0.98, 0.38)
    arrow(ax, o, (o[0] + s[0], o[1] + s[1]), RED, lw=1.1, ls=(0, (3, 2)), alpha=0.75)
    ax.text(o[0] + s[0] + 0.06, o[1] + s[1] - 0.02, "soft penalty:\nstill harmful",
            color=RED, fontsize=6.4, va="center", ha="left", alpha=0.95)

    # projection onto the boundary, then rescale to ||v_t||
    vn = (v[0] ** 2 + v[1] ** 2) ** 0.5
    ax.plot([o[0] + v[0], o[0] + v[0]], [v[1], 0.0], color=MUTED, lw=0.7,
            ls=(0, (1.5, 1.5)), zorder=2)
    ax.add_patch(Arc(o, 2 * vn, 2 * vn, theta1=0,
                     theta2=49, color=MUTED, lw=0.7, ls=(0, (1.5, 1.5)), zorder=2))
    arrow(ax, o, (o[0] + vn, 0.0), BLUE, lw=2.0, z=5)
    ax.text(o[0] + vn - 0.02, -0.07, r"$u_t$ (EXCISE)", color=BLUE, fontsize=8,
            ha="right", va="top", weight="bold")

    box = (f"measured on a CIFAR-10 run ({n} steps):\n"
           f"$v_t$ harmful at {n_conf}/{n} (mean $\\cos = {mean_pre:.2f}$)\n"
           f"$u_t$: $|\\cos(u_t, g_r)| \\leq {max_post:.0e}$")
    box = box.replace("e-07", r"\times 10^{-7}")
    ax.text(-0.62, -0.25, box, fontsize=5.8, color=INK, va="top", ha="left",
            linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbbbbb", lw=0.5))
    ax.set_title("(a) Project the update the optimiser applies", loc="left", pad=3)


def panel_outcome(ax):
    # Table 3, CIFAR-10 class 0: (label, FA, RA, is_ours, label offset in points)
    rows = [("SalUn", 0.00, 97.10, False, (5, -7)),
            ("Bad Teacher", 29.70, 90.20, False, (6, 0)),
            ("SCRUB", 88.80, 97.70, False, (-6, -7)),
            ("NegGrad+", 84.80, 99.30, False, (-6, -8)),
            ("projection only", 18.66, 99.90, False, (6, -7)),
            ("EXCISE", 1.87, 100.00, True, (7, 2))]
    ax.axvspan(-3, 10, ymin=0, ymax=1, color=BLUE_T, zorder=0, lw=0)
    ax.axvline(10, color=INK, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.text(11.5, 89.0, "chance (10%)", fontsize=6.4, color=MUTED, va="bottom")
    for name, fa, ra, ours, (dx, dy) in rows:
        ax.scatter(fa, ra, s=46 if ours else 26, marker="D" if ours else "o",
                   color=BLUE if ours else GREY, edgecolor="white", lw=0.8, zorder=4)
        ha = "left" if dx > 0 else "right"
        ax.annotate(name, (fa, ra), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center", fontsize=6.6, path_effects=HALO, zorder=5,
                    color=BLUE if ours else MUTED, weight="bold" if ours else None)
    ax.set_xlim(-3, 100); ax.set_ylim(88.5, 101.0)
    ax.set_xlabel(r"forget accuracy (%)  $\leftarrow$ forgets more")
    ax.set_ylabel("retain accuracy (%)")
    ax.grid(color="#e9e9e9", lw=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("(b) Forgets to chance, retain set intact", loc="left", pad=3)


def panel_bn(ax):
    # Table 20 (app:bnrobust_perclass), CIFAR-10 classes 0-9, seed 42:
    # frozen pre/post, BN-robust calibrated pre/post.
    frozen_pre = [8.76, 2.40, 6.52, 4.78, 4.84, 7.28, 9.18, 3.38, 4.54, 2.52]
    frozen_post = [99.22, 97.60, 98.30, 94.82, 97.34, 94.70, 99.14, 96.20, 99.26, 98.16]
    robust_pre = [9.22, 6.66, 9.90, 10.00, 10.18, 9.40, 7.30, 9.12, 9.74, 8.92]
    robust_post = [9.24, 6.66, 9.92, 9.98, 10.16, 9.42, 7.30, 9.12, 9.68, 8.92]
    mean = lambda x: sum(x) / len(x)
    groups = [("frozen\nstatistics", RED, frozen_pre, frozen_post),
              ("BN-robust\nEXCISE", BLUE, robust_pre, robust_post)]
    w = 0.34
    for i, (name, c, pre, post) in enumerate(groups):
        for j, (vals, kind) in enumerate(((pre, "before"), (post, "after"))):
            x = i + (j - 0.5) * (w + 0.04)
            m = mean(vals)
            ax.bar(x, m, w, color=c if kind == "after" else "white",
                   edgecolor=c, lw=1.0, hatch=None if kind == "after" else "//////",
                   zorder=3)
            ax.plot([x, x], [min(vals), max(vals)], color=INK, lw=0.7, zorder=4)
            for yv in (min(vals), max(vals)):
                ax.plot([x - 0.05, x + 0.05], [yv, yv], color=INK, lw=0.7, zorder=4)
            ax.text(x, max(vals) + 2.0, f"{m:.1f}", ha="center", va="bottom",
                    fontsize=6.6, color=INK)
        d = mean(post) - mean(pre)
        ax.text(i, 112, f"{d:+.1f} pp" if abs(d) >= 0.05 else "0.0 pp",
                ha="center", va="bottom", fontsize=7.2, weight="bold", color=c)
    ax.axhline(10, color=INK, lw=0.8, ls=(0, (4, 3)), zorder=2)
    ax.text(1.93, 11.5, "chance", fontsize=6.4, color=MUTED, ha="right", va="bottom")
    ax.set_xticks([0, 1]); ax.set_xticklabels([g[0] for g in groups])
    ax.set_xlim(-0.55, 1.95); ax.set_ylim(0, 124)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("forget accuracy (%)")
    ax.grid(axis="y", color="#e9e9e9", lw=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # legend: open = before the attack, filled = after
    h1 = plt.Rectangle((0, 0), 1, 1, fc="white", ec=MUTED, hatch="//////", lw=0.8)
    h2 = plt.Rectangle((0, 0), 1, 1, fc=MUTED, ec=MUTED, lw=0.8)
    ax.legend([h1, h2], ["before recalibration", "after recalibration"],
              frameon=False, fontsize=6.3, loc="center right",
              bbox_to_anchor=(1.03, 0.62), handlelength=1.2, borderaxespad=0)
    ax.set_title("(c) Survives BN recalibration", loc="left", pad=3)


def main():
    fig = plt.figure(figsize=(7.0, 2.45), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 1.0, 0.86], wspace=0.34,
                          left=0.005, right=0.995, top=0.90, bottom=0.17)
    panel_mechanism(fig.add_subplot(gs[0]))
    panel_outcome(fig.add_subplot(gs[1]))
    panel_bn(fig.add_subplot(gs[2]))
    fig.savefig(OUT, dpi=300)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
