"""
Figure: how hard does the retain constraint have to be?

Reads results/strength_sweep.csv -> images/fig_strength.png

The sweep interpolates the projection between "no correction" (s=0) and the
full hard constraint (s=1); s<1 is a soft conflict penalty. Forget accuracy is
0.0% at every setting, so forgetting is complete throughout and the ONLY thing
varying is collateral damage -- a controlled comparison.

Panel (a) shows retain and test accuracy rising continuously with hardness,
under a red->green gradient: the softer the correction, the more damage.
Panel (b) shows why the endpoint is nonetheless special: the residual conflict
is (1-s)*<v,g_r>, so retain-harming steps remain possible at every s<1 and are
excluded only at s=1. The empirical gap between s=0.75 and s=1 is small; the
qualitative difference is not.
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "results/strength_sweep.csv"
OUT = ROOT / "images/fig_strength.png"

RED, GREEN, BLUE, GREY = "#c0392b", "#1e8449", "#2980b9", "#7f8c8d"
CMAP = LinearSegmentedColormap.from_list("dmg", ["#c0392b", "#e67e22",
                                                 "#f1c40f", "#7dcea0", "#1e8449"])


def gradient_fill(ax, x, y, y0, cmap, alpha=0.55):
    """Fill under a curve with a left-to-right colour gradient."""
    grad = np.linspace(0, 1, 512).reshape(1, -1)
    im = ax.imshow(grad, extent=[min(x), max(x), y0, max(y) * 1.02],
                   aspect="auto", cmap=cmap, alpha=alpha, origin="lower",
                   zorder=1)
    poly = ax.fill_between(x, y0, y, facecolor="none", edgecolor="none")
    im.set_clip_path(poly.get_paths()[0], transform=ax.transData)


def main():
    R = sorted((({k: float(v) for k, v in r.items()})
                for r in csv.DictReader(open(SRC))), key=lambda r: r["strength"])
    s = [r["strength"] for r in R]
    ra = [r["RA"] for r in R]
    ta = [r["TA"] for r in R]
    fa = [r["FA"] for r in R]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.2),
                                 gridspec_kw=dict(width_ratios=[1.45, 1]))

    # ---------------- (a) damage vs constraint hardness --------------------
    y0 = 40
    gradient_fill(a1, s, ra, y0, CMAP)
    a1.plot(s, ra, "-o", color="#145a32", lw=2.6, ms=8, zorder=4,
            markeredgecolor="white", markeredgewidth=1.4, label="retain accuracy")
    a1.plot(s, ta, "--s", color=BLUE, lw=2.0, ms=6.5, zorder=4,
            markeredgecolor="white", markeredgewidth=1.2, label="test accuracy")
    for xi, yi in zip(s, ra):
        a1.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9, color="#145a32",
                    fontweight="bold", zorder=5)
    a1.annotate(f"{ta[0]:.1f}", (s[0], ta[0]), textcoords="offset points",
                xytext=(9, 3), fontsize=9, color=BLUE)
    a1.annotate(f"{ta[-1]:.1f}", (s[-1], ta[-1]), textcoords="offset points",
                xytext=(-4, -15), ha="right", fontsize=9, color=BLUE)

    # the controlled variable, stated on the plot
    a1.annotate("forget accuracy = 0.0% at every setting\n"
                "(forgetting is complete throughout —\n"
                "only collateral damage varies)",
                (0.02, 111), fontsize=9, color="0.25", va="top",
                bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.8", lw=0.9),
                zorder=6)
    a1.set_xlabel("projection strength $s$   (0 = soft penalty $\\rightarrow$ 1 = hard constraint)")
    a1.set_ylabel("accuracy (%)")
    a1.set_ylim(y0, 118)
    a1.set_xlim(-0.03, 1.03)
    a1.set_xticks(s)
    a1.set_title("(a) softening the constraint costs retain accuracy", fontsize=11)
    a1.legend(frameon=False, fontsize=9, loc="lower right")
    a1.grid(alpha=0.2, lw=0.6, zorder=0)
    a1.set_axisbelow(True)

    # ---------------- (b) the guarantee is a step, not a slope --------------
    resid = [1 - x for x in s]
    a2.fill_between(s, 0, resid, color=RED, alpha=0.18, zorder=1)
    a2.plot(s, resid, "-o", color=RED, lw=2.4, ms=7, zorder=4,
            markeredgecolor="white", markeredgewidth=1.3)
    a2.plot([1.0], [0.0], "*", color=GREEN, ms=22, zorder=6,
            markeredgecolor="white", markeredgewidth=1.2)
    a2.annotate("$\\langle u,g_r\\rangle = 0$\nguarantee holds",
                (1.0, 0.0), xytext=(0.80, 0.56), ha="center",
                fontsize=9.5, color=GREEN, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.4,
                                shrinkA=4, shrinkB=10))
    a2.annotate("retain-harming steps\nremain possible",
                (0.30, 0.28), fontsize=9.5, color=RED, ha="center")
    a2.annotate(r"$\langle u,g_r\rangle=(1-s)\,\langle v,g_r\rangle$",
                (0.5, 0.93), ha="center", fontsize=10.5, color="0.2")
    a2.set_xlabel("projection strength $s$")
    a2.set_ylabel("residual conflict  (fraction of $\\langle v,g_r\\rangle$)")
    a2.set_xlim(-0.03, 1.03)
    a2.set_ylim(-0.05, 1.06)
    a2.set_xticks(s)
    a2.set_title("(b) …but only $s=1$ excludes them", fontsize=11)
    a2.grid(alpha=0.2, lw=0.6, zorder=0)
    a2.set_axisbelow(True)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"  RA {ra[0]:.2f} -> {ra[-1]:.2f} ; TA {ta[0]:.2f} -> {ta[-1]:.2f} ; "
          f"FA constant at {set(fa)}")


if __name__ == "__main__":
    main()
