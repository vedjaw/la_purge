"""
Figure: what EXCISE achieves, and what it guarantees.

-> images/fig_advantage.png

Panel (a) is measured: where each method lands on CIFAR-10 class 0, forget
accuracy against retain accuracy, with the retrained gold standard marked. The
target region is low FA (at or below the 10% chance level) AND high RA; most
methods miss on one axis or the other.

Panel (b) is definitional, and is the point of the figure. Two methods can look
similar in (a) and still differ in what is *possible*: a soft penalty biases the
optimiser away from retain-harming steps but leaves them in the feasible set,
so a large enough forget-loss reduction still buys one. A projection deletes
that half-space outright. This is Theorem 2(iv), drawn.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "images/fig_advantage.png"

BLUE, RED, GREEN, GREY = "#2980b9", "#c0392b", "#1e8449", "#7f8c8d"
AMBER = "#e67e22"

# CIFAR-10 class 0 (Table 3): name, FA, RA, TA, is_ours
M = [("Retrain\n(gold standard)",  0.00, 100.00, 92.5, "gold"),
     ("SalUn",                     0.00,  97.10, 84.0, "base"),
     ("Bad Teacher",              29.70,  90.20, 82.0, "base"),
     ("SCRUB",                    88.80,  97.70, 93.3, "base"),
     ("NegGrad+",                 84.80,  99.30, 94.2, "base"),
     ("projection only",          18.66,  99.90, 86.2, "abl"),
     ("EXCISE",                    1.87, 100.00, 86.1, "ours")]
STYLE = {"gold": dict(c="#8e44ad", m="*", s=420),
         "base": dict(c=GREY,      m="o", s=110),
         "abl":  dict(c=AMBER,     m="s", s=110),
         "ours": dict(c=GREEN,     m="D", s=190)}


def panel_a(ax):
    ax.axvspan(-4, 10, color=GREEN, alpha=0.06, zorder=0)
    ax.axhspan(97, 101.6, color=GREEN, alpha=0.06, zorder=0)
    ax.axvline(10, color=GREEN, ls="--", lw=1.2, zorder=1)

    for name, fa, ra, ta, kind in M:
        st = STYLE[kind]
        ax.scatter(fa, ra, marker=st["m"], s=st["s"], c=st["c"], zorder=5,
                   edgecolor="white", linewidth=1.5)

    # (text_x, text_y, ha, va, draw_connector)
    lab = {"Retrain\n(gold standard)": (13.0, 101.3, "left", "bottom", True),
           "EXCISE":                    (9.0,  99.55, "left", "top",   True),
           "SalUn":                     (1.8,  96.75, "left", "top",   False),
           "Bad Teacher":              (32.0,  90.20, "left", "center",False),
           "SCRUB":                    (86.5,  97.25, "right","top",   False),
           "NegGrad+":                 (82.5,  99.50, "right","bottom",False),
           "projection only":          (21.0,  99.90, "left", "center",False)}
    for name, fa, ra, ta, kind in M:
        x, y, ha, va, conn = lab[name]
        c = STYLE[kind]["c"]
        if conn:
            ax.plot([fa, x - 0.6 if ha == "left" else x + 0.6], [ra, y],
                    color=c, lw=0.8, ls="-", alpha=0.55, zorder=4)
        ax.annotate(f"{name}   TA {ta:.1f}", (x, y), fontsize=8.6, ha=ha, va=va,
                    color=c, fontweight="bold" if kind == "ours" else "normal",
                    zorder=6)

    ax.annotate("target region:\nforgotten AND intact", (62, 94.5), fontsize=9.5,
                color=GREEN, style="italic", ha="center")
    ax.annotate("chance level (10%)", (11.5, 88.6), fontsize=8.5, color=GREEN,
                va="bottom")
    ax.set_xlabel("forget accuracy (%)   $\\leftarrow$ better")
    ax.set_ylabel("retain accuracy (%)   better $\\rightarrow$")
    ax.set_xlim(-4, 100); ax.set_ylim(88, 102.4)
    ax.set_title("(a) where each method lands  (CIFAR-10, class 0)", fontsize=11)
    ax.grid(alpha=0.2, lw=0.6); ax.set_axisbelow(True)


def half_space(ax, hard):
    """Feasible set for a soft penalty vs a hard constraint."""
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.35); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("0.8")
    if hard:
        ax.axhspan(-1.2, 0, color=GREEN, alpha=0.16)
        ax.axhspan(0, 1.2, color="0.90")
        ax.annotate("UNREACHABLE", (0, 1.02), ha="center", fontsize=9.5,
                    color="0.40", fontweight="bold")
    else:
        grad = np.linspace(0, 1, 256).reshape(-1, 1)
        ax.imshow(grad, extent=[-1.2, 1.2, -1.2, 1.2], origin="lower",
                  cmap="RdYlGn", alpha=0.30, aspect="auto")
        ax.set_aspect("equal")   # imshow's aspect kwarg overrides the axes aspect
        ax.annotate("STILL REACHABLE", (0, 1.02), ha="center", fontsize=9.5,
                    color=RED, fontweight="bold")
    ax.axhline(0, color=GREEN, ls="--", lw=1.3)
    ax.add_patch(FancyArrowPatch((0, 0), (0, -0.8), color=GREEN, lw=2.2,
                                 arrowstyle="-|>", mutation_scale=15))
    ax.annotate("$g_r$", (0.09, -0.72), fontsize=10, color=GREEN)
    ax.add_patch(FancyArrowPatch((0, 0), (0.66, 0.52), color=RED, lw=2.4,
                                 arrowstyle="-|>", mutation_scale=15,
                                 alpha=0.30 if hard else 1.0,
                                 linestyle=":" if hard else "-"))


def main():
    fig, (a1, b1, b2) = plt.subplots(
        1, 3, figsize=(13.2, 4.5), gridspec_kw=dict(width_ratios=[2.0, 0.7, 0.7],
                                                    wspace=0.30))
    panel_a(a1)
    half_space(b1, hard=False)
    half_space(b2, hard=True)
    b1.set_title("soft penalty\n(closest prior work)", fontsize=10, color=RED)
    b2.set_title("hard constraint\n(EXCISE)", fontsize=10, color=GREEN)
    fig.text(0.735, -0.02,
             "(b) a retain-harming step (red) against the constraint $g_r$: a penalty\n"
             "biases the optimiser away from it; a projection deletes the half-space.",
             ha="center", fontsize=9.4, color="0.25")
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
