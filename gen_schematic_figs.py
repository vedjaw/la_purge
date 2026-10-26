"""
Schematic figures. These replace two AAAI diagrams that the current method
has outgrown:

  fig_projection_order.png   replaces fig_purge_vs_salun.png, which drew the
      projection acting on the GRADIENT. The method now projects the
      post-momentum UPDATE, and the difference is the paper's central
      methodological claim, so the figure draws Proposition B.6's
      counterexample directly: projecting the gradient does not constrain the
      vector that is actually applied.

  fig_pipeline.png           replaces fig6_algorithm_overview.png, which showed
      an entropy gate, a representation-erasure term and the update
      theta <- theta - eta*clip(g_f) -- two removed components and a
      superseded update rule.

Both are drawn from the algorithm definition, not from data.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent
IMG = ROOT / "images"
IMG.mkdir(exist_ok=True)

BLUE, RED, GREEN, GREY = "#2980b9", "#c0392b", "#27ae60", "#7f8c8d"
AMBER = "#e67e22"


def arrow(ax, x0, y0, x1, y1, color, lw=2.4, ls="-", alpha=1.0, z=4):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), color=color, lw=lw,
                                 linestyle=ls, alpha=alpha, zorder=z,
                                 arrowstyle="-|>", mutation_scale=17,
                                 shrinkA=0, shrinkB=0))


def projection_order():
    """Proposition B.6: g_r rotates between steps, so a gradient certified at
    step 0 is carried by momentum into step 1, where it is harmful."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.5))
    mu = 0.9
    for ax in (a1, a2):
        ax.set_xlim(-1.35, 1.75)
        ax.set_ylim(-1.5, 1.55)
        ax.set_aspect("equal")
        ax.axhline(0, color="0.75", lw=0.8)
        ax.axvline(0, color="0.75", lw=0.8)
        # step-1 constraint: g_r = (0,-1) -> retain-safe is the LOWER half-plane
        ax.axhspan(-1.5, 0, color=GREEN, alpha=0.07)
        ax.axhline(0, color=GREEN, ls="--", lw=1.2)
        ax.annotate("retain-safe half-space\n" r"$\{u:\langle u,g_r\rangle\geq0\}$",
                    (-1.28, -1.42), fontsize=8.5, color=GREEN, va="bottom")
        arrow(ax, 0, 0, 0, -1.0, GREEN)
        ax.annotate(r"$g_r$ (step 1)", (0.05, -1.0), fontsize=9.5, color=GREEN,
                    va="top")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("0.8")

    # ---- (a) project the gradient, then accumulate momentum ----------------
    arrow(a1, 0, 0, 0, mu, GREY, lw=2.0, ls=":")
    a1.annotate(r"$\mu v_0$" "\n(carried from step 0)", (-0.08, mu * 0.55),
                fontsize=8.8, color=GREY, ha="right")
    arrow(a1, 0, 0, 1.0, 0, BLUE, lw=2.0, ls=":")
    a1.annotate(r"$d_1$  (retain-safe on its own)", (0.30, -0.17),
                fontsize=8.8, color=BLUE)
    arrow(a1, 0, 0, 1.0, mu, RED, lw=2.8)
    a1.annotate(r"$v_1=\mu v_0+d_1$" "\napplied update", (1.03, mu + 0.04),
                fontsize=9.5, color=RED, va="bottom", ha="right")
    a1.annotate(r"$\langle v_1,g_r\rangle=-\mu<0$" "\nretain loss increases",
                (-1.28, 1.42), fontsize=9.5, color=RED, va="top")
    a1.set_title("(a) project the gradient, then add momentum",
                 fontsize=11, color=RED)

    # ---- (b) accumulate momentum, then project ----------------------------
    arrow(a2, 0, 0, 1.0, mu, GREY, lw=1.8, ls=":")
    a2.annotate(r"$v_1$ (pre-projection)", (1.02, mu + 0.03), fontsize=8.8,
                color=GREY, ha="right", va="bottom")
    n = (1.0 ** 2 + mu ** 2) ** 0.5          # projection is magnitude-preserving
    arrow(a2, 0, 0, n, 0, BLUE, lw=2.8)
    a2.annotate(r"$u_1=\widehat{\Pi}_{g_r}(v_1)$" "\napplied update",
                (n + 0.04, 0.10), fontsize=9.5, color=BLUE, va="bottom")
    a2.add_patch(FancyArrowPatch((1.0, mu), (n, 0), color="0.45", lw=1.1,
                                 ls="--", arrowstyle="-|>", mutation_scale=11,
                                 shrinkA=2, shrinkB=2, zorder=3))
    a2.annotate("project", (1.28, mu * 0.55), fontsize=8.5, color="0.45")
    a2.annotate(r"$\langle u_1,g_r\rangle=0$" "\nretain loss non-increasing",
                (-1.28, 1.42), fontsize=9.5, color=BLUE, va="top")
    a2.set_title("(b) add momentum, then project (ours)", fontsize=11,
                 color=BLUE)

    fig.suptitle("Projecting the gradient does not constrain the update that is applied",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(IMG / "fig_projection_order.png", dpi=200, bbox_inches="tight")
    print("wrote fig_projection_order.png")


def pipeline():
    """The four ablation-supported components, and the actual update rule."""
    fig, ax = plt.subplots(figsize=(11, 4.3))
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=9, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                    fc=fc, ec=ec, lw=1.3, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, zorder=3, weight=weight)

    def link(x0, y0, x1, y1, color="0.35", ls="-"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), color=color, lw=1.3,
                                     ls=ls, arrowstyle="-|>", mutation_scale=13,
                                     shrinkA=1, shrinkB=1, zorder=1))

    box(1, 34, 20, 8, "retain batch\n$(x_r,y_r)$", "#eaf4ea", GREEN)
    box(1, 20, 20, 8, "forget batch\n$(x_f,y_f)$", "#fdeeea", RED)
    box(1, 6, 20, 8, "frozen original\n$\\theta_{\\rm orig}$", "#eaf1f8", BLUE)

    box(26, 34, 24, 8, "retain gradient $g_r$\nCE $+\\;\\beta\\,$KD", "#eaf4ea", GREEN)
    box(26, 20, 24, 8, "forget gradient $g_f$\nKL to $p_r$", "#fdeeea", RED)
    box(26, 6, 24, 8, "$p_r$: retain-confusion\n(precomputed once)", "#eaf1f8", BLUE)

    box(55, 20, 20, 8, "clip to $c$\nthen $v\\leftarrow\\mu v+d$", "#fdf3e6", AMBER)
    box(79, 20, 20, 8, "project $v$ onto\n$\\{\\langle u,g_r\\rangle\\geq0\\}$",
        "#e8f0f8", BLUE, weight="bold")
    box(55, 4, 44, 8,
        "$\\theta\\leftarrow\\theta-\\eta\\,u$   +   dual stopping "
        "($\\epsilon_R$, $\\epsilon_F$)", "#f2f2f2", "0.45")

    link(21, 38, 26, 38); link(21, 24, 26, 24); link(21, 10, 26, 10)
    link(50, 24, 55, 24)
    link(38, 14, 38, 20, BLUE, ls=":")
    link(50, 38, 89, 38, GREEN); link(89, 38, 89, 28, GREEN)   # g_r -> projection
    link(75, 24, 79, 24)
    link(89, 20, 89, 12, BLUE); link(89, 12, 77, 12, BLUE)

    ax.text(50, 44.4, "EXCISE: four ablation-supported components",
            ha="center", fontsize=12)
    ax.text(89, 30.2, "$g_r$", ha="center", fontsize=9, color=GREEN)
    fig.tight_layout()
    fig.savefig(IMG / "fig_pipeline.png", dpi=200, bbox_inches="tight")
    print("wrote fig_pipeline.png")


if __name__ == "__main__":
    projection_order()
    pipeline()
