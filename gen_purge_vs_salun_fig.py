"""
Generates `images/fig_purge_vs_salun.png`, the comparison figure used in the
PURGE paper introduction (Figure ref{fig:purge_vs_salun}).

Two side-by-side panels:
  (a) SalUn: saliency-masked random-label update on top-k% of weights. Update
      direction is unconstrained relative to the retain gradient.
  (b) PURGE: forget gradient projected onto the retain-safe half-space; the
      resulting update has non-negative inner product with the retain gradient.

This is a schematic, not a data plot. Runs locally; no GPU needed.

Usage:
    python gen_purge_vs_salun_fig.py
    # writes images/fig_purge_vs_salun.png
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "fig_purge_vs_salun.png")


def arrow(ax, start, end, color, lw=2.4, style="-|>", mutation_scale=18, label=None,
          label_offset=(0.05, 0.05)):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, color=color,
                                 lw=lw, mutation_scale=mutation_scale))
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx + label_offset[0], my + label_offset[1], label,
                color=color, fontsize=11, fontweight="bold")


def setup_axes(ax, title):
    ax.set_xlim(-1.6, 1.8)
    ax.set_ylim(-1.4, 1.8)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=13, fontweight="bold")
    # parameter-space axes
    ax.axhline(0, color="lightgray", lw=0.7, zorder=0)
    ax.axvline(0, color="lightgray", lw=0.7, zorder=0)
    ax.text(1.75, -0.18, r"$\theta_1$", color="gray", fontsize=10)
    ax.text(-0.18, 1.7, r"$\theta_2$", color="gray", fontsize=10)


def draw_retain_gradient(ax, gr_end=(1.0, 0.7)):
    arrow(ax, (0, 0), gr_end, color="#2ecc71", lw=2.6,
          label=r"$g_r$ (retain)", label_offset=(0.05, 0.08))
    return gr_end


def draw_retain_safe_halfspace(ax, gr):
    """Shade the half-space {g : <g, gr> >= 0}."""
    # Perpendicular direction to gr defines the half-plane boundary
    perp = np.array([-gr[1], gr[0]])
    # Two corner points of a polygon shading the half-plane (retain-safe side)
    perp_unit = perp / np.linalg.norm(perp)
    gr_unit = np.array(gr) / np.linalg.norm(gr)
    L = 3.0
    # corners: along +perp, along -perp, and far in +gr direction to close
    p1 = perp_unit * L
    p2 = -perp_unit * L
    p3 = -perp_unit * L + gr_unit * L * 1.8
    p4 = perp_unit * L + gr_unit * L * 1.8
    poly = Polygon([p1, p2, p3, p4], closed=True, alpha=0.10, color="#2ecc71")
    ax.add_patch(poly)
    # dashed boundary line through origin, perpendicular to gr
    ax.plot([-perp_unit[0] * L, perp_unit[0] * L],
            [-perp_unit[1] * L, perp_unit[1] * L],
            color="#2ecc71", lw=1.0, linestyle="--", alpha=0.6)
    ax.text(perp_unit[0] * 1.2, perp_unit[1] * 1.2,
            "retain-safe\nhalf-space", color="#2ecc71",
            fontsize=9, ha="center", alpha=0.9)


def panel_salun(ax):
    setup_axes(ax, "(a) SalUn: masked random-label update")
    gr = draw_retain_gradient(ax)
    draw_retain_safe_halfspace(ax, gr)

    # Draw the parameter-mask intuition: a grid of cells, half highlighted
    mask_xs = np.linspace(-1.3, -0.4, 5)
    mask_ys = np.linspace(-1.2, -0.4, 4)
    for xi in mask_xs:
        for yj in mask_ys:
            on = (xi + yj) > -1.2  # arbitrary half: simulate top-50% saliency
            rect = Rectangle((xi, yj), 0.18, 0.18,
                             facecolor=("#e67e22" if on else "#dddddd"),
                             edgecolor="white", lw=0.4)
            ax.add_patch(rect)
    ax.text(-0.85, -1.32, "top-50% saliency mask",
            fontsize=9, ha="center", color="#e67e22", fontweight="bold")

    # SalUn update: random-label gradient pointing in an arbitrary direction
    # Show one possible direction that crosses INTO the unsafe half-space
    gsal = np.array([-0.7, 0.95])
    arrow(ax, (0, 0), tuple(gsal), color="#e67e22", lw=2.6,
          label=r"$-M\odot g_{\mathrm{RL}}$", label_offset=(-0.4, 0.05))

    # Inner-product sign annotation
    sign = np.dot(gsal, gr)
    txt = r"$\langle g_{\mathrm{SalUn}}, g_r\rangle < 0$"  # in this scenario
    ax.text(-1.4, -0.4, txt, color="#c0392b", fontsize=10, fontweight="bold")
    ax.text(-1.4, -0.65, "(may strictly increase retain loss)",
            color="#c0392b", fontsize=8.5)


def panel_purge(ax):
    setup_axes(ax, "(b) PURGE: projected gradient")
    gr = draw_retain_gradient(ax)
    draw_retain_safe_halfspace(ax, gr)

    gr_arr = np.array(gr)
    # Raw forget gradient that conflicts with retain gradient
    gf_raw = np.array([-0.5, 1.1])
    arrow(ax, (0, 0), tuple(gf_raw), color="#888888", lw=1.8, style="->",
          mutation_scale=14, label=r"$g_f$ (raw)", label_offset=(-0.55, 0.05))

    # Projected forget gradient: subtract the component along gr that goes negative
    inner = np.dot(gf_raw, gr_arr)
    gr_norm_sq = np.dot(gr_arr, gr_arr)
    if inner < 0:
        gf_proj = gf_raw - (inner / gr_norm_sq) * gr_arr
    else:
        gf_proj = gf_raw

    arrow(ax, (0, 0), tuple(gf_proj), color="#e74c3c", lw=2.8,
          label=r"$\tilde g_f$ (projected)", label_offset=(0.05, 0.05))

    # Dotted line indicating the projection subtraction
    proj_along_gr = (inner / gr_norm_sq) * gr_arr
    ax.plot([gf_raw[0], gf_proj[0]],
            [gf_raw[1], gf_proj[1]],
            color="#aaaaaa", lw=1.0, linestyle=":")

    txt = r"$\langle \tilde g_f, g_r\rangle = 0 \geq 0$"
    ax.text(-1.4, -0.4, txt, color="#27ae60", fontsize=10, fontweight="bold")
    ax.text(-1.4, -0.65, "(retain loss non-increasing, Thm. 1)",
            color="#27ae60", fontsize=8.5)


def main():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))
    panel_salun(axL)
    panel_purge(axR)
    fig.suptitle(
        "One-step update: SalUn (mask + random label) vs.\\ PURGE (projection)",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
