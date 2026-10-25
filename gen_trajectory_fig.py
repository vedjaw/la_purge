"""
Figure: distance to a retrained reference along the unlearning trajectory.

Reads results/trajectory/trajectory_sweep.csv and produces images/fig_trajectory.png.

Panel (a) shows Jensen-Shannon divergence to the retrained reference against
optimiser step. The curve is U-shaped: unlearning first moves the model toward
the reference, reaches a minimum, then moves away again. The minimum is marked
and annotated with the forget accuracy at that point.

Panel (b) shows MIA AUROC on the same axis. The curve *crosses* the ideal 0.5
rather than converging to it, so |MIA - 0.5| is minimised at the crossing --
an artefact of where the sweep happens to pass 0.5, not a signal of unlearning
quality. Crossings are marked with the forget accuracy there, which is the
point of the panel: on class 2 the model attains "ideal" MIA while still
classifying 78.6% of the forget set correctly.
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "results/trajectory/trajectory_sweep.csv"
OUT = ROOT / "images/fig_trajectory.png"

COLOURS = {2: "#c0392b", 3: "#2980b9", 4: "#27ae60"}
LABEL = {2: "class 2 (affected)", 3: "class 3 (affected)", 4: "class 4 (control)"}
# Annotation offsets, set per class by hand: the JS minima sit close to the
# axis floor and all three MIA crossings fall within ~16 steps of each other,
# so automatic placement overlaps.
OFF_MIN = {2: (14, -18), 3: (14, -18), 4: (-76, -12)}
OFF_CROSS = {2: (14, 16), 3: (14, -22), 4: (-64, 20)}


def load():
    rows = defaultdict(list)
    for r in csv.DictReader(open(SRC)):
        rows[int(r["forget_class"])].append(
            {k: (v if k == "dataset" else float(v)) for k, v in r.items()})
    for c in rows:
        rows[c].sort(key=lambda r: r["step"])
    return rows


def main():
    rows = load()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.1))

    # ---- (a) JS to the retrained reference -------------------------------
    for c in sorted(rows):
        R = rows[c]
        ax1.plot([r["step"] for r in R], [r["js"] for r in R],
                 color=COLOURS[c], lw=1.7, label=LABEL[c])
        m = min(R, key=lambda r: r["js"])
        ax1.plot(m["step"], m["js"], "o", color=COLOURS[c], ms=8,
                 markeredgecolor="white", markeredgewidth=1.3, zorder=5)
        ax1.annotate(f"FA={m['FA']:.1f}%",
                     (m["step"], m["js"]), textcoords="offset points",
                     xytext=OFF_MIN[c], fontsize=8.5, color=COLOURS[c],
                     arrowprops=dict(arrowstyle="-", lw=0.7,
                                     color=COLOURS[c], alpha=0.8))
    lo = min(r["js"] for R in rows.values() for r in R)
    hi = max(r["js"] for R in rows.values() for r in R)
    ax1.set_ylim(lo - 0.05 * (hi - lo), hi + 0.03 * (hi - lo))  # room for labels
    ax1.set_xlabel("optimiser step")
    ax1.set_ylabel("JS divergence to retrained model")
    ax1.set_title("(a) distance to the retrained reference", fontsize=11)
    ax1.legend(frameon=False, fontsize=9, loc="lower right")
    ax1.grid(alpha=0.25, lw=0.6)

    # ---- (b) MIA, showing the crossing of 0.5 ----------------------------
    for c in sorted(rows):
        R = rows[c]
        ax2.plot([r["step"] for r in R], [r["MIA"] for r in R],
                 color=COLOURS[c], lw=1.7, label=LABEL[c])
        for a, b in zip(R, R[1:]):
            if (a["MIA"] - 0.5) * (b["MIA"] - 0.5) < 0:
                ax2.plot(b["step"], b["MIA"], "s", color=COLOURS[c], ms=7,
                         markeredgecolor="white", markeredgewidth=1.3, zorder=5)
                ax2.annotate(f"FA={b['FA']:.1f}%",
                             (b["step"], b["MIA"]), textcoords="offset points",
                             xytext=OFF_CROSS[c], fontsize=8.5,
                             color=COLOURS[c],
                             arrowprops=dict(arrowstyle="-", lw=0.7,
                                             color=COLOURS[c], alpha=0.8))
                break
    ax2.axhline(0.5, color="0.35", ls="--", lw=1.1)
    ax2.annotate("ideal (0.5)", (0.985, 0.5), xycoords=("axes fraction", "data"),
                 ha="right", va="bottom", fontsize=8.5, color="0.35")
    ax2.set_xlabel("optimiser step")
    ax2.set_ylabel("MIA AUROC")
    ax2.set_title("(b) MIA crosses 0.5 — it does not converge to it", fontsize=11)
    ax2.grid(alpha=0.25, lw=0.6)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}")

    # echo the annotated quantities so the caption can be checked against them
    for c in sorted(rows):
        R = rows[c]
        m = min(R, key=lambda r: r["js"])
        cross = next((b for a, b in zip(R, R[1:])
                      if (a["MIA"] - 0.5) * (b["MIA"] - 0.5) < 0), None)
        print(f"  class {c}: JS min {m['js']:.4f} @ step {int(m['step'])} "
              f"(FA {m['FA']:.2f}%); MIA {R[0]['MIA']:.3f} -> {R[-1]['MIA']:.3f}"
              + (f", crosses 0.5 at FA {cross['FA']:.2f}%" if cross else ""))


if __name__ == "__main__":
    main()
