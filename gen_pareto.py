"""
Generate TA vs MIA AUROC Pareto scatter + FA vs MIA scatter (dual-panel).
Run:  python gen_pareto.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

methods = [
    # (name, TA, FA, RA, MIA, marker, color)
    ("Base (no unlearn)", 93.3, 99.7, 97.2, 0.533, "X", "#7f8c8d"),
    ("Retrain (gold)", 92.5, 0.0, 100.0, 0.500, "D", "#2ecc71"),
    ("SalUn",          84.0, 0.0,  97.1, 0.497, "s", "#3498db"),
    ("Bad Teacher",    82.0, 29.7, 90.2, 0.499, "^", "#9b59b6"),
    ("SCRUB",          93.3, 88.8, 97.7, 0.515, "v", "#e67e22"),
    ("Fast Unlearn",   94.2, 84.8, 99.3, 0.525, "P", "#1abc9c"),
    ("PURGE (ours)",   85.2, 8.9,  97.2, 0.507, "*", "#e74c3c"),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# --- Panel 1: TA vs MIA ---
for name, ta, fa, ra, mia, marker, color in methods:
    size = 250 if "PURGE" in name else 150 if "Retrain" in name or "Base" in name else 100
    ax1.scatter(ta, mia, c=color, s=size, marker=marker, zorder=5,
                edgecolors="black", linewidth=0.6, label=name)

ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4)
ax1.axhspan(0.48, 0.52, color="green", alpha=0.06)
ax1.set_xlabel("Test Accuracy (%)", fontsize=12)
ax1.set_ylabel("MIA AUROC", fontsize=12)
ax1.set_title("(a) Privacy vs. Utility", fontsize=13)
ax1.set_xlim(78, 98)
ax1.set_ylim(0.44, 0.56)
ax1.legend(fontsize=8.5, loc="upper left", framealpha=0.9)

# --- Panel 2: FA vs MIA (the key insight: base model already has MIA ≈ 0.5) ---
for name, ta, fa, ra, mia, marker, color in methods:
    size = 250 if "PURGE" in name else 150 if "Retrain" in name or "Base" in name else 100
    ax2.scatter(fa, mia, c=color, s=size, marker=marker, zorder=5,
                edgecolors="black", linewidth=0.6, label=name)

ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4)
ax2.axhspan(0.48, 0.52, color="green", alpha=0.06)
ax2.axvline(x=10.0, color="gray", linestyle=":", alpha=0.4, label="Random chance (10%)")

ax2.annotate("Base model (no unlearning)\nalready has MIA $\\approx$ 0.5",
             xy=(99.7, 0.533), fontsize=8, fontstyle="italic",
             color="#666666", ha="right", va="bottom",
             xytext=(-15, 5), textcoords="offset points",
             bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8))

ax2.set_xlabel("Forget Accuracy (%) — lower = more forgetting", fontsize=12)
ax2.set_ylabel("MIA AUROC", fontsize=12)
ax2.set_title("(b) Forgetting Quality vs. Privacy", fontsize=13)
ax2.set_xlim(-5, 105)
ax2.set_ylim(0.44, 0.56)
ax2.legend(fontsize=8.5, loc="upper left", framealpha=0.9)

plt.tight_layout()
plt.savefig("images/fig_pareto_frontier.png", dpi=200)
print("Saved: images/fig_pareto_frontier.png")
