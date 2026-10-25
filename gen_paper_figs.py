"""
Regenerate the paper's data figures from the CURRENT result CSVs.

The AAAI figures cannot be reused: they encode numbers from the superseded
implementation, and two of them show components (entropy gate, representation
erasure) and an update rule that the method no longer has. Everything here is
drawn from verified CSVs instead, so each figure matches the tables.

Produces, into images/:
    fig_momentum.png     momentum ablation -- why the projection is applied
                         to the post-momentum update
    fig_guarantee.png    per-step audit of Theorem 1 on a real run
    fig_crossdataset.png forget accuracy against chance, all six datasets
    fig_perclass.png     per-class forget accuracy, MNIST and PathMNIST
"""
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
IMG = ROOT / "images"
IMG.mkdir(exist_ok=True)

BLUE, RED, GREY, GREEN = "#2980b9", "#c0392b", "#7f8c8d", "#27ae60"
CHANCE = {"cifar10": 10.0, "mnist": 10.0, "svhn": 10.0, "stl10": 10.0,
          "pathmnist": 100 / 9, "cifar100": 1.0}
NICE = {"cifar10": "CIFAR-10", "mnist": "MNIST", "svhn": "SVHN",
        "stl10": "STL-10", "pathmnist": "PathMNIST", "cifar100": "CIFAR-100"}
ORDER = ["cifar10", "mnist", "svhn", "pathmnist", "cifar100", "stl10"]


def per_dataset(multiseed, cifar100):
    d = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(multiseed)):
        if r["FA"]:
            for m in ("TA", "FA", "RA", "MIA"):
                d[r["dataset"]][m].append(float(r[m]))
    c = list(csv.DictReader(open(cifar100)))
    for m in ("TA", "FA", "RA", "MIA"):
        d["cifar100"][m] = [float(r[m]) for r in c]
    return {k: {m: st.mean(v) for m, v in mv.items()} for k, mv in d.items()}


def per_class(multiseed, dataset):
    d = defaultdict(list)
    for r in csv.DictReader(open(multiseed)):
        if r["dataset"] == dataset and r["FA"]:
            d[int(r["forget_class"])].append(float(r["FA"]))
    return {k: (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0)
            for k, v in sorted(d.items())}


# ---------------------------------------------------------------- momentum
def fig_momentum(new, old):
    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    xs = range(len(ORDER))
    w = 0.38
    a = [old[d]["FA"] for d in ORDER]
    b = [new[d]["FA"] for d in ORDER]
    ax.bar([x - w / 2 for x in xs], a, w, label=r"$\mu=0$ (analysis-friendly)",
           color=GREY, edgecolor="white")
    ax.bar([x + w / 2 for x in xs], b, w, label="projected momentum (ours)",
           color=BLUE, edgecolor="white")
    for x, d in zip(xs, ORDER):
        ax.plot([x - 0.46, x + 0.46], [CHANCE[d]] * 2, color=RED, lw=1.6,
                zorder=5, solid_capstyle="butt")
    ax.plot([], [], color=RED, lw=1.6, label="chance level (100/K)")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([NICE[d] for d in ORDER], fontsize=9)
    ax.set_ylabel("forget accuracy (%)")
    ax.set_title("Removing momentum breaks forgetting on five of six datasets",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.16))
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(IMG / "fig_momentum.png", dpi=200, bbox_inches="tight")
    print("wrote fig_momentum.png")


# --------------------------------------------------------------- guarantee
def fig_guarantee(perstep):
    R = [{k: float(v) for k, v in r.items()}
         for r in csv.DictReader(open(perstep))]
    R.sort(key=lambda r: r["step"])
    steps = [r["step"] for r in R]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))

    # Normalised alignment cos(.,g_r), NOT the raw inner product: the raw values
    # span six orders of magnitude, and on a log axis the post-projection
    # residual -- which is float32 noise -- would read as a real violation.
    cb = [r["dot_raw"] / (r["g_f_norm"] * r["g_r_norm"]) for r in R]
    ca = [r["dot_post"] / (r["g_f_norm"] * r["g_r_norm"]) for r in R]
    a1.plot(steps, cb, color=GREY, lw=1.4,
            label=r"before projection: $\cos(v_t, g_r)$")
    a1.plot(steps, ca, color=BLUE, lw=1.8,
            label=r"after projection: $\cos(u_t, g_r)$")
    a1.axhline(0, color=RED, ls="--", lw=1.1)
    a1.fill_between(steps, -1, 0, color=RED, alpha=0.05)
    a1.annotate("retain-harming half-space", (steps[len(steps) // 2], -0.11),
                ha="center", va="center", fontsize=8.5, color=RED, alpha=0.85)
    a1.annotate(f"after projection: $|\\cos| \\leq {max(abs(x) for x in ca):.1e}$\n"
                "(float32 noise)", (steps[-1], 0), ha="right", va="bottom",
                xytext=(-6, 10), textcoords="offset points", fontsize=8.5,
                color=BLUE)
    a1.set_ylim(-1.03, 0.22)
    a1.set_xlabel("optimiser step")
    a1.set_ylabel(r"alignment with $g_r$")
    a1.set_title("(a) every step conflicts; projection removes it", fontsize=10.5)
    a1.legend(frameon=False, fontsize=8.5, loc="center right")
    a1.grid(alpha=0.25, lw=0.6)

    norms = [r["g_f_norm"] for r in R]
    a2.plot(steps, norms, color=BLUE, lw=1.6, label=r"$\|u_t\|$ applied")
    a2.axhline(10.0, color=RED, ls="--", lw=1.3,
               label=r"bound $c/(1-\mu)=10$")
    a2.set_ylim(0, 11.5)
    a2.set_xlabel("optimiser step")
    a2.set_ylabel(r"$\|u_t\|$")
    a2.set_title(f"(b) norm bound respected (max {max(norms):.2f})", fontsize=10.5)
    a2.legend(frameon=False, fontsize=8.5, loc="lower right")
    a2.grid(alpha=0.25, lw=0.6)

    fig.tight_layout()
    fig.savefig(IMG / "fig_guarantee.png", dpi=200, bbox_inches="tight")
    print("wrote fig_guarantee.png")


# ------------------------------------------------------------ cross-dataset
def fig_crossdataset(new):
    """FA relative to chance, so datasets with different K are comparable.

    Plotting raw FA does not work here: STL-10 (75.7%) compresses the other
    five into the bottom of the axis, and CIFAR-100's 4.1% is *above* its 1%
    chance level while MNIST's 6.2% is *below* its 10% one -- the raw numbers
    invert the comparison. The ratio FA/(100/K) puts "at chance" at 1.0 for
    every dataset.
    """
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    xs = list(range(len(ORDER)))
    ratio = [new[d]["FA"] / CHANCE[d] for d in ORDER]
    # three regimes, not two: at/below chance; above chance but a large
    # reduction; and outright failure.
    def colour(r):
        return BLUE if r <= 1.0 else ("#e67e22" if r < 6 else RED)
    ax.bar(xs, ratio, 0.6, color=[colour(r) for r in ratio], edgecolor="white")
    ax.axhline(1.0, color="k", lw=1.5, zorder=5)
    ax.annotate("chance level", (len(ORDER) - 0.4, 1.0), fontsize=8.5,
                va="bottom", ha="right", xytext=(0, 3),
                textcoords="offset points")
    for x, d, r in zip(xs, ORDER, ratio):
        ax.annotate(f"{new[d]['FA']:.1f}%", (x, r), ha="center", va="bottom",
                    fontsize=8.5, xytext=(0, 3), textcoords="offset points")
    ax.set_yscale("log")
    ax.set_ylim(0.1, 20)
    ax.set_xticks(xs)
    ax.set_xticklabels([NICE[d] for d in ORDER], fontsize=9)
    ax.set_ylabel(r"forget accuracy $\div$ chance")
    ax.set_title("Forget accuracy relative to chance (labels give raw FA)",
                 fontsize=11)
    import matplotlib.patches as mp
    ax.legend(handles=[mp.Patch(color=BLUE, label="at or below chance"),
                       mp.Patch(color="#e67e22", label="above chance (CIFAR-100)"),
                       mp.Patch(color=RED, label="failure case (STL-10)")],
              frameon=False, fontsize=8.5, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.16))
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(IMG / "fig_crossdataset.png", dpi=200, bbox_inches="tight")
    print("wrote fig_crossdataset.png")


# ---------------------------------------------------------------- per-class
def fig_perclass(multiseed):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.3))
    for ax, ds in zip(axes, ("mnist", "pathmnist")):
        pc = per_class(multiseed, ds)
        ks = list(pc)
        means = [pc[k][0] for k in ks]
        errs = [pc[k][1] for k in ks]
        ax.bar(ks, means, 0.62, yerr=errs, capsize=3, color=BLUE,
               edgecolor="white", error_kw=dict(lw=1, ecolor="0.3"))
        ax.axhline(CHANCE[ds], color=RED, ls="--", lw=1.4,
                   label=f"chance ({CHANCE[ds]:.1f}%)")
        ax.set_xticks(ks)
        ax.set_xlabel("forget class")
        ax.set_ylabel("forget accuracy (%)")
        ax.set_title(f"{NICE[ds]} — every class below chance", fontsize=10.5)
        ax.legend(frameon=False, fontsize=8.5)
        ax.grid(axis="y", alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(IMG / "fig_perclass.png", dpi=200, bbox_inches="tight")
    print("wrote fig_perclass.png")


if __name__ == "__main__":
    MS = ROOT / "results/multiseed/multiseed_all.csv"
    new = per_dataset(MS, ROOT / "results/cifar100/cifar100_summary_projmom.csv")
    old = per_dataset(ROOT / "results/archive_mu0/multiseed_all_MU0.csv",
                      ROOT / "results/archive_mu0/cifar100_summary_MU0.csv")
    fig_momentum(new, old)
    fig_guarantee(ROOT / "results/perstep/perstep_cifar10_c0_s42.csv")
    fig_crossdataset(new)
    fig_perclass(MS)
