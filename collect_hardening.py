"""
Parse logs/hardening/*.log into results/hardening/*.csv, one CSV per phase.

run.py writes no machine-readable output -- metrics are printed and then
scraped, which is what experiments/component_ablation.py already does. This
reuses that module's parse_metrics rather than re-deriving the regexes, so a
change to run.py's print format only has to be fixed in one place.

Also prints the verdict for each phase, because the phases were designed as
yes/no questions and the answer should not need a spreadsheet.
"""
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experiments"))
from component_ablation import parse_metrics          # noqa: E402


def parse_baseline(log_text):
    """Baselines print a different format from run.py.

    run.py:    "Forget Accuracy (FA, class 0): 1.87%"   "MIA (average): 0.529"
    baselines: "Forget Accuracy (UA - want LOW): 1.87%" "MIA Score: 0.529"

    Try the shared patterns first, then the baseline-specific ones, so one
    function reads both and a missing metric stays None rather than silently
    picking up the wrong number.
    """
    out = parse_metrics(log_text)
    alt = {
        "TA":  r"Test Accuracy[^:]*:\s*([\d.]+)%",
        "FA":  r"Forget Accuracy\s*\(UA[^)]*\)\s*:\s*([\d.]+)%",
        "RA":  r"Retain Accuracy\s*\(RA[^)]*\)\s*:\s*([\d.]+)%",
        "MIA": r"MIA Score:\s*([\d.]+)",
    }
    for k, pat in alt.items():
        if out.get(k) is None:
            m = re.search(pat, log_text)
            if m:
                out[k] = float(m.group(1))
    return out

LOGS = ROOT / "logs/hardening"
OUT = ROOT / "results/hardening"

# log-name prefix -> (csv name, phase description)
PHASES = [
    (re.compile(r"^eps_(adaptive|fixed)_t(\d+)$"), "eps_adaptive.csv", "epsilon_F knob"),
    (re.compile(r"^(stl10|cifar100)_"),            "rescue.csv",       "compute rescue"),
    (re.compile(r"^sample_pct"),                   "sample_level.csv", "random-subset forgetting"),
    (re.compile(r"^arch_"),                        "architectures.csv", "architecture transfer"),
    (re.compile(r"^s7_pct"),                       "sample_level_v2.csv", "sample-level, corrected"),
    (re.compile(r"^t8_"),                          "table_grade.csv",   "STL-10 seeds / CIFAR-100 classes"),
    (re.compile(r"^vit_"),                         "vit.csv",           "ViT-B/16 under the current algorithm"),
    (re.compile(r"^mom_"),                         "momentum_rules.csv","momentum dependence across update rules"),
    (re.compile(r"^m14_"),                         "momentum_datasets.csv","momentum x rule x dataset, equal compute"),
    (re.compile(r"^x15_"),                         "crossmethod_sample.csv","sample-level across methods"),
    (re.compile(r"^x16_"),                         "crossmethod_cifar100.csv","CIFAR-100 across methods"),
]
FIELDS = ["run", "TA", "FA", "RA", "MIA", "proj_rate", "epochs_run",
          "stop_reason", "stopped_midepoch", "seconds"]


def collect():
    if not LOGS.is_dir():
        sys.exit(f"no logs at {LOGS} -- run run_hardening.sh first")
    OUT.mkdir(parents=True, exist_ok=True)
    buckets = {}
    for log in sorted(LOGS.glob("*.log")):
        for rx, csv_name, _ in PHASES:
            if rx.match(log.stem):
                txt = log.read_text(errors="replace")
                m = (parse_baseline(txt) if log.stem.startswith(("x15_", "x16_"))
                     else parse_metrics(txt))
                m["run"] = log.stem
                buckets.setdefault(csv_name, []).append(m)
                break

    for csv_name, rows in sorted(buckets.items()):
        path = OUT / csv_name
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path}  ({len(rows)} runs)")
    return buckets


def verdicts(buckets):
    def show(rows, keys=("TA", "FA", "RA", "MIA")):
        for r in sorted(rows, key=lambda x: x["run"]):
            # a metric is None when run.py never printed it (e.g. MIA is skipped
            # for some configurations); None has no numeric format, so render a
            # placeholder rather than formatting it
            vals = "  ".join(
                f"{k} {'--':>6}" if r.get(k) is None else f"{k} {r[k]:>6.2f}"
                for k in keys)
            print(f"    {r['run']:<26} {vals}   [{r.get('stop_reason','?')}]")

    print("\n" + "=" * 68)
    if "eps_adaptive.csv" in buckets:
        rows = buckets["eps_adaptive.csv"]
        print("PHASE 1 — does epsilon_F now behave like a knob?")
        show(rows)
        # how fast does FA actually fall? answers whether a knob is possible
        for lg in sorted(LOGS.glob("eps_adaptive_*.log")):
            tr = [(int(a), float(b)) for a, b in
                  re.findall(r"\[FA-trace\] step (\d+) FA ([\d.]+)%", lg.read_text(errors="replace"))]
            if len(tr) >= 2:
                drops = [(tr[i-1][1] - tr[i][1], tr[i][0]) for i in range(1, len(tr))]
                d, at = max(drops)
                print(f"    {lg.stem}: {len(tr)} steps traced, "
                      f"FA {tr[0][1]:.1f}% -> {tr[-1][1]:.1f}%, "
                      f"largest single-step drop {d:.1f}pp at step {at}")
        ad = [r["FA"] for r in rows if r["run"].startswith("eps_adaptive") and r["FA"] is not None]
        if len(ad) >= 2:
            spread = max(ad) - min(ad)
            print(f"\n  FA spread across adaptive targets: {spread:.2f}pp")
            print("  VERDICT: " + (
                "epsilon_F is a working knob -> replace the retraction in the paper."
                if spread > 3 else
                "still flat -> the collapse is sub-step; report as a STRONGER negative "
                "result (no stopping rule can catch it), and keep the retraction."))

    if "rescue.csv" in buckets:
        rows = buckets["rescue.csv"]
        print("\nPHASE 2 — is STL-10 a method failure or a compute failure?")
        show(rows)
        stl = [r["FA"] for r in rows if r["run"].startswith("stl10") and r["FA"] is not None]
        if stl:
            print(f"\n  best STL-10 FA: {min(stl):.2f}%  (paper currently reports 75.7%)")
            print("  VERDICT: " + (
                "compute failure -> STL-10 moves from 'failure case' to 'needs more steps'."
                if min(stl) < 20 else
                "method failure -> keep it as a limitation, but now it is a TESTED one."))

    if "sample_level.csv" in buckets:
        print("\nPHASE 3 — random-subset forgetting (watch MIA, not FA)")
        show(buckets["sample_level.csv"])
        print("  Target was FA ~ TA, not FA ~ 0. FA far below TA = over-forgetting.")

    if "sample_level_v2.csv" in buckets:
        print("\nPHASE 7 — sample-level, corrected config")
        show(buckets["sample_level_v2.csv"])
        print("  Base TA ~95.3; 'forgotten looks like unseen' means FA ~ TA, not FA ~ 0.")
        print("  FA still 100 => the objective cannot target individual samples.")

    if "table_grade.csv" in buckets:
        rows = buckets["table_grade.csv"]
        print("\nPHASE 8 — table-grade reruns (mean +/- std, as Table 1 reports)")
        show(rows)
        import statistics as st
        for tag, label, cur in (("t8_stl10", "STL-10 (3 seeds)", "75.67"),
                                ("t8_cifar100", "CIFAR-100 (9 classes)", "4.07")):
            g = [r for r in rows if r["run"].startswith(tag)]
            for k in ("TA", "FA", "RA", "MIA"):
                v = [r[k] for r in g if r.get(k) is not None]
                if len(v) >= 2:
                    print(f"    {label:<22} {k:<4} {st.mean(v):7.2f} +/- {st.pstdev(v):.2f}"
                          + (f"   (table currently {cur})" if k == "FA" else ""))

    if "vit.csv" in buckets:
        rows = buckets["vit.csv"]
        print("\nPHASE 10/11 — ViT-B/16 (June run, OLD algorithm: CIFAR-10 FA ~8.5)")
        show(rows)
        import statistics as st
        g = [r["FA"] for r in rows if r["run"].startswith("vit_cifar10") and r["FA"] is not None]
        if len(g) >= 2:
            print(f"    CIFAR-10 ViT FA {st.mean(g):.2f} +/- {st.pstdev(g):.2f} "
                  f"(chance 10.0; June/old-algorithm value was 8.47)")

    if "momentum_rules.csv" in buckets:
        rows = {r["run"]: r for r in buckets["momentum_rules.csv"]}
        print("\nPHASE 12 — does removing momentum break OTHER update rules too?")
        print(f"    {'update rule':<14}{'FA (mu=0.9)':>13}{'FA (mu=0)':>11}{'change':>10}")
        for rule in ("purge", "kluniform", "projonly", "ga"):
            a, b = rows.get(f"mom_{rule}_mu09"), rows.get(f"mom_{rule}_mu00")
            if a and b and a.get("FA") is not None and b.get("FA") is not None:
                d = b["FA"] - a["FA"]
                flag = "  <- breaks" if d > 20 else ""
                print(f"    {rule:<14}{a['FA']:>13.2f}{b['FA']:>11.2f}{d:>+10.2f}{flag}")
        print("    'ga' has no projection: if it breaks too, the dependence is not")
        print("    specific to operating under a hard per-step constraint.")

    if "momentum_datasets.csv" in buckets:
        rows = {r["run"]: r for r in buckets["momentum_datasets.csv"]}
        print("\nPHASE 14 — momentum x update rule, equal compute, on datasets")
        print("           where mu=0 demonstrably breaks PURGE")
        for ds, chance in (("mnist", 10.0), ("svhn", 10.0)):
            print(f"    {ds.upper()}  (chance {chance}%)")
            print(f"      {'rule':<12}{'FA mu=0.9':>11}{'FA mu=0':>10}{'change':>10}")
            for rule in ("purge", "kluniform", "projonly", "ga"):
                a = rows.get(f"m14_{ds}_{rule}_mu09")
                b = rows.get(f"m14_{ds}_{rule}_mu00")
                if a and b and a.get("FA") is not None and b.get("FA") is not None:
                    d = b["FA"] - a["FA"]
                    note = "  breaks" if b["FA"] > chance else ""
                    print(f"      {rule:<12}{a['FA']:>11.2f}{b['FA']:>10.2f}{d:>+10.2f}{note}")
        print("    'ga' runs with NO projection. If it also breaks, the momentum")
        print("    dependence is not specific to the hard per-step constraint.")

    for csvname, title, note in (
            ("crossmethod_sample.csv",
             "PHASE 15 - sample-level forgetting across methods (CIFAR-10, 1%)",
             "PURGE reference (Phase 7, fixed budget): TA 71.87  FA 58.40  RA 73.19.\n"
             "    A method that leaves FA ~ TA with RA intact has succeeded here;\n"
             "    FA far below TA, or RA collapsing, is failure in either direction."),
            ("crossmethod_cifar100.csv",
             "PHASE 16 - CIFAR-100 class 10 across methods (chance 1.0%)",
             "First baseline comparison on CIFAR-100; Table 3 covers CIFAR-10 only.")):
        if csvname in buckets:
            print(f"\n{title}")
            show(buckets[csvname])
            print("    " + note)

    if "architectures.csv" in buckets:
        print("\nPHASE 4 — does it transfer off ResNet-18?")
        show(buckets["architectures.csv"])
    print("=" * 68)


if __name__ == "__main__":
    verdicts(collect())
