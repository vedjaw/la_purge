"""
PathMNIST KL-stopping validation (protocol steps i-iv).

Tests one falsifiable prediction
-------------------------------
Per-class reporting showed that on PathMNIST classes 2 and 3, FA-based stopping
yields MIA AUROC of 0.298 and 0.388 -- far BELOW the ideal 0.5, i.e.
over-forgetting: the model has been pushed PAST the distribution a retrained
model would produce, and is therefore still distinguishable from one, in the
opposite direction to a privacy leak. Forget accuracy cannot see this, because
"moved to p_r" and "moved past p_r" both give FA ~ 0.

PREDICTION: replacing the FA stopping rule with one based on
KL(p_theta(x_f) || p_r) -- which is minimised AT p_r and rises on either side --
will, on the affected classes,
    (1) reduce |MIA - 0.5|,
    (2) reduce Jensen-Shannon divergence to a retrained reference model,
    (3) not materially worsen FA or RA,
and will leave an unaffected control class approximately unchanged.

A control class is included precisely so that a "improvement" that merely
reflects stopping earlier everywhere can be distinguished from a targeted fix.
If the prediction fails, the run reports that; nothing here assumes success.

Stages
------
  1. retrain reference models (skipped if checkpoints exist)   [expensive]
  2. arm A: EXCISE with FA stopping        (the current default)
  3. arm B: EXCISE with KL stopping        (--fa_target -1 --kl_stop_patience N)
  4. evaluate both arms against the references; emit the comparison

Usage
-----
    python experiments/validate_kl_stopping.py                  # 2,3 + control 4
    python experiments/validate_kl_stopping.py --classes 2 3 --control 4
    python experiments/validate_kl_stopping.py --skip_retrain    # references exist
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

os.environ.pop("MKL_THREADING_LAYER", None)
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

# Observed MIA under FA-stopping, from the main cross-dataset run. Used only to
# label classes in the report, never as a substitute for a measurement.
BASELINE_MIA = {0: 0.511, 1: 0.473, 2: 0.298, 3: 0.388, 4: 0.502,
                5: 0.548, 6: 0.541, 7: 0.562, 8: 0.525}


def sh(cmd, log_path, tail_on_error=25):
    """Run a subprocess, tee to a log, and surface the failure tail inline.

    Printing the tail matters: a silent 'exited 1' forces a manual hunt through
    the log file, and the traceback is almost always the only thing needed.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("    $ " + " ".join(str(c) for c in cmd))
    with open(log_path, "w") as f:
        p = subprocess.run([str(c) for c in cmd], stdout=f,
                           stderr=subprocess.STDOUT, text=True)
    text = log_path.read_text()
    if p.returncode != 0 and tail_on_error:
        lines = text.strip().splitlines()
        print(f"    !! exited {p.returncode}; last {min(tail_on_error, len(lines))} "
              f"lines of {log_path}:")
        for ln in lines[-tail_on_error:]:
            print(f"    | {ln}")
    return p.returncode, text


def parse_run(log):
    """Pull metrics + stop diagnostics out of a run.py log."""
    g = lambda p: (float(re.search(p, log).group(1))
                   if re.search(p, log) else None)
    out = {
        "TA": g(r"Test Accuracy\s*\(TA\)\s*:\s*([\d.]+)%"),
        "FA": g(r"Forget Accuracy\s*\(FA[^)]*\)\s*:\s*([\d.]+)%"),
        "RA": g(r"Retain Accuracy\s*\(RA[^)]*\)\s*:\s*([\d.]+)%"),
        "MIA": g(r"MIA \(average\)\s*:\s*([\d.]+)"),
        "epochs_run": len(re.findall(r"Forget CE:", log)),
    }
    # Match the specific terminating messages. "forget KL" alone is not a
    # stop marker: with KL checks enabled it is printed at EVERY check.
    if re.search(r"distribution matched, stopping|past the minimum "
                 r"\(over-forgetting\), stopping|Forget-KL criterion met", log):
        out["stop_reason"] = "kl"
    elif re.search(r"FA target reached|forgetting complete|stopping mid-epoch", log):
        out["stop_reason"] = "fa"
    elif re.search(r"Retain budget exceeded|will stop after this epoch", log):
        out["stop_reason"] = "retain_budget"
    else:
        out["stop_reason"] = "all_epochs"
    # Full KL trajectory from the per-check log line -- this is the U-shaped
    # curve that evidences over-forgetting.
    kls = re.findall(r"forget KL\(p_theta\|\|p_r\) = ([\d.]+)", log)
    out["kl_trace"] = ";".join(kls) if kls else None
    out["kl_min"] = min(map(float, kls)) if kls else None
    out["kl_final"] = float(kls[-1]) if kls else None
    # If the trajectory ended above its minimum, the optimiser went past p_r.
    out["kl_overshoot"] = (round(out["kl_final"] - out["kl_min"], 6)
                           if kls else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pathmnist")
    ap.add_argument("--classes", type=int, nargs="+", default=[2, 3],
                    help="Affected classes (over-forgetting observed).")
    ap.add_argument("--control", type=int, nargs="+", default=[4],
                    help="Control class(es) with MIA already near 0.5.")
    ap.add_argument("--kl_patience", type=int, default=2)
    ap.add_argument("--retrain_epochs", type=int, default=30,
                    help="Epoch budget for each reference model. This also sets "
                         "the cosine schedule horizon (T_max), so it must be the "
                         "budget actually intended: leaving it at the 200-epoch "
                         "default and early-stopping would halt mid-schedule "
                         "while the learning rate is still high, yielding an "
                         "un-annealed and therefore weaker reference.")
    ap.add_argument("--retrain_patience", type=int, default=0,
                    help="Early-stopping patience for the references. Default 0 "
                         "(disabled) so every reference trains the FULL epoch "
                         "budget and the cosine schedule anneals to zero. This "
                         "matters for comparability: patience-stopped references "
                         "halt at different epochs and different learning rates, "
                         "which would confound the affected-vs-control contrast "
                         "with differences in reference quality.")
    ap.add_argument("--retrain_force", action="store_true",
                    help="Retrain references even if checkpoints exist. Use when "
                         "an earlier reference was trained under different "
                         "settings, so all references stay comparable.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_dir", default=str(ROOT / "data"))
    ap.add_argument("--ckpt_dir", default=str(ROOT / "checkpoints"))
    ap.add_argument("--out_dir", default=str(ROOT / "results/kl_stopping"))
    ap.add_argument("--skip_retrain", action="store_true")
    args = ap.parse_args()

    cfg = ROOT / f"configs/{args.dataset}_kl_retain.yaml"
    base = Path(args.ckpt_dir) / f"base_{args.dataset}_resnet18.pth"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_classes = list(args.classes) + list(args.control)

    if not cfg.exists():
        print(f"[ERROR] missing config {cfg}"); return 1
    if not base.exists():
        print(f"[ERROR] missing base checkpoint {base}"); return 1

    # ---- stage 1: references -------------------------------------------------
    print(f"\n{'#' * 72}\n# stage 1/4  retrain references\n{'#' * 72}")
    if args.skip_retrain:
        print("  --skip_retrain given.")
    else:
        cmd = [PY, "-u", ROOT / "experiments/retrain_reference.py",
               "--dataset", args.dataset,
               "--forget_class", *all_classes,
               "--data_dir", args.data_dir,
               "--save_dir", args.ckpt_dir,
               "--epochs", args.retrain_epochs,
               "--patience", args.retrain_patience,
               "--seed", args.seed]
        if args.retrain_force:
            cmd.append("--force")
        rc, _ = sh(cmd, out / "retrain.log")
        if rc != 0:
            print(f"  [WARN] retrain exited {rc}; see {out/'retrain.log'}")

    refs = {c: Path(args.ckpt_dir) / f"retrain_{args.dataset}_c{c}.pth"
            for c in all_classes}
    missing = [c for c, p in refs.items() if not p.exists()]
    if missing:
        print(f"[ERROR] no reference checkpoint for classes {missing}. "
              f"Run stage 1 (drop --skip_retrain).")
        return 1

    # ---- stages 2 & 3: the two arms -----------------------------------------
    runs = {}
    for stage, (arm, extra) in enumerate(
            [("fa_stop", []),
             ("kl_stop", ["--fa_target", "-1.0",
                          "--kl_stop_patience", str(args.kl_patience)])], start=2):
        print(f"\n{'#' * 72}\n# stage {stage}/4  arm '{arm}'\n{'#' * 72}")
        for c in all_classes:
            sd = out / f"{arm}_c{c}"
            cmd = [PY, "-u", ROOT / "run.py",
                   "--config", cfg,
                   "--checkpoint", base,
                   "--data_dir", args.data_dir,
                   "--save_dir", sd,
                   "--forget_class", c,
                   "--seed", args.seed] + extra
            rc, log = sh(cmd, sd / "log.txt")
            m = parse_run(log)
            m.update({"arm": arm, "forget_class": c, "returncode": rc})
            runs[(arm, c)] = m
            print(f"    class {c}: FA={m['FA']} RA={m['RA']} MIA={m['MIA']} "
                  f"epochs={m['epochs_run']} stop={m['stop_reason']}")
            ck = sorted(sd.glob("*.pth"))
            m["ckpt"] = str(ck[-1]) if ck else None
            if not ck:
                print(f"    [WARN] no checkpoint written in {sd}")

    # ---- stage 4: distance to the references --------------------------------
    print(f"\n{'#' * 72}\n# stage 4/4  distance to retrained references\n{'#' * 72}")
    d2r_csv = out / "distance_to_retrained.csv"
    if d2r_csv.exists():
        d2r_csv.unlink()
    for c in all_classes:
        ckpts = [f"{arm}:{runs[(arm, c)]['ckpt']}"
                 for arm in ("fa_stop", "kl_stop")
                 if runs[(arm, c)].get("ckpt")]
        if not ckpts:
            print(f"  [skip] class {c}: no checkpoints"); continue
        sh([PY, "-u", ROOT / "experiments/distance_to_retrained.py",
            "--dataset", args.dataset, "--forget_class", c,
            "--retrained", refs[c],
            "--original", base,
            "--checkpoints", *ckpts,
            "--data_dir", args.data_dir,
            "--out_csv", d2r_csv],
           out / f"d2r_c{c}.log")

    d2r = {}
    if d2r_csv.exists():
        for r in csv.DictReader(open(d2r_csv)):
            d2r[(r["name"], int(r["forget_class"]))] = r

    # ---- report -------------------------------------------------------------
    summary = out / "kl_stopping_summary.csv"
    fields = ["forget_class", "role", "arm", "FA", "RA", "TA", "MIA", "MIA_dev",
              "js", "agree", "kl_to_pr", "epochs_run", "stop_reason",
              "kl_min", "kl_final", "kl_overshoot", "kl_trace"]
    with open(summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in all_classes:
            role = "affected" if c in args.classes else "control"
            for arm in ("fa_stop", "kl_stop"):
                m = runs.get((arm, c), {})
                d = d2r.get((arm, c), {})
                mia = m.get("MIA")
                w.writerow({
                    "forget_class": c, "role": role, "arm": arm,
                    "FA": m.get("FA"), "RA": m.get("RA"), "TA": m.get("TA"),
                    "MIA": mia,
                    "MIA_dev": (abs(float(mia) - 0.5) if mia is not None else None),
                    "js": d.get("js"), "agree": d.get("agree"),
                    "kl_to_pr": d.get("kl_to_pr"),
                    "epochs_run": m.get("epochs_run"),
                    "stop_reason": m.get("stop_reason"),
                    "kl_min": m.get("kl_min"),
                    "kl_final": m.get("kl_final"),
                    "kl_overshoot": m.get("kl_overshoot"),
                    "kl_trace": m.get("kl_trace"),
                })

    print(f"\n{'=' * 100}")
    print("KL-STOPPING VALIDATION — PathMNIST")
    print(f"{'=' * 100}")
    print(f"{'cls':>4} {'role':<9} {'arm':<8} {'FA':>7} {'RA':>7} {'MIA':>8} "
          f"{'|MIA-.5|':>9} {'JS':>8} {'ep':>3} {'stop':>14}")
    print("-" * 100)
    verdict = []
    for c in all_classes:
        role = "affected" if c in args.classes else "control"
        devs = {}
        for arm in ("fa_stop", "kl_stop"):
            m = runs.get((arm, c), {})
            d = d2r.get((arm, c), {})
            mia = m.get("MIA")
            dev = abs(float(mia) - 0.5) if mia is not None else None
            devs[arm] = (dev, float(d["js"]) if d.get("js") else None)
            fmt = lambda v, p=2: f"{v:.{p}f}" if v is not None else "  n/a"
            print(f"{c:>4} {role:<9} {arm:<8} {fmt(m.get('FA')):>7} "
                  f"{fmt(m.get('RA')):>7} {fmt(mia,4):>8} {fmt(dev,4):>9} "
                  f"{fmt(devs[arm][1],4):>8} {str(m.get('epochs_run','?')):>3} "
                  f"{str(m.get('stop_reason','?')):>14}")
        a, b = devs["fa_stop"], devs["kl_stop"]
        if a[0] is not None and b[0] is not None:
            better_mia = b[0] < a[0]
            better_js = (a[1] is not None and b[1] is not None and b[1] < a[1])
            verdict.append((c, role, a, b, better_mia, better_js))
        print()

    print(f"{'=' * 100}\nPREDICTION CHECK\n{'=' * 100}")
    for c, role, a, b, bm, bj in verdict:
        tag = "AS PREDICTED" if (bm and role == "affected") else \
              ("unchanged/ok" if role == "control" else "NOT AS PREDICTED")
        print(f"  class {c} ({role}): |MIA-.5| {a[0]:.4f} -> {b[0]:.4f} "
              f"({'better' if bm else 'worse/equal'});  "
              f"JS {a[1] if a[1] is None else f'{a[1]:.4f}'} -> "
              f"{b[1] if b[1] is None else f'{b[1]:.4f}'} "
              f"({'better' if bj else 'worse/equal'})   [{tag}]")
    aff = [v for v in verdict if v[1] == "affected"]
    if aff:
        n_ok = sum(1 for v in aff if v[4])
        print(f"\n  affected classes improved on |MIA-0.5|: {n_ok}/{len(aff)}")
        if n_ok < len(aff):
            print("  -> prediction NOT fully supported. Report this outcome as-is;")
            print("     do not present KL stopping as a fix on this evidence.")
    print(f"\nSaved: {summary}")
    print(f"       {d2r_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
