#!/usr/bin/env bash
# Collect the run_remaining_todos.sh outputs into ONE transferable archive.
#
# Run from the repo root on the GPU server:   bash collect_todo_results.sh
#
# Several of those runs (the forget-objective comparison, the projection-only
# baseline, the CIFAR-10 BN/GN checkpoints) report their metrics only to stdout,
# and those logs contain absolute cluster paths. This script parses the metrics
# out into a CSV and archives ONLY CSVs, so nothing identifying is transferred.
set -u
cd "$(dirname "$0")"

python - <<'PY'
import csv, glob, os, re

def parse(txt):
    g = lambda p: (float(re.search(p, txt).group(1))
                   if re.search(p, txt) else None)
    d = {
        "TA":  g(r"Test Accuracy\s*\(TA\)\s*:\s*([\d.]+)%"),
        "FA":  g(r"Forget Accuracy\s*\(FA[^)]*\)\s*:\s*([\d.]+)%"),
        "RA":  g(r"Retain Accuracy\s*\(RA[^)]*\)\s*:\s*([\d.]+)%"),
        "MIA": g(r"MIA \(average\)\s*:\s*([\d.]+)"),
        "feat_cos": g(r"Feature [Cc]osine [Ss]im\w*\s*:\s*([\d.]+)"),
        "proj_rate": g(r"Avg Projection Rate\s*:\s*([\d.]+)%"),
        "seconds": g(r"Unlearning completed in\s*([\d.]+)s"),
        "epochs_run": len(re.findall(r"Forget CE:", txt)),
    }
    if re.search(r"FA target reached|forgetting complete|stopping mid-epoch", txt):
        d["stop_reason"] = "fa_target"
    elif re.search(r"Retain budget exceeded|will stop after this epoch", txt):
        d["stop_reason"] = "retain_budget"
    elif re.search(r"distribution matched|past the minimum", txt):
        d["stop_reason"] = "kl"
    else:
        d["stop_reason"] = "all_epochs"
    return d

rows = []
for lg in sorted(glob.glob("results/todo_logs/*.log")):
    name = os.path.basename(lg).rsplit("_", 2)[0]
    if name.startswith("todos_"):        # the master log, not a run
        continue
    try:
        txt = open(lg, errors="ignore").read()
    except OSError:
        continue
    if "Test Accuracy" not in txt:        # not a run.py log (e.g. a driver)
        continue
    r = parse(txt); r["run"] = name
    rows.append(r)

out = "results/todo/todo_runs_summary.csv"
os.makedirs("results/todo", exist_ok=True)
fields = ["run", "TA", "FA", "RA", "MIA", "feat_cos", "proj_rate",
          "epochs_run", "stop_reason", "seconds"]
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for r in sorted(rows, key=lambda x: x["run"]):
        w.writerow({k: r.get(k) for k in fields})
print(f"parsed {len(rows)} runs -> {out}\n")
for r in sorted(rows, key=lambda x: x["run"]):
    print(f"  {r['run']:<26} TA={r['TA']} FA={r['FA']} RA={r['RA']} "
          f"MIA={r['MIA']} stop={r['stop_reason']}")
PY

echo
echo "=== archiving CSVs only (no logs, no checkpoints) ==="
find results results_sweep -name '*.csv' -newermt '-30 days' 2>/dev/null | sort > /tmp/_csvs.txt
tar -czf todo_results.tgz -T /tmp/_csvs.txt 2>/dev/null
echo "files archived:"; sed 's/^/  /' /tmp/_csvs.txt
echo
ls -lh todo_results.tgz | awk '{print "  todo_results.tgz:", $5}'
echo
echo "Transfer todo_results.tgz only. Verify it carries no identifying paths:"
echo "  tar -tzf todo_results.tgz | head"
