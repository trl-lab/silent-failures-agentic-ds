"""Select the high- vs low-memorization task groups for the alteration experiment.

Usage: python scripts/select_memorization_tasks.py [--per-cell 10]
Output: analysis/exposure_probe/selection.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import build_altered_tasks as B  # noqa: E402
from skill_score import meta, skill  # noqa: E402

SET_A = {"kaggle", "mlebench", "openml", "uci"}
PROBE = ROOT / "analysis" / "exposure_probe" / "header_full_gemini-3.6-flash.jsonl"
OUT = ROOT / "analysis" / "exposure_probe" / "selection.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=10, help="tasks per (group x task type)")
    a = ap.parse_args()
    rows = {}
    for l in PROBE.read_text().splitlines():
        r = json.loads(l)
        if "error" not in r:
            rows[r["task"]] = r
    reasons, cands = {}, []
    for tid, r in rows.items():
        if r["guessable"]:
            reasons[tid] = "guessable names"
            continue
        ok, why = B.eligible(r["bucket"], tid)
        if not ok:
            reasons[tid] = why
            continue
        if skill(tid, meta[tid]["metric"], 0.5) is None:
            reasons[tid] = "no skill score for metric"
            continue
        tt = "regression" if "regression" in meta[tid]["task_type"] else "classification"
        cands.append({"task": tid, "bucket": r["bucket"], "set": "A" if r["bucket"] in SET_A else "B",
                      "task_type": tt, "metric": meta[tid]["metric"], "strict": r["strict"], "relaxed": r["relaxed"]})
    df = pd.DataFrame(cands)
    print(f"{len(rows)} probed tasks -> {len(df)} candidates; excluded: "
          + ", ".join(f"{k}={v}" for k, v in pd.Series(reasons).value_counts().items()))
    print(df.groupby(["task_type"]).strict.describe()[["count", "mean", "min", "max"]].round(2).to_string())
    rng = random.Random(0)
    sel = []
    for tt, g in df.groupby("task_type"):
        g = g.assign(_r=[rng.random() for _ in range(len(g))]).sort_values(["strict", "_r"])
        if len(g) < 2 * a.per_cell:
            print(f"WARNING: only {len(g)} {tt} candidates for 2x{a.per_cell}")
        n = min(a.per_cell, len(g) // 2)
        sel += [{**r, "group": "low"} for r in g.head(n).drop(columns="_r").to_dict("records")]
        sel += [{**r, "group": "high"} for r in g.tail(n).drop(columns="_r").to_dict("records")]
    s = pd.DataFrame(sel)
    print("\nSelected (mean strict / relaxed exposure, n, set composition):")
    print(s.groupby(["group", "task_type"]).agg(n=("task", "size"), strict=("strict", "mean"), relaxed=("relaxed", "mean"),
                                               setA=("set", lambda x: int((x == "A").sum()))).round(2).to_string())
    print("\nBy group x set:\n" + s.groupby(["group", "set"]).size().unstack(fill_value=0).to_string())
    OUT.write_text(json.dumps(sel, indent=1))
    print(f"\nwrote {OUT} ({len(sel)} tasks)")


if __name__ == "__main__":
    main()
