"""Column-completion (header) probe on ALL 220 sweep tasks with one LLM (default gemini-3.6-flash).

Same prompt/scoring as probe_exposure.py header_comp, plus a relaxed score:
  strict  = share of the next <=12 column names reproduced at the exact position (case-insensitive)
  relaxed = share of those names reproduced anywhere in the answer, ignoring case/punctuation/spacing
Raw answers and truth are stored so either score can be recomputed. Resumable.

Usage: python scripts/probe_header_full.py [--model gemini/gemini-3.6-flash] [--workers 8]
Output: analysis/exposure_probe/header_full_<model>.jsonl
"""
import argparse
import json
import re
import sys
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import probe_exposure as P  # noqa: E402


IDLE_S = 180


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def all_tasks() -> list:
    ids, seen, out = P.sweep_ids(), set(), []
    for bucket in P.GROUP:
        for p in sorted((ROOT / "datasets" / bucket).glob("*/task.json")):
            if p.parent.name in ids and p.parent.name not in seen:
                seen.add(p.parent.name)
                out.append((bucket, p.parent.name))
    return out


def run(job: dict) -> dict:
    base = {"task": job["task"], "bucket": job["bucket"], "group": job["group"], "model": job["model"], "n_cols": job["n_cols"]}
    try:
        ans = P.ask(job["model"], job["prompt"])
        got = [re.sub(r"^[\-\*\d\.\)\s]+", "", l).strip() for l in ans.splitlines() if l.strip()]
        truth = job["truth"]
        strict = sum(g.lower() == t.lower() for g, t in zip(got, truth)) / len(truth)
        gset = {norm(g) for g in got}
        relaxed = sum(norm(t) in gset for t in truth) / len(truth)
        return {**base, "truth": truth, "answer": ans, "strict": strict, "relaxed": relaxed, "guessable": job["guessable"]}
    except Exception as e:  # noqa: BLE001
        return {**base, "error": f"{type(e).__name__}: {e}"[:200]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini/gemini-3.6-flash")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = P.OUT / f"header_full_{a.model.split('/')[-1]}.jsonl"
    P.OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.read_text().splitlines():
            r = json.loads(l)
            if "error" not in r:
                done.add(r["task"])
    jobs, skipped, mismatched = [], [], []
    for bucket, tid in all_tasks():
        d = ROOT / "datasets" / bucket / tid
        meta_cols = json.loads((d / "task.json").read_text()).get("columns")
        pqp = d / "data" / "train.parquet"
        if pqp.exists():
            names = pq.ParquetFile(pqp).schema.names
            if meta_cols and list(meta_cols) != names:
                mismatched.append(tid)
        elif meta_cols:
            names = list(meta_cols)
        else:  # dataset not stored locally and no column list in task.json
            skipped.append(tid)
            continue
        t = {"train": pd.DataFrame(columns=names)}
        hp, truth, guessable = P.header_prompt(t)
        if hp is None:
            skipped.append(tid)
            continue
        if tid not in done:
            jobs.append({"task": tid, "bucket": bucket, "group": P.GROUP[bucket], "model": a.model,
                         "n_cols": len(names), "prompt": hp, "truth": truth, "guessable": guessable})
    print(f"task.json columns differ from train.parquet for {len(mismatched)} tasks: {mismatched[:5]}", flush=True)
    print(f"{len(all_tasks())} tasks, {len(skipped)} skipped (<6 columns), {len(done)} cached, running {len(jobs)}", flush=True)
    # Write results as they complete (ex.map is ordered, so one hung API call would stall
    # everything behind it). Give up on stragglers after IDLE_S without any completion and
    # exit hard; the run is resumable, so rerun to pick up whatever is left.
    ex = ThreadPoolExecutor(a.workers)
    pending = {ex.submit(run, j) for j in jobs}
    n = 0
    with out.open("a") as f:
        while pending:
            finished, pending = wait(pending, timeout=IDLE_S, return_when=FIRST_COMPLETED)
            if not finished:
                print(f"no completion for {IDLE_S}s; {len(pending)} calls still hung -- rerun to resume", flush=True)
                break
            for fut in finished:
                f.write(json.dumps(fut.result()) + "\n")
                f.flush()
                n += 1
            if n % 20 < len(finished):
                print(f"  {n}/{len(jobs)}", flush=True)
    print("done ->", out, flush=True)
    os._exit(0)  # do not wait on hung API threads


if __name__ == "__main__":
    main()
