"""Pilot: does an LLM recognize / reproduce the schema of our benchmark datasets?

Usage:
  python scripts/probe_exposure.py --dry-run             # print prompts for 2 tasks, no API calls
  python scripts/probe_exposure.py                       # run the pilot (resumable cache)
  python scripts/probe_exposure.py --report              # summarize cached results only
"""
import argparse
import csv
import io
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_vanilla import load_registry, resolve_model_cfg, call_llm  # noqa: E402

OUT = ROOT / "analysis" / "exposure_probe"
RAW = OUT / "pilot_raw.jsonl"
MODELS = ["gemini/gemini-3.6-flash", "gemini/gemini-3.7-flash"]
JUDGE = "gemini/gemini-3.6-flash"
GROUP = {"kaggle": "Kaggle/MLE", "mlebench": "Kaggle/MLE", "openml": "OpenML", "uci": "UCI", "strable": "Tier B"}
QUOTA = {"Kaggle/MLE": 10, "UCI": 4, "OpenML": 8, "Tier B": 12}
POSITIVE = ["titanic", "adult", "house-prices-advanced-regression-techniques", "california_housing"]
MAX_NAMES, MAX_COLS_ROW, N_ROWS, N_COMP = 60, 25, 3, 12


def sweep_ids() -> set:
    return {p.stem for p in (ROOT / "judge_results" / "vanilla").glob("*.json")}


def pick_tasks(seed: int = 0) -> list:
    rng = random.Random(seed)
    ids, pool = sweep_ids(), {g: [] for g in QUOTA}
    for bucket, grp in GROUP.items():
        for p in sorted((ROOT / "datasets" / bucket).glob("*/task.json")):
            if p.parent.name in ids:
                pool[grp].append((bucket, p.parent.name))
    chosen = []
    for grp, n in QUOTA.items():
        forced = [x for x in pool[grp] if x[1] in POSITIVE]
        rest = [x for x in pool[grp] if x[1] not in POSITIVE]
        rng.shuffle(rest)
        chosen += forced + rest[: max(0, n - len(forced))]
    return chosen


def load_task(bucket: str, tid: str) -> dict:
    d = ROOT / "datasets" / bucket / tid
    meta = json.loads((d / "task.json").read_text())
    train = pd.read_parquet(d / "data" / "train.parquet")
    return {"bucket": bucket, "id": tid, "group": GROUP[bucket], "meta": meta, "train": train}


def fmt_cell(v) -> str:
    return "" if pd.isna(v) else str(v)[:40]


def recog_prompt(t: dict, with_rows: bool) -> str:
    cols = list(t["train"].columns)
    names = ", ".join(cols[:MAX_NAMES]) + (f", ... ({len(cols)} columns in total)" if len(cols) > MAX_NAMES else "")
    block = f"Column names (file order): {names}"
    if with_rows:
        sub = t["train"].iloc[:, :MAX_COLS_ROW]
        rows = sub.sample(n=min(N_ROWS, len(sub)), random_state=0)
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(sub.columns)
        for r in rows.itertuples(index=False):
            w.writerow([fmt_cell(v) for v in r])
        block += "\n\nSample rows" + (f" (first {MAX_COLS_ROW} columns)" if len(cols) > MAX_COLS_ROW else "") + ":\n" + buf.getvalue().rstrip()
    return (f"Here is a tabular dataset.\n\n{block}\n\n"
            "Do you recognize this specific dataset? If yes, give its name and where it comes from "
            "(e.g. a Kaggle competition, the UCI repository, OpenML, a government open-data portal). "
            "If you do not recognize it, answer UNKNOWN. Do not guess from the topic alone.\n\n"
            "Answer on one line in exactly this format: <dataset name> | <source> | <confidence 0-100>")


def header_prompt(t: dict):
    cols = list(t["train"].columns)
    m = min(N_COMP, len(cols) - 3)
    if m < 3:
        return None, None, None
    truth = cols[3:3 + m]
    guessable = sum(bool(re.fullmatch(r"\D*\d+", c.strip())) for c in truth) / m >= 0.8
    p = (f"A tabular dataset file has {len(cols)} columns. Its first 3 columns, in file order, are:\n"
         f"{', '.join(cols[:3])}\n\n"
         f"List the next {m} column names in file order, one per line, names only. If you recall the "
         "actual dataset, reproduce its real column names; otherwise give your best guess.")
    return p, truth, guessable


def judge_prompt(t: dict, answer: str) -> str:
    m = t["meta"]
    url = m.get("kaggle_url") or m.get("hf_url") or ""
    return (f"Ground-truth dataset: id={t['id']}; name=\"{m.get('display_name')}\"; origin={m.get('origin')}; url={url}.\n"
            "A model was shown only the column names (and possibly a few rows) of a dataset and asked to identify it. "
            f"Its answer:\n\"\"\"{answer}\"\"\"\n\n"
            "Does the answer identify THE SAME dataset (same name, a clear alias, or the same competition or table family)? "
            "A merely similar topic does not count. If the answer is UNKNOWN, reply unknown.\n"
            "Reply with exactly one word: match, partial, wrong, or unknown.")


def ask(model: str, prompt: str) -> str:
    cfg = resolve_model_cfg(model, load_registry())
    cfg.update({"temperature": 0.0, "max_tokens": 4096})
    text, _ = call_llm([{"role": "user", "content": prompt}], cfg)
    return (text or "").strip()


def run_one(job: dict) -> dict:
    try:
        ans = ask(job["model"], job["prompt"])
        out = {**{k: job[k] for k in ("task", "group", "model", "probe")}, "answer": ans}
        if job["probe"].startswith("recog"):
            out["judge"] = re.sub(r"[^a-z]", "", ask(JUDGE, judge_prompt(job["t"], ans)).lower()) or "error"
        else:
            got = [re.sub(r"^[\-\*\d\.\)\s]+", "", l).strip() for l in ans.splitlines() if l.strip()]
            truth = job["truth"]
            out["score"] = sum(g.lower() == t.lower() for g, t in zip(got, truth)) / len(truth)
            out["n_truth"], out["guessable"] = len(truth), job["guessable"]
        return out
    except Exception as e:  # noqa: BLE001
        return {**{k: job[k] for k in ("task", "group", "model", "probe")}, "error": f"{type(e).__name__}: {e}"[:200]}


def build_jobs(tasks: list) -> list:
    jobs = []
    for bucket, tid in tasks:
        t = load_task(bucket, tid)
        hp, truth, guessable = header_prompt(t)
        for model in MODELS:
            base = {"task": tid, "group": t["group"], "model": model, "t": t}
            jobs.append({**base, "probe": "recog_header", "prompt": recog_prompt(t, False)})
            jobs.append({**base, "probe": "recog_rows", "prompt": recog_prompt(t, True)})
            if hp:
                jobs.append({**base, "probe": "header_comp", "prompt": hp, "truth": truth, "guessable": guessable})
    return jobs


def report() -> None:
    rows = [json.loads(l) for l in RAW.read_text().splitlines()] if RAW.exists() else []
    df = pd.DataFrame(rows)
    if df.empty:
        print("no cached results")
        return
    print(f"{len(df)} cached results; errors: {int(df.get('error', pd.Series(dtype=object)).notna().sum())}")
    df = df[df.get("error").isna()] if "error" in df else df
    order = ["Kaggle/MLE", "UCI", "OpenML", "Tier B"]
    for probe in ("recog_header", "recog_rows"):
        d = df[df.probe == probe]
        print(f"\n[{probe}] judge verdict by group x model (n = tasks)")
        print("| Group | Model | n | match | partial | wrong | unknown |\n|---|---|---|---|---|---|---|")
        for g in order:
            for m in MODELS:
                x = d[(d.group == g) & (d.model == m)]
                if len(x):
                    c = x.judge.value_counts()
                    print(f"| {g} | {m.split('/')[-1]} | {len(x)} | " + " | ".join(f"{c.get(k, 0)}" for k in ("match", "partial", "wrong", "unknown")) + " |")
    d = df[(df.probe == "header_comp") & (~df.guessable.astype(bool))]
    print(f"\n[header_comp] mean share of next columns reproduced in exact position (excl. {int(df[(df.probe == 'header_comp')].guessable.astype(bool).sum())} name+number datasets)")
    print("| Group | Model | n | mean | tasks with >=50% |\n|---|---|---|---|---|")
    for g in order:
        for m in MODELS:
            x = d[(d.group == g) & (d.model == m)]
            if len(x):
                print(f"| {g} | {m.split('/')[-1]} | {len(x)} | {x.score.mean():.2f} | {int((x.score >= 0.5).sum())} |")
    print("\nPositive controls and best cases (per task, gemini-3.6 | 3.7):")
    piv = df.pivot_table(index="task", columns=["probe", "model"], values="score", aggfunc="first")
    for tid in POSITIVE:
        r = {}
        for probe in ("recog_header", "recog_rows"):
            for m in MODELS:
                x = df[(df.task == tid) & (df.probe == probe) & (df.model == m)]
                r[f"{probe[-6:]}:{m[-9:-6]}"] = x.judge.iloc[0] if len(x) else "-"
        for m in MODELS:
            x = df[(df.task == tid) & (df.probe == "header_comp") & (df.model == m)]
            r[f"hdr:{m[-9:-6]}"] = f"{x.score.iloc[0]:.2f}" if len(x) else "-"
        print(f"  {tid[:32]:32s} {r}")
    print("\nAll non-'unknown' recognition answers that the judge marked match/partial (task | probe | model | answer):")
    for r in df[df.probe.str.startswith("recog") & df.judge.isin(["match", "partial"])].itertuples():
        print(f"  {r.group:10s} {r.task[:30]:30s} {r.probe:12s} {r.model.split('/')[-1]:16s} {r.judge:7s} {r.answer[:90]!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.report:
        report()
        return
    tasks = pick_tasks()
    jobs = build_jobs(tasks)
    print(f"{len(tasks)} tasks ({', '.join(f'{g}={sum(1 for b, _ in tasks if GROUP[b] == g)}' for g in QUOTA)}), {len(jobs)} probe calls "
          f"(+{sum(j['probe'].startswith('recog') for j in jobs)} judge calls)")
    if a.dry_run:
        seen = set()
        for j in jobs:
            if j["probe"] not in seen and j["task"] in ("titanic", "us-school-bus-fleet") or (j["group"] == "Tier B" and j["probe"] not in seen):
                seen.add(j["probe"])
                print(f"\n----- {j['task']} | {j['probe']} -----\n{j['prompt']}")
            if len(seen) == 3:
                break
        print("\n(dry run: no API calls made)")
        return
    done = set()
    if RAW.exists():
        for l in RAW.read_text().splitlines():
            r = json.loads(l)
            if "error" not in r:
                done.add((r["task"], r["model"], r["probe"]))
    todo = [j for j in jobs if (j["task"], j["model"], j["probe"]) not in done]
    print(f"{len(done)} already cached, running {len(todo)}")
    with ThreadPoolExecutor(a.workers) as ex, RAW.open("a") as f:
        for i, out in enumerate(ex.map(run_one, todo), 1):
            f.write(json.dumps(out) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    report()


if __name__ == "__main__":
    main()
