"""
FM02 Player Rep Norm — Metric Calculator

Reads FM02_player_rep_norm_eval.csv and computes three metrics:
  1. Score              — average data-handling adoption rate (FM02_02/03/04 successful runs only)
  2. Hint Effectiveness — hint compliance rate (FM02_03 + FM02_04 successful runs only)
  3. Execution Error Rate — fraction of all runs where pipeline_failed=True

Usage:
  Single-file mode (print results to stdout only):
    python scripts/fm02_metric.py --csv llmjudge/judge_results/GPT4oDsstar/FM02_player_rep_norm_eval.csv

  Batch mode (scan all subdirs under llmjudge/judge_results, write to table.csv):
    python scripts/fm02_metric.py
    python scripts/fm02_metric.py --judge-root llmjudge/judge_results --out llmjudge/table.csv

  No-hint-only mode (scan all subdirs, compute FM02_02 no-hint score via score_per_run(), write to llmjudge/table.csv):
    python scripts/fm02_metric.py --nohint

  Output behaviour (batch mode):
    APPENDS to llmjudge/table.csv — run fm01_metric.py first to create/reset the file.
"""

import argparse
import csv
from pathlib import Path


def parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def load_csv(path: str) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def get_variant(task_id: str) -> str:
    """Extract variant (01/02/03/04) from task_id.

    Handles both separator styles:
      FM02_player_rep_norm_01_20260611_...  (DSstar)
      FM02_player_rep_norm_01-20260611_...  (dsguru)
    """
    parts = task_id.split("_")
    # parts: ['FM02', 'player', 'rep', 'norm', '01[...]', ...]
    return parts[4][:2]


def score_per_run(row: dict) -> float:
    """
    Score = 1/2 * handles_na_string_in_international_caps
           + 1/2 * handles_unit_suffix_in_height
    """
    na   = parse_bool(row["handles_na_string_in_international_caps"])
    unit = parse_bool(row["handles_unit_suffix_in_height"])
    return 0.5 * na + 0.5 * unit


def hint_score_per_run(row: dict, variant: str) -> float | None:
    """
    FM02_01: clean data baseline — not applicable
    FM02_02: no hint              — not applicable
    FM02_03: hint = "some column may contain a unit suffix"
             → score = 1 if handles_unit_suffix_in_height
    FM02_04: hint = strip '185cm' AND treat 'na' as missing
             → score = 1/2 each
    """
    if variant in ("01", "02"):
        return None

    na   = parse_bool(row["handles_na_string_in_international_caps"])
    unit = parse_bool(row["handles_unit_suffix_in_height"])

    if variant == "03":
        return float(unit)

    if variant == "04":
        return 0.5 * na + 0.5 * unit

    return None


def compute_metrics(rows: list[dict]) -> dict:
    # ── overall_score + hint_gain: run-weighted (task02:03:04 = 7:2:1) ──
    # task01 (clean data) is excluded — behavioral indicators not applicable.
    # pipeline_failed=True → score=0 (penalised, not excluded).
    by_variant: dict[str, list[dict]] = {"02": [], "03": [], "04": []}
    for row in rows:
        v = get_variant(row["task_id"])
        if v in by_variant:
            by_variant[v].append(row)
    for lst in by_variant.values():
        lst.sort(key=lambda r: r["task_id"])

    TASK_WEIGHTS = {"02": 0.7, "03": 0.2, "04": 0.1}
    n_runs = max((len(lst) for lst in by_variant.values()), default=0)
    run_scores             = []
    hint_gain_strong_vals  = []   # score_04 - score_02
    hint_gain_mild_vals    = []   # score_03 - score_02
    for i in range(n_runs):
        task_scores: dict[str, float] = {}
        for v in ("02", "03", "04"):
            lst = by_variant[v]
            if i < len(lst):
                row = lst[i]
                task_scores[v] = 0.0 if parse_bool(row["pipeline_failed"]) else score_per_run(row)
        run_score = sum(TASK_WEIGHTS[v] * task_scores[v] for v in task_scores)
        run_scores.append(run_score)
        if "02" in task_scores and "04" in task_scores:
            hint_gain_strong_vals.append(task_scores["04"] - task_scores["02"])
        if "02" in task_scores and "03" in task_scores:
            hint_gain_mild_vals.append(task_scores["03"] - task_scores["02"])

    overall_score    = sum(run_scores) / len(run_scores) if run_scores else 0.0
    hint_gain_strong = sum(hint_gain_strong_vals) / len(hint_gain_strong_vals) if hint_gain_strong_vals else 0.0
    hint_gain_mild   = sum(hint_gain_mild_vals)   / len(hint_gain_mild_vals)   if hint_gain_mild_vals   else 0.0

    # ── execution_error_rate (all 4 variants) ────────────────────────
    per_variant_failed = {"01": 0, "02": 0, "03": 0, "04": 0}
    per_variant_total  = {"01": 0, "02": 0, "03": 0, "04": 0}
    score_variants     = {"02": [], "03": [], "04": []}
    total_runs  = len(rows)
    failed_runs = 0

    for row in rows:
        variant = get_variant(row["task_id"])
        failed  = parse_bool(row["pipeline_failed"])
        per_variant_total[variant] += 1
        if failed:
            failed_runs += 1
            per_variant_failed[variant] += 1
            continue
        if variant in score_variants:
            score_variants[variant].append(score_per_run(row))

    error_rate = failed_runs / total_runs if total_runs else 0

    return {
        "overall_score":        overall_score,
        "hint_gain_strong":     hint_gain_strong,
        "hint_gain_mild":       hint_gain_mild,
        "execution_error_rate": error_rate,
        "per_variant_score":    {v: (sum(s)/len(s) if s else None) for v, s in score_variants.items()},
        "per_variant_error":    {v: per_variant_failed[v]/per_variant_total[v] for v in per_variant_total},
        "n_score":  sum(len(s) for s in score_variants.values()),
        "n_total":  total_runs,
        "n_failed": failed_runs,
    }


def print_results(m: dict) -> None:
    print("=" * 52)
    print("FM02 Player Rep Norm — Metrics")
    print("=" * 52)

    print(f"\n[1] Score  (n_success={m['n_score']}, FM02_02/03/04 only)")
    for v in ("02", "03", "04"):
        val = m["per_variant_score"][v]
        label = f"{val:.3f}" if val is not None else "—"
        print(f"    FM02_{v}: {label}")
    print(f"    Overall: {m['overall_score']:.3f}")

    print(f"\n[2] Hint Gain  (avg score delta per run, baseline=FM02_02)")
    print(f"    hint_gain_strong (task04 - task02): {m['hint_gain_strong']:.3f}")
    print(f"    hint_gain_mild   (task03 - task02): {m['hint_gain_mild']:.3f}")

    print(f"\n[3] Execution Error Rate  (n_total={m['n_total']}, n_failed={m['n_failed']})")
    for v in ("01", "02", "03", "04"):
        r = m["per_variant_error"][v]
        print(f"    FM02_{v}: {r:.1%}")
    print(f"    Overall: {m['execution_error_rate']:.1%}")

    print("=" * 52)


_AGENT_SUFFIXES = ("Dsstar", "FewShot", "OneShot")


def parse_dir_name(dirname: str) -> tuple[str, str]:
    """Split 'GPT4oOneShot' → (llm='GPT4o', agent='OneShot')."""
    for suffix in _AGENT_SUFFIXES:
        if dirname.endswith(suffix):
            return dirname[: -len(suffix)], suffix
    return dirname, "Unknown"


def batch_main(judge_root: str, out_csv: str) -> None:
    root = Path(judge_root)
    results = []

    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        csv_path = subdir / "FM02_player_rep_norm_eval.csv"
        if not csv_path.exists():
            continue

        llm, agent = parse_dir_name(subdir.name)
        rows = load_csv(str(csv_path))
        m = compute_metrics(rows)

        pv_score = m["per_variant_score"]
        pv_error = m["per_variant_error"]
        results.append({
            "failure_mode": "fm02",
            "agent": agent,
            "llm": llm,
            "overall_score": round(m["overall_score"], 3),
            "hint_gain_strong": round(m["hint_gain_strong"], 3),
            "hint_gain_mild": round(m["hint_gain_mild"], 3),
            "execution_error_rate": round(m["execution_error_rate"], 3),
            "task2_score": round(pv_score["02"], 3) if pv_score["02"] is not None else "",
            "task3_score": round(pv_score["03"], 3) if pv_score["03"] is not None else "",
            "task4_score": round(pv_score["04"], 3) if pv_score["04"] is not None else "",
            "task1_exe_error_rate": round(pv_error["01"], 3),
            "task2_exe_error_rate": round(pv_error["02"], 3),
            "task3_exe_error_rate": round(pv_error["03"], 3),
            "task4_exe_error_rate": round(pv_error["04"], 3),
        })

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["failure_mode", "agent", "llm", "overall_score",
                        "hint_gain_strong", "hint_gain_mild", "execution_error_rate",
                        "task2_score", "task3_score", "task4_score",
                        "task1_exe_error_rate", "task2_exe_error_rate",
                        "task3_exe_error_rate", "task4_exe_error_rate"],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Appended {len(results)} rows to {out_path}")
    for r in results:
        print(f"  fm02 | {r['agent']:10s} | {r['llm']:15s} | "
              f"score={r['overall_score']:.3f}  "
              f"gain_strong={r['hint_gain_strong']:.3f}  gain_mild={r['hint_gain_mild']:.3f}  "
              f"err={r['execution_error_rate']:.3f}")


def compute_nohint_score(rows: list[dict]) -> float:
    """
    Average score_per_run() over FM02_02 (no-hint) runs only.
    Failed runs contribute a score of 0, matching the convention used for
    overall_score in compute_metrics().
    """
    scores = []
    for row in rows:
        if get_variant(row["task_id"]) != "02":
            continue
        s = 0.0 if parse_bool(row["pipeline_failed"]) else score_per_run(row)
        scores.append(s)
    return sum(scores) / len(scores) if scores else 0.0


def nohint_main(judge_root: str, out_csv: str) -> None:
    root = Path(judge_root)
    results = []

    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        csv_path = subdir / "FM02_player_rep_norm_eval.csv"
        if not csv_path.exists():
            continue

        llm, agent = parse_dir_name(subdir.name)
        rows = load_csv(str(csv_path))
        nohint_score = compute_nohint_score(rows)

        results.append({
            "failure_mode": "fm02",
            "agent": agent,
            "llm": llm,
            "nohint_score": round(nohint_score, 3),
        })

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["failure_mode", "agent", "llm", "nohint_score"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote {len(results)} rows to {out_path}")
    for r in results:
        print(f"  fm02 | {r['agent']:10s} | {r['llm']:15s} | nohint_score={r['nohint_score']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Compute FM02 metrics from eval CSV")
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to a single FM02_player_rep_norm_eval.csv (omit for batch mode)",
    )
    parser.add_argument(
        "--judge-root",
        default="llmjudge/judge_results",
        help="Root directory containing per-configuration subdirectories (batch mode)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path (default: llmjudge/table.csv)",
    )
    parser.add_argument(
        "--nohint",
        action="store_true",
        help="Only compute the FM02_02 no-hint score (via score_per_run()) across all configs, output llmjudge/table.csv",
    )
    args = parser.parse_args()

    if args.csv:
        path = Path(args.csv)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        rows = load_csv(str(path))
        metrics = compute_metrics(rows)
        print_results(metrics)
    elif args.nohint:
        nohint_main(args.judge_root, args.out or "llmjudge/table.csv")
    else:
        batch_main(args.judge_root, args.out or "llmjudge/table.csv")


if __name__ == "__main__":
    main()