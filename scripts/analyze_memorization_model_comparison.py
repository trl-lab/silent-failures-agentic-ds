"""Paired comparison of column-completion (header) memorization exposure between two probed models,
via scipy's Wilcoxon signed-rank test -- the nonparametric paired test appropriate here since
strict/relaxed scores are bounded in [0,1] and not normally distributed (heavy mass at 0.0 and
1.0), matching the test used by
analyze_memorization_delta.py.

Usage: python scripts/analyze_memorization_model_comparison.py --model-a gemini-3.6-flash --model-b gemini-3.7-flash
Output: results/memorization_model_comparison_<a>_vs_<b>.csv, printed summary + Wilcoxon test
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
PROBE_DIR = ROOT / "analysis" / "exposure_probe"


def load(model: str) -> pd.DataFrame:
    path = PROBE_DIR / f"header_full_{model}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run scripts/probe_header_full.py --model gemini/{model} first")
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    if "error" in df.columns:
        n_err = df["error"].notna().sum()
        if n_err:
            print(f"  {model}: {n_err} tasks with a probe error, dropped")
            df = df[df["error"].isna()]
    return df[["task", "strict", "relaxed", "guessable"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-a", default="gemini-3.6-flash")
    ap.add_argument("--model-b", default="gemini-3.7-flash")
    args = ap.parse_args()

    a_name, b_name = args.model_a, args.model_b
    a = load(a_name).rename(columns={"strict": "strict_a", "relaxed": "relaxed_a", "guessable": "guessable_a"})
    b = load(b_name).rename(columns={"strict": "strict_b", "relaxed": "relaxed_b", "guessable": "guessable_b"})

    merged = a.merge(b, on="task", how="inner")
    n_a_only = len(set(a["task"]) - set(b["task"]))
    n_b_only = len(set(b["task"]) - set(a["task"]))
    if n_a_only or n_b_only:
        print(f"  {n_a_only} tasks only in {a_name}, {n_b_only} only in {b_name} -- excluded from the paired test")

    # guessable should agree between models (it's a property of the task's column names,
    # not the model) -- flag if it doesn't rather than silently picking one side's value.
    disagree = merged["guessable_a"] != merged["guessable_b"]
    if disagree.any():
        print(f"  WARNING: {disagree.sum()} tasks have mismatched 'guessable' flags between models "
              f"(should be task-only, not model-dependent) -- treating as guessable if EITHER says so")
    merged["guessable"] = merged["guessable_a"] | merged["guessable_b"]

    n_total = len(merged)
    n_guessable = int(merged["guessable"].sum())
    scored = merged[~merged["guessable"]].copy()
    n_scored = len(scored)
    print(f"\n{n_total} matched tasks, {n_guessable} excluded as trivially guessable (name+number), "
          f"{n_scored} scored\n")

    results = []
    for metric in ("strict", "relaxed"):
        col_a, col_b = f"{metric}_a", f"{metric}_b"
        diffs = scored[col_a] - scored[col_b]
        n_nonzero = int((diffs != 0).sum())
        mean_a, mean_b = scored[col_a].mean(), scored[col_b].mean()
        median_a, median_b = scored[col_a].median(), scored[col_b].median()
        if n_nonzero == 0:
            stat, p = float("nan"), float("nan")
            note = "all differences are zero -- Wilcoxon undefined"
        else:
            stat, p = stats.wilcoxon(scored[col_a], scored[col_b], zero_method="wilcox")
            note = ""
        direction = a_name if mean_a > mean_b else (b_name if mean_b > mean_a else "tie")
        results.append({
            "metric": metric, "n": n_scored, "n_nonzero_diff": n_nonzero,
            f"mean_{a_name}": round(mean_a, 4), f"mean_{b_name}": round(mean_b, 4),
            f"median_{a_name}": round(median_a, 4), f"median_{b_name}": round(median_b, 4),
            "higher_exposure_model": direction,
            "wilcoxon_stat": round(stat, 2) if stat == stat else stat,
            "p_value": round(p, 5) if p == p else p,
            "significant_at_0.05": bool(p < 0.05) if p == p else False,
            "note": note,
        })
        print(f"[{metric}] n={n_scored} ({n_nonzero} nonzero diffs) | "
              f"mean {a_name}={mean_a:.4f} vs {b_name}={mean_b:.4f} | "
              f"median {a_name}={median_a:.4f} vs {b_name}={median_b:.4f} | "
              f"Wilcoxon W={stat if stat == stat else 'n/a'}, p={p if p == p else 'n/a'} "
              f"({'SIGNIFICANT' if (p == p and p < 0.05) else 'not significant'} at alpha=0.05) {note}")

    out = ROOT / "results" / f"memorization_model_comparison_{a_name}_vs_{b_name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
