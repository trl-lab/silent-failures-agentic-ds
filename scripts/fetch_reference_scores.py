#!/usr/bin/env python3
"""Fetch reference leaderboard scores for all benchmark tasks and write a `leaderboard` field into
each task.json.

Usage
-----
  python scripts/fetch_reference_scores.py                # all sources
  python scripts/fetch_reference_scores.py --source mlebench openml
  python scripts/fetch_reference_scores.py --task champs-scalar-coupling
  python scripts/fetch_reference_scores.py --dry-run
"""

import argparse
import io
import json
import math
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALL_SOURCES  = ["strable", "openml", "uci", "kaggle", "mlebench"]

MLEBENCH_LB_URL = (
    "https://media.githubusercontent.com/media/openai/mle-bench/main"
    "/mlebench/competitions/{slug}/leaderboard.csv"
)
MLEBENCH_CFG_URL = (
    "https://raw.githubusercontent.com/openai/mle-bench/main"
    "/mlebench/competitions/{slug}/config.yaml"
)

# OpenML API metric names for each of our metric identifiers
OPENML_METRIC = {
    "accuracy": "predictive_accuracy",
    "roc_auc":  "area_under_roc_curve",
    "rmse":     "root_mean_squared_error",
}
OPENML_TASK_TYPE_NAME = {
    "b-classification": "SUPERVISED_CLASSIFICATION",
    "m-classification": "SUPERVISED_CLASSIFICATION",
    "regression":       "SUPERVISED_REGRESSION",
}


# ---------------------------------------------------------------------------
# Medal threshold helpers  (Kaggle standard, as used by MLE-bench)
# ---------------------------------------------------------------------------

def _medal_positions(n: int) -> dict:
    """Return 1-indexed leaderboard positions for bronze / silver / gold."""
    if n < 250:
        return dict(bronze=math.ceil(0.40 * n),
                    silver=math.ceil(0.20 * n),
                    gold=math.ceil(0.10 * n))
    elif n < 1000:
        return dict(bronze=100, silver=50,
                    gold=10 + math.ceil(0.002 * n))
    else:
        return dict(bronze=math.ceil(0.10 * n),
                    silver=math.ceil(0.05 * n),
                    gold=10 + math.ceil(0.002 * n))


def _score_at(scores: list, pos: int):
    idx = pos - 1
    return float(scores[idx]) if idx < len(scores) else None


def _lb_entry(scores: list, source: str, note: str, metric: str) -> dict:
    """Build a full leaderboard dict from a best-first sorted score list."""
    n   = len(scores)
    pos = _medal_positions(n)
    return {
        "source":             source,
        "note":               note,
        "leaderboard_metric": metric,
        "top1_score":         float(scores[0]) if scores else None,
        "bronze_threshold":   _score_at(scores, pos["bronze"]),
        "silver_threshold":   _score_at(scores, pos["silver"]),
        "gold_threshold":     _score_at(scores, pos["gold"]),
        "n_teams":            n,
    }


def _null_entry(reason: str = "No public per-dataset leaderboard available") -> dict:
    return {
        "source":             None,
        "note":               reason,
        "leaderboard_metric": None,
        "top1_score":         None,
        "bronze_threshold":   None,
        "silver_threshold":   None,
        "gold_threshold":     None,
        "n_teams":            None,
    }


# ---------------------------------------------------------------------------
# MLE-bench  (leaderboard CSVs on GitHub via media.githubusercontent.com)
# ---------------------------------------------------------------------------

def _fetch_mlebench_metric(slug: str) -> str:
    """Fetch the native grader metric name from MLE-bench config.yaml."""
    try:
        r = requests.get(MLEBENCH_CFG_URL.format(slug=slug), timeout=15)
        r.raise_for_status()
        for line in r.text.splitlines():
            line = line.strip()
            if line.startswith("name:"):
                return line.split("name:", 1)[1].strip()
    except Exception:
        pass
    return "unknown"


def fetch_mlebench(task_id: str) -> dict:
    metric = _fetch_mlebench_metric(task_id)
    url    = MLEBENCH_LB_URL.format(slug=task_id)
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df = df[df["hasScore"] == True].copy()
        # Leaderboard is already in position order (best = index 0)
        scores = df["score"].tolist()
        if not scores:
            return _null_entry("MLE-bench leaderboard empty or no valid scores")
        return _lb_entry(
            scores,
            source="mlebench_github",
            note=(
                "MLE-bench private-leaderboard snapshot "
                "(test split ~10% of original Kaggle train; ours is 20%)"
            ),
            metric=metric,
        )
    except Exception as exc:
        print(f"    WARN MLE-bench fetch failed for {task_id}: {exc}")
        return _null_entry(f"MLE-bench fetch error: {exc}")


# ---------------------------------------------------------------------------
# OpenML  (best known evaluation via the openml Python package)
# ---------------------------------------------------------------------------

def fetch_openml(tj: dict) -> dict:
    try:
        import openml
    except ImportError:
        return _null_entry("openml package not installed")

    did       = tj.get("openml_did")
    task_type = tj.get("task_type", "regression")
    metric    = tj.get("metric", "rmse")
    higher    = tj.get("metric_higher_is_better", True)

    if not did:
        return _null_entry("No openml_did in task.json")

    openml_metric = OPENML_METRIC.get(metric)
    if not openml_metric:
        return _null_entry(f"No OpenML metric mapping for '{metric}'")

    tt_name = OPENML_TASK_TYPE_NAME.get(task_type, "SUPERVISED_REGRESSION")
    tt = getattr(openml.tasks.TaskType, tt_name)

    try:
        # Find tasks that use this dataset
        tasks_df = openml.tasks.list_tasks(
            data_id=did,
            task_type=tt,
            output_format="dataframe",
        )
        if tasks_df is None or tasks_df.empty:
            return _null_entry(f"No OpenML tasks found for did={did}")

        # Pick the task with the most runs (most-evaluated = most representative)
        task_id = int(tasks_df.index[0])
        if "NumberOfInstances" in tasks_df.columns:
            # Proxy: just take first row; OpenML returns by popularity
            pass

        # Fetch evaluations
        evals_df = openml.evaluations.list_evaluations(
            function=openml_metric,
            tasks=[task_id],
            size=200,
            output_format="dataframe",
        )
        if evals_df is None or evals_df.empty:
            return _null_entry(f"No OpenML evaluations for task {task_id}")

        values = [float(v) for v in evals_df["value"].tolist()]
        n_raw = len(values)
        if not higher:
            values = [v for v in values if v > 0]
            if not values:
                return _null_entry(
                    f"All {n_raw} OpenML evaluations for task {task_id} were non-positive "
                    "(degenerate uploads) -- no usable reference"
                )

        best = max(values) if higher else min(values)
        return {
            "source":             "openml_api",
            "note": (
                f"Best known result on OpenML task {task_id} "
                "(cross-validation; may differ from our 80/20 hold-out)"
            ),
            "leaderboard_metric": openml_metric,
            "openml_task_id":     task_id,
            "top1_score":         best,
            # How many evaluations that "best" was drawn from, after dropping
            # degenerate ones. A top1 backed by 1-2 runs is far weaker evidence
            # than one backed by 200 -- carry it so downstream analysis can
            # weight or exclude thin references rather than trusting them all
            # equally.
            "n_evals":            len(values),
            "n_evals_raw":        n_raw,
            "bronze_threshold":   None,
            "silver_threshold":   None,
            "gold_threshold":     None,
            "n_teams":            None,
        }

    except Exception as exc:
        print(f"    WARN OpenML fetch failed for did={did}: {exc}")
        return _null_entry(f"OpenML fetch error: {exc}")


# ---------------------------------------------------------------------------
# Kaggle  (public leaderboard download via kaggle package)
# ---------------------------------------------------------------------------

_kaggle_api = None


def _get_kaggle_api():
    global _kaggle_api
    if _kaggle_api is None:
        from kaggle import KaggleApi
        api = KaggleApi()
        api.authenticate()
        _kaggle_api = api
    return _kaggle_api


def fetch_kaggle(task_id: str, tj: dict = None) -> dict:
    import tempfile

    try:
        api = _get_kaggle_api()
    except Exception as exc:
        return _null_entry(f"Kaggle auth failed: {exc}")

    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            api.competition_leaderboard_download(
                competition=task_id, path=tmp, quiet=True
            )
            # Result may be a zip or a bare CSV
            zips = list(tmp_path.glob("*.zip"))
            csvs = list(tmp_path.glob("*.csv"))

            if zips:
                with zipfile.ZipFile(zips[0]) as zf:
                    name = zf.namelist()[0]
                    df = pd.read_csv(zf.open(name))
            elif csvs:
                df = pd.read_csv(csvs[0])
            else:
                return _null_entry("Kaggle leaderboard download returned no files")

        df.columns = [c.lower() for c in df.columns]
        score_col = next((c for c in df.columns if "score" in c), None)
        if score_col is None:
            return _null_entry("No score column found in Kaggle leaderboard CSV")

        df = df[df[score_col].notna()]
        scores = df[score_col].tolist()
        if not scores:
            return _null_entry("Kaggle leaderboard has no valid scores")

        # Kaggle leaderboard metric matches our task metric in most cases;
        # flag it explicitly so comparison scripts can detect any mismatch.
        task_metric = tj.get("metric", "unknown") if isinstance(tj, dict) else "unknown"
        return _lb_entry(
            scores,
            source="kaggle_api",
            note=(
                "Kaggle public leaderboard "
                "(official Kaggle test set; different from our 80/20 split); "
                "metric assumed to match task metric — verify if scores look off"
            ),
            metric=task_metric,
        )

    except Exception as exc:
        print(f"    WARN Kaggle fetch failed for {task_id}: {exc}")
        return _null_entry(f"Kaggle fetch error: {exc}")


# ---------------------------------------------------------------------------
# Task iteration
# ---------------------------------------------------------------------------

def iter_tasks(source_filter=None, task_filter=None):
    for src in ALL_SOURCES:
        if source_filter and src not in source_filter:
            continue
        src_dir = PROJECT_ROOT / "datasets" / src
        if not src_dir.exists():
            continue
        for task_path in sorted(src_dir.glob("*/task.json")):
            tid = task_path.parent.name
            if task_filter and tid != task_filter:
                continue
            tj = json.loads(task_path.read_text())
            tj["_src"]  = src
            tj["_path"] = task_path
            yield tj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source", nargs="+", choices=ALL_SOURCES,
        help="Limit to specific sources (default: all)",
    )
    parser.add_argument(
        "--task", help="Process a single task by ID",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print results without writing to task.json",
    )
    args = parser.parse_args()

    source_filter = set(args.source) if args.source else None
    n_ok = n_null = n_err = 0

    for tj in iter_tasks(source_filter, args.task):
        src     = tj["_src"]
        tid     = tj["id"]
        path    = tj["_path"]

        print(f"[{src:10s}] {tid}")

        if src in ("strable", "uci"):
            entry = _null_entry()
            n_null += 1

        elif src == "openml":
            entry = fetch_openml(tj)
            time.sleep(0.3)
            n_ok += 1 if entry["source"] else 0
            n_null += 1 if not entry["source"] else 0

        elif src == "mlebench":
            entry = fetch_mlebench(tid)
            time.sleep(0.2)
            n_ok += 1 if entry["source"] else 0
            n_null += 1 if not entry["source"] else 0

        elif src == "kaggle":
            entry = fetch_kaggle(tid, tj)
            time.sleep(0.5)
            n_ok += 1 if entry["source"] else 0
            n_null += 1 if not entry["source"] else 0

        else:
            entry = _null_entry()
            n_null += 1

        score_str = f"{entry['top1_score']:.4g}" if entry["top1_score"] is not None else "N/A"
        print(f"  → top1={score_str}  source={entry['source']}")

        if not args.dry_run:
            clean = {k: v for k, v in tj.items() if not k.startswith("_")}
            clean["leaderboard"] = entry
            path.write_text(json.dumps(clean, indent=2) + "\n")

    print(f"\nDone.  fetched={n_ok}  null={n_null}")


if __name__ == "__main__":
    main()
