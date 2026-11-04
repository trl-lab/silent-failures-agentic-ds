"""fetch_kaggle_notebooks.py

Output structure:
    analysis/kaggle_notebooks/
        <competition>/
            metadata.json          # top-N kernel metadata + leaderboard top-K
            <kernel-ref>/          # pulled notebook files
            ...

Usage:
    python scripts/fetch_kaggle_notebooks.py
    python scripts/fetch_kaggle_notebooks.py --competition ieee-fraud-detection --top-n 5
    python scripts/fetch_kaggle_notebooks.py --results results/tables_zero_shot_vs_few_shot_vs_few_shot.csv
"""

import argparse
import csv
import json
import time
from pathlib import Path

import requests
from kaggle import KaggleApi

RESULTS_CSV = Path(__file__).parent.parent / "results" / "tables_zero_shot_vs_few_shot_vs_few_shot.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "analysis" / "kaggle_notebooks"

TOP_N_KERNELS = 1    # notebooks to download per competition
TOP_K_LB = 10        # leaderboard rows to record in metadata
SLEEP_BETWEEN = 2    # seconds between API calls (rate limiting)

KAGGLE_API_BASE = "https://www.kaggle.com/api/v1"


def get_competitions_from_csv(csv_path: Path) -> dict[str, float]:
    """Return {competition_slug: lb_top1_score} for Kaggle tasks with matched LB."""
    seen = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if (row["source"] == "kaggle"
                    and row["metric_match"] == "True"
                    and row["lb_top1"]
                    and row["task_id"] not in seen):
                seen[row["task_id"]] = {
                    "lb_top1": float(row["lb_top1"]),
                    "metric": row["metric"],
                    "higher_is_better": row["higher"] == "True",
                    "medal": row.get("medal", ""),
                }
    return seen


def fetch_leaderboard(api: KaggleApi, competition: str, top_k: int) -> list[dict]:
    try:
        entries = api.competition_leaderboard_view(competition)
        rows = []
        for e in entries[:top_k]:
            rows.append({
                "team_name": e._team_name,
                "score": str(e._score),
                "submission_date": str(e._submission_date),
            })
        return rows
    except Exception as ex:
        print(f"  [warn] leaderboard unavailable for {competition}: {ex}")
        return []


def fetch_kernels(api: KaggleApi, competition: str, top_n: int) -> list[dict]:
    """Fetch top-N notebooks sorted by best public submission score, falling back to vote count."""
    creds = json.load(open(Path.home() / ".kaggle" / "kaggle.json"))
    auth = (creds["username"], creds["key"])

    for sort_by in ("scoreDescending", "voteCount"):
        try:
            r = requests.get(
                f"{KAGGLE_API_BASE}/kernels/list",
                params={"competition": competition, "sortBy": sort_by, "pageSize": top_n},
                auth=auth,
                timeout=15,
            )
            r.raise_for_status()
            kernels = r.json()
            if not kernels:
                continue
            result = []
            for k in kernels:
                result.append({
                    "ref": k["ref"],
                    "title": k["title"],
                    "author": k["author"],
                    "total_votes": k["totalVotes"],
                    "last_run_time": k.get("lastRunTime", ""),
                    "sort_by": sort_by,
                })
            return result
        except Exception as ex:
            print(f"  [warn] kernel list ({sort_by}) failed for {competition}: {ex}")

    return []


def pull_kernel(api: KaggleApi, kernel_ref: str, output_dir: Path) -> bool:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        api.kernels_pull(kernel_ref, path=str(output_dir), metadata=True, quiet=True)
        return True
    except Exception as ex:
        print(f"    [warn] could not pull {kernel_ref}: {ex}")
        return False


def process_competition(
    api: KaggleApi,
    competition: str,
    lb_info: dict,
    out_root: Path,
    top_n: int,
    top_k: int,
):
    comp_dir = out_root / competition
    meta_path = comp_dir / "metadata.json"

    if meta_path.exists():
        print(f"  [skip] {competition} — already downloaded")
        return

    comp_dir.mkdir(parents=True, exist_ok=True)
    print(f"  {competition}")

    leaderboard = fetch_leaderboard(api, competition, top_k)
    time.sleep(SLEEP_BETWEEN)

    kernels = fetch_kernels(api, competition, top_n)
    time.sleep(SLEEP_BETWEEN)

    pulled = []
    for k in kernels:
        ref = k["ref"]
        kernel_dir = comp_dir / ref.replace("/", "__")
        print(f"    pulling {ref} (sorted by {k['sort_by']}, {k['total_votes']} votes) …")
        ok = pull_kernel(api, ref, kernel_dir)
        pulled.append({**k, "downloaded": ok})
        time.sleep(SLEEP_BETWEEN)

    metadata = {
        "competition": competition,
        "lb_top1": lb_info["lb_top1"],
        "metric": lb_info["metric"],
        "higher_is_better": lb_info["higher_is_better"],
        "medal": lb_info["medal"],
        "leaderboard_top": leaderboard,
        "notebooks": pulled,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"    saved metadata → {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Download top Kaggle notebooks per competition.")
    parser.add_argument("--results", default=str(RESULTS_CSV), help="Path to results CSV")
    parser.add_argument("--competition", default=None, help="Run for a single competition slug")
    parser.add_argument("--top-n", type=int, default=TOP_N_KERNELS, help="Notebooks per competition")
    parser.add_argument("--top-k-lb", type=int, default=TOP_K_LB, help="Leaderboard rows to record")
    parser.add_argument("--output", default=str(OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()

    api = KaggleApi()
    api.authenticate()


    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.competition:
        competitions = {args.competition: {"lb_top1": None, "metric": "", "higher_is_better": True, "medal": ""}}
    else:
        competitions = get_competitions_from_csv(Path(args.results))
        print(f"Found {len(competitions)} competitions with matched LB scores.\n")

    for i, (comp, lb_info) in enumerate(competitions.items(), 1):
        print(f"[{i}/{len(competitions)}]", end=" ")
        process_competition(api, comp, lb_info, out_root, args.top_n, args.top_k_lb)

    print(f"\nDone. Notebooks saved to {out_root}/")


if __name__ == "__main__":
    main()
