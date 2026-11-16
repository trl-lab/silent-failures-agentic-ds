#!/usr/bin/env python3
"""Fetch and prepare the predict-energy-behavior-of-prosumers (Enefit) task as a multi-table dataset,
same rationale as fetch_home_credit.py.

Output structure:
  datasets/kaggle/predict-energy-behavior-of-prosumers/
    data/
      train/train.csv                 <80% of train.csv rows, with target>
      test/test.csv                   <20%, target dropped, sequential "id">
      client.csv                      <shared reference tables, unfiltered>
      electricity_prices.csv
      forecast_weather.csv
      historical_weather.csv
      gas_prices.csv
      weather_station_to_county_mapping.csv
    evaluation/
      labels.parquet   # id, target
      metric.py
    task.json
    description.md
    sample_submission.csv

Usage:
  python scripts/fetch_energy_behavior.py
"""

import json
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

COMP_ID = "predict-energy-behavior-of-prosumers"
RANDOM_STATE = 42
TEST_SIZE = 0.20

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASK_DIR = PROJECT_ROOT / "datasets" / "kaggle" / COMP_ID

REFERENCE_FILES = [
    "client.csv",
    "electricity_prices.csv",
    "forecast_weather.csv",
    "historical_weather.csv",
    "gas_prices.csv",
    "weather_station_to_county_mapping.csv",
]

METRIC_MAE = """\
from sklearn.metrics import mean_absolute_error

def evaluate(y_true, y_pred):
    score = float(mean_absolute_error(y_true, y_pred))
    return {"score": score, "metric": "mae", "higher_is_better": False}
"""


def download_file(api, remote_name: str, dest_dir: Path, max_retries: int = 3) -> Path:
    """Download one file, unzipping it first if Kaggle wrapped it despite
    keeping the original extension. Retries on transient failures where the
    API call reports success but doesn't actually write the file."""
    out_path = dest_dir / remote_name
    last_err = None
    for attempt in range(1, max_retries + 1):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            try:
                api.competition_download_file(COMP_ID, remote_name, path=tmp, quiet=True)
                dpath = tmp / remote_name
                if not dpath.exists():
                    raise FileNotFoundError(
                        f"Kaggle API reported success but {remote_name} was not "
                        f"written to {tmp}"
                    )
                if zipfile.is_zipfile(dpath):
                    with zipfile.ZipFile(dpath) as z:
                        inner = z.namelist()[0]
                        with z.open(inner) as src, open(out_path, "wb") as dst:
                            dst.write(src.read())
                else:
                    dpath.replace(out_path)
                return out_path
            except Exception as exc:
                last_err = exc
                print(f"    [retry {attempt}/{max_retries}] {remote_name} failed: {exc}")
    raise RuntimeError(f"Could not download {remote_name} after {max_retries} attempts") from last_err


def main():
    from kaggle import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print("Kaggle API authenticated.\n")

    (TASK_DIR / "data" / "train").mkdir(parents=True, exist_ok=True)
    (TASK_DIR / "data" / "test").mkdir(parents=True, exist_ok=True)
    (TASK_DIR / "evaluation").mkdir(parents=True, exist_ok=True)

    raw_dir = TASK_DIR / "_raw_download"
    raw_dir.mkdir(exist_ok=True)

    all_files = ["train.csv"] + REFERENCE_FILES
    local_paths = {}
    for i, fname in enumerate(all_files, 1):
        local = raw_dir / fname
        if local.exists():
            print(f"  [{i}/{len(all_files)}] {fname} — already downloaded, skipping")
        else:
            print(f"  [{i}/{len(all_files)}] downloading {fname} ...")
            download_file(api, fname, raw_dir)
        local_paths[fname] = local

    # --- Split train.csv rows 80/20 (regression: plain random split) ---
    train_df = pd.read_csv(local_paths["train.csv"], low_memory=False)
    print(f"\ntrain.csv: {train_df.shape}")

    before = len(train_df)
    train_df = train_df.dropna(subset=["target"]).reset_index(drop=True)
    if len(train_df) < before:
        print(f"  dropped {before - len(train_df)} rows with missing target")

    train_idx, test_idx = train_test_split(
        train_df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )
    train_split = train_df.loc[train_idx].reset_index(drop=True)
    test_split = train_df.loc[test_idx].reset_index(drop=True)

    train_split.to_csv(TASK_DIR / "data" / "train" / "train.csv", index=False)

    test_out = test_split.drop(columns=["target"]).copy()
    test_out.insert(0, "id", range(len(test_out)))
    test_out.to_csv(TASK_DIR / "data" / "test" / "test.csv", index=False)

    print(f"train split: {len(train_split)}  test split: {len(test_split)}")

    # --- Copy reference tables as-is (shared, not split — no target leakage) ---
    for fname in REFERENCE_FILES:
        src = local_paths[fname]
        dst = TASK_DIR / "data" / fname
        dst.write_bytes(src.read_bytes())
        print(f"  copied {fname} ({dst.stat().st_size / 1e6:.1f} MB)")

    # --- evaluation/labels.parquet, aligned to test_out's "id" order ---
    import pyarrow as pa
    import pyarrow.parquet as pq

    labels_df = pd.DataFrame({"id": range(len(test_split)), "target": test_split["target"].values})
    pq.write_table(pa.Table.from_pandas(labels_df, preserve_index=False),
                    TASK_DIR / "evaluation" / "labels.parquet")

    (TASK_DIR / "evaluation" / "metric.py").write_text(METRIC_MAE, encoding="utf-8")

    pd.DataFrame({"id": range(len(test_split)), "target": [0.0] * len(test_split)}).to_csv(
        TASK_DIR / "sample_submission.csv", index=False
    )

    # --- task.json ---
    task_meta = {
        "id": COMP_ID,
        "display_name": "Enefit — Predict Energy Behavior of Prosumers",
        "source_paper": "new",
        "origin": "kaggle",
        "kaggle_url": f"https://www.kaggle.com/competitions/{COMP_ID}",
        "task_type": "regression",
        "target_column": "target",
        "metric": "mae",
        "metric_higher_is_better": False,
        "n_train": len(train_split),
        "n_test": len(test_split),
        "domain": "energy",
        "modality": "tab",
        "kaggle_category": "Featured",
        "multi_table": True,
        "tables": ["train.csv (split)", "test.csv (split)"] + REFERENCE_FILES,
        "notes": (
            "Multi-table task: train.csv/test.csv (one row per "
            "county/is_business/product_type/is_consumption/datetime prediction "
            "target) only has 9 columns — no weather or price signal. Real "
            "features require joining the reference tables (client, "
            "electricity_prices, forecast_weather, historical_weather, "
            "gas_prices, weather_station_to_county_mapping — the latter two "
            "needed to map weather stations to counties). Reference tables are "
            "NOT split by our 80/20 holdout since they carry no per-row target "
            "information; only train.csv/test.csv are. Agent must join them "
            "itself on county/is_business/product_type/data_block_id/datetime "
            "as appropriate."
        ),
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "strategy": "random (regression, no stratification)",
        },
    }
    (TASK_DIR / "task.json").write_text(json.dumps(task_meta, indent=2), encoding="utf-8")

    (TASK_DIR / "description.md").write_text(
        "# Enefit — Predict Energy Behavior of Prosumers\n\n"
        "Predict hourly energy production/consumption (`target`, MWh) for Estonian "
        "prosumers. Multi-table dataset: `train.csv`/`test.csv` only carries "
        "`county`, `is_business`, `product_type`, `is_consumption`, `datetime`, "
        "`data_block_id`, `row_id`, `prediction_unit_id` — no weather or price "
        "data. Separate reference tables (`client.csv`, `electricity_prices.csv`, "
        "`forecast_weather.csv`, `historical_weather.csv`, `gas_prices.csv`, "
        "`weather_station_to_county_mapping.csv`) must be joined yourself to get "
        "the real predictive signal.\n\n"
        "**Evaluation metric:** mae\n\n"
        f"*Source: Kaggle competition — {COMP_ID}*\n",
        encoding="utf-8",
    )

    print(f"\nDone. Task dir: {TASK_DIR}")
    print(f"Raw downloaded files kept at: {raw_dir} (safe to delete once satisfied)")


if __name__ == "__main__":
    main()
