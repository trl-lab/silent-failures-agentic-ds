# Set 3 — Human Solution Comparison (HSC)

For each task: compare agent-generated code against top Kaggle notebooks to measure the optimality gap and identify missing best practices.

## Tasks

| # | Task | Complexity | Type | Metric | Agent run | LB open | Notebooks |
|---|------|-----------|------|--------|-----------|---------|-----------|
| 1 | store-sales-time-series-forecasting | standard | TS regression | RMSLE | ✓ | ✓ | ✓ fetched |
| 2 | house-prices-advanced-regression-techniques | standard | regression | RMSLE | ✓ | ✓ | ✓ fetched |
| 3 | ashrae-energy-prediction | standard | TS regression | RMSLE | ✓ | ✗ | ✓ fetched |
| 4 | ieee-fraud-detection | standard | binary classification | ROC AUC | ✓ | ✗ | ✓ fetched |
| 5 | equity-post-HCT-survival-predictions | **high** | survival analysis | C-index (ROC AUC proxy) | ✗ | ✗ | ✓ fetched (3rd + 4th place) |
| 6 | nfl-big-data-bowl-2026-prediction | **high** | trajectory prediction | MAE | ✗ | ✗ | ✓ fetched (qualitative only) |

## Notes

- **equity**: data ready (23k train / 5.76k test split). Agent run needed. Pipeline uses ROC AUC; true metric is C-index.
- **nfl**: trajectory prediction — incompatible with our predictions.csv pipeline. Qualitative notebook comparison only.

## Folder structure

```
hsc/
├── hsc_tasks.json          # task registry with metadata
├── notebooks/
│   ├── store-sales-time-series-forecasting/   # downloaded
│   ├── house-prices-advanced-regression-techniques/
│   ├── ashrae-energy-prediction/
│   ├── ieee-fraud-detection/
│   └── mitsui-commodity-prediction-challenge/
└── README.md
```

## Notes on mitsui

Mitsui uses a **non-standard Kaggle Evaluation Gateway** (gRPC). The agent cannot simply produce `predictions.csv` — it must serve responses to a streaming API. For HSC purposes this task is **qualitative only**: pull the 3rd-place notebook (AYUSH KHAIRE), compare approaches, identify what our agent would likely miss. Numeric score comparison is deferred until a gateway adapter is built.

## Next steps

1. Run `fetch_kaggle_notebooks.py` for the 4 pending tasks
2. For each notebook: identify 2–3 practices that are *necessary* (removing them breaks score) vs *present but not required*
3. Run `hsc_comparison.ipynb` per task to compare agent trace vs expert notebook
