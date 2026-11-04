_test_preds}
)

output.to_csv("predictions.csv", index=False)
```

Let's check if there are any subtle bugs or warnings in this code:
1. `X_tr[col].cat.codes`: pandas `Series.cat.codes` returns integer codes (-1 for missing, 0, 1, 2... for categories). Converting to `int` gives integer array. Safe for XGBoost / CatBoost.
2. `callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]` works in lightgbm >= 3.0.
3. `xgb.XGBRegressor` with `early_stopping_rounds=100` and `eval_set=[(X_va_num, y_va)]` works in modern xgboost.
4. `cb.CatBoostRegressor` with `early_stopping_rounds=100` and `eval_set=(X_va_num, y_va)` works in catboost.
5. `output.to_csv("predictions.csv", index=False)` writes two columns `id` and `capacity_mw`.
6. Requirement check:
   - Save predictions to `predictions.csv` in current working directory
   - Exactly two columns: "id" (integer row index starting from 0) and "capacity_mw"
   - Output only the Python code, no explanation or markdown fences.

Let's double-check all requirements and ensure clean code