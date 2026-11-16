to `model_xgb.fit(...)` works cleanly.
Yes! `model_xgb.fit(X_tr_xgb, y_tr, eval_set=[(X_va_xgb, y_va)], early_stopping_rounds=50, verbose=False)`.

Let's check if `xgboost` handles missing values in numerical columns automatically:
Yes, XGBoost natively handles `np.nan` in numerical columns.
Does CatBoost handle `np.nan` in numerical columns?
Yes, CatBoost handles NaN in numerical columns.
Does LightGBM handle `np.nan` in numerical columns?
Yes, LightGBM handles NaN in numerical columns natively.

Wait, what about `cat_cols` with `np.nan`?
We already replaced `NaN` with `'missing'` string for categorical columns before setting dtypes/encoding, so categoricals have no NaNs.

Let's double check index alignment and IDs:
Output file must be `predictions.csv` with exactly two columns: `id` and `CAPACITY`.
`id` is integer row index starting from 0: `np.arange(len(final_test_preds))`.
`CAPACITY` is the predicted capacity values.

Let's check requirement list:
- Runs without any user input or internet access
- Save predictions to `predictions.csv` in current working directory
- Output ONLY the Python code, no explanation or markdown fences.

Let's review the required output format:
"Output