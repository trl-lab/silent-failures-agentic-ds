Now let's verify sample weights:
The Kaggle metric is Weighted Mean Absolute Error (WMAE):
$$WMAE = \frac{\sum_i w_i |y_i - \hat{y}_i|}{\sum_i w_i}$$
where $w_i = 5$ if week is a holiday, $1$ otherwise.

When training LightGBM, CatBoost, and XGBoost with MAE loss (`objective='regression_l1'` or `objective='mae'` or `loss_function='MAE'`):
Passing `sample_weight = np.where(df['IsHoliday'] == 1, 5.0, 1.0)` instructs the GBDT to optimize weighted MAE directly.

Let's test model selection & hyperparameter tuning for maximum accuracy:
1. **LightGBM**:
   - `objective`: `'regression_l1'` (MAE loss)
   - `metric`: `'mae'`
   - `n_estimators`: 1500
   - `learning_rate`: 0.05
   - `num_leaves`: 127
   - `min_child_samples`: 20
   - `subsample`: 0.8
   - `colsample_bytree`: 0.8
   - `random_state`: 42

2. **CatBoost**:
   - `loss_function`: `'MAE'`
   - `iterations`: 1500
   - `learning_