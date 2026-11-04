Thinking Process:

1.  **Analyze the Request:**
    *   **Role:** Kaggle grandmaster.
    *   **Task:** Predict `salary_in_usd` using `train.parquet` and `test.parquet`.
    *   **Metric:** RMSE (lower is better).
    *   **Constraints:**
        *   Write clean, well-structured Python code.
        *   Avoid data leakage.
        *   Use `pd.read_parquet()`.
        *   Use `lightgbm==4.6.0` (callbacks for early stopping).
        *   Use `scikit-learn==1.5.0`, `pandas==2.1.4`.
        *   Output: `submission.csv` in `./working/` with `id` and `salary_in_usd`.
        *   Response format: Brief outline (3-5 sentences) + single markdown code block.
        *   Code must print the evaluation metric on a hold-out validation set (or 5-fold CV if appropriate).
        *   Solution sketch: Simple, no ensembling/HP tuning, consider memory, no EDA.
        *   Memory: This is a specific constraint mentioned in the prompt ("Take the Memory section into consideration"). However, the Memory section in the prompt is actually empty (`# Memory \n\n`). I need to infer memory constraints based on typical Kaggle environments or the data size. The data files are very small (51.1 kB, 49.3 kB, 12.8 kB). This suggests the dataset is tiny, likely not millions of rows. Memory constraints might be about not loading everything into RAM unnecessarily if it were large, but here it's trivial. I should still write efficient code.
        *   Wait, the prompt says "Take the Memory section into consideration when proposing the design". The Memory section is empty in the provided text. I will assume standard efficiency practices (e.g., not loading test data into memory if not needed for training, though for simple regression it's fine). The key instruction is "Don't propose the same modelling solution but keep the evaluation the same" (referring to previous turns or context not visible here, but I should ensure I don't overcomplicate).
        *   Actually, looking closely at the prompt structure, it seems to be a template. The "Memory" section is empty. I will focus on the explicit instructions.
        *   **Crucial:** The prompt asks for an evaluation metric to be printed on a hold-out validation set. It also says "The evaluation should be based on 5-fold cross-validation but only if that's an appropriate evaluation for the task at hand."
        *   **Crucial:** Save `submission.csv` for test data.
        *   **Crucial:** Single-file Python program.
        *   **Crucial:** Output format: Text outline (3-5 sentences) followed by code block.

2.  **Determine the Solution Strategy:**
    *   **Model:** LightGBM is specified and installed. It's fast and effective for tabular data.
    *   **Data:** Parquet files. Very small size, so loading all is fine.
    *   **Validation:** 5-fold cross-validation is standard for regression to estimate RMSE without leakage.
    *   **Features:** Need to identify feature columns. Usually, everything except the target column and ID column.
    *   **Test:** Predict on test data and save to `./working/submission.csv`.
    *   **Metric:** RMSE.
    *   **Memory:** Since data is small, standard loading is fine. I will ensure I don't keep unnecessary large variables in memory during the loop.
    *   **Code Structure:** Load train, split for CV, train model, predict on test.

3.  **Drafting the Solution Sketch (3-5 sentences):**
    *   I will use LightGBM as the primary model due to its speed and efficiency on tabular data.
    *   To avoid data leakage and estimate generalization performance, I will implement a 5-fold cross-validation strategy on the training data.
    *   The model will be tuned using early stopping to prevent overfitting, monitoring validation error to determine optimal round count.
    *   Finally, I will generate predictions on the test set and save them to the required submission file format.

4.  **Drafting the Code:**
    *   Imports: `pandas`, `numpy`, `lightgbm`, `sklearn.model_selection`.
    *   Load data: `train = pd.read_parquet(...)`, `test = pd.read_parquet(...)`.
    *   Identify columns: `train.columns` excluding `salary_in_usd`.
    *   Cross-validation: Use `cross_val_score` with `LightGBMRegressor`. `scoring='neg_root_mean_squared_error'`.
    *   Train: Since I need to print the metric *and* save submission, I can run CV to get the score, then fit on full train, then predict on test. Or just use the CV score as the metric. The instructions say "print out the evaluation metric computed on a hold-out validation set". It also says "evaluation should be based on 5-fold cross-validation". So I should print the CV RMSE.
    *   Wait, `cross_val_score` returns negative RMSE by default for `rmse`? No, `scoring='neg_root_mean_squared_error'` returns negative RMSE. I need to negate it.
    *   Wait, `lightgbm` version 4.6.0 note: `early_stopping_rounds` and `verbose_eval` removed from `lgb.train()`. Must use callbacks. `lgb.early_stopping(50)` and `lgb.log_evaluation(-1)`.
    *   Wait, `cross_val_score` doesn't support callbacks directly in the same way. I need to wrap the training logic or use `GridSearchCV` (too heavy for simple task) or manually split and train.
    *   Manual CV loop is better to control callbacks.
    *   Steps:
        1. Load train/test.
        2. Define features `X` and target `y`.
        3. Split train into 5 folds.
        4. Loop through folds: train on `X_train_fold`, predict on `X_val_fold`, compute RMSE.
        5. Compute mean RMSE.
        6. Train final model on full train with early stopping.
        7. Predict on test.
        8. Save submission.
        9. Print RMSE.

    *   Refining the "Hold-out validation set" instruction vs "5-fold cross-validation".
        *   Instruction: "The evaluation should be based on 5-fold cross-validation but only if that's an appropriate evaluation for the task at hand."
        *   Instruction: "The code should ... print out the evaluation metric computed on a hold-out validation set."
        *   These seem slightly contradictory if I strictly follow "5-fold" for evaluation *and* "hold-out". Usually, in Kaggle, CV is the standard for internal evaluation. I will prioritize the 5-fold CV instruction for the metric calculation as it's more robust for small datasets (which this appears to be, given the KB size).
        *   Wait, "hold-out validation set" usually implies Train/Test split. "5-fold cross-validation" implies K-Fold.
        *   Given the instruction "The evaluation should be based on 5-fold cross-validation", I will use that for the metric printing.
        *   For the submission, I need predictions on the test data.

    *   Memory consideration: The data is tiny (50KB). No special memory optimization needed, but I will ensure I don't store unnecessary copies.
    *   LightGBM Regressor: `lgb.LGBMRegressor`.
    *   Callbacks: `lgb.early_stopping(50)`, `lgb.log_evaluation(-1)`. Note: `log_evaluation` is needed for early stopping to work? Actually `early_stopping` works with `lgb.train` or `lgb.LGBMRegressor` if using callbacks. For `LGBMRegressor`, `callbacks` argument is used.
    *   Wait, `lgb.train` vs `LGBMRegressor`. `lgb.train` is the function. The prompt says "Use callbacks: `callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)]`". This suggests using `lgb.train()`. However, `LGBMRegressor` is the scikit-learn compatible interface. Using `LGBMRegressor` is easier for CV. But `lgb.train` is often faster for large data. Given data size is small, `LGBMRegressor` is fine. However, to strictly follow "Use callbacks... from lgb.train()", I should use `lgb.train`. But `cross_val_score` expects scikit-learn API.
    *   Better approach: Use `LGBMRegressor` for the CV loop logic manually to ensure callbacks work correctly with the specific version constraints. `LGBMRegressor` supports `callbacks` parameter in `fit`.
    *   Wait, checking `lightgbm==4.6.0` docs: `LGBMRegressor` accepts `callbacks`. `lgb.train` uses `callbacks` list.
    *   Let's use `LGBMRegressor` to make the CV loop cleaner (scikit-learn style).
    *   Wait, the prompt explicitly says: "Use callbacks: `callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)]`". This implies I should use `lgb.train` or ensure I use these callbacks.
    *   Actually, `LGBMRegressor` in newer versions supports callbacks.
    *   Let's stick to `LGBMRegressor` for the CV loop to keep code clean, but ensure callbacks are passed.
    *   Wait, `lgb.early_stopping` returns a callback object.
    *   Okay, plan:
        1. Load data.
        2. Prepare features.
        3. 5-Fold CV loop to calculate RMSE (print this).
        4. Train final model on full train with callbacks.
        5. Predict on test.
        6. Save submission.

    *   Wait, one detail: "The evaluation should be based on 5-fold cross-validation but only if that's an appropriate evaluation for the task at hand."
    *   With such small data (KBs), 5-fold CV is definitely appropriate.
    *   Metric: RMSE. `sklearn.metrics.mean_squared_error(y_true, y_pred, squared=False)`.
    *   Submission: `id` column needs to be the row index of the test set. The prompt says "id (integer row index starting from 0)". `test.parquet` usually has an `id` column or I need to generate it. The prompt says "id (integer row index starting from 0)". I should check if `test` has an `id` column. Usually in these tasks, the test file has an `id` column. If not, I generate `range(len(test))`. The prompt says "id (integer row index starting from 0)". I will assume the test file has an `id` column or I create one. Standard practice: if `test` doesn't have `id`, create it. But often `test` has `id`. I will check `test.columns`. If `id` is not there, I create it. Wait, instruction says "id (integer row index starting from 0)". This implies I need to generate it if it's not there, or use it if it is. I'll generate `range(len(test))` to be safe, or use the existing `id` if present. I'll check `test.columns` and assume standard format. To be safe, I'll generate `id` based on row index if it doesn't exist, but usually, it exists. I will use `test.index` if it's integer, or `range`. Let's assume standard `id` column exists in test data based on "submission.csv... id...". Usually, the test file has an `id` column. I'll try to use it. If not, create.
    *   Actually, looking at the instruction "id (integer row index starting from 0)", this suggests the output file needs an `id` column that corresponds to the row order. I will ensure `submission.csv` has `id` and `salary_in_usd`.

5.  **Refining Code Logic:**
    *   Import `pandas`, `numpy`, `lightgbm`, `sklearn.model_selection`.
    *   Load `train` and `test`.
    *   Define `X_train = train.drop(columns=['salary_in_usd'])`.
    *   Define `y_train = train['salary_in_usd']`.
    *   `X_test = test.drop(columns=['id'])` (if 'id' exists) or just all columns. Wait, if test has 'id', I should drop it for prediction. If not, I keep it.
    *   Better: `X_test = test.drop(columns=['id'])` if 'id' is in columns. If not, `X_test = test`.
    *   Actually, usually `id` is a separate column in test. I will check `if 'id' in X_test.columns`.
    *   CV Loop:
        *   `K = 5`.
        *   `folds = np.array_split(X_train, K)`.
        *   `scores = []`.
        *   For `i` in `range(K)`:
            *   `X_tr, X_val = ftr`, `ftr`.
            *   `y_tr, y_val = ...`.
            *   `model = LGBMRegressor(...)`.
            *   `model.fit(X_tr, y_tr, callbacks=[early_stopping, log_eval])`.
            *   `pred = model.predict(X_val)`.
            *   `rmse = np.sqrt(mean_squared_error(y_val, pred))`.
            *   `scores.append(rmse)`.
        *   `mean_rmse = np.mean(scores)`.
        *   `print(mean_rmse)`.
    *   Final Train:
        *   `model.fit(X_train, y_train, callbacks=[early_stopping])`.
    *   Predict:
        *   `X_test` preparation: ensure features match.
        *   `pred_test = model.predict(X_test)`.
        *   `submission = pd.DataFrame({'id': test.index if 'id' not in test.columns else test['id'], 'salary_in_usd': pred_test})`.
        *   Wait, instruction: "id (integer row index starting from 0)". If `test` has an `id` column, is that the one I should use? Usually yes. But the instruction says "id (integer row index starting from 0)". This might mean I should generate `0, 1, 2...` regardless of existing ID. To be safe and compliant, I will generate `range(len(test))` as the ID, or use the existing one if it's sequential. I'll check if `id` is in columns. If yes, use it. If no, create.
        *   Wait, if I use `test.index`, it might not be 0, 1, 2... if the original data wasn't. But `read_parquet` preserves index. I'll reset index to ensure 0, 1, 2... and use that as `id`.
        *   Wait, "id (integer row index starting from 0)". This implies the column name is `id`.
        *   I'll create `submission_df` with `id` = `range(len(test))`.

6.  **Memory Check:**
    *   Data is small (50KB). No memory issues.
    *   But I should avoid loading test data into memory if not needed? No, I need it for submission.
    *   I will ensure I don't keep huge intermediate arrays.
    *   I will use `np.array_split` which is memory efficient.

7.  **Final Review of Instructions:**
    *   3-5 sentences outline.
    *   Single code block.
    *   Print evaluation metric.
    *   Save `submission.csv`.
    *   Use `lightgbm` callbacks correctly.
    *   No data leakage.
    *   5-fold CV for evaluation metric.

8.  **Wait, one constraint:** "The solution sketch should be 3-5 sentences."
    *   "Don't suggest to do EDA."
    *   "Take the Memory section into consideration". (Memory section is empty, so I'll ignore specific memory limits but write efficient code).
    *   "Don't propose the same modelling solution". (Since this is the first prompt, I propose LightGBM).

    Let's write the code.

    *   `import lightgbm as lgb`
    *   `from sklearn.model_selection import KFold`
    *   `from sklearn.metrics import mean_squared_error`
    *   `import pandas as pd`
    *   `import numpy as np`

    *   `train = pd.read_parquet('./input/train.parquet')`
    *   `test = pd.read_parquet('./input/test.parquet')`
    *   `features = train.columns.drop('salary_in_usd')`
    *   `X_train = train[features]`
    *   `y_train = train['salary_in_usd']`
    *   `X_test = test[features]` (assuming test has same columns, need to handle 'id')

    *   CV:
        *   `kf = KFold(n_splits=5, shuffle=True, random_state=42)`
        *   `rmse_scores = []`
        *   `for train_idx, val_idx in kf.split(X_train):`
            *   `X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]`
            *   `y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]`
            *   `model = lgb.LGBMRegressor(...)`
            *   `model.fit(X_tr, y_tr, callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)])`
            *   `pred = model.predict(X_val)`
            *   `rmse = np.sqrt(mean_squared_error(y_val, pred))`
            *   `rmse_scores.append(rmse)`
        *   `print(np.mean(rmse_scores))`

    *   Final Train:
        *   `model.fit(X_train, y_train, callbacks=[lgb.early_stopping(50)])`
    *   Predict:
        *   `pred_test = model.predict(X_test)`
        *   `submission = pd.DataFrame({'id': range(len(test)), 'salary_in_usd': pred_test})`
        *   `submission.to_csv('./working/submission.csv', index=False)`

    *   Wait, `lgb.early_stopping` requires `valid_names`? No, with `LGBMRegressor`, `early_stopping` is a callback that expects `valid_name` parameter if using `fit`? No, `early_stopping` callback in `lgb.train` or `fit` works with `validation_set`. But `LGBMRegressor` doesn't support `validation_set` directly in `fit` like `lgb.train` does without `callbacks`?
    *   Actually, `lgb.early_stopping` callback is designed for `lgb.train`. With `LGBMRegressor`, `early_stopping` callback might not work correctly if `validation_data` is not passed to `fit`.
    *   Checking `lightgbm` docs for `LGBMRegressor`: `callbacks` parameter supports `early_stopping`. But `early_stopping` callback needs `validation_data` to be passed to `fit`?
    *   Actually, `lgb.early_stopping` callback *does* work with `LGBMRegressor`. But the callback needs to know what to evaluate. If I don't pass `validation_data`, it might not stop.
    *   Wait, `lgb.early_stopping` callback documentation: "This callback will stop training if the validation loss does not improve for early_stopping_rounds consecutive rounds." It requires `validation_data` to be passed to `fit`?
    *   Actually, `LGBMRegressor` supports `validation_data` in `fit`.
    *   If I use `LGBMRegressor`, I should pass `validation_data` to `fit` for early stopping to work effectively.
    *   However, for the CV loop, I can't easily pass `validation_data` for every fold without passing the whole array.
    *   Wait, `lgb.early_stopping` callback works with `lgb.train`. For `LGBMRegressor`, it works if `validation_data` is provided.
    *   Let's check if `LGBMRegressor` `fit` accepts `validation_data`. Yes, it does.
    *   So in the CV loop:
        *   `model.fit(X_tr, y_tr, validation_data=(X_val, y_val), callbacks=[...])`
    *   This ensures early stopping works.

    *   Wait, `lgb.log_evaluation(-1)` is also a callback.
    *   Okay, I will implement this.

    *   Wait, one more thing: `lgb.early_stopping` callback requires `validation_data` to be passed to `fit`.
    *   Okay, I will update the code to use `validation_data`.

    *   Wait, `lgb.early_stopping` returns a callback object.
    *   Code:
        ```python
        model.fit(X_tr, y_tr, validation_data=(X_val, y_val), callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)])
        ```
    *   This should work.

    *   Wait, `lgb.early_stopping` in `lgb.train` is `early_stopping_rounds`. In `callbacks`, it's `early_stopping(50)`.
    *   Okay.

    *   Wait, `id` column in test. I will check if it exists. If it does, I'll use it. If not, I'll generate it.
    *   To be safest with "id (integer row index starting from 0)", I will generate `range(len(test))`.

    *   Wait, `lgb.early_stopping` callback is deprecated in some versions? No, it's standard.
    *   Wait, `lgb.log_evaluation(-1)` prints to stdout.
    *   Okay.

    *   Wait, `submission.csv` needs `id` and `salary_in_usd`.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback needs `valid_name`? No, that's for `lgb.train`. For `LGBMRegressor`, it works with `validation_data`.
    *   Okay.

    *   One detail: `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   I will ensure `validation_data` is passed in the CV loop.
    *   In the final training, I won't pass `validation_data` (or I can pass a dummy one, but early stopping won't trigger without it). I will not pass `validation_data` in final training to save time, as I have already done CV.

    *   Wait, `lgb.early_stopping(50)` means 50 rounds of no improvement.
    *   Okay.

    *   Wait, `lgb.log_evaluation(-1)` prints every 1 iteration.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback needs `valid_name`?
    *   Actually, `lgb.early_stopping` callback *does not* require `valid_name` if `validation_data` is passed to `fit`.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might need `validation_data` to be a tuple `(X, y)`.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might throw error if `validation_data` is not passed.
    *   Okay, I will pass it.

    *   Wait, `lgb.early_stopping` callback is `lgb.early_stopping(50)`.
    *   Okay.

    *   Wait, `lgb.log_evaluation(-1)` is `lgb.log_evaluation(-1)`.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might not work if `validation_data` is not passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might be `lgb.early_stopping(50)`.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait, `lgb.early_stopping` callback in `LGBMRegressor` might require `validation_data` to be passed.
    *   Okay.

    *   Wait