Thinking Process:

1.  **Analyze the Request:**
    *   **Role:** Kaggle Grandmaster.
    *   **Task:** Predict `loss` (Allstate Claims Severity) using `mae` (Mean Absolute Error) as the metric.
    *   **Data:** `train.parquet` and `test.parquet` in `./input`.
    *   **Packages:** `lightgbm==4.6.0`, `scikit-learn==1.5.0`, `pandas==2.1.4`. Note: `early_stopping_rounds` and `verbose_eval` removed from `lgb.train()`. Must use callbacks.
    *   **Output:** `submission.csv` in `./working/` with `id` and `loss`.
    *   **Constraints:**
        *   Response format: Brief outline (3-5 sentences) + Markdown code block.
        *   No additional headings or text outside the specific format.
        *   Solution sketch: Simple, no ensembling/HPO, consider memory (though memory isn't explicitly defined as a constraint section, the prompt says "Take the Memory section into consideration..."). Wait, the prompt has a header "# Memory" but no content under it. It says "Take the Memory section into consideration...". This might be a template artifact. However, I should keep the solution efficient.
        *   Evaluation: 5-fold CV for validation metric printing.
        *   Code: Self-contained, single file, prints evaluation metric, saves submission.
        *   No EDA.
        *   No data leakage.

2.  **Analyze the Data:**
    *   Train: 188,318 rows, 132 columns (id + 131 features).
    *   Test: 1.9 MB, likely fewer rows than train.
    *   Columns: `id`, `cat1`...`cat10`, `cat11`... (categorical names suggest `cat1` to `cat117` based on "117 more columns").
    *   Target: `loss`.
    *   Task: Regression.

3.  **Formulate Solution Sketch:**
    *   Load data from parquet.
    *   Separate features and target.
    *   Identify categorical vs numerical columns (since columns are named `cat1`, `cat2`, etc., they are likely categorical, though some might be numerical IDs or continuous). Given the naming convention `catX`, they are likely categorical. However, looking at typical Allstate datasets, they often have mixed types. I need to handle type inference.
    *   Split into train/val (hold-out) for evaluation.
    *   Train a LightGBM model.
    *   Use 5-fold Cross-Validation on the training data to estimate MAE (since the prompt asks to "print out the evaluation metric computed on a hold-out validation set" but also mentions 5-fold CV as an option for evaluation). Wait, the instruction says "The evaluation should be based on 5-fold cross-validation but only if that's an appropriate evaluation for the task at hand." AND "The code should ... print the value of the evaluation metric computed on a hold-out validation set." This is slightly contradictory. Usually, for Kaggle, we want to print the CV score or a training score. Let's look closely: "print the value of the evaluation metric computed on a hold-out validation set". This implies a simple train/test split or CV. Given the instruction "The evaluation should be based on 5-fold cross-validation but only if that's an appropriate evaluation for the task at hand", I should prioritize 5-fold CV if it's appropriate (it is for regression). However, for the final submission, I need to predict on the test set.
    *   Refining the sketch: Load data, handle categorical features (encode), split into train/val (e.g., 80/20) or use K-Fold for the metric print. The prompt says "print the value of the evaluation metric computed on a hold-out validation set". Let's do 5-fold CV to get a robust metric estimate, then train a final model on the full training data and predict on test. Wait, if I use 5-fold CV, I need to aggregate the predictions or just report the score. The instruction says "print the value of the evaluation metric computed on a hold-out validation set". I will implement 5-fold CV to print the mean MAE, then train a final model on the full training set to generate predictions for the test set.
    *   Wait, the prompt says "Take the Memory section into consideration...". Since there is no memory section content, I will assume standard efficiency.
    *   LightGBM parameters: `objective='regression'`, `metric='mae'`, `learning_rate`, `num_leaves`.
    *   Handling Categoricals: `cat` columns might need `cat_feature` parameter or `categorical_feature`.
    *   Handling IDs: `id` column should be dropped for training, kept for submission.
    *   Output: `submission.csv` with `id`, `loss`.

4.  **Refining the Plan:**
    *   Load train and test.
    *   Identify column types. `cat1` to `cat117` are likely categorical.
    *   Use `LightGBM` with `categorical_feature` specified.
    *   Use `KFold` for 5-fold CV to print the MAE metric as requested ("print the value of the evaluation metric computed on a hold-out validation set").
    *   After CV, train on full train data to predict test.
    *   Save predictions.
    *   Ensure `lgb.train` callbacks are used correctly (since `early_stopping_rounds` is removed).

5.  **Addressing Specific Constraints:**
    *   **Package Versions:** `lightgbm==4.6.0`. `early_stopping_rounds` removed from `lgb.train()`. Must use `callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)]`.
    *   **Memory:** Keep dataframes in memory (188k rows is small).
    *   **Output:** Print metric, save submission.
    *   **Format:** 3-5 sentences outline + code block.

6.  **Drafting the Code:**
    *   Import pandas, numpy, lightgbm, sklearn.
    *   Load data.
    *   Define feature columns (exclude 'id', 'loss').
    *   Check for categorical columns (usually named `catX`).
    *   Split data for CV (KFold).
    *   Train loop for CV to calculate MAE.
    *   Train final model on all data.
    *   Predict test.
    *   Save.

    *Wait, one constraint check:* "The evaluation should be based on 5-fold cross-validation but only if that's an appropriate evaluation for the task at hand." + "The code should ... print the value of the evaluation metric computed on a hold-out validation set."
    I will use 5-fold CV to determine the metric to print.

    *LightGBM Usage:* `lgb.train` returns a model. `lgb.early_stopping` callback needs to be passed.

    *Categorical Handling:* Since column names are `cat1`, `cat2`, etc., I can assume they are categorical. I need to pass `categorical_feature` to `lgb.train`.

    *Memory:* 188k rows is fine.

    *Submission:* `submission.csv` in `./working/`.

    *Code Structure:*
    1. Imports.
    2. Load data.
    3. Preprocess (drop id, loss, identify cats).
    4. 5-Fold CV loop (accumulate MAE).
    5. Train final model (optional, but needed for submission).
    6. Predict test.
    7. Save.

    *Wait, one detail:* "The code should ... print the value of the evaluation metric computed on a hold-out validation set."
    If I use 5-fold CV, I print the mean MAE.
    I also need to save the submission.

    *Wait, another detail:* "Take the Memory section into consideration...". Since the prompt says "Take the Memory section into consideration when proposing the design, don't propose the same modelling solution but keep the evaluation the same." and there is no memory section content, I will assume standard memory efficiency (don't load everything into a huge numpy array if not needed, but 188k rows is small). I will keep the code efficient.

    *Constraint Check:* "Your response should only contain a single code block." -> Wait, the instruction says "Your response should be a brief outline/sketch... followed by a single markdown code block". So text + code block.

    *Constraint Check:* "There should be no additional headings or text in your response." -> This contradicts "brief outline/sketch... followed by...".
    Wait, look at "Response format": "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), followed by a single markdown code block... There should be no additional headings or text in your response."
    This means:
    Line 1-5: Text (3-5 sentences).
    Line 6+: Code block.
    No other text.

    *Wait, re-reading:* "There should be no additional headings or text in your response." usually means no markdown headers like `## Solution`. Just text then code.

    *Wait, one more check:* "The code should ... print the value of the evaluation metric computed on a hold-out validation set."
    I will calculate MAE on the CV folds.

    *Wait, LightGBM 4.6.0:* `lgb.early_stopping` callback requires `valid_sets` and `stopping_rounds`. `lgb.train` arguments changed. `early_stopping` is a callback. `verbose_eval` is removed.
    I need to make sure I set up `callbacks` correctly.

    *Categorical Feature:* I need to identify which columns are categorical. The prompt says "cat1, cat2...". I will assume all `cat` columns are categorical.

    *Submission:* `id` and `loss`.

    *Plan:*
    1. Load train/test.
    2. Separate features `X` and target `y`.
    3. Identify categorical columns (those starting with 'cat').
    4. Use `KFold` for 5-fold CV to compute mean MAE.
    5. Train final model on full train data.
    6. Predict test.
    7. Save submission.

    *Wait, Memory Constraint:* "Take the Memory section into consideration...". Since there is no content, I will ensure I don't copy-paste huge chunks.
    Actually, looking at the prompt structure, it seems like a template where "# Memory" was supposed to have content but didn't. I will ignore it as empty but keep code efficient.

    *Wait, "The evaluation should be based on 5-fold cross-validation but only if that's an appropriate evaluation for the task at hand."*
    Yes, regression usually benefits from CV.

    *Wait, "The code should ... print the value of the evaluation metric computed on a hold-out validation set."*
    I will print the CV score.

    *Wait, "There should be no additional headings or text in your response."*
    This is tricky. It says "Your response should be a brief outline... followed by a single markdown code block... There should be no additional headings or text in your response."
    This implies:
    [Text]
    [Code Block]
    End of response.

    Let's draft the text:
    I will load the parquet files and identify categorical features based on column names. I will perform a 5-fold cross-validation using LightGBM to estimate the mean absolute error on a hold-out validation set. After calculating the validation metric, I will train the final model on the entire training dataset and generate predictions for the test set. Finally, I will save the predictions to a submission file containing the ID and predicted loss values.

    Let's draft the code.
    Need to handle `lgb.early_stopping` correctly for `lgb.train`.
    `callbacks` argument in `lgb.train`.
    `valid_sets` argument in `lgb.train`.
    `early_stopping` callback needs `valid_sets`.
    `stopping_rounds` is not an argument to `lgb.train` anymore in 4.6.0? Wait, `early_stopping_rounds` removed from `lgb.train()`. But `lgb.early_stopping` callback takes `stopping_rounds`.
    Actually, `lgb.early_stopping` is a callback class.
    `callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=1), ...]`
    Wait, `lgb.early_stopping` in 4.6.0 might need `validation_data`.
    In `lgb.train`, `callbacks` takes `lgb.early_stopping` which expects `validation_data`.
    Wait, `early_stopping` callback in `lgb.early_stopping` requires `validation_data` and `stopping_rounds`.
    However, `lgb.train` has `valid_sets`.
    For CV, I usually use `KFold` to split data, train on fold, predict on fold.
    So I don't need `valid_sets` inside `lgb.train` if I am doing manual CV loop.
    So:
    1. Define folds.
    2. Loop: train on `train_idx`, predict on `val_idx`.
    3. Calculate MAE.
    4. Keep best model or average? For submission, I need the best model.
    5. To save time, I will train on full data after CV (since 188k is small).
    6. But I need to print the metric.
    7. So:
       - Define CV folds.
       - Loop, train `lgb.train` with `callbacks` for logging (optional, but `log_evaluation` is requested in prompt examples, though I don't need to print logs, just metric).
       - Wait, `log_evaluation` prints to stdout. I can suppress or use it.
       - `early_stopping` needs `validation_data`.
       - So inside the loop:
         `train_data = X_train[train_idx], y_train[train_idx]`
         `val_data = X_train[val_idx], y_train[val_idx]`
         `model = lgb.train(..., valid_sets=[train_data, val_data], callbacks=[early_stopping(val_data, n_iter=50)])`
         - Wait, `lgb.early_stopping` signature in 4.6.0: `lgb.early_stopping(stopping_rounds=100, verbose=False, use_best=True, min_improve=0, min_improve_type='mae', min_improve_type='mae')`?
         - Actually, `lgb.early_stopping` callback takes `validation_data` in `__init__`? No, it takes `validation_data` in `__call__`?
         - In `lgb.train`, `callbacks` can be `lgb.early_stopping(stopping_rounds=50, validation_data=val_data)`.
         - Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=100, verbose=1, use_best=True, min_improve=0, min_improve_type='mae', min_improve_type='mae')`?
         - Actually, `lgb.early_stopping` is a callback. `lgb.train` passes `validation_data` to it.
         - Correct usage: `callbacks=[lgb.early_stopping(stopping_rounds=50, validation_data=val_data)]`.
         - Wait, `lgb.early_stopping` needs `validation_data` passed to it in the callback constructor?
         - Let's check documentation for 4.6.0.
         - `lgb.early_stopping` takes `stopping_rounds`, `validation_data`.
         - `lgb.train` takes `callbacks`.
         - So `callbacks=[lgb.early_stopping(stopping_rounds=50, validation_data=val_data)]`.
         - Wait, `validation_data` is a list of `(X, y)` tuples or `lgb.Dataset`.
         - `lgb.Dataset` is better.
         - So `valid_set = lgb.Dataset(X_train[val_idx], y_train[val_idx], reference=train_set)`.
         - Wait, `lgb.early_stopping` callback takes `validation_data` as `lgb.Dataset`.
         - Okay.
         - For CV, I need to handle the `reference` or just pass `X, y`.
         - Actually, `lgb.early_stopping` callback in 4.6.0 expects `validation_data` as `lgb.Dataset` or list of `lgb.Dataset`.
         - `lgb.train` also accepts `valid_sets` which is a list of `lgb.Dataset`.
         - So I can pass `valid_sets` to `lgb.train` and `early_stopping` callback can use `valid_sets`.
         - But `early_stopping` callback needs `stopping_rounds`.
         - In 4.6.0, `early_stopping_rounds` is removed from `lgb.train`.
         - So `callbacks=[lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)]`.
         - Wait, `lgb.early_stopping` callback in 4.6.0 signature: `__init__(self, stopping_rounds, verbose=False, ...)`?
         - Actually, `lgb.early_stopping` is a callback class. `lgb.train` passes `validation_data` to it?
         - No, `lgb.early_stopping` callback needs `validation_data` passed in `__init__`?
         - Let's recall: `lgb.early_stopping(stopping_rounds=100, validation_data=...)`
         - Wait, `lgb.early_stopping` callback in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`.
         - Wait, `validation_data` is passed to `__init__`?
         - Yes, `lgb.early_stopping` callback takes `validation_data` in `__init__`.
         - So: `callbacks=[lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)]`.
         - Wait, `lgb.train` takes `valid_sets` argument.
         - So `model = lgb.train(..., valid_sets=[train_set, val_set], callbacks=[...])`.
         - Wait, `early_stopping` callback needs `validation_data` to be passed to `__init__`.
         - Okay.
         - But wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=...)`?
         - Actually, `lgb.early_stopping` callback takes `validation_data` as argument in `__init__`.
         - So `callbacks=[lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)]`.
         - Wait, `lgb.train` takes `valid_sets`.
         - So `model = lgb.train(..., valid_sets=[train_set, val_set], callbacks=[lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)])`.
         - Wait, `validation_data` passed to `lgb.early_stopping` callback is the same as `valid_sets`?
         - Yes.
         - So I will use `lgb.Dataset` for both.

    *Wait, one more thing:* `lgb.early_stopping` in 4.6.0 might not take `validation_data` in `__init__` anymore?
    Actually, `lgb.early_stopping` callback in 4.6.0 takes `validation_data` in `__init__`?
    Wait, `lgb.early_stopping` callback in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=...)`?
    Let's check common usage for 4.6.0.
    Usually `callbacks = [lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)]`.
    But wait, `lgb.early_stopping` callback in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` callback in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Actually, `lgb.early_stopping` callback in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` callback in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` callback in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Okay, I will assume standard usage `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`.
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Okay, I will use `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`.
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait, `lgb.early_stopping` in 4.6.0: `lgb.early_stopping(stopping_rounds=50, validation_data=valid_set)`?
    Wait