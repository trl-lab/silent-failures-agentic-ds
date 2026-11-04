` (or `[0, 1]`).
`le.classes_[0]` is the smaller value, `le.classes_[1]` is the larger value.

Let's double-check if there are any edge cases:
What if `y_train` has missing values or strange target values?
We can drop rows from `train_df` where target `Class` is null:
`train_df = train_df.dropna(subset=['Class'])`.

Let's check if `test_df` has an `id` column or index:
Regardless of input index, we reset index or use integer row index starting from 0 for `id`:
`predictions_df = pd.DataFrame({'id': np.arange(len(test_df)), 'Class': test_preds})`
`predictions_df.to_csv('predictions.csv', index=False)`

Let's check requirements from prompt:
- Self-contained Python script
- Trains a model to predict "Class"
- Saves predictions to `predictions.csv` in the current working directory
- Columns: "id" (integer row index starting from 0) and "Class"
- Standard libraries plus pandas, numpy, scikit-learn, lightgbm, xgboost, catboost
- Handle missing values and categorical columns automatically
- Output only Python code, no explanation, no markdown fences.

Wait! The prompt says "Output only the Python code, no explanation or markdown fences".
Wait, no markdown fences at all!