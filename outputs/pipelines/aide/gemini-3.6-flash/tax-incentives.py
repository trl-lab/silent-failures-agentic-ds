import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# Load train and test datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "credit_amount"
train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)
y = train_df[target_col].values

# Prepare combined dataframe for consistent categorical transformation
n_train = len(train_df)
test_ids = test_df["id"].values if "id" in test_df.columns else np.arange(len(test_df))

df_all = pd.concat(
    [
        train_df.drop(columns=[target_col]),
        test_df.drop(columns=["id"] if "id" in test_df.columns else []),
    ],
    axis=0,
    ignore_index=True,
)

# Identify features
id_cols = ["id"]
feat_cols = [c for c in df_all.columns if c not in id_cols]

# Type casting and missing value fill
for c in feat_cols:
    if (
        df_all[c].dtype == "object"
        or df_all[c].dtype.name == "category"
        or df_all[c].dtype == "bool"
    ):
        df_all[c] = df_all[c].fillna("missing").astype(str)
    else:
        df_all[c] = pd.to_numeric(df_all[c], errors="coerce")

cat_cols = [
    c
    for c in feat_cols
    if df_all[c].dtype == "object" or df_all[c].dtype.name == "category"
]

# Create pairwise categorical interaction features among key categorical variables
cat_subset = sorted(cat_cols, key=lambda c: df_all[c].nunique())[:6]
new_interaction_cols = []
for i in range(len(cat_subset)):
    for j in range(i + 1, len(cat_subset)):
        c1, c2 = cat_subset[i], cat_subset[j]
        col_name = f"{c1}_x_{c2}"
        df_all[col_name] = df_all[c1] + "__" + df_all[c2]
        new_interaction_cols.append(col_name)

all_cat_cols = cat_cols + new_interaction_cols

# Frequency encoding for all categorical columns
for c in all_cat_cols:
    freq = df_all[c].value_counts()
    df_all[f"{c}_freq"] = df_all[c].map(freq)

# Ensure string type for all categorical columns passed to CatBoost
for c in all_cat_cols:
    df_all[c] = df_all[c].fillna("missing").astype(str)

# Split back to train and test
X_train = df_all.iloc[:n_train].copy()
X_test = df_all.iloc[n_train:].copy()

# 5-Fold Cross Validation Setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42 + fold,
        verbose=0,
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        cat_features=all_cat_cols,
        early_stopping_rounds=100,
        verbose=False,
    )

    oof_preds[val_idx] = np.clip(model.predict(X_va), 0, None)
    test_preds += np.clip(model.predict(X_test), 0, None) / kf.n_splits

# Out-Of-Fold Evaluation
oof_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"Validation RMSE: {oof_rmse:.4f}")

# Output test predictions
submission = pd.DataFrame({"id": test_ids, "credit_amount": test_preds})
submission.to_csv("./working/submission.csv", index=False)
