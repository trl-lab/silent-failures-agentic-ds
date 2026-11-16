import os
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

# Load input datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Ensure proper row indexing and ID assignment
if "id" not in test_df.columns:
    test_df["id"] = np.arange(len(test_df))
if "id" not in train_df.columns:
    train_df["id"] = np.arange(len(train_df))

target_col = "salary_in_usd"
y_log = np.log1p(train_df[target_col].values)


# Job role categorization helper
def categorize_job(title):
    title = str(title).lower()
    if any(
        k in title
        for k in ["machine learning", "ml", "ai", "deep learning", "vision", "nlp"]
    ):
        return "ML_AI_Engineer"
    elif any(k in title for k in ["scientist", "science", "research"]):
        return "Data_Scientist"
    elif any(
        k in title for k in ["engineer", "pipeline", "etl", "architect", "database"]
    ):
        return "Data_Engineer"
    elif any(k in title for k in ["analyst", "analytics", "bi"]):
        return "Data_Analyst"
    elif any(
        k in title for k in ["lead", "head", "director", "manager", "vp", "chief"]
    ):
        return "Leadership"
    else:
        return "Other"


# Combine datasets for uniform feature extraction
df_all = pd.concat(
    [train_df.drop(columns=[target_col], errors="ignore"), test_df], ignore_index=True
)

# Domain-specific feature engineering
df_all["job_category"] = df_all["job_title"].apply(categorize_job)
df_all["exp_level_num"] = (
    df_all["experience_level"].map({"EN": 1, "MI": 2, "SE": 3, "EX": 4}).fillna(2)
)
df_all["company_size_num"] = (
    df_all["company_size"].map({"S": 1, "M": 2, "L": 3}).fillna(2)
)
df_all["is_same_country"] = (
    df_all["employee_residence"] == df_all["company_location"]
).astype(int)
df_all["is_us_company"] = (df_all["company_location"] == "US").astype(int)
df_all["is_us_residence"] = (df_all["employee_residence"] == "US").astype(int)

# Frequency encodings
cat_cols = [
    "experience_level",
    "employment_type",
    "job_title",
    "employee_residence",
    "company_location",
    "company_size",
    "job_category",
]

for col in cat_cols:
    freq = df_all[col].value_counts()
    df_all[f"{col}_freq"] = df_all[col].map(freq)
    df_all[col] = df_all[col].astype(str)

train_feat = df_all.iloc[: len(train_df)].copy().reset_index(drop=True)
test_feat = df_all.iloc[len(train_df) :].copy().reset_index(drop=True)

# Out-Of-Fold Target Encoding to prevent leakage
kf = KFold(n_splits=5, shuffle=True, random_state=42)
te_cols = ["job_title", "company_location", "employee_residence", "job_category"]

for col in te_cols:
    train_feat[f"{col}_te"] = np.nan
    test_feat[f"{col}_te"] = 0.0

for train_idx, val_idx in kf.split(train_feat):
    tr = train_feat.iloc[train_idx]
    va = train_feat.iloc[val_idx]

    for col in te_cols:
        te_map = pd.Series(y_log[train_idx]).groupby(tr[col]).mean()
        train_feat.loc[val_idx, f"{col}_te"] = va[col].map(te_map)

global_mean = y_log.mean()
for col in te_cols:
    train_feat[f"{col}_te"] = train_feat[f"{col}_te"].fillna(global_mean)
    full_te_map = pd.Series(y_log).groupby(train_feat[col]).mean()
    test_feat[f"{col}_te"] = test_feat[col].map(full_te_map).fillna(global_mean)

# Define feature subset
features = [c for c in train_feat.columns if c not in ["id", target_col]]

oof_preds = np.zeros(len(train_feat))
test_preds = np.zeros(len(test_feat))

# Train CatBoost Regressor across 5 folds
for fold, (train_idx, val_idx) in enumerate(kf.split(train_feat)):
    X_tr, y_tr = train_feat.iloc[train_idx][features], y_log[train_idx]
    X_va, y_va = train_feat.iloc[val_idx][features], y_log[val_idx]

    model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        cat_features=cat_cols,
        random_seed=42 + fold,
        verbose=0,
    )
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=100)

    oof_preds[val_idx] = model.predict(X_va)
    test_preds += model.predict(test_feat[features]) / kf.n_splits

# Compute CV RMSE metric
cv_rmse = root_mean_squared_error(y_log, oof_preds)
print(f"Validation RMSE: {cv_rmse:.4f}")

# Convert log scale back to USD and save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": test_df["id"], "salary_in_usd": np.expm1(test_preds)})
submission.to_csv("./working/submission.csv", index=False)
