import os
import warnings
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# 2. Domain-Specific Feature Engineering
keyword_cols = [c for c in train_df.columns if "Medical_Keyword" in c]
med_hist_cols = [
    c
    for c in train_df.columns
    if "Medical_History" in c and train_df[c].dtype != "object"
]

for df in [train_df, test_df]:
    # Physical metric interactions
    df["BMI_Age"] = df["BMI"] * df["Ins_Age"]
    df["BMI_Ht"] = df["BMI"] / (df["Ht"] + 1e-5)
    df["Wt_Ht"] = df["Wt"] / (df["Ht"] + 1e-5)
    df["Wt_Age"] = df["Wt"] * df["Ins_Age"]
    df["Wt_BMI"] = df["Wt"] * df["BMI"]

    # Medical Keyword aggregations
    df["Medical_Keyword_sum"] = df[keyword_cols].sum(axis=1)
    df["Medical_Keyword_mean"] = df[keyword_cols].mean(axis=1)
    df["Medical_Keyword_count"] = (df[keyword_cols] > 0).sum(axis=1)

    # Medical History aggregations
    df["Med_Hist_null_count"] = df[med_hist_cols].isnull().sum(axis=1)

    # Row-level missingness
    df["Null_count"] = df.isnull().sum(axis=1)

    # Product info parsing
    p2_str = df["Product_Info_2"].astype(str)
    df["Product_Info_2_char"] = p2_str.str[0]
    df["Product_Info_2_num"] = pd.to_numeric(p2_str.str[1:], errors="coerce").fillna(-1)

# Categorical Feature Frequency & Numeric Encoding
cat_cols = train_df.select_dtypes(include=["object", "category"]).columns.tolist()
for c in cat_cols:
    freq = train_df[c].value_counts().to_dict()
    train_df[c + "_freq"] = train_df[c].map(freq).fillna(0)
    test_df[c + "_freq"] = test_df[c].map(freq).fillna(0)

    categories = {val: i for i, val in enumerate(train_df[c].unique())}
    train_df[c] = train_df[c].map(categories).fillna(-1).astype(int)
    test_df[c] = test_df[c].map(categories).fillna(-1).astype(int)

# Group Mean Aggregations
for agg_col in ["Product_Info_2", "Product_Info_2_char"]:
    for num_col in ["BMI", "Ins_Age", "Medical_Keyword_sum", "Wt"]:
        grp = train_df.groupby(agg_col)[num_col].mean().to_dict()
        train_df[f"{agg_col}_{num_col}_mean"] = train_df[agg_col].map(grp)
        test_df[f"{agg_col}_{num_col}_mean"] = (
            test_df[agg_col].map(grp).fillna(train_df[num_col].mean())
        )

features = [c for c in train_df.columns if c not in ["Id", "Response"] and c in test_df]
X = train_df[features].copy()
y = train_df["Response"].values.astype(float)
X_test = test_df[features].copy()

# 3. Stratified 5-Fold Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y.astype(int)), start=1):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.035,
        depth=6,
        l2_leaf_reg=3,
        loss_function="RMSE",
        random_seed=42 + fold,
        verbose=0,
    )

    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=60)

    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / skf.n_splits


# 4. Powell Threshold Optimization for QWK
def qwk_obj(thresholds, preds, targets):
    sorted_t = np.sort(thresholds)
    preds_discrete = np.digitize(preds, sorted_t) + 1
    return -cohen_kappa_score(targets, preds_discrete, weights="quadratic")


init_thresholds = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
res = minimize(qwk_obj, init_thresholds, args=(oof_preds, y), method="Powell")
opt_thresholds = np.sort(res.x)

oof_discrete = np.digitize(oof_preds, opt_thresholds) + 1
final_qwk = cohen_kappa_score(y, oof_discrete, weights="quadratic")

print(f"Validation QWK Score: {final_qwk:.5f}")

# 5. Generate and Save Test Predictions
test_discrete = np.digitize(test_preds, opt_thresholds) + 1
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_discrete)), "Response": test_discrete.astype(int)}
)
submission.to_csv("./working/submission.csv", index=False)
