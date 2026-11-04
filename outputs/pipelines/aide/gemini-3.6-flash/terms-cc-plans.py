import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

# 1. Load Datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "APR"
id_col = "id" if "id" in test_df.columns else None

train_df["is_train"] = 1
test_df["is_train"] = 0
if target_col in test_df.columns:
    test_df = test_df.drop(columns=[target_col])

full_df = pd.concat([train_df, test_df], ignore_index=True)

# 2. Automated Feature Engineering and Preprocessing
numeric_cols = []
cat_cols = []
date_cols = []

for col in full_df.columns:
    if col in [target_col, id_col, "is_train"]:
        continue

    if full_df[col].dtype == "object":
        sample_str = full_df[col].dropna().astype(str).head(50)
        cleaned_sample = (
            sample_str.str.replace("$", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        is_num = pd.to_numeric(
            cleaned_sample, errors="coerce"
        ).notna().sum() > 0.8 * len(cleaned_sample)
        if is_num:
            full_df[col] = (
                full_df[col]
                .astype(str)
                .str.replace("$", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            full_df[col] = pd.to_numeric(full_df[col], errors="coerce")
        else:
            try:
                parsed_dates = pd.to_datetime(full_df[col], errors="coerce")
                if parsed_dates.notna().sum() > 0.5 * len(full_df):
                    full_df[col + "_year"] = parsed_dates.dt.year
                    full_df[col + "_month"] = parsed_dates.dt.month
                    full_df[col + "_day"] = parsed_dates.dt.day
                    full_df[col + "_dayofweek"] = parsed_dates.dt.dayofweek
                    full_df[col + "_dayofyear"] = parsed_dates.dt.dayofyear
                    full_df[col + "_is_month_end"] = (
                        parsed_dates.dt.is_month_end.astype(int)
                    )
                    date_cols.append(col)
                    continue
            except Exception:
                pass

    if pd.api.types.is_numeric_dtype(full_df[col]):
        numeric_cols.append(col)
    else:
        cat_cols.append(col)

full_df = full_df.drop(columns=date_cols)

# Label Encoding for categoricals
for col in cat_cols:
    full_df[col] = full_df[col].astype(str).fillna("missing")
    le = LabelEncoder()
    full_df[col + "_le"] = le.fit_transform(full_df[col])

# Separate Train and Test splits
train_processed = (
    full_df[full_df["is_train"] == 1].drop(columns=["is_train"]).reset_index(drop=True)
)
test_processed = (
    full_df[full_df["is_train"] == 0]
    .drop(columns=["is_train", target_col], errors="ignore")
    .reset_index(drop=True)
)

feature_cols = [c for c in train_processed.columns if c not in [target_col, id_col]]
lgb_cat_cols = [col for col in cat_cols if col in feature_cols]

for col in lgb_cat_cols:
    train_processed[col] = train_processed[col].astype("category")
    test_processed[col] = test_processed[col].astype("category")

X = train_processed[feature_cols]
y = train_processed[target_col]
X_test = test_processed[feature_cols]

numeric_and_le_cols = [c for c in feature_cols if c not in lgb_cat_cols]

# 3. 5-Fold Cross Validation & Multi-Model Ensembling
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(train_processed))
oof_xgb = np.zeros(len(train_processed))
oof_cat = np.zeros(len(train_processed))

test_preds_lgb = np.zeros(len(test_processed))
test_preds_xgb = np.zeros(len(test_processed))
test_preds_cat = np.zeros(len(test_processed))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

    # Model 1: LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42 + fold,
        verbosity=-1,
    )
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    oof_lgb[val_idx] = lgb_model.predict(X_va)
    test_preds_lgb += lgb_model.predict(X_test) / kf.n_splits

    # Model 2: XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=6,
        random_state=42 + fold,
        enable_categorical=True,
        tree_method="hist",
        early_stopping_rounds=50,
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict(X_va)
    test_preds_xgb += xgb_model.predict(X_test) / kf.n_splits

    # Model 3: CatBoost
    cat_model = CatBoostRegressor(
        iterations=1200,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        early_stopping_rounds=50,
        verbose=0,
    )
    cat_model.fit(
        X_tr[numeric_and_le_cols],
        y_tr,
        eval_set=(X_va[numeric_and_le_cols], y_va),
        verbose=False,
    )
    oof_cat[val_idx] = cat_model.predict(X_va[numeric_and_le_cols])
    test_preds_cat += cat_model.predict(X_test[numeric_and_le_cols]) / kf.n_splits

# 4. Meta-Learner Blending
oof_matrix = np.column_stack([oof_lgb, oof_xgb, oof_cat])
test_matrix = np.column_stack([test_preds_lgb, test_preds_xgb, test_preds_cat])

meta_model = Ridge(alpha=1.0, positive=True)
meta_model.fit(oof_matrix, y)

oof_blend = meta_model.predict(oof_matrix)
test_blend = meta_model.predict(test_matrix)

final_rmse = root_mean_squared_error(y, oof_blend)
print(f"Validation RMSE: {final_rmse:.5f}")

# 5. Format and Save Final Output
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame()
if id_col and id_col in test_df.columns:
    submission["id"] = test_df[id_col].astype(int)
else:
    submission["id"] = np.arange(len(test_blend), dtype=int)

submission["APR"] = test_blend
submission.to_csv("./working/submission.csv", index=False)
