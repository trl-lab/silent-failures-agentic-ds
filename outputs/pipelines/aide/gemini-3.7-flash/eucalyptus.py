import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "Utility"
le_target = LabelEncoder()
y_train = le_target.fit_transform(train_df[target_col].astype(str))

n_train = len(train_df)
df = pd.concat([train_df.drop(columns=[target_col]), test_df], ignore_index=True)

# 2. Domain & Missingness Feature Engineering
morph_cols = [
    "DBH",
    "Ht",
    "Surv",
    "Vig",
    "Ins_res",
    "Stem_Fm",
    "Crown_Fm",
    "Brnch_Fm",
]
env_cols = ["Rainfall", "Altitude", "Frosts"]

for col in morph_cols + env_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Explicit missingness patterns reflecting mortality / data absence
for col in morph_cols:
    df[f"{col}_isna"] = df[col].isna().astype(np.float32)
df["Morph_Missing_Count"] = df[[f"{c}_isna" for c in morph_cols]].sum(axis=1)

# Morphology & composite quality indicators
df["Slenderness"] = df["Ht"] / (df["DBH"] + 1e-4)
df["Volume_Proxy"] = (df["DBH"] ** 2) * df["Ht"]
df["Total_Form"] = (
    df["Stem_Fm"].fillna(0) + df["Crown_Fm"].fillna(0) + df["Brnch_Fm"].fillna(0)
)
df["Mean_Form"] = df[["Stem_Fm", "Crown_Fm", "Brnch_Fm"]].mean(axis=1)
df["Vitality_Score"] = df["Surv"] * df["Vig"]
df["Health_Index"] = df["Vig"] * (df["Ins_res"] + 1e-4)
df["Growth_Index"] = df["Ht"] * df["Vig"]

# Environmental interactions
df["Rain_per_Alt"] = df["Rainfall"] / (df["Altitude"] + 1.0)
df["Frost_Alt_Stress"] = df["Frosts"] * (df["Altitude"] + 1.0)
df["Rain_per_Frost"] = df["Rainfall"] / (np.abs(df["Frosts"]) + 1.0)

# Categorical column handling & Frequency Encodings
cat_cols = ["Abbrev", "Locality", "Map_Ref", "Latitude", "Sp", "Rep", "PMCno", "Year"]
for col in cat_cols:
    df[col] = df[col].astype(str)
    freq = df[col].value_counts()
    df[f"{col}_freq"] = df[col].map(freq)
    df[col] = LabelEncoder().fit_transform(df[col])

X_train_full = df.iloc[:n_train].copy().reset_index(drop=True)
X_test_full = df.iloc[n_train:].copy().reset_index(drop=True)
num_classes = len(np.unique(y_train))

# 3. Model Training with In-Fold Species Relative Benchmark Features
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb = np.zeros((n_train, num_classes))
oof_xgb = np.zeros((n_train, num_classes))
oof_cat = np.zeros((n_train, num_classes))

test_preds_lgb = np.zeros((len(X_test_full), num_classes))
test_preds_xgb = np.zeros((len(X_test_full), num_classes))
test_preds_cat = np.zeros((len(X_test_full), num_classes))

stat_cols = ["Ht", "DBH", "Vig", "Total_Form"]

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train)):
    X_tr = X_train_full.iloc[train_idx].copy()
    y_tr = y_train[train_idx]
    X_va = X_train_full.iloc[val_idx].copy()
    y_va = y_train[val_idx]
    X_te = X_test_full.copy()

    # Species-relative benchmark features computed strictly inside the fold
    for stat_col in stat_cols:
        sp_mean = X_tr.groupby("Sp")[stat_col].mean()
        overall_mean = X_tr[stat_col].mean()

        for d in (X_tr, X_va, X_te):
            mapped_mean = d["Sp"].map(sp_mean).fillna(overall_mean)
            d[f"Sp_mean_{stat_col}"] = mapped_mean
            d[f"Rel_diff_{stat_col}"] = d[stat_col] - mapped_mean
            d[f"Rel_ratio_{stat_col}"] = d[stat_col] / (mapped_mean + 1e-4)

    # 1. LightGBM Classifier
    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    lgb_val = lgb.Dataset(X_va, label=y_va, reference=lgb_train)

    lgb_params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": 4,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 8,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "random_state": 42 + fold,
        "verbosity": -1,
    }

    model_lgb = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=1200,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
    )
    oof_lgb[val_idx] = model_lgb.predict(X_va, num_iteration=model_lgb.best_iteration)
    test_preds_lgb += (
        model_lgb.predict(X_te, num_iteration=model_lgb.best_iteration) / n_splits
    )

    # 2. XGBoost Classifier
    model_xgb = xgb.XGBClassifier(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42 + fold,
        eval_metric="mlogloss",
        early_stopping_rounds=60,
        tree_method="hist",
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict_proba(X_va)
    test_preds_xgb += model_xgb.predict_proba(X_te) / n_splits

    # 3. CatBoost Classifier
    model_cat = CatBoostClassifier(
        iterations=1200,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=3.0,
        random_seed=42 + fold,
        early_stopping_rounds=60,
        verbose=False,
    )
    model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    oof_cat[val_idx] = model_cat.predict_proba(X_va)
    test_preds_cat += model_cat.predict_proba(X_te) / n_splits

# 4. Optimal Weighted Blending & Evaluation
weights = [0.35, 0.30, 0.35]
oof_ensemble = (oof_lgb * weights[0]) + (oof_xgb * weights[1]) + (oof_cat * weights[2])
oof_preds = np.argmax(oof_ensemble, axis=1)
accuracy = accuracy_score(y_train, oof_preds)

print(f"5-Fold Cross-Validation Accuracy: {accuracy:.5f}")

# 5. Generate Test Predictions & Save Submission
test_ensemble = (
    (test_preds_lgb * weights[0])
    + (test_preds_xgb * weights[1])
    + (test_preds_cat * weights[2])
)
final_test_classes = np.argmax(test_ensemble, axis=1)
final_test_labels = le_target.inverse_transform(final_test_classes)

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(final_test_labels)), "Utility": final_test_labels}
)
submission.to_csv("./working/submission.csv", index=False)
print(f"Submission successfully saved with shape: {submission.shape}")
