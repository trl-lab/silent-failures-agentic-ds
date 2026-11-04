import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Encode target labels
target_col = "Class"
le = LabelEncoder()
y = le.fit_transform(train_df[target_col])
num_classes = len(le.classes_)


# Feature engineering function
def engineer_features(df):
    data = df.copy()

    # Area and perimeter ratios
    data["Log_Area"] = np.log1p(data["Area"])
    data["Log_Perimeter"] = np.log1p(data["Perimeter"])
    data["Log_ConvexArea"] = np.log1p(data["ConvexArea"])
    data["Convex_Area_Diff"] = data["ConvexArea"] - data["Area"]
    data["Convex_Area_Ratio"] = data["Area"] / (data["ConvexArea"] + 1e-6)

    # Axis and diameter ratios
    data["Major_to_Equiv"] = data["MajorAxisLength"] / (data["EquivDiameter"] + 1e-6)
    data["Minor_to_Equiv"] = data["MinorAxisLength"] / (data["EquivDiameter"] + 1e-6)
    data["Perimeter_to_Equiv"] = data["Perimeter"] / (
        np.pi * data["EquivDiameter"] + 1e-6
    )

    # Elliptical approximations
    ellipse_area = (np.pi * data["MajorAxisLength"] * data["MinorAxisLength"]) / 4.0
    data["Area_to_Ellipse"] = data["Area"] / (ellipse_area + 1e-6)

    # Shape factor interactions
    data["SF1_x_SF2"] = data["ShapeFactor1"] * data["ShapeFactor2"]
    data["SF3_to_SF4"] = data["ShapeFactor3"] / (data["ShapeFactor4"] + 1e-6)
    data["Aspect_x_Eccentricity"] = data["AspectRatio"] * data["Eccentricity"]

    return data


# Identify and engineer feature columns
raw_features = [col for col in train_df.columns if col != target_col]
X_train_eng = engineer_features(train_df[raw_features])
X_test_eng = engineer_features(test_df[raw_features])

# 5-fold Stratified Cross-Validation setup
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

oof_preds_xgb = np.zeros((len(train_df), num_classes))
oof_preds_lgb = np.zeros((len(train_df), num_classes))
oof_preds_cat = np.zeros((len(train_df), num_classes))

test_preds_xgb = np.zeros((len(test_df), num_classes))
test_preds_lgb = np.zeros((len(test_df), num_classes))
test_preds_cat = np.zeros((len(test_df), num_classes))

xgb_params = {
    "objective": "multi:softprob",
    "num_class": num_classes,
    "eval_metric": "mlogloss",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 2,
    "random_state": SEED,
    "n_estimators": 1200,
    "tree_method": "hist",
    "n_jobs": -1,
}

lgb_params = {
    "objective": "multiclass",
    "num_class": num_classes,
    "metric": "multi_logloss",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "random_state": SEED,
    "n_estimators": 1200,
    "n_jobs": -1,
    "verbose": -1,
}

cat_params = {
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "iterations": 1200,
    "learning_rate": 0.05,
    "depth": 6,
    "random_seed": SEED,
    "verbose": False,
    "thread_count": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_eng, y)):
    X_tr, y_tr = X_train_eng.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train_eng.iloc[val_idx], y[val_idx]

    # 1. XGBoost
    model_xgb = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=50)
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_preds_xgb[val_idx] = model_xgb.predict_proba(X_va)
    test_preds_xgb += model_xgb.predict_proba(X_test_eng) / n_splits

    # 2. LightGBM
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_preds_lgb[val_idx] = model_lgb.predict_proba(X_va)
    test_preds_lgb += model_lgb.predict_proba(X_test_eng) / n_splits

    # 3. CatBoost
    model_cat = CatBoostClassifier(**cat_params, early_stopping_rounds=50)
    model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    oof_preds_cat[val_idx] = model_cat.predict_proba(X_va)
    test_preds_cat += model_cat.predict_proba(X_test_eng) / n_splits

# Ensemble probability predictions with equal weighting
oof_ensemble = (oof_preds_xgb + oof_preds_lgb + oof_preds_cat) / 3.0
test_ensemble = (test_preds_xgb + test_preds_lgb + test_preds_cat) / 3.0

# Compute and display individual and ensemble Out-of-Fold accuracy
acc_xgb = accuracy_score(y, np.argmax(oof_preds_xgb, axis=1))
acc_lgb = accuracy_score(y, np.argmax(oof_preds_lgb, axis=1))
acc_cat = accuracy_score(y, np.argmax(oof_preds_cat, axis=1))
oof_accuracy = accuracy_score(y, np.argmax(oof_ensemble, axis=1))

print(f"XGBoost OOF Accuracy: {acc_xgb:.5f}")
print(f"LightGBM OOF Accuracy: {acc_lgb:.5f}")
print(f"CatBoost OOF Accuracy: {acc_cat:.5f}")
print(f"Ensemble OOF Accuracy: {oof_accuracy:.5f}")

# Prepare and save final submission
os.makedirs("./working", exist_ok=True)
final_test_classes = le.inverse_transform(np.argmax(test_ensemble, axis=1))

submission = pd.DataFrame({"id": np.arange(len(test_df)), "Class": final_test_classes})
submission.to_csv("./working/submission.csv", index=False)
print("Submission successfully saved to ./working/submission.csv")
