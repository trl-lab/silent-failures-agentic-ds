import json
import os
from typing import Any, Dict, Optional

from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

# =========================================================================
# 0. Setup Directories and Target Mappings
# =========================================================================
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

TARGET_MAP = {
    "Insufficient_Weight": 0,
    "Normal_Weight": 1,
    "Overweight_Level_I": 2,
    "Overweight_Level_II": 3,
    "Obesity_Type_I": 4,
    "Obesity_Type_II": 5,
    "Obesity_Type_III": 6,
}
REV_TARGET_MAP = {v: k for k, v in TARGET_MAP.items()}
NUM_CLASSES = len(TARGET_MAP)

# =========================================================================
# 1. Feature Engineering
# =========================================================================


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # Anthropometric & Physical Measurements
    data["BMI"] = data["Weight"] / (data["Height"] ** 2)
    data["Ponderal_Index"] = data["Weight"] / (data["Height"] ** 3)
    data["BMI_Prime"] = data["BMI"] / 25.0
    data["Body_Shape_Index"] = data["Weight"] / (data["Height"] ** 1.5)
    data["Weight_to_Height"] = data["Weight"] / data["Height"]
    data["Height_Weight_Product"] = data["Height"] * data["Weight"]

    # Ideal Body Weight (Devine formula approximation)
    is_male = (data["Gender"] == "Male").astype(float)
    height_inches = data["Height"] * 39.3700787
    inches_over_5ft = np.maximum(0, height_inches - 60.0)
    ideal_weight = (50.0 * is_male + 45.5 * (1.0 - is_male)) + 2.3 * inches_over_5ft
    data["Ideal_Weight"] = ideal_weight
    data["Weight_Diff_Ideal"] = data["Weight"] - ideal_weight
    data["Weight_Ratio_Ideal"] = data["Weight"] / (ideal_weight + 1e-5)

    # WHO BMI Classification Categories
    data["WHO_BMI_Cat"] = np.select(
        [
            data["BMI"] < 18.5,
            (data["BMI"] >= 18.5) & (data["BMI"] < 25.0),
            (data["BMI"] >= 25.0) & (data["BMI"] < 30.0),
            (data["BMI"] >= 30.0) & (data["BMI"] < 35.0),
            (data["BMI"] >= 35.0) & (data["BMI"] < 40.0),
            data["BMI"] >= 40.0,
        ],
        [0, 1, 2, 3, 4, 5],
        default=1,
    ).astype(float)

    # Ordinal and Binary Survey Encodings
    caec_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    calc_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    binary_map = {"no": 0, "yes": 1}
    gender_map = {"Male": 0, "Female": 1}

    data["CAEC_Num"] = data["CAEC"].map(caec_map).fillna(1).astype(float)
    data["CALC_Num"] = data["CALC"].map(calc_map).fillna(0).astype(float)
    data["Gender_Num"] = data["Gender"].map(gender_map).fillna(0).astype(float)
    data["FAVC_Num"] = data["FAVC"].map(binary_map).fillna(0).astype(float)
    data["SMOKE_Num"] = data["SMOKE"].map(binary_map).fillna(0).astype(float)
    data["SCC_Num"] = data["SCC"].map(binary_map).fillna(0).astype(float)
    data["Family_Overweight_Num"] = (
        data["family_history_with_overweight"].map(binary_map).fillna(0).astype(float)
    )

    # Transportation physical intensity score
    mtrans_activity_map = {
        "Walking": 3.0,
        "Bike": 3.0,
        "Public_Transportation": 2.0,
        "Motorbike": 1.0,
        "Automobile": 1.0,
    }
    data["MTRANS_Active_Score"] = (
        data["MTRANS"].map(mtrans_activity_map).fillna(2.0).astype(float)
    )

    # Behavioral & Lifestyle Interaction Features
    data["Water_per_kg"] = data["CH2O"] / (data["Weight"] + 1e-5)
    data["Water_per_BMI"] = data["CH2O"] / (data["BMI"] + 1e-5)
    data["Activity_to_Sedentary"] = data["FAF"] / (data["TUE"] + 0.1)
    data["Total_Active_Score"] = (
        (data["FAF"] * 2.0) + data["MTRANS_Active_Score"] - (data["TUE"] * 1.5)
    )
    data["Meal_Snack_Ratio"] = data["NCP"] / (data["CAEC_Num"] + 1.0)
    data["Healthy_Habit_Score"] = (
        (data["FCVC"] * 1.5)
        + (data["CH2O"] * 1.0)
        + (data["SCC_Num"] * 2.0)
        - (data["FAVC_Num"] * 2.0)
        - (data["CALC_Num"] * 1.0)
    )

    # Non-linear & polynomial anthropometric features
    data["Height_Sq"] = data["Height"] ** 2
    data["Height_Cube"] = data["Height"] ** 3
    data["Weight_Sq"] = data["Weight"] ** 2
    data["Log_BMI"] = np.log1p(data["BMI"])
    data["Log_Weight"] = np.log1p(data["Weight"])
    data["Log_Age"] = np.log1p(data["Age"])

    # Gender-Anthropometric Interactions (resolving Type II vs Type III boundary)
    data["Gender_Height"] = data["Gender_Num"] * data["Height"]
    data["Gender_Weight"] = data["Gender_Num"] * data["Weight"]
    data["Gender_Weight_to_Height"] = data["Gender_Num"] * data["Weight_to_Height"]
    data["Gender_Ideal_Weight"] = data["Gender_Num"] * ideal_weight
    data["Gender_Weight_Diff_Ideal"] = data["Gender_Num"] * data["Weight_Diff_Ideal"]
    data["Gender_Age"] = data["Gender_Num"] * data["Age"]

    # Metabolic Balance Proxies
    data["Calorie_Proxy"] = (
        (data["FAVC_Num"] * 2.0)
        + data["CAEC_Num"]
        + (data["NCP"] * 0.5)
        + (data["CALC_Num"] * 0.5)
    )
    data["Energy_Expenditure_Proxy"] = (
        (data["FAF"] * 1.5) + data["MTRANS_Active_Score"] - (data["TUE"] * 0.8)
    )
    data["Metabolic_Balance"] = data["Calorie_Proxy"] / (
        data["Energy_Expenditure_Proxy"] + 1.0
    )

    # Age and physiological interactions
    data["Age_BMI"] = data["Age"] * data["BMI"]
    data["Age_FAF"] = data["Age"] * data["FAF"]
    data["Gender_BMI"] = data["Gender_Num"] * data["BMI"]
    data["Family_History_BMI"] = data["Family_Overweight_Num"] * data["BMI"]
    data["Family_History_Weight"] = data["Family_Overweight_Num"] * data["Weight"]

    # Demographic Cohort Aggregated Statistics
    age_cohort = pd.cut(
        data["Age"],
        bins=[0, 18, 25, 30, 40, 50, 100],
        labels=[0, 1, 2, 3, 4, 5],
    ).astype(float)
    data["Age_Cohort"] = age_cohort
    group_bmi_mean = data.groupby(["Gender_Num", "Age_Cohort"], observed=False)["BMI"].transform("mean")
    group_bmi_std = data.groupby(["Gender_Num", "Age_Cohort"], observed=False)["BMI"].transform("std").fillna(0)
    group_weight_mean = data.groupby(["Gender_Num", "Age_Cohort"], observed=False)["Weight"].transform("mean")
    group_weight_std = data.groupby(["Gender_Num", "Age_Cohort"], observed=False)["Weight"].transform("std").fillna(0)
    data["Group_BMI_Mean"] = group_bmi_mean
    data["Group_BMI_Std"] = group_bmi_std
    data["Group_Weight_Mean"] = group_weight_mean
    data["Group_Weight_Std"] = group_weight_std
    data["BMI_Diff_Group_Mean"] = data["BMI"] - group_bmi_mean
    data["Weight_Diff_Group_Mean"] = data["Weight"] - group_weight_mean

    # Discretized survey approximations
    data["FCVC_Round"] = np.round(data["FCVC"])
    data["NCP_Round"] = np.round(data["NCP"])
    data["CH2O_Round"] = np.round(data["CH2O"])
    data["FAF_Round"] = np.round(data["FAF"])
    data["TUE_Round"] = np.round(data["TUE"])

    # One-hot encoding for MTRANS categorical feature
    mtrans_dummies = pd.get_dummies(data["MTRANS"], prefix="MTRANS", dtype=float)
    for col in [
        "MTRANS_Automobile",
        "MTRANS_Bike",
        "MTRANS_Motorbike",
        "MTRANS_Public_Transportation",
        "MTRANS_Walking",
    ]:
        if col not in mtrans_dummies.columns:
            mtrans_dummies[col] = 0.0
    data = pd.concat([data, mtrans_dummies], axis=1)

    # Drop raw categorical columns
    cols_to_drop = [
        "Gender",
        "family_history_with_overweight",
        "FAVC",
        "CAEC",
        "SMOKE",
        "SCC",
        "CALC",
        "MTRANS",
    ]
    data = data.drop(columns=cols_to_drop, errors="ignore")

    return data


# =========================================================================
# 2. Model Factory Definitions
# =========================================================================


def get_lgb_model(
    custom_params: Optional[Dict[str, Any]] = None,
) -> lgb.LGBMClassifier:
    default_params = {
        "objective": "multiclass",
        "num_class": NUM_CLASSES,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if custom_params:
        default_params.update(custom_params)
    return lgb.LGBMClassifier(**default_params)


def get_xgb_model(
    custom_params: Optional[Dict[str, Any]] = None,
) -> xgb.XGBClassifier:
    default_params = {
        "objective": "multi:softprob",
        "num_class": NUM_CLASSES,
        "eval_metric": "mlogloss",
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 2,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
        "early_stopping_rounds": 50,
    }
    if custom_params:
        default_params.update(custom_params)
    return xgb.XGBClassifier(**default_params)


def get_catboost_model(
    custom_params: Optional[Dict[str, Any]] = None,
) -> CatBoostClassifier:
    default_params = {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "iterations": 2000,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "random_seed": 42,
        "thread_count": -1,
        "verbose": False,
        "early_stopping_rounds": 50,
    }
    if custom_params:
        default_params.update(custom_params)
    return CatBoostClassifier(**default_params)


# =========================================================================
# 3. Execution: Data Loading, Splitting, and Model Training
# =========================================================================
print("Loading raw datasets...")
raw_train = pd.read_parquet("./input/train.parquet")
raw_test = pd.read_parquet("./input/test.parquet")

train_df = engineer_features(raw_train)
test_df = engineer_features(raw_test)

train_df["target"] = train_df["NObeyesdad"].map(TARGET_MAP)
train_df = train_df.drop(columns=["NObeyesdad"])

# Load and process original survey dataset for fold-wise training augmentation
orig_path = "./input/original/train.csv"
if os.path.exists(orig_path):
    print(f"Loading and augmenting with original survey data from {orig_path}...")
    raw_orig = pd.read_csv(orig_path)
    if "id" not in raw_orig.columns:
        raw_orig["id"] = -1
    orig_df = engineer_features(raw_orig)
    if "NObeyesdad" in orig_df.columns:
        orig_df["target"] = orig_df["NObeyesdad"].map(TARGET_MAP)
        orig_df = orig_df.drop(columns=["NObeyesdad"])
        orig_df = orig_df.dropna(subset=["target"])
        orig_df["target"] = orig_df["target"].astype(int)
    print(f"Original dataset processed: {len(orig_df)} records.")
else:
    orig_df = pd.DataFrame()

feature_cols = sorted(
    [c for c in train_df.columns if c not in ["id", "target", "fold"]]
)
print(f"Dataset processed: {len(feature_cols)} features engineered.")

# Align columns for original dataset if present
if len(orig_df) > 0:
    for col in feature_cols:
        if col not in orig_df.columns:
            orig_df[col] = 0.0

# Create 5-Fold Stratified Split
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
train_df["fold"] = -1
for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df["target"])):
    train_df.loc[val_idx, "fold"] = fold

X_test = test_df[feature_cols].values

oof_lgb = np.zeros((len(train_df), NUM_CLASSES))
oof_xgb = np.zeros((len(train_df), NUM_CLASSES))
oof_cat = np.zeros((len(train_df), NUM_CLASSES))

test_lgb = np.zeros((len(test_df), NUM_CLASSES))
test_xgb = np.zeros((len(test_df), NUM_CLASSES))
test_cat = np.zeros((len(test_df), NUM_CLASSES))

print(
    f"Starting {n_splits}-Fold Cross-Validation training (LGBM, XGBoost, CatBoost)..."
)

for fold in range(n_splits):
    train_mask = train_df["fold"] != fold
    val_mask = train_df["fold"] == fold

    fold_train_df = train_df.loc[train_mask]
    if len(orig_df) > 0:
        fold_train_df = pd.concat([fold_train_df, orig_df], ignore_index=True)

    X_train, y_train = (
        fold_train_df[feature_cols].values,
        fold_train_df["target"].values.astype(int),
    )
    X_val, y_val = (
        train_df.loc[val_mask, feature_cols].values,
        train_df.loc[val_mask, "target"].values.astype(int),
    )

    # 1. Train LightGBM
    lgb_clf = get_lgb_model({"random_state": 42 + fold})
    lgb_callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    lgb_clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=lgb_callbacks,
    )
    oof_lgb[val_mask] = lgb_clf.predict_proba(X_val)
    test_lgb += lgb_clf.predict_proba(X_test) / n_splits

    # 2. Train XGBoost
    xgb_clf = get_xgb_model({"random_state": 42 + fold})
    xgb_clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    oof_xgb[val_mask] = xgb_clf.predict_proba(X_val)
    test_xgb += xgb_clf.predict_proba(X_test) / n_splits

    # 3. Train CatBoost
    cat_clf = get_catboost_model({"random_seed": 42 + fold})
    cat_clf.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        verbose=False,
    )
    oof_cat[val_mask] = cat_clf.predict_proba(X_val)
    test_cat += cat_clf.predict_proba(X_test) / n_splits

    fold_blend = (
        (oof_lgb[val_mask] + oof_xgb[val_mask] + oof_cat[val_mask]) / 3.0
    )
    fold_acc = accuracy_score(y_val, np.argmax(fold_blend, axis=1))
    print(f"Fold {fold} - Multi-Class Accuracy: {fold_acc:.5f}")

# =========================================================================
# 4. Evaluation and Prediction Submission
# =========================================================================
y_true = train_df["target"].values.astype(int)

# Optimize ensemble weights directly using SLSQP
def objective(weights):
    w = np.array(weights)
    w = np.maximum(0, w)
    w = w / (np.sum(w) + 1e-12)
    blend = w[0] * oof_lgb + w[1] * oof_xgb + w[2] * oof_cat
    eps = 1e-15
    blend = np.clip(blend, eps, 1.0 - eps)
    blend = blend / blend.sum(axis=1, keepdims=True)
    return log_loss(y_true, blend)

init_weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
constraints = {"type": "eq", "fun": lambda w: sum(w) - 1.0}

opt_res = minimize(
    objective,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)

opt_weights = np.maximum(0, opt_res.x)
opt_weights = opt_weights / np.sum(opt_weights)
print(f"Optimal Ensemble Weights: LGBM={opt_weights[0]:.4f}, XGB={opt_weights[1]:.4f}, CatBoost={opt_weights[2]:.4f}")

oof_blend = (
    opt_weights[0] * oof_lgb
    + opt_weights[1] * oof_xgb
    + opt_weights[2] * oof_cat
)

lgb_acc = accuracy_score(y_true, np.argmax(oof_lgb, axis=1))
xgb_acc = accuracy_score(y_true, np.argmax(oof_xgb, axis=1))
cat_acc = accuracy_score(y_true, np.argmax(oof_cat, axis=1))
overall_score = accuracy_score(y_true, np.argmax(oof_blend, axis=1))

print(f"LGBM OOF Accuracy:     {lgb_acc:.5f}")
print(f"XGBoost OOF Accuracy:  {xgb_acc:.5f}")
print(f"CatBoost OOF Accuracy: {cat_acc:.5f}")
print(f"Ensemble OOF Accuracy: {overall_score:.5f}")

# Generate Test Predictions
test_blend = (
    opt_weights[0] * test_lgb
    + opt_weights[1] * test_xgb
    + opt_weights[2] * test_cat
)
test_pred_indices = np.argmax(test_blend, axis=1)
test_pred_labels = [REV_TARGET_MAP[idx] for idx in test_pred_indices]

submission_df = pd.DataFrame(
    {"id": test_df["id"].astype(int), "NObeyesdad": test_pred_labels}
)

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)

# Submission Integrity Verification
assert len(submission_df) == len(
    test_df
), f"Submission length mismatch: {len(submission_df)} vs {len(test_df)}"
assert list(submission_df.columns) == [
    "id",
    "NObeyesdad",
], f"Incorrect columns: {submission_df.columns.tolist()}"
assert (
    submission_df["NObeyesdad"].isna().sum() == 0
), "Submission contains null predictions."

print(
    f"Submission successfully generated at {submission_path} with {len(submission_df)} rows."
)
print(f"Final Validation Score: {overall_score}")
