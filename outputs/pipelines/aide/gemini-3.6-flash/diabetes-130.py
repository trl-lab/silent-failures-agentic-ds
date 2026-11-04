import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

test_ids = np.arange(len(test_df))

# Target processing
target_col = "readmitted"
le_target = LabelEncoder()
y = le_target.fit_transform(train_df[target_col])
train_df = train_df.drop(columns=[target_col])

# Combine for uniform preprocessing
df = pd.concat([train_df, test_df], ignore_index=True)
n_train = len(train_df)

# Clean missing values
df = df.replace(
    {"?": np.nan, "Unknown/Invalid": np.nan, "None": np.nan, "none": np.nan}
)

# Feature 1: Explicit clinical status indicators from integer IDs
if "discharge_disposition_id" in df.columns:
    expired_hospice_ids = [11, 13, 14, 19, 20, 21, 28]
    df["is_expired_hospice"] = (
        df["discharge_disposition_id"].isin(expired_hospice_ids).astype(int)
    )
    df["is_discharged_home"] = (df["discharge_disposition_id"] == 1).astype(int)

if "admission_type_id" in df.columns and "admission_source_id" in df.columns:
    df["is_emergency_admission"] = (
        (df["admission_type_id"] == 1) | (df["admission_source_id"] == 7)
    ).astype(int)

# Feature 2: Age mapping to numerical midpoints
age_map = {
    "[0-10)": 5,
    "[10-20)": 15,
    "[20-30)": 25,
    "[30-40)": 35,
    "[40-50)": 45,
    "[50-60)": 55,
    "[60-70)": 65,
    "[70-80)": 75,
    "[80-90)": 85,
    "[90-100)": 95,
}
df["age_num"] = df["age"].map(age_map).fillna(55).astype(float)


# Feature 3: ICD-9 Grouping function
def map_icd9(code):
    if pd.isna(code):
        return "Missing"
    code_str = str(code).strip()
    if code_str.startswith("V") or code_str.startswith("E"):
        return "Supplementary"
    try:
        val = float(code_str)
        if (390 <= val <= 459) or val == 785:
            return "Circulatory"
        elif (460 <= val <= 519) or val == 786:
            return "Respiratory"
        elif (520 <= val <= 579) or val == 787:
            return "Digestive"
        elif 250 <= val < 251:
            return "Diabetes"
        elif 800 <= val <= 999:
            return "Injury"
        elif 710 <= val <= 739:
            return "Musculoskeletal"
        elif (580 <= val <= 629) or val == 788:
            return "Genitourinary"
        elif 140 <= val <= 239:
            return "Neoplasms"
        else:
            return "Other"
    except ValueError:
        return "Other"


for col in ["diag_1", "diag_2", "diag_3"]:
    if col in df.columns:
        df[f"{col}_group"] = df[col].apply(map_icd9)

# Feature 4: Visit aggregations
df["total_visits"] = (
    df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
)
df["has_past_inpatient"] = (df["number_inpatient"] > 0).astype(int)
df["has_past_emergency"] = (df["number_emergency"] > 0).astype(int)

# Feature 5: Medication active & modification counts
med_cols = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

existing_med_cols = [c for c in med_cols if c in df.columns]
if existing_med_cols:
    df["med_num_active"] = (df[existing_med_cols] != "No").sum(axis=1)
    df["med_num_up"] = (df[existing_med_cols] == "Up").sum(axis=1)
    df["med_num_down"] = (df[existing_med_cols] == "Down").sum(axis=1)
    df["med_num_steady"] = (df[existing_med_cols] == "Steady").sum(axis=1)
    df["med_num_changes"] = df["med_num_up"] + df["med_num_down"]

# Feature 6: Ratios & intensity
df["labs_per_day"] = df["num_lab_procedures"] / (df["time_in_hospital"] + 1e-5)
df["meds_per_day"] = df["num_medications"] / (df["time_in_hospital"] + 1e-5)
df["procs_per_day"] = df["num_procedures"] / (df["time_in_hospital"] + 1e-5)
df["total_procedures"] = df["num_lab_procedures"] + df["num_procedures"]

# Feature 7: Patient-level encounter statistics across dataset if present
if "patient_nbr" in df.columns:
    df["patient_encounter_count"] = df.groupby("patient_nbr")["patient_nbr"].transform(
        "count"
    )
    df["patient_stay_sum"] = df.groupby("patient_nbr")["time_in_hospital"].transform(
        "sum"
    )
    df["patient_stay_mean"] = df.groupby("patient_nbr")["time_in_hospital"].transform(
        "mean"
    )
    df["patient_inpatient_max"] = df.groupby("patient_nbr")[
        "number_inpatient"
    ].transform("max")

# Drop unnecessary identifier columns
drop_cols = ["encounter_id", "patient_nbr"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Explicit Categorical Processing (convert nominal IDs to categoricals)
id_cat_cols = [
    "discharge_disposition_id",
    "admission_type_id",
    "admission_source_id",
]
for c in id_cat_cols:
    if c in df.columns:
        df[c] = df[c].fillna(-1).astype(str)

# Find all string/object categorical columns
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

# Label encode for unified matrix representations while tracking categorical column names
for c in cat_cols:
    df[c] = df[c].fillna("Missing").astype(str)
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c])

# Split back to train and test sets
X_train = df.iloc[:n_train].copy()
X_test = df.iloc[n_train:].copy()

# Setup 5-fold Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds_lgb = np.zeros((n_train, 3))
oof_preds_cat = np.zeros((n_train, 3))
test_preds_lgb = np.zeros((len(X_test), 3))
test_preds_cat = np.zeros((len(X_test), 3))

lgb_params = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    # Train LightGBM with explicit categorical features
    dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain, categorical_feature=cat_cols)

    model_lgb = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    oof_preds_lgb[val_idx] = model_lgb.predict(X_va)
    test_preds_lgb += model_lgb.predict(X_test) / skf.n_splits

    # Train CatBoost with explicit categorical features
    cat_model = CatBoostClassifier(
        iterations=800,
        learning_rate=0.04,
        depth=6,
        loss_function="MultiClass",
        cat_features=cat_cols,
        random_seed=42,
        verbose=0,
    )
    cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50)

    oof_preds_cat[val_idx] = cat_model.predict_proba(X_va)
    test_preds_cat += cat_model.predict_proba(X_test) / skf.n_splits

# Ensemble predictions (50/50 blend)
oof_blend = 0.5 * oof_preds_lgb + 0.5 * oof_preds_cat
test_blend = 0.5 * test_preds_lgb + 0.5 * test_preds_cat


# Class multiplier calibration optimization for accuracy score
def loss_func(weights):
    w1, w2, w3 = weights
    weighted_oof = oof_blend * np.array([w1, w2, w3])
    preds = np.argmax(weighted_oof, axis=1)
    return -accuracy_score(y, preds)


res = minimize(loss_func, [1.0, 1.0, 1.0], method="Nelder-Mead")
best_weights = res.x

oof_weighted = oof_blend * np.array(best_weights)
oof_pred_labels = np.argmax(oof_weighted, axis=1)
cv_accuracy = accuracy_score(y, oof_pred_labels)

# Generate final test predictions
test_weighted = test_blend * np.array(best_weights)
test_pred_labels = np.argmax(test_weighted, axis=1)
test_pred_strings = le_target.inverse_transform(test_pred_labels)

# Save submission.csv
submission = pd.DataFrame({"id": test_ids, "readmitted": test_pred_strings})
os.makedirs("./working", exist_ok=True)
submission.to_csv("./working/submission.csv", index=False)

print(f"Validation Accuracy: {cv_accuracy:.5f}")
