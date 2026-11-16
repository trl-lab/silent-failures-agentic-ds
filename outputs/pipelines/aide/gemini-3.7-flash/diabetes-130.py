import os
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# 1. Load Data
os.makedirs("./working", exist_ok=True)
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "readmitted"
y_raw = train_df[target_col].copy()

# Encode target
target_le = LabelEncoder()
y = target_le.fit_transform(y_raw)

# Separate features
X_train_raw = train_df.drop(columns=[target_col], errors="ignore")
X_test_raw = test_df.copy()

n_train = len(train_df)
df_all = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)

# Clean missing values
df_all.replace("?", np.nan, inplace=True)
df_all.replace("None", np.nan, inplace=True)
df_all.replace("Unknown/Invalid", np.nan, inplace=True)

# 2. Patient-Level Longitudinal & Historical Features
if "patient_nbr" in df_all.columns:
    patient_counts = df_all["patient_nbr"].value_counts()
    df_all["patient_encounter_count"] = df_all["patient_nbr"].map(patient_counts)
    df_all["is_repeat_patient"] = (df_all["patient_encounter_count"] > 1).astype(int)

    if "encounter_id" in df_all.columns:
        df_all["encounter_id_num"] = pd.to_numeric(
            df_all["encounter_id"], errors="coerce"
        ).fillna(0)
        df_all["patient_encounter_seq"] = (
            df_all.groupby("patient_nbr")["encounter_id_num"]
            .rank(method="first")
            .astype(int)
        )
        df_all["is_first_encounter"] = (df_all["patient_encounter_seq"] == 1).astype(
            int
        )
        df_all.drop(columns=["encounter_id_num"], inplace=True)

    patient_time_mean = df_all.groupby("patient_nbr")["time_in_hospital"].transform(
        "mean"
    )
    patient_num_meds_mean = df_all.groupby("patient_nbr")["num_medications"].transform(
        "mean"
    )
    patient_num_labs_mean = df_all.groupby("patient_nbr")[
        "num_lab_procedures"
    ].transform("mean")

    df_all["patient_time_in_hospital_mean"] = patient_time_mean
    df_all["patient_num_meds_mean"] = patient_num_meds_mean
    df_all["patient_num_labs_mean"] = patient_num_labs_mean


# 3. Clinical & ICD-9 Domain Features
def categorize_icd9(val):
    if pd.isna(val):
        return "Missing"
    val_str = str(val).strip()
    if val_str.startswith("V") or val_str.startswith("v"):
        return "Supplementary_V"
    if val_str.startswith("E") or val_str.startswith("e"):
        return "External_E"
    try:
        code = float(val_str)
        if 390 <= code <= 459 or code == 785:
            return "Circulatory"
        elif 460 <= code <= 519 or code == 786:
            return "Respiratory"
        elif 520 <= code <= 579 or code == 787:
            return "Digestive"
        elif 250 <= code < 251:
            return "Diabetes"
        elif 800 <= code <= 999:
            return "Injury"
        elif 710 <= code <= 739:
            return "Musculoskeletal"
        elif 580 <= code <= 629 or code == 788:
            return "Genitourinary"
        elif 140 <= code <= 239:
            return "Neoplasms"
        else:
            return "Other"
    except ValueError:
        return "Other"


for diag in ["diag_1", "diag_2", "diag_3"]:
    if diag in df_all.columns:
        df_all[f"{diag}_cat"] = df_all[diag].apply(categorize_icd9)
        code_freq = df_all[diag].value_counts(normalize=True).to_dict()
        df_all[f"{diag}_freq"] = df_all[diag].map(code_freq).fillna(0).astype(float)

# Discharge Status Flags
if "discharge_disposition_id" in df_all.columns:
    disp_num = pd.to_numeric(
        df_all["discharge_disposition_id"], errors="coerce"
    ).fillna(-1)
    df_all["is_expired_or_hospice"] = disp_num.isin(
        [11, 13, 14, 19, 20, 21, 25]
    ).astype(int)
    df_all["is_discharged_home"] = (disp_num == 1).astype(int)

# Medication usage features
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
present_med_cols = [c for c in med_cols if c in df_all.columns]

active_count = np.zeros(len(df_all), dtype=int)
change_count = np.zeros(len(df_all), dtype=int)
up_count = np.zeros(len(df_all), dtype=int)
down_count = np.zeros(len(df_all), dtype=int)

for col in present_med_cols:
    vals = df_all[col].astype(str).str.lower()
    active_count += vals.isin(["up", "down", "steady"]).astype(int)
    change_count += vals.isin(["up", "down"]).astype(int)
    up_count += (vals == "up").astype(int)
    down_count += (vals == "down").astype(int)

df_all["num_active_meds"] = active_count
df_all["num_med_changes"] = change_count
df_all["num_med_up"] = up_count
df_all["num_med_down"] = down_count

# Healthcare visits metrics
outpatient = (
    df_all["number_outpatient"].fillna(0)
    if "number_outpatient" in df_all.columns
    else 0
)
emergency = (
    df_all["number_emergency"].fillna(0) if "number_emergency" in df_all.columns else 0
)
inpatient = (
    df_all["number_inpatient"].fillna(0) if "number_inpatient" in df_all.columns else 0
)

df_all["total_visits"] = outpatient + emergency + inpatient
df_all["has_prior_inpatient"] = (inpatient > 0).astype(int)
df_all["has_prior_emergency"] = (emergency > 0).astype(int)
df_all["inpatient_ratio"] = inpatient / (df_all["total_visits"] + 1e-4)

# Interaction ratios
if "num_lab_procedures" in df_all.columns and "time_in_hospital" in df_all.columns:
    df_all["labs_per_day"] = df_all["num_lab_procedures"] / (
        df_all["time_in_hospital"] + 1e-5
    )
    df_all["meds_per_day"] = df_all["num_medications"] / (
        df_all["time_in_hospital"] + 1e-5
    )
    df_all["procs_per_day"] = df_all["num_procedures"] / (
        df_all["time_in_hospital"] + 1e-5
    )
    df_all["med_to_lab_ratio"] = (df_all["num_medications"] + 1) / (
        df_all["num_lab_procedures"] + 1
    )

# Age numeric mapping
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
if "age" in df_all.columns:
    df_all["age_num"] = df_all["age"].map(age_map).fillna(55).astype(float)

# Lab result indicators
if "A1Cresult" in df_all.columns:
    a1c_map = {"None": 0, "Norm": 1, ">7": 2, ">8": 3}
    df_all["a1c_ordinal"] = df_all["A1Cresult"].map(a1c_map).fillna(0)

if "max_glu_serum" in df_all.columns:
    glu_map = {"None": 0, "Norm": 1, ">200": 2, ">300": 3}
    df_all["glu_ordinal"] = df_all["max_glu_serum"].map(glu_map).fillna(0)

# Drop raw IDs
drop_cols = ["encounter_id", "patient_nbr"]
df_all.drop(columns=[c for c in drop_cols if c in df_all.columns], inplace=True)

# Categorical column handling
cat_cols = list(df_all.select_dtypes(include=["object", "category"]).columns)
for col in cat_cols:
    df_all[col] = df_all[col].astype(str).fillna("Missing")
    le = LabelEncoder()
    df_all[col] = le.fit_transform(df_all[col])

X = df_all.iloc[:n_train].copy()
X_test = df_all.iloc[n_train:].copy()

# 4. Stratified K-Fold Cross-Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
num_classes = len(np.unique(y))

oof_lgb = np.zeros((n_train, num_classes))
oof_cb = np.zeros((n_train, num_classes))
test_preds_lgb = np.zeros((len(X_test), num_classes))
test_preds_cb = np.zeros((len(X_test), num_classes))

lgb_params = {
    "objective": "multiclass",
    "num_class": num_classes,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    # LightGBM
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)
    model_lgb = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=600,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    oof_lgb[val_idx] = model_lgb.predict(X_va, num_iteration=model_lgb.best_iteration)
    test_preds_lgb += (
        model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration) / n_splits
    )

    # CatBoost
    model_cb = CatBoostClassifier(
        iterations=600,
        learning_rate=0.05,
        depth=6,
        loss_function="MultiClass",
        random_seed=42 + fold,
        early_stopping_rounds=50,
        verbose=False,
        thread_count=-1,
    )
    model_cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    oof_cb[val_idx] = model_cb.predict_proba(X_va)
    test_preds_cb += model_cb.predict_proba(X_test) / n_splits

# Blend OOF predictions
oof_blend = 0.5 * oof_lgb + 0.5 * oof_cb
oof_pred_labels = np.argmax(oof_blend, axis=1)
cv_accuracy = accuracy_score(y, oof_pred_labels)

print(f"5-Fold CV Accuracy: {cv_accuracy:.5f}")

# 5. Predict on Test and Save Submission
test_blend = 0.5 * test_preds_lgb + 0.5 * test_preds_cb
test_pred_labels = np.argmax(test_blend, axis=1)
test_pred_raw = target_le.inverse_transform(test_pred_labels)

submission = pd.DataFrame({"id": np.arange(len(test_df)), "readmitted": test_pred_raw})

submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully with shape:", submission.shape)
