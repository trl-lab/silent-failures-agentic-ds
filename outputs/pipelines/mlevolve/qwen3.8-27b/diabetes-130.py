import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

# ============================================================
# 1. LOAD DATA
# ============================================================
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Target distribution:\n{train_df['readmitted'].value_counts()}")

# ============================================================
# 2. IDENTIFY COLUMNS
# ============================================================
id_cols = ["encounter_id", "patient_nbr"]
target_col = "readmitted"

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

ordinal_med_map = {"no": 0, "no change": 1, "st": 2, "up": 3, "down": 4}

lab_cols = ["max_glu_serum", "A1Cresult"]

count_cols = [
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "time_in_hospital",
]

high_card_cols = ["diag_1", "diag_2", "diag_3"]

low_card_cols = [
    "race",
    "gender",
    "age",
    "weight",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "payer_code",
    "medical_specialty",
]

feature_cols = [c for c in train_df.columns if c not in id_cols + [target_col]]
print(f"Total feature columns: {len(feature_cols)}")

# ============================================================
# 3. SPLIT TRAIN INTO TRAIN/VAL (BEFORE any processing)
# ============================================================
train_part, val_part = train_test_split(
    train_df, test_size=0.2, random_state=42, stratify=train_df[target_col]
)
print(f"Train part: {train_part.shape}, Val part: {val_part.shape}")


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
def engineer_features(df, is_train=False, freq_maps=None, cat_maps=None):
    df = df.copy()

    # 4a. Ordinal encoding of medication columns
    for col in med_cols:
        if col in df.columns:
            df[col] = df[col].map(ordinal_med_map).fillna(0).astype(int)

    # 4b. Medication composite features
    df["num_meds_prescribed"] = (df[med_cols] > 0).sum(axis=1)
    df["num_meds_up"] = (df[med_cols] == 3).sum(axis=1)
    df["num_meds_down"] = (df[med_cols] == 4).sum(axis=1)
    df["num_meds_st"] = (df[med_cols] == 2).sum(axis=1)
    df["med_activity_score"] = df[med_cols].sum(axis=1)
    df["has_insulin"] = (df["insulin"] > 0).astype(int)
    df["has_metformin"] = (df["metformin"] > 0).astype(int)
    df["insulin_metformin_combo"] = (
        (df["insulin"] > 0) & (df["metformin"] > 0)
    ).astype(int)

    # 4c. Lab result encoding
    glu_map = {"none": 0, "normal": 1, ">200": 2, ">300": 3}
    df["max_glu_serum_enc"] = df["max_glu_serum"].map(glu_map).fillna(0).astype(int)

    a1c_map = {"none": 0, "normal": 1, ">7": 2, ">8": 3}
    df["A1Cresult_enc"] = df["A1Cresult"].map(a1c_map).fillna(0).astype(int)

    df["labs_performed"] = (
        (df["max_glu_serum_enc"] > 0) | (df["A1Cresult_enc"] > 0)
    ).astype(int)
    df["glycemic_control_score"] = df["max_glu_serum_enc"] + df["A1Cresult_enc"]

    # 4d. Visit history features
    df["total_visits_prev_year"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )
    df["emergency_ratio"] = np.where(
        df["total_visits_prev_year"] > 0,
        df["number_emergency"] / df["total_visits_prev_year"],
        0,
    )
    df["inpatient_ratio"] = np.where(
        df["total_visits_prev_year"] > 0,
        df["number_inpatient"] / df["total_visits_prev_year"],
        0,
    )
    df["high_utilization"] = (df["total_visits_prev_year"] > 10).astype(int)

    # 4e. Encounter complexity features
    df["total_procedures"] = df["num_lab_procedures"] + df["num_procedures"]
    df["lab_to_proc_ratio"] = np.where(
        df["num_procedures"] > 0, df["num_lab_procedures"] / df["num_procedures"], 0
    )
    df["diagnosis_density"] = np.where(
        df["number_diagnoses"] > 0,
        df["number_diagnoses"] / (df["time_in_hospital"] + 1),
        0,
    )
    df["meds_per_day"] = np.where(
        df["time_in_hospital"] > 0, df["num_medications"] / df["time_in_hospital"], 0
    )

    # 4f. Frequency encoding for high-cardinality diagnosis codes
    if is_train:
        freq_maps = {}
        for col in high_card_cols:
            freq = df[col].value_counts(normalize=True).to_dict()
            freq_maps[col] = freq
            df[f"{col}_freq"] = df[col].map(freq).fillna(0)
    else:
        for col in high_card_cols:
            df[f"{col}_freq"] = df[col].map(freq_maps[col]).fillna(0)

    # 4g. Encode low-cardinality categoricals
    for col in low_card_cols:
        if col in df.columns:
            if is_train:
                df[col] = df[col].astype("category")
                df[f"{col}_code"] = df[col].cat.codes.astype(int)
            else:
                if cat_maps is not None and col in cat_maps:
                    df[f"{col}_code"] = df[col].map(cat_maps[col]).fillna(-1).astype(int)
                else:
                    df[f"{col}_code"] = df[col].astype("category").cat.codes.fillna(-1).astype(int)

    # 4h. Age and weight numeric encoding
    def age_to_num(age_str):
        try:
            s = str(age_str).strip("[]()")
            parts = s.split(",")
            if len(parts) == 2:
                return (float(parts[0].strip()) + float(parts[1].strip())) / 2
            return 0
        except:
            return 0

    df["age_num"] = df["age"].apply(age_to_num)

    def weight_to_num(w):
        try:
            w = str(w).strip()
            if w.startswith(">"):
                return float(w[1:]) + 50
            elif w == "" or w == "nan":
                return 0
            else:
                return float(w)
        except:
            return 0

    df["weight_num"] = df["weight"].apply(weight_to_num)
    df["weight_missing"] = (
        df["weight"].isna()
        | (df["weight"].astype(str).str.strip() == "")
        | (df["weight"].astype(str) == "nan")
    ).astype(int)

    # 4i. Interaction features
    df["stay_x_meds"] = df["time_in_hospital"] * df["num_meds_prescribed"]
    df["stay_x_labs"] = df["time_in_hospital"] * df["num_lab_procedures"]
    df["visits_x_stay"] = df["total_visits_prev_year"] * df["time_in_hospital"]

    # 4j. Binary flags for special admission/discharge
    df["is_emergency_admission"] = (df["admission_type_id"] == 1).astype(int)
    df["expired_during_stay"] = (df["discharge_disposition_id"] == 3).astype(int)
    df["transferred_out"] = df["discharge_disposition_id"].isin([2, 11]).astype(int)

    return df, freq_maps, cat_maps


# Apply feature engineering to train_part (fit)
train_part, freq_maps, _ = engineer_features(train_part, is_train=True)

# Store category mappings from train (value -> code)
cat_maps = {}
for col in low_card_cols:
    if col in train_part.columns and train_part[col].dtype == "category":
        cat_maps[col] = {cat: code for code, cat in enumerate(train_part[col].cat.categories)}

# Apply to val_part (transform)
val_part, _, _ = engineer_features(
    val_part, is_train=False, freq_maps=freq_maps, cat_maps=cat_maps
)

# Apply to test_df (transform)
test_df, _, _ = engineer_features(
    test_df, is_train=False, freq_maps=freq_maps, cat_maps=cat_maps
)

# ============================================================
# 5. DEFINE FINAL FEATURE SET
# ============================================================
drop_cols = (
    id_cols
    + [target_col]
    + med_cols
    + lab_cols
    + high_card_cols
    + low_card_cols
    + ["age", "weight"]
)

train_features = train_part.drop(
    columns=[c for c in drop_cols if c in train_part.columns]
)
val_features = val_part.drop(columns=[c for c in drop_cols if c in val_part.columns])
test_features = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

# Ensure all are numeric
for col in train_features.columns:
    if not pd.api.types.is_numeric_dtype(train_features[col]):
        train_features[col] = pd.to_numeric(train_features[col], errors="coerce")
        val_features[col] = pd.to_numeric(val_features[col], errors="coerce")
        test_features[col] = pd.to_numeric(test_features[col], errors="coerce")

# Final safety: force all columns to float
train_features = train_features.astype(float)
val_features = val_features.astype(float)
test_features = test_features.astype(float)

# Fill NaN and replace infinities
train_features = train_features.fillna(-1).replace([np.inf, -np.inf], 0)
val_features = val_features.fillna(-1).replace([np.inf, -np.inf], 0)
test_features = test_features.fillna(-1).replace([np.inf, -np.inf], 0)

print(f"\nFinal feature count: {train_features.shape[1]}")
print(f"Train features shape: {train_features.shape}")
print(f"Val features shape: {val_features.shape}")
print(f"Test features shape: {test_features.shape}")

# ============================================================
# 6. TARGETS
# ============================================================
y_train = train_part[target_col].values
y_val = val_part[target_col].values

print(f"Target classes: {sorted(train_part[target_col].unique())}")
print(f"Target distribution (train): {pd.Series(y_train).value_counts().to_dict()}")
print(f"Target distribution (val): {pd.Series(y_val).value_counts().to_dict()}")

# ============================================================
# 7. TRAIN LIGHTGBM MODEL
# ============================================================
model = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=3,
    n_estimators=3000,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=7,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

print("\nTraining LightGBM with early stopping on validation logloss...")
model.fit(
    train_features,
    y_train,
    eval_set=[(val_features, y_val)],
    eval_metric="multi_logloss",
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=500),
    ],
)

best_iter = model.best_iteration_
best_logloss = model.best_score_["valid_0"]["multi_logloss"]
print(
    f"Training complete. Best iteration: {best_iter}, Best logloss: {best_logloss:.4f}"
)

# ============================================================
# 8. VALIDATION EVALUATION (official metric: accuracy)
# ============================================================
val_pred_proba = model.predict_proba(val_features)
val_pred_idx = val_pred_proba.argmax(axis=1)
val_pred_labels = model.classes_[val_pred_idx]

val_accuracy = accuracy_score(y_val, val_pred_labels)
print(f"\nValidation Accuracy: {val_accuracy:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_val, val_pred_labels, target_names=model.classes_))

# ============================================================
# 9. TEST INFERENCE & SUBMISSION
# ============================================================
test_pred_proba = model.predict_proba(test_features)
test_pred_idx = test_pred_proba.argmax(axis=1)
test_pred_labels = model.classes_[test_pred_idx]

submission = pd.DataFrame(
    {"id": np.arange(len(test_pred_labels)), "readmitted": test_pred_labels}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(f"Prediction distribution:\n{submission['readmitted'].value_counts()}")

# ============================================================
# 10. FEATURE IMPORTANCE (diagnostic)
# ============================================================
fi = pd.Series(model.feature_importances_, index=train_features.columns).sort_values(
    ascending=False
)
print(f"\nTop 10 features:")
print(fi.head(10))

# ============================================================
# 11. FINAL SCORE
# ============================================================
score = val_accuracy
print(f"Final Validation Score: {score}")
