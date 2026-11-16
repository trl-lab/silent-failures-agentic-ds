import os
import random
import re
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)

# ---------------------------------------------------------
# 1. Directory and Data Loading
# ---------------------------------------------------------
INPUT_DIR = "./input"
SUBMISSION_DIR = "./submission"

os.makedirs(SUBMISSION_DIR, exist_ok=True)

train_path = os.path.join(INPUT_DIR, "train.parquet")
test_path = os.path.join(INPUT_DIR, "test.parquet")

df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

# ---------------------------------------------------------
# 2. Clinical Domain Helper Functions & Feature Engineering
# ---------------------------------------------------------
MEDICATION_COLS = [
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

AGE_MAP = {
    "[0-10)": 5,
    "[0, 10)": 5,
    "[10-20)": 15,
    "[10, 20)": 15,
    "[20-30)": 25,
    "[20, 30)": 25,
    "[30-40)": 35,
    "[30, 40)": 35,
    "[40-50)": 45,
    "[40, 50)": 45,
    "[50-60)": 55,
    "[50, 60)": 55,
    "[60-70)": 65,
    "[60, 70)": 65,
    "[70-80)": 75,
    "[70, 80)": 75,
    "[80-90)": 85,
    "[80, 90)": 85,
    "[90-100)": 95,
    "[90, 100)": 95,
}

A1C_MAP = {"none": 0, "None": 0, "norm": 1, "Norm": 1, ">7": 2, ">8": 3}
GLU_MAP = {"none": 0, "None": 0, "norm": 1, "Norm": 1, ">200": 2, ">300": 3}


def parse_icd9_category(code_str):
    """Maps ICD-9 diagnostic codes into standard clinical diagnostic categories."""
    if pd.isna(code_str) or str(code_str).strip() in ["?", "", "nan", "None"]:
        return "Missing"

    code = str(code_str).strip()
    if code.startswith("V") or code.startswith("v"):
        return "Supplementary_V"
    if code.startswith("E") or code.startswith("e"):
        return "External_E"

    try:
        val = float(re.findall(r"^\d+", code)[0])
    except (IndexError, ValueError):
        return "Other"

    if (390 <= val <= 459) or val == 785:
        return "Circulatory"
    elif (460 <= val <= 519) or val == 786:
        return "Respiratory"
    elif (520 <= val <= 579) or val == 787:
        return "Digestive"
    elif val == 250 or (250 <= val < 251):
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


def is_diabetes_diag(code_str):
    if pd.isna(code_str):
        return 0
    return int(str(code_str).strip().startswith("250"))


def is_circulatory_diag(code_str):
    if pd.isna(code_str):
        return 0
    code = str(code_str).strip()
    try:
        val = float(re.findall(r"^\d+", code)[0])
        return int((390 <= val <= 459) or val == 785)
    except (IndexError, ValueError):
        return 0


def is_renal_diag(code_str):
    if pd.isna(code_str):
        return 0
    code = str(code_str).strip()
    try:
        val = float(re.findall(r"^\d+", code)[0])
        return int(580 <= val <= 589)
    except (IndexError, ValueError):
        return 0


def get_icd9_prefix(code_str):
    if pd.isna(code_str) or str(code_str).strip() in ["?", "", "nan", "None"]:
        return "Missing"
    s = str(code_str).strip()
    return s.split(".")[0]


def is_chf(code_str):
    if pd.isna(code_str):
        return 0
    return int(str(code_str).strip().startswith("428"))


def is_ckd(code_str):
    if pd.isna(code_str):
        return 0
    return int(str(code_str).strip().startswith("585"))


def is_pneumonia(code_str):
    if pd.isna(code_str):
        return 0
    return int(str(code_str).strip().startswith("486"))


def is_fluid_electrolyte(code_str):
    if pd.isna(code_str):
        return 0
    return int(str(code_str).strip().startswith("276"))


def is_hypertension(code_str):
    if pd.isna(code_str):
        return 0
    return int(str(code_str).strip().startswith("401"))


def is_ami(code_str):
    if pd.isna(code_str):
        return 0
    return int(str(code_str).strip().startswith("410"))


def is_cerebrovascular(code_str):
    if pd.isna(code_str):
        return 0
    code = str(code_str).strip()
    try:
        val = float(re.findall(r"^\d+", code)[0])
        return int(430 <= val <= 438)
    except (IndexError, ValueError):
        return 0


def calculate_charlson_score(df):
    """Calculates Charlson Comorbidity Index (CCI) proxy score from diag 1, 2, 3."""
    cci = np.zeros(len(df), dtype=float)
    for d_col in ["diag_1", "diag_2", "diag_3"]:
        if d_col not in df.columns:
            continue
        code_str = df[d_col].fillna("").astype(str).str.strip()

        mi = code_str.str.startswith(("410", "412"))
        chf = code_str.str.startswith("428")
        pvd = code_str.str.startswith(("440", "441", "442", "443", "444", "447", "785.4", "V43.4"))
        cvd = code_str.str.startswith(("430", "431", "432", "433", "434", "435", "436", "437", "438"))
        dementia = code_str.str.startswith("290")
        copd = code_str.str.startswith(("490", "491", "492", "493", "494", "495", "496", "500", "501", "502", "503", "504", "505"))
        rheum = code_str.str.startswith(("710", "714", "725"))
        pud = code_str.str.startswith(("531", "532", "533", "534"))
        mild_liver = code_str.str.startswith(("571.2", "571.5", "571.6", "571.4"))
        dm_uncomp = code_str.str.startswith(("250.0", "250.1", "250.2", "250.3"))
        dm_comp = code_str.str.startswith(("250.4", "250.5", "250.6", "250.7", "250.8", "250.9"))
        renal = code_str.str.startswith(("582", "583", "585", "586", "588"))
        cancer = code_str.apply(lambda x: bool(re.match(r"^(1[4-9][0-9]|20[0-8])", str(x))))
        sev_liver = code_str.str.startswith(("572.2", "572.3", "572.4", "572.8"))
        metastatic = code_str.str.startswith(("196", "197", "198", "199"))

        cci += mi.astype(int) * 1 + chf.astype(int) * 1 + pvd.astype(int) * 1 + cvd.astype(int) * 1 + dementia.astype(int) * 1
        cci += copd.astype(int) * 1 + rheum.astype(int) * 1 + pud.astype(int) * 1 + mild_liver.astype(int) * 1
        cci += dm_uncomp.astype(int) * 1 + dm_comp.astype(int) * 2 + renal.astype(int) * 2 + cancer.astype(int) * 2
        cci += sev_liver.astype(int) * 3 + metastatic.astype(int) * 6

    return cci


def engineer_features(df_input):
    """Performs domain-specific feature engineering on Diabetes 130-US encounters."""
    df = df_input.copy()

    for col in df.columns:
        if df[col].dtype == object or df[col].dtype.name == "category":
            df[col] = df[col].replace(
                ["?", "None", "unknown/invalid", "Unknown/Invalid", ""], np.nan
            )

    # Demographics and Age Encoding
    df["age_num"] = (
        df["age"].map(AGE_MAP).fillna(65).astype(float)
        if "age" in df.columns
        else 65.0
    )
    df["gender_clean"] = (
        df["gender"].fillna("Unknown").astype(str)
        if "gender" in df.columns
        else "Unknown"
    )
    df["race_clean"] = (
        df["race"].fillna("Missing").astype(str)
        if "race" in df.columns
        else "Missing"
    )
    df["weight_missing"] = (
        df["weight"].isna().astype(int) if "weight" in df.columns else 1
    )

    # Diagnostic ICD-9 Categorization & Comorbidities
    for d_col in ["diag_1", "diag_2", "diag_3"]:
        if d_col in df.columns:
            df[f"{d_col}_prefix"] = df[d_col].apply(get_icd9_prefix)
            df[f"{d_col}_cat"] = df[d_col].apply(parse_icd9_category)
            df[f"{d_col}_is_diabetes"] = df[d_col].apply(is_diabetes_diag)
            df[f"{d_col}_is_circulatory"] = df[d_col].apply(is_circulatory_diag)
            df[f"{d_col}_is_renal"] = df[d_col].apply(is_renal_diag)
            df[f"{d_col}_is_chf"] = df[d_col].apply(is_chf)
            df[f"{d_col}_is_ckd"] = df[d_col].apply(is_ckd)
            df[f"{d_col}_is_pneumonia"] = df[d_col].apply(is_pneumonia)
            df[f"{d_col}_is_fluid_electrolyte"] = df[d_col].apply(is_fluid_electrolyte)
            df[f"{d_col}_is_hypertension"] = df[d_col].apply(is_hypertension)
            df[f"{d_col}_is_ami"] = df[d_col].apply(is_ami)
            df[f"{d_col}_is_cerebrovascular"] = df[d_col].apply(is_cerebrovascular)
        else:
            df[f"{d_col}_prefix"] = "Missing"
            df[f"{d_col}_cat"] = "Missing"
            df[f"{d_col}_is_diabetes"] = 0
            df[f"{d_col}_is_circulatory"] = 0
            df[f"{d_col}_is_renal"] = 0
            df[f"{d_col}_is_chf"] = 0
            df[f"{d_col}_is_ckd"] = 0
            df[f"{d_col}_is_pneumonia"] = 0
            df[f"{d_col}_is_fluid_electrolyte"] = 0
            df[f"{d_col}_is_hypertension"] = 0
            df[f"{d_col}_is_ami"] = 0
            df[f"{d_col}_is_cerebrovascular"] = 0

    df["total_diabetes_diagnoses"] = (
        df["diag_1_is_diabetes"] + df["diag_2_is_diabetes"] + df["diag_3_is_diabetes"]
    )
    df["total_circulatory_diagnoses"] = (
        df["diag_1_is_circulatory"]
        + df["diag_2_is_circulatory"]
        + df["diag_3_is_circulatory"]
    )
    df["total_renal_diagnoses"] = (
        df["diag_1_is_renal"] + df["diag_2_is_renal"] + df["diag_3_is_renal"]
    )
    df["has_chf"] = (
        (df["diag_1_is_chf"] + df["diag_2_is_chf"] + df["diag_3_is_chf"]) > 0
    ).astype(int)
    df["has_ckd"] = (
        (df["diag_1_is_ckd"] + df["diag_2_is_ckd"] + df["diag_3_is_ckd"]) > 0
    ).astype(int)
    df["has_pneumonia"] = (
        (
            df["diag_1_is_pneumonia"]
            + df["diag_2_is_pneumonia"]
            + df["diag_3_is_pneumonia"]
        )
        > 0
    ).astype(int)
    df["has_fluid_electrolyte"] = (
        (
            df["diag_1_is_fluid_electrolyte"]
            + df["diag_2_is_fluid_electrolyte"]
            + df["diag_3_is_fluid_electrolyte"]
        )
        > 0
    ).astype(int)
    df["has_hypertension"] = (
        (
            df["diag_1_is_hypertension"]
            + df["diag_2_is_hypertension"]
            + df["diag_3_is_hypertension"]
        )
        > 0
    ).astype(int)
    df["has_ami"] = (
        (df["diag_1_is_ami"] + df["diag_2_is_ami"] + df["diag_3_is_ami"]) > 0
    ).astype(int)
    df["has_cerebrovascular"] = (
        (
            df["diag_1_is_cerebrovascular"]
            + df["diag_2_is_cerebrovascular"]
            + df["diag_3_is_cerebrovascular"]
        )
        > 0
    ).astype(int)
    df["high_risk_comorbidity_count"] = (
        df["has_chf"]
        + df["has_ckd"]
        + df["has_pneumonia"]
        + df["has_fluid_electrolyte"]
        + df["has_hypertension"]
        + df["has_ami"]
        + df["has_cerebrovascular"]
    )
    df["comorbidity_dm_circulatory"] = (
        (df["total_diabetes_diagnoses"] > 0) & (df["total_circulatory_diagnoses"] > 0)
    ).astype(int)
    df["comorbidity_dm_renal"] = (
        (df["total_diabetes_diagnoses"] > 0) & (df["total_renal_diagnoses"] > 0)
    ).astype(int)

    # Medication Dynamics and Regimen Complexity
    existing_meds = [m for m in MEDICATION_COLS if m in df.columns]

    med_active_counts = np.zeros(len(df), dtype=int)
    med_up_counts = np.zeros(len(df), dtype=int)
    med_down_counts = np.zeros(len(df), dtype=int)
    med_steady_counts = np.zeros(len(df), dtype=int)

    for med in existing_meds:
        med_val = df[med].astype(str).str.strip().str.capitalize()
        med_active_counts += (med_val.isin(["Steady", "Up", "Down"])).astype(int)
        med_up_counts += (med_val == "Up").astype(int)
        med_down_counts += (med_val == "Down").astype(int)
        med_steady_counts += (med_val == "Steady").astype(int)

    df["med_num_active"] = med_active_counts
    df["med_num_dosage_up"] = med_up_counts
    df["med_num_dosage_down"] = med_down_counts
    df["med_num_dosage_steady"] = med_steady_counts
    df["med_total_dosage_changes"] = med_up_counts + med_down_counts
    df["med_polypharmacy"] = (df["med_num_active"] >= 3).astype(int)

    if "insulin" in df.columns:
        ins_val = df["insulin"].astype(str).str.strip().str.capitalize()
        df["insulin_active"] = (ins_val.isin(["Steady", "Up", "Down"])).astype(int)
        df["insulin_dose_change"] = (ins_val.isin(["Up", "Down"])).astype(int)
        df["insulin_status"] = ins_val
    else:
        df["insulin_active"] = 0
        df["insulin_dose_change"] = 0
        df["insulin_status"] = "No"

    if "metformin" in df.columns:
        met_val = df["metformin"].astype(str).str.strip().str.capitalize()
        df["metformin_active"] = (met_val.isin(["Steady", "Up", "Down"])).astype(int)
    else:
        df["metformin_active"] = 0

    df["insulin_and_oral_combo"] = (
        (df["insulin_active"] == 1)
        & ((df["med_num_active"] - df["insulin_active"]) > 0)
    ).astype(int)

    # Glycemic Testing & Lab Results
    if "A1Cresult" in df.columns:
        df["a1c_clean"] = df["A1Cresult"].fillna("none").astype(str).str.lower()
        df["a1c_score"] = df["a1c_clean"].map(A1C_MAP).fillna(0).astype(int)
    else:
        df["a1c_score"] = 0
    df["a1c_tested"] = (df["a1c_score"] > 0).astype(int)
    df["a1c_high"] = (df["a1c_score"] >= 2).astype(int)

    if "max_glu_serum" in df.columns:
        df["glu_clean"] = df["max_glu_serum"].fillna("none").astype(str).str.lower()
        df["glu_score"] = df["glu_clean"].map(GLU_MAP).fillna(0).astype(int)
    else:
        df["glu_score"] = 0
    df["glu_tested"] = (df["glu_score"] > 0).astype(int)
    df["glu_high"] = (df["glu_score"] >= 2).astype(int)
    df["glycemic_monitored"] = (df["a1c_tested"] | df["glu_tested"]).astype(int)

    # Healthcare Utilization Ratios and Hospital Stay Intensity
    outpatient = (
        df["number_outpatient"].fillna(0) if "number_outpatient" in df.columns else 0
    )
    emergency = (
        df["number_emergency"].fillna(0) if "number_emergency" in df.columns else 0
    )
    inpatient = (
        df["number_inpatient"].fillna(0) if "number_inpatient" in df.columns else 0
    )

    df["total_past_visits"] = outpatient + emergency + inpatient
    df["has_past_inpatient"] = (inpatient > 0).astype(int)
    df["has_past_emergency"] = (emergency > 0).astype(int)
    df["inpatient_visit_ratio"] = inpatient / (df["total_past_visits"] + 1.0)
    df["emergency_visit_ratio"] = emergency / (df["total_past_visits"] + 1.0)
    df["weighted_utilization_risk"] = (
        2.0 * inpatient + 1.5 * emergency + 0.5 * outpatient
    )

    stay = (
        df["time_in_hospital"].clip(lower=1) if "time_in_hospital" in df.columns else 1
    )
    num_labs = df["num_lab_procedures"] if "num_lab_procedures" in df.columns else 0
    num_procs = df["num_procedures"] if "num_procedures" in df.columns else 0
    num_meds = df["num_medications"] if "num_medications" in df.columns else 0
    num_diags = df["number_diagnoses"] if "number_diagnoses" in df.columns else 0

    df["lab_procedures_per_day"] = num_labs / stay
    df["medications_per_day"] = num_meds / stay
    df["procedures_per_day"] = num_procs / stay
    df["diagnoses_per_day"] = num_diags / stay
    df["clinical_intensity_score"] = (num_labs + 2.0 * num_procs + num_meds) / stay
    df["procedure_to_med_ratio"] = num_procs / (num_meds + 1.0)

    # Admission and Discharge Clinical Groupings
    if "discharge_disposition_id" in df.columns:
        df["discharge_is_home"] = (
            df["discharge_disposition_id"].isin([1, 6, 8]).astype(int)
        )
        df["discharge_is_transfer"] = (
            df["discharge_disposition_id"].isin([2, 3, 4, 5, 22, 23, 24]).astype(int)
        )
        df["discharge_is_expired_or_hospice"] = (
            df["discharge_disposition_id"].isin([11, 13, 14, 19, 20, 21]).astype(int)
        )
        df["discharge_is_ama"] = (df["discharge_disposition_id"] == 7).astype(int)
    else:
        df["discharge_is_home"] = 0
        df["discharge_is_transfer"] = 0
        df["discharge_is_expired_or_hospice"] = 0
        df["discharge_is_ama"] = 0

    if "admission_type_id" in df.columns:
        df["admission_is_emergency"] = (df["admission_type_id"] == 1).astype(int)
        df["admission_is_urgent"] = (df["admission_type_id"] == 2).astype(int)
        df["admission_is_elective"] = (df["admission_type_id"] == 3).astype(int)
    else:
        df["admission_is_emergency"] = 0
        df["admission_is_urgent"] = 0
        df["admission_is_elective"] = 0

    if "admission_source_id" in df.columns:
        df["admission_from_er"] = (df["admission_source_id"] == 7).astype(int)
        df["admission_from_referral"] = (
            df["admission_source_id"].isin([1, 2, 3]).astype(int)
        )
    else:
        df["admission_from_er"] = 0
        df["admission_from_referral"] = 0

    # Binary Medication & Change Flags
    if "change" in df.columns:
        df["change_clean"] = df["change"].astype(str).str.strip().str.lower()
        df["change_flag"] = (
            df["change_clean"].isin(["ch", "change", "yes", "1"]).astype(int)
        )
    else:
        df["change_flag"] = 0

    if "diabetesMed" in df.columns:
        df["diabetesMed_clean"] = df["diabetesMed"].astype(str).str.strip().str.lower()
        df["diabetesMed_flag"] = df["diabetesMed_clean"].isin(["yes", "1"]).astype(int)
    else:
        df["diabetesMed_flag"] = 0

    if "medical_specialty" in df.columns:
        df["med_specialty_clean"] = (
            df["medical_specialty"].fillna("Missing").astype(str)
        )
    else:
        df["med_specialty_clean"] = "Missing"

    if "payer_code" in df.columns:
        df["payer_code_clean"] = df["payer_code"].fillna("Missing").astype(str)
    else:
        df["payer_code_clean"] = "Missing"

    # Charlson Comorbidity Index
    df["charlson_comorbidity_index"] = calculate_charlson_score(df)

    # LACE Index Proxy calculation
    # L (Length of stay)
    lace_l = pd.cut(
        stay,
        bins=[-1, 0, 1, 2, 3, 6, 13, 100],
        labels=[0, 1, 2, 3, 4, 5, 7],
    ).astype(float).fillna(0)

    # A (Acuity: Emergency admission = 3)
    lace_a = (df["admission_is_emergency"] * 3.0).astype(float)

    # C (Comorbidity proxy)
    lace_c = pd.cut(
        df["charlson_comorbidity_index"],
        bins=[-1, 0, 1, 2, 3, 100],
        labels=[0, 1, 2, 3, 5],
    ).astype(float).fillna(0)

    # E (Emergency visits in past 6 months)
    lace_e = pd.cut(
        emergency,
        bins=[-1, 0, 1, 2, 3, 100],
        labels=[0, 1, 2, 3, 4],
    ).astype(float).fillna(0)

    df["lace_index_proxy"] = lace_l + lace_a + lace_c + lace_e
    df["lace_high_risk"] = (df["lace_index_proxy"] >= 10).astype(int)

    # Uncontrolled Diabetes Risk Flags
    df["uncontrolled_dm_no_med_change"] = (
        ((df["a1c_high"] == 1) | (df["glu_high"] == 1)) & (df["change_flag"] == 0)
    ).astype(int)
    df["uncontrolled_dm_with_change"] = (
        ((df["a1c_high"] == 1) | (df["glu_high"] == 1)) & (df["change_flag"] == 1)
    ).astype(int)

    # Categorical interaction combos
    if "admission_source_id" in df.columns and "discharge_disposition_id" in df.columns:
        df["adm_source_discharge_disp"] = (
            df["admission_source_id"].astype(str)
            + "_"
            + df["discharge_disposition_id"].astype(str)
        )
    else:
        df["adm_source_discharge_disp"] = "Missing"

    if "admission_type_id" in df.columns and "admission_source_id" in df.columns:
        df["adm_type_adm_source"] = (
            df["admission_type_id"].astype(str)
            + "_"
            + df["admission_source_id"].astype(str)
        )
    else:
        df["adm_type_adm_source"] = "Missing"

    if "diag_1_cat" in df.columns and "age_num" in df.columns:
        df["diag1_cat_age"] = (
            df["diag_1_cat"].astype(str) + "_" + df["age_num"].astype(str)
        )
    else:
        df["diag1_cat_age"] = "Missing"

    return df


train_feat = engineer_features(df_train)
test_feat = engineer_features(df_test)

# Patient encounter frequency
if "patient_nbr" in df_train.columns and "patient_nbr" in df_test.columns:
    all_patients = pd.concat([df_train["patient_nbr"], df_test["patient_nbr"]])
    patient_counts = all_patients.value_counts()
    train_feat["patient_encounter_count"] = (
        train_feat["patient_nbr"].map(patient_counts).fillna(1)
    )
    test_feat["patient_encounter_count"] = (
        test_feat["patient_nbr"].map(patient_counts).fillna(1)
    )

# Frequency encoding for high-cardinality categorical features
freq_cols = [
    "medical_specialty",
    "payer_code",
    "diag_1_prefix",
    "diag_2_prefix",
    "diag_3_prefix",
]
for col in freq_cols:
    if col in train_feat.columns and col in test_feat.columns:
        combined_series = pd.concat(
            [train_feat[col].astype(str), test_feat[col].astype(str)]
        )
        freq_map = combined_series.value_counts(normalize=True).to_dict()
        train_feat[f"{col}_freq"] = (
            train_feat[col].astype(str).map(freq_map).fillna(0).astype(float)
        )
        test_feat[f"{col}_freq"] = (
            test_feat[col].astype(str).map(freq_map).fillna(0).astype(float)
        )

# Grouped Aggregation Statistics across Diagnostic and Age Cohorts
for grp in ["diag_1_cat", "age_num"]:
    for num_col in ["time_in_hospital", "num_lab_procedures", "num_medications", "number_diagnoses"]:
        if grp in train_feat.columns and num_col in train_feat.columns:
            grp_mean = train_feat.groupby(grp)[num_col].mean().to_dict()
            grp_std = train_feat.groupby(grp)[num_col].std().to_dict()

            train_feat[f"{num_col}_mean_by_{grp}"] = train_feat[grp].map(grp_mean).fillna(0).astype(float)
            test_feat[f"{num_col}_mean_by_{grp}"] = test_feat[grp].map(grp_mean).fillna(0).astype(float)

            train_feat[f"{num_col}_std_by_{grp}"] = train_feat[grp].map(grp_std).fillna(0).astype(float)
            test_feat[f"{num_col}_std_by_{grp}"] = test_feat[grp].map(grp_std).fillna(0).astype(float)

            train_feat[f"{num_col}_diff_from_mean_{grp}"] = (
                train_feat[num_col] - train_feat[f"{num_col}_mean_by_{grp}"]
            )
            test_feat[f"{num_col}_diff_from_mean_{grp}"] = (
                test_feat[num_col] - test_feat[f"{num_col}_mean_by_{grp}"]
            )

# Target Setup and Fold Splitting
target_col = "readmitted"
target_encoder = LabelEncoder()
y_train = target_encoder.fit_transform(train_feat[target_col].astype(str))
target_classes = list(target_encoder.classes_)
num_classes = len(target_classes)

n_folds = 5
if "patient_nbr" in train_feat.columns:
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    split_gen = splitter.split(
        train_feat, y_train, groups=train_feat["patient_nbr"].values
    )
else:
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    split_gen = splitter.split(train_feat, y_train)

fold_assignments = np.zeros(len(train_feat), dtype=int)
for fold, (trn_idx, val_idx) in enumerate(split_gen):
    fold_assignments[val_idx] = fold

# Leak-Free Out-of-Fold Multi-Class Target Encoding
te_cols = [
    "medical_specialty",
    "payer_code",
    "diag_1_prefix",
    "diag_2_prefix",
    "diag_3_prefix",
    "adm_source_discharge_disp",
]

for col in te_cols:
    if col not in train_feat.columns or col not in test_feat.columns:
        continue
    for c in range(num_classes):
        train_feat[f"{col}_te_{c}"] = 0.0
        test_feat[f"{col}_te_{c}"] = 0.0
        prior_c = np.mean(y_train == c)

        for fold in range(n_folds):
            trn_m = fold_assignments != fold
            val_m = fold_assignments == fold

            trn_series = train_feat.loc[trn_m, col].astype(str)
            trn_y = (y_train[trn_m] == c).astype(float)

            counts = trn_series.value_counts()
            pos_counts = trn_series[trn_y == 1].value_counts()

            smoothed = (pos_counts + 10.0 * prior_c) / (counts + 10.0)
            smooth_dict = smoothed.to_dict()

            train_feat.loc[val_m, f"{col}_te_{c}"] = (
                train_feat.loc[val_m, col].astype(str).map(smooth_dict).fillna(prior_c).astype(float)
            )
            test_feat[f"{col}_te_{c}"] += (
                test_feat[col].astype(str).map(smooth_dict).fillna(prior_c).astype(float) / n_folds
            )

exclude_cols = [
    "encounter_id",
    "patient_nbr",
    target_col,
    "weight",
    "diag_1",
    "diag_2",
    "diag_3",
    "gender",
    "race",
    "age",
    "A1Cresult",
    "max_glu_serum",
    "change",
    "diabetesMed",
    "medical_specialty",
    "payer_code",
    "a1c_clean",
    "glu_clean",
    "change_clean",
    "diabetesMed_clean",
]

feature_cols = [c for c in train_feat.columns if c not in exclude_cols]

# Categorical Label Encoding & Numerical Median Imputation
for c in feature_cols:
    if train_feat[c].dtype == object or train_feat[c].dtype.name == "category":
        train_feat[c] = train_feat[c].fillna("Missing").astype(str)
        test_feat[c] = test_feat[c].fillna("Missing").astype(str)
        le = LabelEncoder()
        combined_vals = np.unique(
            np.concatenate([train_feat[c].unique(), test_feat[c].unique()])
        )
        le.fit(combined_vals)
        train_feat[c] = le.transform(train_feat[c])
        test_feat[c] = le.transform(test_feat[c])
    else:
        med_val = train_feat[c].median()
        train_feat[c] = train_feat[c].fillna(med_val)
        test_feat[c] = test_feat[c].fillna(med_val)

X_train_df = train_feat[feature_cols].copy()
X_test_df = test_feat[feature_cols].copy()


# ---------------------------------------------------------
# 3. Model Definitions
# ---------------------------------------------------------
def get_lightgbm_model(num_classes=3, seed=42):
    params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "min_child_samples": 25,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
    }
    return lgb.LGBMClassifier(**params, n_estimators=1200)


def get_xgboost_model(num_classes=3, seed=42):
    return XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        learning_rate=0.03,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.5,
        random_state=seed,
        n_estimators=1000,
        early_stopping_rounds=50,
        n_jobs=-1,
        tree_method="hist",
    )


def get_catboost_model(num_classes=3, seed=42):
    return CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        iterations=1000,
        learning_rate=0.04,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=seed,
        verbose=False,
        thread_count=-1,
        early_stopping_rounds=50,
    )


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return x + self.block(x)


class ClinicalTabularResNet(nn.Module):
    def __init__(
        self, input_dim, num_classes=3, hidden_dim=128, num_blocks=2, dropout_rate=0.2
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        h = self.input_layer(x)
        for block in self.res_blocks:
            h = block(h)
        return self.head(h)


# ---------------------------------------------------------
# 4. Training & Validation Loop
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_tr_all = X_train_df.values
y_tr_all = y_train
X_te_all = X_test_df.values

oof_probs_lgb = np.zeros((len(X_tr_all), num_classes))
oof_probs_xgb = np.zeros((len(X_tr_all), num_classes))
oof_probs_cat = np.zeros((len(X_tr_all), num_classes))
oof_probs_nn = np.zeros((len(X_tr_all), num_classes))

test_probs_lgb = np.zeros((len(X_te_all), num_classes))
test_probs_xgb = np.zeros((len(X_te_all), num_classes))
test_probs_cat = np.zeros((len(X_te_all), num_classes))
test_probs_nn = np.zeros((len(X_te_all), num_classes))

for fold in range(n_folds):
    trn_mask = fold_assignments != fold
    val_mask = fold_assignments == fold

    X_trn, y_trn = X_tr_all[trn_mask], y_tr_all[trn_mask]
    X_val, y_val = X_tr_all[val_mask], y_tr_all[val_mask]

    # --- LightGBM ---
    lgb_model = get_lightgbm_model(num_classes=num_classes, seed=42 + fold)
    lgb_model.fit(
        X_trn,
        y_trn,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    oof_probs_lgb[val_mask] = lgb_model.predict_proba(X_val)
    test_probs_lgb += lgb_model.predict_proba(X_te_all) / n_folds

    # --- XGBoost ---
    xgb_model = get_xgboost_model(num_classes=num_classes, seed=42 + fold)
    xgb_model.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], verbose=False)
    oof_probs_xgb[val_mask] = xgb_model.predict_proba(X_val)
    test_probs_xgb += xgb_model.predict_proba(X_te_all) / n_folds

    # --- CatBoost ---
    cat_model = get_catboost_model(num_classes=num_classes, seed=42 + fold)
    cat_model.fit(X_trn, y_trn, eval_set=(X_val, y_val), verbose=False)
    oof_probs_cat[val_mask] = cat_model.predict_proba(X_val)
    test_probs_cat += cat_model.predict_proba(X_te_all) / n_folds

    # --- Tabular ResNet ---
    scaler = StandardScaler()
    X_trn_scaled = scaler.fit_transform(X_trn)
    X_val_scaled = scaler.transform(X_val)
    X_te_scaled = scaler.transform(X_te_all)

    train_dataset = TensorDataset(
        torch.tensor(X_trn_scaled, dtype=torch.float32),
        torch.tensor(y_trn, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    nn_model = ClinicalTabularResNet(
        input_dim=X_trn.shape[1],
        num_classes=num_classes,
        hidden_dim=128,
        num_blocks=2,
        dropout_rate=0.2,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)

    best_val_loss = float("inf")
    best_nn_val_probs = None

    for epoch in range(15):
        nn_model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = nn_model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        val_loss = 0.0
        val_probs_list = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = nn_model(batch_x)
                val_loss += criterion(logits, batch_y).item() * len(batch_y)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                val_probs_list.append(probs)

        val_loss /= len(val_dataset)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_nn_val_probs = np.vstack(val_probs_list)

    oof_probs_nn[val_mask] = best_nn_val_probs

    nn_model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(X_te_scaled, dtype=torch.float32).to(device)
        test_logits = nn_model(test_tensor)
        nn_test_probs = torch.softmax(test_logits, dim=1).cpu().numpy()
        test_probs_nn += nn_test_probs / n_folds

# ---------------------------------------------------------
# 5. Ensemble Optimization & Final Submission
# ---------------------------------------------------------
# Step 1: Optimize model blending weights to minimize out-of-fold log-loss
def blend_loss(weights):
    w = np.exp(weights) / np.sum(np.exp(weights))
    blend = (
        w[0] * oof_probs_lgb
        + w[1] * oof_probs_xgb
        + w[2] * oof_probs_cat
        + w[3] * oof_probs_nn
    )
    blend = np.clip(blend, 1e-15, 1.0 - 1e-15)
    blend = blend / np.sum(blend, axis=1, keepdims=True)
    return log_loss(y_tr_all, blend)


res_weights = minimize(blend_loss, [0.0, 0.0, 0.0, 0.0], method="Nelder-Mead")
opt_w = np.exp(res_weights.x) / np.sum(np.exp(res_weights.x))

oof_blended_probs = (
    opt_w[0] * oof_probs_lgb
    + opt_w[1] * oof_probs_xgb
    + opt_w[2] * oof_probs_cat
    + opt_w[3] * oof_probs_nn
)
test_blended_probs = (
    opt_w[0] * test_probs_lgb
    + opt_w[1] * test_probs_xgb
    + opt_w[2] * test_probs_cat
    + opt_w[3] * test_probs_nn
)

# Step 2: Optimize decision boundary class multipliers to maximize overall accuracy
def threshold_obj(multipliers):
    m = np.maximum(multipliers, 1e-4)
    scaled_probs = oof_blended_probs * m
    preds = np.argmax(scaled_probs, axis=1)
    return -accuracy_score(y_tr_all, preds)


best_score = -1.0
best_mult = np.array([1.0, 1.0, 1.0])
candidates = [
    [1.0, 1.0, 1.0],
    [1.0, 1.1, 1.2],
    [1.0, 0.9, 1.3],
    [1.0, 1.2, 0.9],
    [0.9, 1.1, 1.3],
    [1.1, 1.0, 1.2],
    [1.0, 1.3, 1.5],
]

for init in candidates:
    res = minimize(
        threshold_obj,
        init,
        method="Nelder-Mead",
        options={"maxiter": 500, "xatol": 1e-3, "fatol": 1e-4},
    )
    acc = -res.fun
    if acc > best_score:
        best_score = acc
        best_mult = np.maximum(res.x, 1e-4)

opt_mult = best_mult
oof_predictions = np.argmax(oof_blended_probs * opt_mult, axis=1)
score = accuracy_score(y_tr_all, oof_predictions)

test_predictions_idx = np.argmax(test_blended_probs * opt_mult, axis=1)
test_predicted_labels = [target_classes[idx] for idx in test_predictions_idx]

submission_df = pd.DataFrame(
    {"id": np.arange(len(test_predicted_labels)), "readmitted": test_predicted_labels}
)

submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission_df.to_csv(submission_path, index=False)

print(f"Final Validation Score: {score}")
