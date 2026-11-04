import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Ensure output directory exists
os.makedirs("./working", exist_ok=True)

# Load datasets
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Label encoding target
target_col = "NObeyesdad"
target_encoder = LabelEncoder()
y_train = target_encoder.fit_transform(train_df[target_col])
n_classes = len(target_encoder.classes_)


def engineer_features(df):
    data = df.copy()

    # 1. Anthropometric & Body Composition Features
    data["BMI"] = data["Weight"] / (data["Height"] ** 2)
    data["Ponderal_Index"] = data["Weight"] / (data["Height"] ** 3)
    data["BSA"] = (
        0.007184 * ((data["Height"] * 100) ** 0.725) * (data["Weight"] ** 0.425)
    )

    data["Ideal_Weight"] = 22.5 * (data["Height"] ** 2)
    data["Weight_Difference"] = data["Weight"] - data["Ideal_Weight"]
    data["Weight_Ratio"] = data["Weight"] / (data["Ideal_Weight"] + 1e-6)

    # WHO BMI Classification Categories
    bmi_conditions = [
        (data["BMI"] < 18.5),
        (data["BMI"] >= 18.5) & (data["BMI"] < 25.0),
        (data["BMI"] >= 25.0) & (data["BMI"] < 27.0),
        (data["BMI"] >= 27.0) & (data["BMI"] < 30.0),
        (data["BMI"] >= 30.0) & (data["BMI"] < 35.0),
        (data["BMI"] >= 35.0) & (data["BMI"] < 40.0),
        (data["BMI"] >= 40.0),
    ]
    bmi_choices = [0, 1, 2, 3, 4, 5, 6]
    data["WHO_BMI_Cat"] = np.select(bmi_conditions, bmi_choices, default=1)

    # 2. Gender Anthropometric Interactions
    gender_num = (data["Gender"] == "Male").astype(float)
    data["Gender_BMI"] = gender_num * data["BMI"]
    data["Gender_Weight"] = gender_num * data["Weight"]
    data["Gender_Height"] = gender_num * data["Height"]

    # 3. Lifestyle, Diet & Physical Activity Interactions
    data["Activity_to_Sedentary"] = data["FAF"] / (data["TUE"] + 0.1)
    data["Calorie_Burn_Index"] = data["FAF"] * data["Weight"]
    data["Hydration_per_Weight"] = data["CH2O"] / (data["Weight"] + 1e-5)
    data["Veg_Meal_Ratio"] = data["FCVC"] / (data["NCP"] + 0.1)

    data["Age_BMI"] = data["Age"] * data["BMI"]
    data["Age_FAF"] = data["Age"] * (data["FAF"] + 0.1)
    data["Age_TUE"] = data["Age"] * (data["TUE"] + 0.1)

    # Ordinal mappings
    caec_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    calc_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    data["CAEC_num"] = data["CAEC"].map(caec_map).fillna(1).astype(float)
    data["CALC_num"] = data["CALC"].map(calc_map).fillna(1).astype(float)

    favc_binary = (data["FAVC"] == "yes").astype(float)
    scc_binary = (data["SCC"] == "no").astype(float)
    smoke_binary = (data["SMOKE"] == "yes").astype(float)
    fam_hist = (data["family_history_with_overweight"] == "yes").astype(float)
    data["Unhealthy_Score"] = (
        favc_binary + scc_binary + smoke_binary + fam_hist + data["CAEC_num"]
    )

    return data


# Feature transformation
drop_cols = ["id", target_col] if target_col in train_df.columns else ["id"]
train_feat = engineer_features(train_df).drop(columns=drop_cols)
test_feat = engineer_features(test_df).drop(columns=["id"])

# Categorical column handling
cat_cols = [
    "Gender",
    "family_history_with_overweight",
    "FAVC",
    "CAEC",
    "SMOKE",
    "SCC",
    "CALC",
    "MTRANS",
]

train_feat_cat = train_feat.copy()
test_feat_cat = test_feat.copy()

for col in cat_cols:
    train_feat[col] = train_feat[col].astype("category")
    test_feat[col] = test_feat[col].astype("category")
    train_feat_cat[col] = train_feat_cat[col].astype(str)
    test_feat_cat[col] = test_feat_cat[col].astype(str)

# Stratified K-Fold setup
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

oof_xgb = np.zeros((len(train_feat), n_classes))
oof_lgb = np.zeros((len(train_feat), n_classes))
oof_cat = np.zeros((len(train_feat), n_classes))

test_xgb = np.zeros((len(test_feat), n_classes))
test_lgb = np.zeros((len(test_feat), n_classes))
test_cat = np.zeros((len(test_feat), n_classes))

# Model hyperparameter specifications
xgb_params = {
    "objective": "multi:softprob",
    "num_class": n_classes,
    "eval_metric": "mlogloss",
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "enable_categorical": True,
    "random_state": SEED,
    "n_estimators": 1200,
}

lgb_params = {
    "objective": "multiclass",
    "num_class": n_classes,
    "metric": "multi_logloss",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "random_state": SEED,
    "n_estimators": 1200,
    "verbose": -1,
}

cat_params = {
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "learning_rate": 0.04,
    "depth": 6,
    "iterations": 1200,
    "random_seed": SEED,
    "verbose": False,
}

# Cross-Validation Training Loop
for fold, (train_idx, val_idx) in enumerate(skf.split(train_feat, y_train)):
    # Data partitions
    X_tr_xgb, y_tr = train_feat.iloc[train_idx], y_train[train_idx]
    X_va_xgb, y_va = train_feat.iloc[val_idx], y_train[val_idx]
    X_tr_cat = train_feat_cat.iloc[train_idx]
    X_va_cat = train_feat_cat.iloc[val_idx]

    # 1. XGBoost
    model_xgb = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=50)
    model_xgb.fit(X_tr_xgb, y_tr, eval_set=[(X_va_xgb, y_va)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict_proba(X_va_xgb)
    test_xgb += model_xgb.predict_proba(test_feat) / n_splits

    # 2. LightGBM
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(
        X_tr_xgb,
        y_tr,
        eval_set=[(X_va_xgb, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    oof_lgb[val_idx] = model_lgb.predict_proba(X_va_xgb)
    test_lgb += model_lgb.predict_proba(test_feat) / n_splits

    # 3. CatBoost
    model_cat = CatBoostClassifier(
        **cat_params, cat_features=cat_cols, early_stopping_rounds=50
    )
    model_cat.fit(X_tr_cat, y_tr, eval_set=(X_va_cat, y_va), verbose=False)
    oof_cat[val_idx] = model_cat.predict_proba(X_va_cat)
    test_cat += model_cat.predict_proba(test_feat_cat) / n_splits

# Individual Model Accuracies
acc_xgb = accuracy_score(y_train, np.argmax(oof_xgb, axis=1))
acc_lgb = accuracy_score(y_train, np.argmax(oof_lgb, axis=1))
acc_cat = accuracy_score(y_train, np.argmax(oof_cat, axis=1))

print(f"XGBoost OOF Accuracy: {acc_xgb:.5f}")
print(f"LightGBM OOF Accuracy: {acc_lgb:.5f}")
print(f"CatBoost OOF Accuracy: {acc_cat:.5f}")

# Weighted Ensemble
oof_ensemble = 0.35 * oof_xgb + 0.35 * oof_lgb + 0.30 * oof_cat
test_ensemble = 0.35 * test_xgb + 0.35 * test_lgb + 0.30 * test_cat

ensemble_acc = accuracy_score(y_train, np.argmax(oof_ensemble, axis=1))
print(f"Validation Ensemble Accuracy: {ensemble_acc:.5f}")

# Generate test predictions and write submission file
test_preds = np.argmax(test_ensemble, axis=1)
test_labels = target_encoder.inverse_transform(test_preds)

sub = pd.DataFrame({"id": test_df["id"].values, target_col: test_labels})

sub.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
