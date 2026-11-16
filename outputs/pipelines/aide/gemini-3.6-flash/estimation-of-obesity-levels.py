import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "NObeyesdad"


# Feature Engineering
def engineer_features(df):
    df = df.copy()

    # Biometric features
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)
    df["Ponderal_Index"] = df["Weight"] / (df["Height"] ** 3)
    df["BSA"] = 0.007184 * (df["Weight"] ** 0.425) * ((df["Height"] * 100) ** 0.725)

    # Basal Metabolic Rate (Mifflin-St Jeor proxy)
    gender_num = (df["Gender"] == "Male").astype(int)
    df["BMR"] = (
        10 * df["Weight"]
        + 6.25 * (df["Height"] * 100)
        - 5 * df["Age"]
        + (5 * gender_num - 161 * (1 - gender_num))
    )

    # Lifestyle interaction ratios
    df["Calorie_Burn_Ratio"] = (df["FAF"] + 1e-3) / (df["TUE"] + 1e-3)
    df["Hydration_Per_Weight"] = df["CH2O"] / df["Weight"]
    df["Meals_Hydration_Product"] = df["NCP"] * df["CH2O"]

    # Rounding signals (distinguishing integer records from synthetic fractional entries)
    for col in ["FCVC", "NCP", "CH2O", "FAF", "TUE", "Age"]:
        if col in df.columns:
            df[f"{col}_is_int"] = (df[col] % 1 == 0).astype(int)

    return df


train_df = engineer_features(train_df)
test_df = engineer_features(test_df)

features = [col for col in train_df.columns if col != target_col]

# Ordinal encode categorical features for uniform handling across model architectures
cat_cols = (
    train_df[features].select_dtypes(include=["object", "category"]).columns.tolist()
)
ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

train_df[cat_cols] = ordinal_encoder.fit_transform(train_df[cat_cols])
test_df[cat_cols] = ordinal_encoder.transform(test_df[cat_cols])

# Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(train_df[target_col])
X = train_df[features].copy()
X_test = test_df[features].copy()

num_classes = len(label_encoder.classes_)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros((len(train_df), num_classes))
test_preds = np.zeros((len(test_df), num_classes))

# LightGBM Parameters
lgb_params = {
    "objective": "multiclass",
    "num_class": num_classes,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "random_state": 42,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    # 1. LightGBM Model
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    model_lgb = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1200,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    val_prob_lgb = model_lgb.predict(X_val)
    test_prob_lgb = model_lgb.predict(X_test)

    # 2. CatBoost Model
    model_cat = CatBoostClassifier(
        iterations=1200,
        learning_rate=0.03,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        random_seed=42,
        early_stopping_rounds=50,
        verbose=False,
    )
    model_cat.fit(X_train, y_train, eval_set=(X_val, y_val))
    val_prob_cat = model_cat.predict_proba(X_val)
    test_prob_cat = model_cat.predict_proba(X_test)

    # 3. XGBoost Model
    model_xgb = XGBClassifier(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=50,
        eval_metric="mlogloss",
    )
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    val_prob_xgb = model_xgb.predict_proba(X_val)
    test_prob_xgb = model_xgb.predict_proba(X_test)

    # Soft voting ensemble for this fold
    val_prob_blend = (val_prob_lgb + val_prob_cat + val_prob_xgb) / 3.0
    test_prob_blend = (test_prob_lgb + test_prob_cat + test_prob_xgb) / 3.0

    oof_preds[val_idx] = val_prob_blend
    test_preds += test_prob_blend / skf.n_splits

oof_pred_labels = oof_preds.argmax(axis=1)
oof_accuracy = accuracy_score(y, oof_pred_labels)
print(f"OOF Accuracy: {oof_accuracy:.5f}")

# Save test set predictions
test_pred_labels = label_encoder.inverse_transform(test_preds.argmax(axis=1))
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_df)), "NObeyesdad": test_pred_labels}
)
submission.to_csv("./working/submission.csv", index=False)
