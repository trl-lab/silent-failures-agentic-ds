import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb

try:
    import catboost as cb

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "target"
feature_cols = [col for col in train_df.columns if col != target_col]

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(train_df[target_col].astype(str))
num_classes = len(le.classes_)


# 2. Comprehensive Domain Feature Engineering
def engineer_features(df):
    df_feat = df.copy()

    # Bounding Box features
    if "X_Maximum" in df_feat.columns and "X_Minimum" in df_feat.columns:
        df_feat["X_Range"] = df_feat["X_Maximum"] - df_feat["X_Minimum"]
        df_feat["X_Center"] = (df_feat["X_Maximum"] + df_feat["X_Minimum"]) / 2.0
    if "Y_Maximum" in df_feat.columns and "Y_Minimum" in df_feat.columns:
        df_feat["Y_Range"] = df_feat["Y_Maximum"] - df_feat["Y_Minimum"]
        df_feat["Y_Center"] = (df_feat["Y_Maximum"] + df_feat["Y_Minimum"]) / 2.0

    if "X_Range" in df_feat.columns and "Y_Range" in df_feat.columns:
        df_feat["Aspect_Ratio"] = (df_feat["X_Range"] + 1e-5) / (
            df_feat["Y_Range"] + 1e-5
        )
        df_feat["Box_Area"] = df_feat["X_Range"] * df_feat["Y_Range"]
        df_feat["BBox_Diagonal"] = np.sqrt(
            df_feat["X_Range"] ** 2 + df_feat["Y_Range"] ** 2
        )
        if "Pixels_Areas" in df_feat.columns:
            df_feat["Pixel_Density_In_Box"] = (df_feat["Pixels_Areas"] + 1e-5) / (
                df_feat["Box_Area"] + 1e-5
            )
            df_feat["Area_to_Box_Area_Diff"] = (
                df_feat["Box_Area"] - df_feat["Pixels_Areas"]
            )

    # Conveyer-relative spatial coordinates
    if "Length_of_Conveyer" in df_feat.columns:
        if "X_Center" in df_feat.columns:
            df_feat["X_Center_Conveyer_Ratio"] = df_feat["X_Center"] / (
                df_feat["Length_of_Conveyer"] + 1e-5
            )
        if "X_Minimum" in df_feat.columns:
            df_feat["X_Min_Conveyer_Ratio"] = df_feat["X_Minimum"] / (
                df_feat["Length_of_Conveyer"] + 1e-5
            )
        if "X_Maximum" in df_feat.columns:
            df_feat["X_Max_Conveyer_Ratio"] = df_feat["X_Maximum"] / (
                df_feat["Length_of_Conveyer"] + 1e-5
            )
        if "X_Range" in df_feat.columns:
            df_feat["X_Range_Conveyer_Ratio"] = df_feat["X_Range"] / (
                df_feat["Length_of_Conveyer"] + 1e-5
            )

    # Luminosity features & Dynamic Contrast
    if (
        "Maximum_of_Luminosity" in df_feat.columns
        and "Minimum_of_Luminosity" in df_feat.columns
    ):
        df_feat["Luminosity_Range"] = (
            df_feat["Maximum_of_Luminosity"] - df_feat["Minimum_of_Luminosity"]
        )
        df_feat["Luminosity_Mid"] = (
            df_feat["Maximum_of_Luminosity"] + df_feat["Minimum_of_Luminosity"]
        ) / 2.0
        df_feat["Luminosity_Contrast"] = (
            df_feat["Maximum_of_Luminosity"] - df_feat["Minimum_of_Luminosity"]
        ) / (df_feat["Maximum_of_Luminosity"] + df_feat["Minimum_of_Luminosity"] + 1e-5)
    if "Sum_of_Luminosity" in df_feat.columns and "Pixels_Areas" in df_feat.columns:
        df_feat["Mean_Luminosity_Per_Pixel"] = df_feat["Sum_of_Luminosity"] / (
            df_feat["Pixels_Areas"] + 1e-5
        )

    # Perimeter and Compactness
    if "X_Perimeter" in df_feat.columns and "Y_Perimeter" in df_feat.columns:
        total_perimeter = df_feat["X_Perimeter"] + df_feat["Y_Perimeter"] + 1e-5
        df_feat["Total_Perimeter"] = total_perimeter
        df_feat["Perimeter_Ratio"] = (df_feat["X_Perimeter"] + 1e-5) / (
            df_feat["Y_Perimeter"] + 1e-5
        )
        if "Pixels_Areas" in df_feat.columns:
            df_feat["Area_Perimeter_Ratio"] = df_feat["Pixels_Areas"] / total_perimeter
            df_feat["Shape_Circularity"] = (4.0 * np.pi * df_feat["Pixels_Areas"]) / (
                total_perimeter**2 + 1e-5
            )

    # Log indices interactions
    if "Log_X_Index" in df_feat.columns and "Log_Y_Index" in df_feat.columns:
        df_feat["Log_Index_Ratio"] = (df_feat["Log_X_Index"] + 1e-5) / (
            df_feat["Log_Y_Index"] + 1e-5
        )
        df_feat["Log_Index_Sum"] = df_feat["Log_X_Index"] + df_feat["Log_Y_Index"]

    if "Edges_X_Index" in df_feat.columns and "Edges_Y_Index" in df_feat.columns:
        df_feat["Edges_Ratio"] = (df_feat["Edges_X_Index"] + 1e-5) / (
            df_feat["Edges_Y_Index"] + 1e-5
        )

    # Physical volume interaction
    if "Pixels_Areas" in df_feat.columns and "Steel_Plate_Thickness" in df_feat.columns:
        df_feat["Fault_Volume_Proxy"] = (
            df_feat["Pixels_Areas"] * df_feat["Steel_Plate_Thickness"]
        )
        df_feat["Area_Thickness_Ratio"] = df_feat["Pixels_Areas"] / (
            df_feat["Steel_Plate_Thickness"] + 1e-5
        )

    return df_feat


X = engineer_features(train_df[feature_cols]).fillna(0)
X_test = engineer_features(test_df[feature_cols]).fillna(0)

# 3. Stratified 5-Fold Cross-Validation Setup
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

models_list = ["lgb", "xgb", "rf", "et"]
if HAS_CATBOOST:
    models_list.append("cb")

oof_preds = {m: np.zeros((len(X), num_classes)) for m in models_list}
test_preds = {m: np.zeros((len(X_test), num_classes)) for m in models_list}

lgb_params = {
    "objective": "multiclass",
    "num_class": num_classes,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 27,
    "max_depth": 6,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 18,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

xgb_params = {
    "objective": "multi:softprob",
    "num_class": num_classes,
    "eval_metric": "mlogloss",
    "learning_rate": 0.03,
    "max_depth": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.75,
    "min_child_weight": 2,
    "gamma": 0.1,
    "random_state": 42,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_f, y_train_f = X.iloc[train_idx], y[train_idx]
    X_val_f, y_val_f = X.iloc[val_idx], y[val_idx]

    # --- LightGBM ---
    trn_data = lgb.Dataset(X_train_f, label=y_train_f)
    val_data = lgb.Dataset(X_val_f, label=y_val_f, reference=trn_data)

    model_lgb = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=1500,
        valid_sets=[trn_data, val_data],
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_preds["lgb"][val_idx] = model_lgb.predict(
        X_val_f, num_iteration=model_lgb.best_iteration
    )
    test_preds["lgb"] += (
        model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration) / n_splits
    )

    # --- XGBoost ---
    dtrain = xgb.DMatrix(X_train_f, label=y_train_f)
    dval = xgb.DMatrix(X_val_f, label=y_val_f)
    dtest = xgb.DMatrix(X_test)

    model_xgb = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=1500,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=60,
        verbose_eval=False,
    )
    oof_preds["xgb"][val_idx] = model_xgb.predict(dval)
    test_preds["xgb"] += model_xgb.predict(dtest) / n_splits

    # --- CatBoost ---
    if HAS_CATBOOST:
        model_cb = cb.CatBoostClassifier(
            iterations=1500,
            learning_rate=0.03,
            depth=5,
            loss_function="MultiClass",
            eval_metric="MultiClass",
            random_seed=42 + fold,
            early_stopping_rounds=60,
            verbose=False,
            thread_count=-1,
        )
        model_cb.fit(X_train_f, y_train_f, eval_set=(X_val_f, y_val_f), verbose=False)
        oof_preds["cb"][val_idx] = model_cb.predict_proba(X_val_f)
        test_preds["cb"] += model_cb.predict_proba(X_test) / n_splits

    # --- Random Forest ---
    model_rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, random_state=42 + fold, n_jobs=-1
    )
    model_rf.fit(X_train_f, y_train_f)
    oof_preds["rf"][val_idx] = model_rf.predict_proba(X_val_f)
    test_preds["rf"] += model_rf.predict_proba(X_test) / n_splits

    # --- Extra Trees ---
    model_et = ExtraTreesClassifier(
        n_estimators=300, max_depth=14, random_state=42 + fold, n_jobs=-1
    )
    model_et.fit(X_train_f, y_train_f)
    oof_preds["et"][val_idx] = model_et.predict_proba(X_val_f)
    test_preds["et"] += model_et.predict_proba(X_test) / n_splits

# 4. Meta-Learner Stacking
oof_features = np.hstack([oof_preds[m] for m in models_list])
test_features = np.hstack([test_preds[m] for m in models_list])

meta_oof_preds = np.zeros((len(X), num_classes))
meta_test_preds = np.zeros((len(X_test), num_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(oof_features, y)):
    meta_model = LogisticRegression(C=0.5, max_iter=1000, random_state=42 + fold)
    meta_model.fit(oof_features[train_idx], y[train_idx])
    meta_oof_preds[val_idx] = meta_model.predict_proba(oof_features[val_idx])
    meta_test_preds += meta_model.predict_proba(test_features) / n_splits

final_cv_acc = accuracy_score(y, np.argmax(meta_oof_preds, axis=1))
print(f"Validation Accuracy: {final_cv_acc:.5f}")

# 5. Save final predictions
os.makedirs("./working", exist_ok=True)
test_labels = le.inverse_transform(np.argmax(meta_test_preds, axis=1))
submission = pd.DataFrame({"id": np.arange(len(test_df)), "target": test_labels})
submission.to_csv("./working/submission.csv", index=False)
print("Saved predictions to ./working/submission.csv successfully.")
