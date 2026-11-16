import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Load train and test datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")


# Enhanced Feature Engineering function for geometric, spatial, and sensor attributes
def feature_engineering(df):
    df = df.copy()

    if "V1" in df.columns:
        x_min, x_max = df["V1"], df["V2"]
        y_min, y_max = df["V3"], df["V4"]
        pixels_area = df["V5"]
        x_perim, y_perim = df["V6"], df["V7"]
        sum_lum, min_lum, max_lum = df["V8"], df["V9"], df["V10"]
        conveyer_len = df["V11"]
        thickness = df["V14"]
    elif "X_Minimum" in df.columns:
        x_min, x_max = df["X_Minimum"], df["X_Maximum"]
        y_min, y_max = df["Y_Minimum"], df["Y_Maximum"]
        pixels_area = df["Pixels_Areas"]
        x_perim, y_perim = df["X_Perimeter"], df["Y_Perimeter"]
        sum_lum = df["Sum_of_Luminosity"]
        min_lum = df["Minimum_of_Luminosity"]
        max_lum = df["Maximum_of_Luminosity"]
        conveyer_len = df["Length_of_Conveyer"]
        thickness = df["Steel_Plate_Thickness"]
    else:
        return df

    x_len = x_max - x_min
    y_len = y_max - y_min
    bbox_area = x_len * y_len

    df["X_Length"] = x_len
    df["Y_Length"] = y_len
    df["BBox_Area"] = bbox_area
    df["Aspect_Ratio"] = x_len / (y_len + 1e-6)
    df["Fill_Ratio"] = pixels_area / (bbox_area + 1e-6)
    df["Lum_Range"] = max_lum - min_lum
    df["Lum_Density"] = sum_lum / (pixels_area + 1e-6)
    df["Perim_Ratio"] = x_perim / (y_perim + 1e-6)
    df["Area_Perim_Ratio"] = pixels_area / (x_perim + y_perim + 1e-6)

    # Additional geometric & spatial interactions
    df["X_Center"] = (x_min + x_max) / 2.0
    df["Y_Center"] = (y_min + y_max) / 2.0
    df["X_Conveyer_Ratio"] = x_len / (conveyer_len + 1e-6)
    df["Compactness"] = ((x_perim + y_perim) ** 2) / (pixels_area + 1e-6)
    df["Thickness_Area_Ratio"] = pixels_area / (thickness + 1e-6)

    return df


# Prepare features and encode targets
y_raw = train_df["target"]
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

X_train_raw = train_df.drop(columns=["target"])
X_test_raw = test_df.copy()
if "target" in X_test_raw.columns:
    X_test_raw = X_test_raw.drop(columns=["target"])

X_train = feature_engineering(X_train_raw)
X_test = feature_engineering(X_test_raw)

# Setup 5-fold Stratified CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
num_classes = len(label_encoder.classes_)

oof_probs = np.zeros((len(X_train), num_classes))
test_probs = np.zeros((len(X_test), num_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    fold_val_preds = []
    fold_test_preds = []

    # 1. CatBoost Classifier
    try:
        from catboost import CatBoostClassifier

        cb = CatBoostClassifier(
            iterations=1200,
            learning_rate=0.03,
            depth=6,
            loss_function="MultiClass",
            random_seed=42 + fold,
            verbose=0,
        )
        cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50)
        fold_val_preds.append(cb.predict_proba(X_va))
        fold_test_preds.append(cb.predict_proba(X_test))
    except ImportError:
        pass

    # 2. XGBoost Classifier
    try:
        from xgboost import XGBClassifier

        xgb = XGBClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=6,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42 + fold,
            tree_method="hist",
            early_stopping_rounds=50,
            n_jobs=-1,
        )
        xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        fold_val_preds.append(xgb.predict_proba(X_va))
        fold_test_preds.append(xgb.predict_proba(X_test))
    except ImportError:
        pass

    # 3. LightGBM Classifier
    try:
        import lightgbm as lgb

        lgbm = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=6,
            num_leaves=31,
            objective="multiclass",
            random_state=42 + fold,
            n_jobs=-1,
            verbose=-1,
        )
        lgbm.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        fold_val_preds.append(lgbm.predict_proba(X_va))
        fold_test_preds.append(lgbm.predict_proba(X_test))
    except ImportError:
        pass

    # Fallback if specific packages are unavailable
    if not fold_val_preds:
        from sklearn.ensemble import HistGradientBoostingClassifier

        hgb = HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.03,
            max_depth=6,
            random_state=42 + fold,
            early_stopping=True,
        )
        hgb.fit(X_tr, y_tr)
        fold_val_preds.append(hgb.predict_proba(X_va))
        fold_test_preds.append(hgb.predict_proba(X_test))

    # Average fold predictions across models
    fold_val_prob = np.mean(fold_val_preds, axis=0)
    fold_test_prob = np.mean(fold_test_preds, axis=0)

    oof_probs[val_idx] = fold_val_prob
    test_probs += fold_test_prob / skf.n_splits

# Compute Out-of-Fold accuracy score
oof_preds = np.argmax(oof_probs, axis=1)
cv_accuracy = accuracy_score(y, oof_preds)

# Format test predictions and export
test_preds = np.argmax(test_probs, axis=1)
test_pred_labels = label_encoder.inverse_transform(test_preds)

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_pred_labels)), "target": test_pred_labels}
)
submission.to_csv("./working/submission.csv", index=False)

print(f"Validation Accuracy: {cv_accuracy:.5f}")
