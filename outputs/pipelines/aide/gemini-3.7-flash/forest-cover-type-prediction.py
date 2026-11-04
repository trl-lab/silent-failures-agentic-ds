import os
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Load data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop Id column if present
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Euclidean distance to hydrology
    df["Euclidean_Distance_To_Hydrology"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    )

    # Manhattan distance to hydrology
    df["Manhattan_Distance_To_Hydrology"] = np.abs(
        df["Horizontal_Distance_To_Hydrology"]
    ) + np.abs(df["Vertical_Distance_To_Hydrology"])

    # Hydrology slope angle (in degrees)
    df["Hydrology_Slope_Deg"] = np.degrees(
        np.arctan2(
            df["Vertical_Distance_To_Hydrology"],
            df["Horizontal_Distance_To_Hydrology"] + 1e-5,
        )
    )

    # Elevation relative to hydrology
    df["Elevation_Minus_VD_Hydrology"] = (
        df["Elevation"] - df["Vertical_Distance_To_Hydrology"]
    )
    df["Elevation_Plus_VD_Hydrology"] = (
        df["Elevation"] + df["Vertical_Distance_To_Hydrology"]
    )

    # Pairwise distance interactions
    df["Hydro_Fire_Sum"] = (
        df["Horizontal_Distance_To_Hydrology"]
        + df["Horizontal_Distance_To_Fire_Points"]
    )
    df["Hydro_Fire_Diff"] = np.abs(
        df["Horizontal_Distance_To_Hydrology"]
        - df["Horizontal_Distance_To_Fire_Points"]
    )
    df["Hydro_Road_Sum"] = (
        df["Horizontal_Distance_To_Hydrology"] + df["Horizontal_Distance_To_Roadways"]
    )
    df["Hydro_Road_Diff"] = np.abs(
        df["Horizontal_Distance_To_Hydrology"] - df["Horizontal_Distance_To_Roadways"]
    )
    df["Road_Fire_Sum"] = (
        df["Horizontal_Distance_To_Roadways"] + df["Horizontal_Distance_To_Fire_Points"]
    )
    df["Road_Fire_Diff"] = np.abs(
        df["Horizontal_Distance_To_Roadways"] - df["Horizontal_Distance_To_Fire_Points"]
    )

    # Mean distances to amenities
    df["Mean_Distance_Amenities"] = (
        df["Horizontal_Distance_To_Hydrology"]
        + df["Horizontal_Distance_To_Roadways"]
        + df["Horizontal_Distance_To_Fire_Points"]
    ) / 3.0
    df["Mean_Hydro_Road"] = (
        df["Horizontal_Distance_To_Hydrology"] + df["Horizontal_Distance_To_Roadways"]
    ) / 2.0
    df["Mean_Hydro_Fire"] = (
        df["Horizontal_Distance_To_Hydrology"]
        + df["Horizontal_Distance_To_Fire_Points"]
    ) / 2.0

    # Aspect trigonometric projections & slope interactions
    aspect_rad = np.radians(df["Aspect"])
    df["Aspect_Sin"] = np.sin(aspect_rad)
    df["Aspect_Cos"] = np.cos(aspect_rad)
    df["Slope_Aspect_Cos"] = df["Slope"] * df["Aspect_Cos"]
    df["Slope_Aspect_Sin"] = df["Slope"] * df["Aspect_Sin"]

    # Hillshade contrasts and summary stats
    df["Hillshade_9am_minus_3pm"] = df["Hillshade_9am"] - df["Hillshade_3pm"]
    df["Hillshade_9am_minus_Noon"] = df["Hillshade_9am"] - df["Hillshade_Noon"]
    df["Hillshade_Noon_minus_3pm"] = df["Hillshade_Noon"] - df["Hillshade_3pm"]
    hillshades = df[["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]]
    df["Hillshade_Mean"] = hillshades.mean(axis=1)
    df["Hillshade_Std"] = hillshades.std(axis=1)
    df["Hillshade_Max"] = hillshades.max(axis=1)
    df["Hillshade_Min"] = hillshades.min(axis=1)

    # Condensed categorical indices for Wilderness Area and Soil Type
    wild_cols = [c for c in df.columns if c.startswith("Wilderness_Area")]
    if wild_cols:
        df["Wilderness_Area_Index"] = np.argmax(df[wild_cols].values, axis=1)

    soil_cols = [c for c in df.columns if c.startswith("Soil_Type")]
    if soil_cols:
        df["Soil_Type_Index"] = np.argmax(df[soil_cols].values, axis=1)
        df["Soil_Type_Sum"] = df[soil_cols].sum(axis=1)

    return df


# Separate target and prepare features
target_col = "Cover_Type"
y = train_df[target_col].values

X_train_raw = train_df.drop(columns=[target_col])
X_test_raw = test_df.copy()

X_train_fe = engineer_features(X_train_raw)
X_test_fe = engineer_features(X_test_raw)

# Ensure consistent feature alignment
feature_cols = [c for c in X_train_fe.columns if c in X_test_fe.columns]
X = X_train_fe[feature_cols]
X_test = X_test_fe[feature_cols]

# 5-fold Stratified Cross-Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

oof_probs = np.zeros((len(X), 7), dtype=float)
test_prob_accum = np.zeros((len(X_test), 7), dtype=float)

print(
    f"Starting {n_splits}-fold cross-validation with ExtraTrees + RandomForest ensemble..."
)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    # Model 1: Extra Trees Classifier
    et_model = ExtraTreesClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_SEED + fold,
        n_jobs=-1,
    )
    et_model.fit(X_tr, y_tr)
    val_probs_et = et_model.predict_proba(X_va)
    test_probs_et = et_model.predict_proba(X_test)

    # Model 2: Random Forest Classifier
    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_SEED + fold,
        n_jobs=-1,
    )
    rf_model.fit(X_tr, y_tr)
    val_probs_rf = rf_model.predict_proba(X_va)
    test_probs_rf = rf_model.predict_proba(X_test)

    # Weighted ensemble blend (0.65 ExtraTrees + 0.35 RandomForest)
    val_probs_blend = 0.65 * val_probs_et + 0.35 * val_probs_rf
    test_probs_blend = 0.65 * test_probs_et + 0.35 * test_probs_rf

    oof_probs[val_idx] = val_probs_blend
    test_prob_accum += test_probs_blend / n_splits

    fold_preds = np.argmax(val_probs_blend, axis=1) + 1
    fold_acc = accuracy_score(y_va, fold_preds)
    print(f"Fold {fold + 1} Accuracy: {fold_acc:.5f}")

# Overall OOF Accuracy
oof_preds = np.argmax(oof_probs, axis=1) + 1
overall_acc = accuracy_score(y, oof_preds)
print(f"\nOverall Out-of-Fold Accuracy: {overall_acc:.5f}")

# Final Test Predictions
final_test_preds = np.argmax(test_prob_accum, axis=1) + 1

# Prepare and save submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(final_test_preds)), "Cover_Type": final_test_preds}
)

submission_path = "./working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Saved submission to {submission_path} with shape {submission.shape}")
