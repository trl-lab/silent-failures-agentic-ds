import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import os

# Create directories
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Class distribution:\n{train_df['class'].value_counts().sort_index()}")

# Define sensor angles (in degrees) based on the specification
sensor_angles_deg = [
    180,
    -165,
    -150,
    -135,
    -120,
    -105,
    -90,
    -75,
    -60,
    -45,
    -30,
    -15,
    0,
    15,
    30,
    45,
    60,
    75,
    90,
    105,
    120,
    135,
    150,
    165,
]
sensor_angles_rad = np.radians(sensor_angles_deg)
# Dynamically determine sensor columns (all columns except the target)
sensor_cols = [c for c in train_df.columns if c != "class"]
print(f"Sensor columns: {sensor_cols[:5]}... (total {len(sensor_cols)})")


def engineer_features(df, sensor_cols):
    """Create spatial/geometric features from the 24 circularly-arranged ultrasound sensors."""
    features = pd.DataFrame(index=df.index)

    sensors = df[sensor_cols].values  # shape: (n, 24)
    angles = sensor_angles_rad  # shape: (24,)
    n_sensors = len(sensor_cols)

    # === Basic statistics ===
    features["sensor_mean"] = sensors.mean(axis=1)
    features["sensor_std"] = sensors.std(axis=1)
    features["sensor_min"] = sensors.min(axis=1)
    features["sensor_max"] = sensors.max(axis=1)
    features["sensor_median"] = np.median(sensors, axis=1)
    features["sensor_range"] = sensors.max(axis=1) - sensors.min(axis=1)
    features["sensor_skew"] = pd.DataFrame(sensors, index=df.index).skew(axis=1).values

    # === Directional features (angular sectors) ===
    angles_norm = np.mod(sensor_angles_deg, 360)

    front_mask = (angles_norm >= 120) & (angles_norm <= 240)
    right_mask = (angles_norm >= 30) & (angles_norm <= 150)
    left_mask = (angles_norm >= 210) & (angles_norm <= 330)
    back_mask = (angles_norm <= 60) | (angles_norm >= 300)

    front_narrow_mask = (angles_norm >= 150) & (angles_norm <= 210)
    right_narrow_mask = (angles_norm >= 60) & (angles_norm <= 120)
    left_narrow_mask = (angles_norm >= 240) & (angles_norm <= 300)
    back_narrow_mask = (angles_norm <= 30) | (angles_norm >= 330)

    for name, mask in [
        ("front", front_mask),
        ("right", right_mask),
        ("left", left_mask),
        ("back", back_mask),
        ("front_n", front_narrow_mask),
        ("right_n", right_narrow_mask),
        ("left_n", left_narrow_mask),
        ("back_n", back_narrow_mask),
    ]:
        if mask.any():
            sector_sensors = sensors[:, mask]
            features[f"{name}_min"] = sector_sensors.min(axis=1)
            features[f"{name}_mean"] = sector_sensors.mean(axis=1)
            features[f"{name}_max"] = sector_sensors.max(axis=1)
            features[f"{name}_std"] = sector_sensors.std(axis=1)

    # === Angular projection features ===
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    features["proj_x"] = sensors @ sin_angles
    features["proj_y"] = sensors @ cos_angles
    features["proj_mag"] = np.sqrt(features["proj_x"] ** 2 + features["proj_y"] ** 2)
    features["proj_angle"] = np.arctan2(features["proj_x"], features["proj_y"])

    # === Asymmetry features ===
    right_sensors = sensors[:, 13:24]
    left_sensors = sensors[:, 1:13]

    features["right_mean"] = right_sensors.mean(axis=1)
    features["left_mean"] = left_sensors.mean(axis=1)
    features["right_min"] = right_sensors.min(axis=1)
    features["left_min"] = left_sensors.min(axis=1)
    features["right_std"] = right_sensors.std(axis=1)
    features["left_std"] = left_sensors.std(axis=1)
    features["asymmetry_mean"] = features["right_mean"] - features["left_mean"]
    features["asymmetry_min"] = features["right_min"] - features["left_min"]
    features["asymmetry_ratio"] = features["right_mean"] / (
        features["left_mean"] + 1e-8
    )

    # === Ratio features ===
    front_min = features["front_min"].values
    right_min = features["right_min"].values
    left_min = features["left_min"].values
    back_min = features["back_min"].values

    features["front_right_ratio"] = front_min / (right_min + 1e-8)
    features["front_left_ratio"] = front_min / (left_min + 1e-8)
    features["right_left_ratio"] = right_min / (left_min + 1e-8)
    features["front_back_ratio"] = front_min / (back_min + 1e-8)
    features["front_right_n_ratio"] = features["front_n_min"] / (
        features["right_n_min"] + 1e-8
    )
    features["front_left_n_ratio"] = features["front_n_min"] / (
        features["left_n_min"] + 1e-8
    )

    # === Gradient features (adjacent sensor differences) ===
    for i in range(n_sensors):
        j = (i + 1) % n_sensors
        features[f"grad_{i:02d}_{j:02d}"] = sensors[:, i] - sensors[:, j]

    # === Proximity features ===
    for threshold in [10, 20, 30, 50]:
        features[f"close_count_{threshold}"] = (sensors < threshold).sum(axis=1)
        features[f"close_front_{threshold}"] = (sensors[:, front_mask] < threshold).sum(
            axis=1
        )
        features[f"close_right_{threshold}"] = (sensors[:, right_mask] < threshold).sum(
            axis=1
        )
        features[f"close_left_{threshold}"] = (sensors[:, left_mask] < threshold).sum(
            axis=1
        )

    # === Second-order interactions ===
    features["front_x_right"] = front_min * right_min
    features["front_x_left"] = front_min * left_min
    features["right_x_left"] = right_min * left_min

    # === Front emphasis features (first sensor is front) ===
    us1 = sensors[:, 0]
    features["us1_vs_mean"] = us1 / (features["sensor_mean"] + 1e-8)
    features["us1_vs_min"] = us1 / (features["sensor_min"] + 1e-8)

    # === Additional sensor index features ===
    features["sensor_0"] = sensors[:, 0]
    features["sensor_6"] = sensors[:, 6]
    features["sensor_12"] = sensors[:, 12]
    features["sensor_18"] = sensors[:, 18]

    # === Inverse distance features ===
    features["front_inv"] = 1.0 / (front_min + 1e-8)
    features["right_inv"] = 1.0 / (right_min + 1e-8)
    features["left_inv"] = 1.0 / (left_min + 1e-8)
    features["back_inv"] = 1.0 / (back_min + 1e-8)

    # === Quadratic features ===
    features["front_min_sq"] = front_min**2
    features["right_min_sq"] = right_min**2
    features["left_min_sq"] = left_min**2

    return features


# Apply feature engineering
print("Engineering features for train...")
train_features = engineer_features(train_df, sensor_cols)
print(f"Train features shape: {train_features.shape}")

print("Engineering features for test...")
test_features = engineer_features(test_df, sensor_cols)
print(f"Test features shape: {test_features.shape}")

# Combine raw sensor features with engineered features
X_train = pd.concat([train_df[sensor_cols], train_features], axis=1)
X_test = pd.concat([test_df[sensor_cols], test_features], axis=1)
y_train = train_df["class"]

# Handle any potential NaN/inf values
X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

# Create train/validation split (stratified)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

print(f"\nFinal shapes:")
print(f"  Train: {X_tr.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"  Val class distribution:\n{y_val.value_counts().sort_index()}")

# === Model Training ===
model = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=4,
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=6,
    min_child_samples=10,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

callbacks = [lgb.early_stopping(50), lgb.log_evaluation(-1)]

model.fit(
    X_tr,
    y_tr,
    eval_set=[(X_val, y_val)],
    eval_metric="multi_error",
    callbacks=callbacks,
)

print(f"Best iteration: {model.best_iteration_}")

# Evaluate on validation set using the official metric (accuracy)
val_pred = model.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)
print(f"Validation Accuracy: {val_acc:.4f}")

# Generate test predictions
test_pred = model.predict(X_test)

# Create and save submission
submission = pd.DataFrame({"id": range(len(test_pred)), "class": test_pred})
submission.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved: {len(submission)} predictions")
print(
    f"Test prediction distribution: {pd.Series(test_pred).value_counts().sort_index().to_dict()}"
)

print(f"Final Validation Score: {val_acc}")
