import os
import catboost as cb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Set random seed for reproducibility
np.random.seed(42)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "class"
feature_cols = [c for c in train_df.columns if c != target_col]

# Reference angles for US1 to US24 in degrees
angles_deg = np.array(
    [
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
)
angles_rad = np.radians(angles_deg)


def extract_features(df):
    data = df[feature_cols].values.astype(np.float32)

    # Cartesian coordinates (x, y) relative to robot body
    x_coords = data * np.cos(angles_rad)
    y_coords = data * np.sin(angles_rad)

    # Global spatial aggregates
    mean_x = np.mean(x_coords, axis=1, keepdims=True)
    mean_y = np.mean(y_coords, axis=1, keepdims=True)
    std_x = np.std(x_coords, axis=1, keepdims=True)
    std_y = np.std(y_coords, axis=1, keepdims=True)

    # Radial distance statistics
    min_dist = np.min(data, axis=1, keepdims=True)
    max_dist = np.max(data, axis=1, keepdims=True)
    mean_dist = np.mean(data, axis=1, keepdims=True)
    std_dist = np.std(data, axis=1, keepdims=True)

    # Direction of nearest obstacle
    min_idx = np.argmin(data, axis=1, keepdims=True)
    closest_angle = angles_rad[min_idx.squeeze(-1)][:, None]

    # Spatial derivatives around the sensor ring
    diff_forward = np.diff(data, axis=1, append=data[:, :1])
    second_diff = data - 0.5 * (np.roll(data, 1, axis=1) + np.roll(data, -1, axis=1))

    # 8 Directional Sectors (3 sensors per sector)
    sector_means, sector_mins = [], []
    for s in range(8):
        sec_indices = [(s * 3 + i) % 24 for i in range(3)]
        sec_data = data[:, sec_indices]
        sector_means.append(np.mean(sec_data, axis=1, keepdims=True))
        sector_mins.append(np.min(sec_data, axis=1, keepdims=True))

    sector_means = np.hstack(sector_means)
    sector_mins = np.hstack(sector_mins)

    # Symmetry comparisons (Left vs Right)
    left_sensors = data[:, 13:24]
    right_sensors = data[:, 1:12][:, ::-1]
    lr_diff = left_sensors - right_sensors
    lr_mean_diff = np.mean(left_sensors, axis=1, keepdims=True) - np.mean(
        right_sensors, axis=1, keepdims=True
    )
    lr_min_diff = np.min(left_sensors, axis=1, keepdims=True) - np.min(
        right_sensors, axis=1, keepdims=True
    )

    # Temporal short-term memory features (Lags and Differences)
    lag1 = np.vstack([data[0:1], data[:-1]])
    lag2 = np.vstack([data[0:1], data[0:1], data[:-2]])

    temp_diff1 = data - lag1
    temp_diff2 = data - lag2

    return np.hstack(
        [
            data,
            x_coords,
            y_coords,
            mean_x,
            mean_y,
            std_x,
            std_y,
            min_dist,
            max_dist,
            mean_dist,
            std_dist,
            closest_angle,
            diff_forward,
            second_diff,
            sector_means,
            sector_mins,
            lr_diff,
            lr_mean_diff,
            lr_min_diff,
            lag1,
            lag2,
            temp_diff1,
            temp_diff2,
        ]
    )


X_train = extract_features(train_df)
X_test = extract_features(test_df)

le = LabelEncoder()
y_train = le.fit_transform(train_df[target_col].values)

n_classes = len(le.classes_)
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros((len(X_train), n_classes))
test_preds = np.zeros((len(X_test), n_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_va, y_va = X_train[val_idx], y_train[val_idx]

    model = cb.CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        random_seed=42 + fold,
        verbose=0,
    )

    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50)

    oof_preds[val_idx] = model.predict_proba(X_va)
    test_preds += model.predict_proba(X_test) / n_splits

oof_class_preds = np.argmax(oof_preds, axis=1)
accuracy = accuracy_score(y_train, oof_class_preds)
print(f"Validation Accuracy: {accuracy:.5f}")

# Save test predictions
os.makedirs("./working", exist_ok=True)
final_preds_encoded = np.argmax(test_preds, axis=1)
final_preds = le.inverse_transform(final_preds_encoded)

sub = pd.DataFrame({"id": np.arange(len(test_df)), "class": final_preds})

sub.to_csv("./working/submission.csv", index=False)
