import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# Load train and test data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Separate target and feature columns
target_col = "class"
feature_cols = [c for c in train_df.columns if c != target_col]

# Target encoding mapping
unique_classes = np.sort(train_df[target_col].unique())
class_to_idx = {c: i for i, c in enumerate(unique_classes)}
idx_to_class = {i: c for i, c in enumerate(unique_classes)}

y = train_df[target_col].map(class_to_idx).values


def extract_spatial_spectral_features(df, feat_cols):
    X_raw = df[feat_cols].values
    n_samples = len(X_raw)

    # Reshape to (N, 9 pixels, 4 bands)
    X_grid = X_raw.reshape(n_samples, 9, 4)

    extracted_features = []

    # Original raw features
    extracted_features.append(X_raw)

    # Central pixel (index 4) features
    center_pixel = X_grid[:, 4, :]
    extracted_features.append(center_pixel)

    # Band-wise summary statistics across 9 pixels
    band_means = np.mean(X_grid, axis=1)
    band_stds = np.std(X_grid, axis=1)
    band_mins = np.min(X_grid, axis=1)
    band_maxs = np.max(X_grid, axis=1)
    band_ranges = band_maxs - band_mins

    extracted_features.extend(
        [band_means, band_stds, band_mins, band_maxs, band_ranges]
    )

    # Central pixel differences from neighborhood stats
    center_diff_mean = center_pixel - band_means
    center_diff_min = center_pixel - band_mins
    center_diff_max = center_pixel - band_maxs
    extracted_features.extend([center_diff_mean, center_diff_min, center_diff_max])

    # Spatial gradients: mean absolute difference from center to 8 surrounding pixels
    neighbor_indices = [0, 1, 2, 3, 5, 6, 7, 8]
    neighbors = X_grid[:, neighbor_indices, :]
    spatial_diffs = np.abs(neighbors - center_pixel[:, np.newaxis, :])
    spatial_grad_mean = np.mean(spatial_diffs, axis=1)
    spatial_grad_max = np.max(spatial_diffs, axis=1)
    extracted_features.extend([spatial_grad_mean, spatial_grad_max])

    # Spectral vegetation & moisture indices across pixels
    # Band 0: Green, Band 1: Red, Band 2: NIR1, Band 3: NIR2
    eps = 1e-5

    # Center pixel indices
    ndvi1_center = (center_pixel[:, 2] - center_pixel[:, 1]) / (
        center_pixel[:, 2] + center_pixel[:, 1] + eps
    )
    ndvi2_center = (center_pixel[:, 3] - center_pixel[:, 1]) / (
        center_pixel[:, 3] + center_pixel[:, 1] + eps
    )
    ndwi_center = (center_pixel[:, 0] - center_pixel[:, 2]) / (
        center_pixel[:, 0] + center_pixel[:, 2] + eps
    )
    red_green_center = center_pixel[:, 1] / (center_pixel[:, 0] + eps)
    nir_red_center = center_pixel[:, 2] / (center_pixel[:, 1] + eps)

    center_indices = np.column_stack(
        [ndvi1_center, ndvi2_center, ndwi_center, red_green_center, nir_red_center]
    )
    extracted_features.append(center_indices)

    # Average spectral indices across all 9 pixels
    ndvi1_grid = (X_grid[:, :, 2] - X_grid[:, :, 1]) / (
        X_grid[:, :, 2] + X_grid[:, :, 1] + eps
    )
    ndvi2_grid = (X_grid[:, :, 3] - X_grid[:, :, 1]) / (
        X_grid[:, :, 3] + X_grid[:, :, 1] + eps
    )

    extracted_features.append(np.mean(ndvi1_grid, axis=1, keepdims=True))
    extracted_features.append(np.mean(ndvi2_grid, axis=1, keepdims=True))
    extracted_features.append(np.std(ndvi1_grid, axis=1, keepdims=True))

    return np.hstack(extracted_features)


# Generate domain features
X_train = extract_spatial_spectral_features(train_df, feature_cols)
X_test = extract_spatial_spectral_features(test_df, feature_cols)

# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds_prob = np.zeros((len(train_df), len(unique_classes)))
test_preds_prob = np.zeros((len(test_df), len(unique_classes)))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train[train_idx], y[train_idx]
    X_va, y_va = X_train[val_idx], y[val_idx]

    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=8,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42 + fold,
    )

    model.fit(X_tr, y_tr)

    oof_preds_prob[val_idx] = model.predict_proba(X_va)
    test_preds_prob += model.predict_proba(X_test) / skf.n_splits

# Evaluate Out-Of-Fold accuracy
oof_preds = np.argmax(oof_preds_prob, axis=1)
accuracy = accuracy_score(y, oof_preds)
print(f"Validation Accuracy: {accuracy:.5f}")

# Prepare submission
test_preds_class_idx = np.argmax(test_preds_prob, axis=1)
test_preds_original_class = [idx_to_class[idx] for idx in test_preds_class_idx]

submission_df = pd.DataFrame(
    {
        "id": np.arange(len(test_preds_original_class)),
        "class": test_preds_original_class,
    }
)

submission_df.to_csv("./working/submission.csv", index=False)
