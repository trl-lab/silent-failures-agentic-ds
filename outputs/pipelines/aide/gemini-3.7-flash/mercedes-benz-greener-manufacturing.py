import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.random_projection import GaussianRandomProjection
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Separate target and drop ID column if present
y = train_df["y"].values
if "ID" in train_df.columns:
    train_df = train_df.drop(columns=["ID"])
if "ID" in test_df.columns:
    test_df = test_df.drop(columns=["ID"])

train_features = train_df.drop(columns=["y"])
test_features = test_df.copy()

# 2. Encode categorical features
cat_cols = train_features.select_dtypes(include=["object", "category"]).columns.tolist()
for col in cat_cols:
    le = LabelEncoder()
    # Fit on combined unique values to handle unseen categories cleanly
    combined = pd.concat(
        [train_features[col].astype(str), test_features[col].astype(str)], axis=0
    )
    le.fit(combined)
    train_features[col] = le.transform(train_features[col].astype(str))
    test_features[col] = le.transform(test_features[col].astype(str))

# 3. Remove zero-variance (constant) features
constant_cols = [c for c in train_features.columns if train_features[c].std() == 0]
train_features = train_features.drop(columns=constant_cols)
test_features = test_features.drop(columns=constant_cols)

# 4. Extract decomposition features from binary features
binary_cols = [c for c in train_features.columns if c not in cat_cols]
n_comp = 10

pca = PCA(n_components=n_comp, random_state=42)
svd = TruncatedSVD(n_components=n_comp, random_state=42)
ica = FastICA(n_components=n_comp, random_state=42, max_iter=1000)
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=42)

decomp_models = [("pca", pca), ("svd", svd), ("ica", ica), ("grp", grp)]

for name, model in decomp_models:
    train_decomp = model.fit_transform(train_features[binary_cols])
    test_decomp = model.transform(test_features[binary_cols])
    for i in range(n_comp):
        train_features[f"{name}_{i}"] = train_decomp[:, i]
        test_features[f"{name}_{i}"] = test_decomp[:, i]

X = train_features.values
X_test = test_features.values

# 5. 5-Fold Cross-Validation with LightGBM
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_preds = np.zeros(len(y))
test_preds = np.zeros(len(test_features))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.01,
    "max_depth": 3,
    "num_leaves": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "min_child_samples": 20,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_va, y_va = X[val_idx], y[val_idx]

    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)

    model = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=2000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    val_pred = model.predict(X_va, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_pred
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_splits

    fold_r2 = r2_score(y_va, val_pred)
    print(f"Fold {fold + 1} R2: {fold_r2:.5f}")

overall_r2 = r2_score(y, oof_preds)
print(f"Overall Out-of-Fold R2: {overall_r2:.5f}")

# 6. Save test predictions
os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame({"id": np.arange(len(test_preds)), "y": test_preds})
sub.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully with shape:", sub.shape)
