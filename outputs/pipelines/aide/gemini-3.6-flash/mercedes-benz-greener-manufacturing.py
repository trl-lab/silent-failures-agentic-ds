import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.decomposition import PCA, FastICA, TruncatedSVD, FactorAnalysis
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# Load datasets
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Extract target and align features
y_train = train["y"].values
if "y" in test.columns:
    test = test.drop(columns=["y"])

train_features = train.drop(columns=["y", "ID"], errors="ignore")
test_features = test.drop(columns=["ID"], errors="ignore")

# Combine datasets for consistent transformations
df_all = pd.concat([train_features, test_features], axis=0).reset_index(drop=True)
n_train = len(train)

# Identify string categorical features
cat_cols = [c for c in df_all.columns if df_all[c].dtype == "object"]

# Construct categorical pairwise interactions
df_all["X0_X1"] = df_all["X0"].astype(str) + "_" + df_all["X1"].astype(str)
df_all["X0_X2"] = df_all["X0"].astype(str) + "_" + df_all["X2"].astype(str)
df_all["X0_X5"] = df_all["X0"].astype(str) + "_" + df_all["X5"].astype(str)
df_all["X1_X2"] = df_all["X1"].astype(str) + "_" + df_all["X2"].astype(str)

all_cat_cols = cat_cols + ["X0_X1", "X0_X2", "X0_X5", "X1_X2"]

# Frequency encoding
for col in all_cat_cols:
    freq = df_all[col].value_counts(normalize=True)
    df_all[f"{col}_freq"] = df_all[col].map(freq)

# Identify binary indicator features
binary_cols = [
    c for c in train_features.columns if c not in cat_cols and c != "y" and c != "ID"
]

# Remove constant / zero variance features
binary_stds = df_all[binary_cols].std()
valid_binary_cols = binary_stds[binary_stds > 0].index.tolist()

# Binary row aggregations
df_all["bin_sum"] = df_all[valid_binary_cols].sum(axis=1)
df_all["bin_mean"] = df_all[valid_binary_cols].mean(axis=1)
df_all["bin_std"] = df_all[valid_binary_cols].std(axis=1)
df_all["bin_skew"] = df_all[valid_binary_cols].skew(axis=1)

# Scaled binary features for matrix decompositions
scaler = StandardScaler()
X_binary_scaled = scaler.fit_transform(df_all[valid_binary_cols].fillna(0))

# Matrix decompositions & manifold features
n_comp = 10
pca = PCA(n_components=n_comp, random_state=42)
pca_feats = pca.fit_transform(X_binary_scaled)

ica = FastICA(n_components=n_comp, random_state=42, max_iter=1000)
ica_feats = ica.fit_transform(X_binary_scaled)

tsvd = TruncatedSVD(n_components=n_comp, random_state=42)
tsvd_feats = tsvd.fit_transform(X_binary_scaled)

fa = FactorAnalysis(n_components=n_comp, random_state=42)
fa_feats = fa.fit_transform(X_binary_scaled)

grp = GaussianRandomProjection(n_components=n_comp, random_state=42)
grp_feats = grp.fit_transform(X_binary_scaled)

# K-Means clustering distance features
kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
kmeans_dist = kmeans.fit_transform(X_binary_scaled)

for i in range(n_comp):
    df_all[f"pca_{i}"] = pca_feats[:, i]
    df_all[f"ica_{i}"] = ica_feats[:, i]
    df_all[f"tsvd_{i}"] = tsvd_feats[:, i]
    df_all[f"fa_{i}"] = fa_feats[:, i]
    df_all[f"grp_{i}"] = grp_feats[:, i]

for i in range(8):
    df_all[f"kmeans_dist_{i}"] = kmeans_dist[:, i]

# Ordinal encoding for categoricals
for col in all_cat_cols:
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df_all[f"{col}_encoded"] = oe.fit_transform(df_all[[col]])

df_all_numeric = df_all.drop(columns=all_cat_cols)

# Separate train and test sets
X_train_full = df_all_numeric.iloc[:n_train].copy()
X_test_full = df_all_numeric.iloc[n_train:].copy()

# Setup 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Out-of-fold target encoding for categoricals
X_train_encoded = X_train_full.copy()
X_test_encoded = X_test_full.copy()

target_enc_cols = ["X0", "X1", "X2", "X5", "X0_X1", "X0_X2"]
for col in target_enc_cols:
    X_train_encoded[f"{col}_te_mean"] = np.nan

for train_idx, val_idx in kf.split(X_train_full):
    for col in target_enc_cols:
        cat_vals_tr = (
            train_features.iloc[train_idx][col]
            if col in train_features.columns
            else df_all.iloc[:n_train].iloc[train_idx][col]
        )
        means_map = pd.Series(y_train[train_idx]).groupby(cat_vals_tr).mean()

        cat_vals_val = (
            train_features.iloc[val_idx][col]
            if col in train_features.columns
            else df_all.iloc[:n_train].iloc[val_idx][col]
        )
        X_train_encoded.iloc[
            val_idx, X_train_encoded.columns.get_loc(f"{col}_te_mean")
        ] = cat_vals_val.map(means_map)

for col in target_enc_cols:
    cat_vals = (
        train_features[col]
        if col in train_features.columns
        else df_all.iloc[:n_train][col]
    )
    global_map = pd.Series(y_train).groupby(cat_vals).mean()
    X_train_encoded[f"{col}_te_mean"] = X_train_encoded[f"{col}_te_mean"].fillna(
        y_train.mean()
    )
    test_cats = (
        test_features[col]
        if col in test_features.columns
        else df_all.iloc[n_train:][col]
    )
    X_test_encoded[f"{col}_te_mean"] = (
        test_cats.map(global_map).fillna(y_train.mean()).values
    )

# Define base learning models
models = {
    "lgb": lambda: lgb.LGBMRegressor(
        n_estimators=800,
        max_depth=3,
        num_leaves=7,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42,
        verbose=-1,
    ),
    "xgb": lambda: xgb.XGBRegressor(
        n_estimators=800,
        max_depth=2,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42,
        n_jobs=-1,
    ),
    "cat": lambda: cb.CatBoostRegressor(
        iterations=800,
        depth=3,
        learning_rate=0.02,
        l2_leaf_reg=5,
        verbose=0,
        random_seed=42,
    ),
    "et": lambda: ExtraTreesRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    ),
    "hgb": lambda: HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.02, max_iter=300, random_state=42
    ),
    "ridge": lambda: Ridge(alpha=100.0, random_state=42),
    "knn": lambda: KNeighborsRegressor(n_neighbors=30, weights="distance", n_jobs=-1),
}

oof_preds = {name: np.zeros(n_train) for name in models}
test_preds = {name: np.zeros(len(X_test_encoded)) for name in models}

# Cap y_train during fitting to stabilize against severe outlier gradients
y_train_clipped = np.clip(y_train, a_min=None, a_max=135.0)

# Scaled inputs for linear and distance-based models
scaler_scaled = StandardScaler()
X_tr_scaled = scaler_scaled.fit_transform(X_train_encoded.fillna(0))
X_te_scaled = scaler_scaled.transform(X_test_encoded.fillna(0))

for name in models:
    for train_idx, val_idx in kf.split(X_train_encoded):
        if name in ["ridge", "knn"]:
            X_tr, y_tr = X_tr_scaled[train_idx], y_train_clipped[train_idx]
            X_val, X_te = X_tr_scaled[val_idx], X_te_scaled
        else:
            X_tr, y_tr = X_train_encoded.iloc[train_idx], y_train_clipped[train_idx]
            X_val, X_te = X_train_encoded.iloc[val_idx], X_test_encoded

        model = models[name]()
        model.fit(X_tr, y_tr)

        oof_preds[name][val_idx] = model.predict(X_val)
        test_preds[name] += model.predict(X_te) / kf.n_splits

# Level-2 Stacking Meta-Model
X_oof = np.column_stack([oof_preds[name] for name in models])
X_test_meta = np.column_stack([test_preds[name] for name in models])

meta_model = RidgeCV(alphas=np.logspace(-2, 3, 20), cv=5)
meta_model.fit(X_oof, y_train_clipped)

oof_meta = meta_model.predict(X_oof)
final_test_preds = meta_model.predict(X_test_meta)

overall_r2 = r2_score(y_train, oof_meta)
print(f"Validation Metric (Stacked Ensemble OOF R2): {overall_r2:.5f}")

# Save submission file
sub = pd.DataFrame({"id": np.arange(len(final_test_preds)), "y": final_test_preds})
sub.to_csv("./working/submission.csv", index=False)
