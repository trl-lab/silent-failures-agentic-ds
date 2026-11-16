import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

train_path = "/path/to/user/workspace/ds_failures/datasets/openml/energy-efficiency/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/openml/energy-efficiency/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "heating_load"
y = train_df[target_col].values

drop_cols = [c for c in [target_col, "cooling_load"] if c in train_df.columns]
feature_cols = [c for c in train_df.columns if c not in drop_cols]

X_train = train_df[feature_cols].copy()
X_test = test_df[[c for c in feature_cols if c in test_df.columns]].copy()


def create_features(df):
    df = df.copy()
    if "surface_area" in df.columns and "wall_area" in df.columns:
        if "roof_area" in df.columns:
            df["floor_area"] = df["surface_area"] - df["wall_area"] - df["roof_area"]
            df["roof_to_surface"] = df["roof_area"] / (df["surface_area"] + 1e-6)
        df["wall_to_surface"] = df["wall_area"] / (df["surface_area"] + 1e-6)
    if "surface_area" in df.columns and "overall_height" in df.columns:
        df["volume_est"] = df["surface_area"] * df["overall_height"]
    if "glazing_area" in df.columns and "wall_area" in df.columns:
        df["glazing_wall"] = df["glazing_area"] * df["wall_area"]
    if "glazing_area" in df.columns and "surface_area" in df.columns:
        df["glazing_surface"] = df["glazing_area"] * df["surface_area"]
    if "relative_compactness" in df.columns and "overall_height" in df.columns:
        df["compactness_height"] = df["relative_compactness"] * df["overall_height"]
    return df


X_train = create_features(X_train)
X_test = create_features(X_test)

object_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
if object_cols:
    combined = pd.concat([X_train, X_test], axis=0)
    combined = pd.get_dummies(combined, columns=object_cols, drop_first=True)
    X_train = combined.iloc[: len(X_train)].copy()
    X_test = combined.iloc[len(X_train) :].copy()

for col in X_train.columns:
    if X_train[col].isnull().any() or X_test[col].isnull().any():
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med)
        X_test[col] = X_test[col].fillna(med)

models = {
    "cat": CatBoostRegressor(
        verbose=0, random_seed=42, iterations=800, learning_rate=0.03, depth=6
    ),
    "xgb": XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        random_state=42,
        verbosity=0,
    ),
    "lgb": LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    ),
    "gbr": GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.04, max_depth=4, random_state=42
    ),
    "hgb": HistGradientBoostingRegressor(max_iter=300, random_state=42),
    "rf": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    "et": ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    "mlp": make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(128, 64),
            max_iter=1000,
            random_state=42,
            early_stopping=True,
        ),
    ),
}

kf = KFold(n_splits=10, shuffle=True, random_state=42)

oof_preds = {m: np.zeros(len(X_train)) for m in models}
test_preds = {m: np.zeros(len(X_test)) for m in models}

X_tr_np = X_train.values
X_te_np = X_test.values

for train_idx, val_idx in kf.split(X_tr_np, y):
    X_tr, y_tr = X_tr_np[train_idx], y[train_idx]
    X_va, y_va = X_tr_np[val_idx], y[val_idx]

    for m_name, model in models.items():
        model.fit(X_tr, y_tr)
        oof_preds[m_name][val_idx] = model.predict(X_va)
        test_preds[m_name] += model.predict(X_te_np) / kf.n_splits

OOF_mat = np.column_stack([oof_preds[m] for m in models])
TEST_mat = np.column_stack([test_preds[m] for m in models])

meta_model = Ridge(alpha=1.0)
meta_model.fit(OOF_mat, y)
final_preds = meta_model.predict(TEST_mat)

sub = pd.DataFrame({"id": np.arange(len(final_preds)), "heating_load": final_preds})

sub.to_csv("predictions.csv", index=False)