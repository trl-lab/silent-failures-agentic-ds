import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

train_path = "/path/to/user/workspace/ds_failures/datasets/uci/wine-quality/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/uci/wine-quality/data/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "quality"
y = train_df[target_col].values

train_len = len(train_df)
df_all = pd.concat([train_df.drop(columns=[target_col]), test_df.drop(columns=[target_col], errors='ignore')], ignore_index=True)

if 'color' in df_all.columns:
    color_map = {'red': 0, 'white': 1}
    df_all['color'] = df_all['color'].map(color_map).fillna(-1)

for col in df_all.columns:
    df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

df_all['free_sulfur_ratio'] = df_all['free_sulfur_dioxide'] / (df_all['total_sulfur_dioxide'] + 1e-5)
df_all['bound_sulfur_dioxide'] = df_all['total_sulfur_dioxide'] - df_all['free_sulfur_dioxide']
df_all['total_acidity'] = df_all['fixed_acidity'] + df_all['volatile_acidity'] + df_all['citric_acid']
df_all['sugar_to_chlorides'] = df_all['residual_sugar'] / (df_all['chlorides'] + 1e-5)
df_all['alcohol_density_ratio'] = df_all['alcohol'] / (df_all['density'] + 1e-5)

X = df_all.iloc[:train_len].copy()
X_test = df_all.iloc[train_len:].copy()

kf = KFold(n_splits=10, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))
cb_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbosity=-1
    )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    lgb_preds += lgb_model.predict(X_test) / kf.n_splits

    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        early_stopping_rounds=50,
        verbosity=0
    )
    xgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    xgb_preds += xgb_model.predict(X_test) / kf.n_splits

    cb_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        verbose=0
    )
    cb_model.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=50,
        verbose=False
    )
    cb_preds += cb_model.predict(X_test) / kf.n_splits

final_preds = 0.4 * lgb_preds + 0.3 * xgb_preds + 0.3 * cb_preds

sub = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'quality': final_preds
})
sub.to_csv("predictions.csv", index=False)