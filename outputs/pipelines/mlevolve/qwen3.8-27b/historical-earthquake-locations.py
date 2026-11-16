import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
train = pd.read_parquet('./input/train.parquet')
test = pd.read_parquet('./input/test.parquet')

target_col = 'Damage Description (total)'

# ---------------------------------------------------------------------------
# Separate target / features (drop id-like columns)
# ---------------------------------------------------------------------------
y = train[target_col].values
X = train.drop(columns=[target_col]).copy()

id_like = [c for c in X.columns if c.lower() in ('id', 'building_id')]
X = X.drop(columns=id_like)
test_features = test.drop(columns=[c for c in test.columns if c.lower() in ('id', 'building_id')],
                          errors='ignore').copy()

# Keep only columns present in both sets
common_cols = [c for c in X.columns if c in test_features.columns]
X = X[common_cols]
test_features = test_features[common_cols]

# ---------------------------------------------------------------------------
# Categorical handling (fit on TRAIN ONLY to avoid leakage)
# ---------------------------------------------------------------------------
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
low_card_cat_cols = []
for col in cat_cols:
    if X[col].nunique(dropna=True) <= 500:
        train_categories = list(X[col].dropna().unique())
        categories = train_categories + ['__UNSEEN__']

        X[col] = (
            X[col]
            .astype('category')
            .cat.set_categories(categories)
            .fillna('__UNSEEN__')
        )

        test_col = (
            test_features[col]
            .astype('category')
            .cat.set_categories(categories)
        )
        test_col = test_col.where(test_col.isin(train_categories), '__UNSEEN__')
        test_features[col] = test_col.astype('category').cat.set_categories(categories)
        low_card_cat_cols.append(col)
    else:
        freq = X[col].value_counts(normalize=True)
        X[col] = X[col].map(freq).fillna(0.0).astype(float)
        test_features[col] = test_features[col].map(freq).fillna(0.0).astype(float)

# ---------------------------------------------------------------------------
# Safe domain features (fit on TRAIN ONLY)
# ---------------------------------------------------------------------------
def _numeric_series(df, col):
    return pd.to_numeric(df[col], errors='coerce').astype(float)

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
mag_cols = [c for c in numeric_cols if 'mag' in c.lower() or 'magnitude' in c.lower()]
depth_cols = [c for c in numeric_cols if 'depth' in c.lower()]
error_cols = [c for c in numeric_cols if 'err' in c.lower() or 'error' in c.lower()]

processed_numeric_cols = set()
for col in mag_cols:
    if col in processed_numeric_cols:
        continue
    processed_numeric_cols.add(col)
    s_train = _numeric_series(X, col)
    s_test = _numeric_series(test_features, col)
    X[f'{col}_logabs'] = np.log1p(np.abs(s_train))
    test_features[f'{col}_logabs'] = np.log1p(np.abs(s_test))
    X[f'{col}_sq'] = s_train ** 2
    test_features[f'{col}_sq'] = s_test ** 2

for col in depth_cols + error_cols:
    if col in processed_numeric_cols:
        continue
    processed_numeric_cols.add(col)
    s_train = _numeric_series(X, col)
    s_test = _numeric_series(test_features, col)
    X[f'{col}_logabs'] = np.log1p(np.abs(s_train))
    test_features[f'{col}_logabs'] = np.log1p(np.abs(s_test))

if mag_cols and depth_cols:
    mag_col = mag_cols[0]
    depth_col = depth_cols[0]
    s_mag_train = _numeric_series(X, mag_col)
    s_mag_test = _numeric_series(test_features, mag_col)
    s_depth_train = _numeric_series(X, depth_col)
    s_depth_test = _numeric_series(test_features, depth_col)
    X[f'{depth_col}_per_{mag_col}'] = np.divide(
        s_depth_train, s_mag_train, out=np.full_like(s_depth_train, np.nan, dtype=float), where=(s_mag_train != 0)
    )
    test_features[f'{depth_col}_per_{mag_col}'] = np.divide(
        s_depth_test, s_mag_test, out=np.full_like(s_depth_test, np.nan, dtype=float), where=(s_mag_test != 0)
    )

datetime_cols = [c for c in X.columns if pd.api.types.is_datetime64_any_dtype(X[c])]
for col in datetime_cols:
    train_dt = pd.to_datetime(X[col], errors='coerce')
    test_dt = pd.to_datetime(test_features[col], errors='coerce')
    if train_dt.notna().any():
        train_min = train_dt.min()
        X[f'{col}_year'] = train_dt.dt.year
        X[f'{col}_month'] = train_dt.dt.month
        X[f'{col}_day'] = train_dt.dt.day
        X[f'{col}_hour'] = train_dt.dt.hour
        X[f'{col}_dayofweek'] = train_dt.dt.dayofweek
        X[f'{col}_seconds_since_train_min'] = (train_dt - train_min).dt.total_seconds()

        test_dt = pd.to_datetime(test_features[col], errors='coerce')
        test_features[f'{col}_year'] = test_dt.dt.year
        test_features[f'{col}_month'] = test_dt.dt.month
        test_features[f'{col}_day'] = test_dt.dt.day
        test_features[f'{col}_hour'] = test_dt.dt.hour
        test_features[f'{col}_dayofweek'] = test_dt.dt.dayofweek
        test_features[f'{col}_seconds_since_train_min'] = (test_dt - train_min).dt.total_seconds()

lat_cols = [c for c in numeric_cols if 'lat' in c.lower()]
lon_cols = [c for c in numeric_cols if 'lon' in c.lower() or 'lng' in c.lower()]
if lat_cols and lon_cols:
    lat_col = lat_cols[0]
    lon_col = lon_cols[0]
    lat_train = _numeric_series(X, lat_col)
    lon_train = _numeric_series(X, lon_col)
    lat_test = _numeric_series(test_features, lat_col)
    lon_test = _numeric_series(test_features, lon_col)

    centroid_lat = lat_train.mean()
    centroid_lon = lon_train.mean()
    X[f'{lat_col}_dist_to_train_centroid'] = np.sqrt(
        (lat_train - centroid_lat) ** 2 + (lon_train - centroid_lon) ** 2
    )
    test_features[f'{lat_col}_dist_to_train_centroid'] = np.sqrt(
        (lat_test - centroid_lat) ** 2 + (lon_test - centroid_lon) ** 2
    )

    for geo_col in [lat_col, lon_col]:
        train_geo = _numeric_series(X, geo_col)
        test_geo = _numeric_series(test_features, geo_col)
        if train_geo.notna().sum() > 1:
            edges = np.unique(np.quantile(train_geo.dropna().to_numpy(), np.linspace(0, 1, 21)))
            if len(edges) > 1:
                X[f'{geo_col}_bin'] = pd.cut(train_geo, bins=edges, labels=False, include_lowest=True)
                test_features[f'{geo_col}_bin'] = pd.cut(test_geo, bins=edges, labels=False, include_lowest=True)

# ---------------------------------------------------------------------------
# Validation split (done BEFORE any modelling)
# ---------------------------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------
num_class = len(np.unique(y))
params = {
    'objective': 'multiclass',
    'num_class': num_class,
    'metric': 'multi_logloss',
    'learning_rate': 0.02,
    'num_leaves': 127,
    'n_estimators': 3000,
    'min_child_samples': 20,
    'subsample': 0.8,
    'subsample_freq': 1,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'verbose': -1,
}

if low_card_cat_cols:
    params['categorical_feature'] = low_card_cat_cols

train_class_counts = pd.Series(y_train).value_counts()
if len(train_class_counts) > 1 and train_class_counts.max() / train_class_counts.min() > 5:
    params['class_weight'] = 'balanced'

val_models = []
val_proba = np.zeros((X_val.shape[0], num_class), dtype=float)
for seed in [42, 43, 44]:
    model = lgb.LGBMClassifier(**{**params, 'random_state': seed})
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    val_models.append(model)
    val_proba += model.predict_proba(X_val)

val_pred = val_models[0].classes_[val_proba.argmax(axis=1)]
val_accuracy = accuracy_score(y_val, val_pred)
print(f'Final Validation Score: {val_accuracy:.4f}')

# ---------------------------------------------------------------------------
# Retrain 3-seed ensemble on the FULL training set
# ---------------------------------------------------------------------------
best_iters = [getattr(m, 'best_iteration_', params['n_estimators']) for m in val_models]
final_n_estimators = max(1, int(round(np.mean(best_iters))))

final_models = []
test_proba = np.zeros((test_features.shape[0], num_class), dtype=float)
for seed in [42, 43, 44]:
    model = lgb.LGBMClassifier(**{**params, 'random_state': seed, 'n_estimators': final_n_estimators})
    model.fit(X, y, callbacks=[lgb.log_evaluation(0)])
    final_models.append(model)
    test_proba += model.predict_proba(test_features)

test_pred = final_models[0].classes_[test_proba.argmax(axis=1)]

os.makedirs('./submission', exist_ok=True)
submission = pd.DataFrame({
    'id': np.arange(len(test_pred)),
    target_col: test_pred,
})
submission.to_csv('./submission/submission.csv', index=False)
print('Submission saved to ./submission/submission.csv')