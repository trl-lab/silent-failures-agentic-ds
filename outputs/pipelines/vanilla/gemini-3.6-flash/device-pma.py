import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

warnings.filterwarnings("ignore")

train_df = pd.read_parquet(
    "/path/to/user/workspace/ds_failures/datasets/strable/device-pma/data/train.parquet"
)
test_df = pd.read_parquet(
    "/path/to/user/workspace/ds_failures/datasets/strable/device-pma/data/test.parquet"
)

target_col = "decision_code"

y_raw = train_df[target_col]
train_df = train_df.drop(columns=[target_col])

le_target = LabelEncoder()
y = le_target.fit_transform(y_raw.astype(str))

features = [c for c in train_df.columns if c in test_df.columns]
X_train = train_df[features].copy()
X_test = test_df[features].copy()

date_cols = []
text_cols = []
cat_cols = []

for col in features:
  if pd.api.types.is_datetime64_any_dtype(
      X_train[col]
  ) or pd.api.types.is_timedelta64_dtype(X_train[col]):
    date_cols.append(col)
  elif pd.api.types.is_numeric_dtype(X_train[col]):
    continue
  else:
    if "date" in col.lower() or "time" in col.lower():
      date_cols.append(col)
    else:
      sample_vals = X_train[col].dropna().astype(str)
      if len(sample_vals) > 0 and sample_vals.str.len().mean() > 50:
        text_cols.append(col)
      else:
        cat_cols.append(col)

for col in date_cols:
  for df in [X_train, X_test]:
    dt = pd.to_datetime(df[col], errors="coerce")
    df[f"{col}_year"] = dt.dt.year
    df[f"{col}_month"] = dt.dt.month
    df[f"{col}_day"] = dt.dt.day
    df[f"{col}_dayofweek"] = dt.dt.dayofweek
    df[f"{col}_dayofyear"] = dt.dt.dayofyear
    dt_num = pd.to_numeric(dt) / 1e9
    dt_num[dt.isna()] = np.nan
    df[f"{col}_timestamp"] = dt_num
  X_train = X_train.drop(columns=[col])
  X_test = X_test.drop(columns=[col])

for col in text_cols:
  tfidf = TfidfVectorizer(max_features=20, stop_words="english")
  tr_text = X_train[col].fillna("").astype(str)
  te_text = X_test[col].fillna("").astype(str)
  tr_tfidf = tfidf.fit_transform(tr_text).toarray()
  te_tfidf = tfidf.transform(te_text).toarray()
  for i in range(tr_tfidf.shape[1]):
    X_train[f"{col}_tfidf_{i}"] = tr_tfidf[:, i]
    X_test[f"{col}_tfidf_{i}"] = te_tfidf[:, i]
  X_train = X_train.drop(columns=[col])
  X_test = X_test.drop(columns=[col])

for col in cat_cols:
  freq = X_train[col].value_counts(normalize=True).to_dict()
  X_train[f"{col}_freq"] = X_train[col].map(freq).fillna(0)
  X_test[f"{col}_freq"] = X_test[col].map(freq).fillna(0)

  le = LabelEncoder()
  combined = pd.concat([X_train[col], X_test[col]]).astype(str).fillna("missing")
  le.fit(combined)
  X_train[col] = le.transform(X_train[col].astype(str).fillna("missing"))
  X_test[col] = le.transform(X_test[col].astype(str).fillna("missing"))

for col in X_train.columns:
  if not pd.api.types.is_numeric_dtype(X_train[col]):
    le = LabelEncoder()
    combined = pd.concat([X_train[col], X_test[col]]).astype(str)
    le.fit(combined)
    X_train[col] = le.transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

X_train = X_train.apply(pd.to_numeric, errors="coerce")
X_test = X_test.apply(pd.to_numeric, errors="coerce")

clean_cols = [
    str(c)
    .replace("[", "_")
    .replace("]", "_")
    .replace("<", "_")
    .replace(":", "_")
    .replace("{", "_")
    .replace("}", "_")
    .replace('"', "_")
    .replace("'", "_")
    for c in X_train.columns
]
X_train.columns = clean_cols
X_test.columns = clean_cols

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_preds = np.zeros(len(X_test))
cb_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y)):
  X_tr, y_tr = X_train.iloc[tr_idx], y[tr_idx]
  X_val, y_val = X_train.iloc[val_idx], y[val_idx]

  lgb_model = lgb.LGBMClassifier(
      n_estimators=1000,
      learning_rate=0.03,
      num_leaves=31,
      random_state=42 + fold,
      verbosity=-1,
      subsample=0.8,
      colsample_bytree=0.8,
  )
  lgb_model.fit(
      X_tr,
      y_tr,
      eval_set=[(X_val, y_val)],
      callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
  )
  lgb_preds += lgb_model.predict_proba(X_test)[:, 1] / n_splits

  cb_model = cb.CatBoostClassifier(
      iterations=1000,
      learning_rate=0.04,
      random_seed=42 + fold,
      verbose=0,
      early_stopping_rounds=50,
  )
  cb_model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
  cb_preds += cb_model.predict_proba(X_test)[:, 1] / n_splits

  xgb_model = xgb.XGBClassifier(
      n_estimators=1000,
      learning_rate=0.03,
      max_depth=6,
      subsample=0.8,
      colsample_bytree=0.8,
      random_state=42 + fold,
      eval_metric="auc",
      early_stopping_rounds=50,
  )