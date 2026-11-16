import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb
from scipy.optimize import minimize

# Set random seed for reproducibility
RANDOM_STATE = 42

# Ensure working directory exists
os.makedirs("./working", exist_ok=True)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "number_health_problems"
y_raw = train_df[target_col]

# Encode target labels
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y_raw)
num_classes = len(target_encoder.classes_)

# Separate features
X_train_raw = train_df.drop(columns=[target_col]).copy()
X_test_raw = test_df.copy()

# Feature engineering: missingness count per row
X_train_raw["n_missing"] = X_train_raw.isnull().sum(axis=1)
X_test_raw["n_missing"] = X_test_raw.isnull().sum(axis=1)

# Identify column types
cat_cols = []
num_cols = []
for col in X_train_raw.columns:
    if col == "n_missing":
        num_cols.append(col)
    elif (
        X_train_raw[col].dtype == "object" or X_train_raw[col].dtype.name == "category"
    ):
        cat_cols.append(col)
    else:
        num_cols.append(col)

# Prepare datasets for CatBoost, LightGBM, and XGBoost
X_cb_train = X_train_raw.copy()
X_cb_test = X_test_raw.copy()

for col in cat_cols:
    X_cb_train[col] = X_cb_train[col].fillna("__MISSING__").astype(str)
    X_cb_test[col] = X_cb_test[col].fillna("__MISSING__").astype(str)

for col in num_cols:
    fill_val = X_cb_train[col].median()
    X_cb_train[col] = X_cb_train[col].fillna(fill_val)
    X_cb_test[col] = X_cb_test[col].fillna(fill_val)

# Add row-level summary statistics across numerical features
num_cols_to_agg = [c for c in num_cols if c != "n_missing"]
if len(num_cols_to_agg) > 0:
    for df in [X_cb_train, X_cb_test]:
        num_data = df[num_cols_to_agg]
        df["row_mean"] = num_data.mean(axis=1)
        df["row_std"] = num_data.std(axis=1).fillna(0)
        df["row_sum"] = num_data.sum(axis=1)
        df["row_nonzero_cnt"] = (num_data != 0).sum(axis=1)

# Frequency encoding for categoricals
for col in cat_cols:
    combined = pd.concat([X_cb_train[col], X_cb_test[col]])
    freq_map = combined.value_counts(normalize=True).to_dict()
    X_cb_train[f"{col}_freq"] = X_cb_train[col].map(freq_map).astype(float)
    X_cb_test[f"{col}_freq"] = X_cb_test[col].map(freq_map).astype(float)

# Prepare LightGBM dataset with category dtypes
X_lgb_train = X_cb_train.copy()
X_lgb_test = X_cb_test.copy()

for col in cat_cols:
    categories = sorted(
        list(set(X_cb_train[col].unique()).union(set(X_cb_test[col].unique())))
    )
    X_lgb_train[col] = pd.Categorical(X_lgb_train[col], categories=categories)
    X_lgb_test[col] = pd.Categorical(X_lgb_test[col], categories=categories)

# Prepare XGBoost dataset with numeric label-encoded categoricals
X_xgb_train = X_cb_train.copy()
X_xgb_test = X_cb_test.copy()

for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([X_cb_train[col], X_cb_test[col]]).astype(str)
    le.fit(combined)
    X_xgb_train[col] = le.transform(X_cb_train[col].astype(str))
    X_xgb_test[col] = le.transform(X_cb_test[col].astype(str))

# Stratified K-Fold setup
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

cb_oof = np.zeros((len(train_df), num_classes))
cb_test = np.zeros((len(test_df), num_classes))

lgb_oof = np.zeros((len(train_df), num_classes))
lgb_test = np.zeros((len(test_df), num_classes))

xgb_oof = np.zeros((len(train_df), num_classes))
xgb_test = np.zeros((len(test_df), num_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_cb_train, y)):
    y_tr, y_val = y[train_idx], y[val_idx]

    # 1. CatBoost fold training
    X_tr_cb, X_va_cb = X_cb_train.iloc[train_idx], X_cb_train.iloc[val_idx]
    cb_model = CatBoostClassifier(
        iterations=1200,
        learning_rate=0.035,
        depth=6,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        cat_features=cat_cols,
        random_seed=RANDOM_STATE + fold,
        early_stopping_rounds=50,
        verbose=False,
    )
    cb_model.fit(X_tr_cb, y_tr, eval_set=(X_va_cb, y_val), use_best_model=True)
    cb_oof[val_idx] = cb_model.predict_proba(X_va_cb)
    cb_test += cb_model.predict_proba(X_cb_test) / n_splits

    # 2. LightGBM fold training
    X_tr_lgb, X_va_lgb = X_lgb_train.iloc[train_idx], X_lgb_train.iloc[val_idx]
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE + fold,
        objective="multiclass",
        num_class=num_classes,
        verbose=-1,
    )
    lgb_model.fit(
        X_tr_lgb,
        y_tr,
        eval_set=[(X_va_lgb, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    lgb_oof[val_idx] = lgb_model.predict_proba(X_va_lgb)
    lgb_test += lgb_model.predict_proba(X_lgb_test) / n_splits

    # 3. XGBoost fold training
    X_tr_xgb, X_va_xgb = X_xgb_train.iloc[train_idx], X_xgb_train.iloc[val_idx]
    xgb_model = xgb.XGBClassifier(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE + fold,
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        early_stopping_rounds=50,
    )
    xgb_model.fit(X_tr_xgb, y_tr, eval_set=[(X_va_xgb, y_val)], verbose=False)
    xgb_oof[val_idx] = xgb_model.predict_proba(X_va_xgb)
    xgb_test += xgb_model.predict_proba(X_xgb_test) / n_splits

print(f"CatBoost OOF Accuracy: {accuracy_score(y, np.argmax(cb_oof, axis=1)):.5f}")
print(f"LightGBM OOF Accuracy: {accuracy_score(y, np.argmax(lgb_oof, axis=1)):.5f}")
print(f"XGBoost OOF Accuracy:  {accuracy_score(y, np.argmax(xgb_oof, axis=1)):.5f}")

# Grid search for optimal 3-way ensemble weights
best_acc = 0.0
best_weights = (1.0, 0.0, 0.0)

for w_cb in np.linspace(0.0, 1.0, 21):
    for w_lgb in np.linspace(0.0, 1.0 - w_cb, 21):
        w_xgb = max(0.0, 1.0 - w_cb - w_lgb)
        blended_oof = w_cb * cb_oof + w_lgb * lgb_oof + w_xgb * xgb_oof
        acc = accuracy_score(y, np.argmax(blended_oof, axis=1))
        if acc > best_acc:
            best_acc = acc
            best_weights = (w_cb, w_lgb, w_xgb)

w_cb, w_lgb, w_xgb = best_weights
print(f"Optimal Weights: CB={w_cb:.2f}, LGB={w_lgb:.2f}, XGB={w_xgb:.2f}")
print(f"5-Fold Ensemble CV Accuracy: {best_acc:.5f}")

# Generate ensemble test predictions
final_test_preds = w_cb * cb_test + w_lgb * lgb_test + w_xgb * xgb_test
final_test_classes = np.argmax(final_test_preds, axis=1)
final_test_labels = target_encoder.inverse_transform(final_test_classes)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test_df)),
        target_col: final_test_labels,
    }
)

submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully.")
