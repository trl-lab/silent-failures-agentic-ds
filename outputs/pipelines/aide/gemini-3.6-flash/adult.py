import warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

try:
    from catboost import CatBoostClassifier

    HAS_CAT = True
except ImportError:
    HAS_CAT = False

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")

# 1. Load data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

# Clean string columns
str_cols = [
    c
    for c in train.columns
    if train[c].dtype == "object" or isinstance(train[c].dtype, pd.CategoricalDtype)
]
for col in str_cols:
    train[col] = train[col].astype(str).str.strip()
    if col in test.columns:
        test[col] = test[col].astype(str).str.strip()

# Target processing
y = (
    train["income"]
    .astype(str)
    .str.strip()
    .str.replace(".", "", regex=False)
    .apply(lambda x: 1 if ">50K" in x else 0)
    .values
)

train_features = train.drop(columns=["income"], errors="ignore")
test_features = test.drop(columns=["income"], errors="ignore")

# Combine datasets for consistent feature engineering
df = pd.concat([train_features, test_features], ignore_index=True)

# Domain numeric features
df["net_capital"] = df["capital-gain"] - df["capital-loss"]
df["has_capital_gain"] = (df["capital-gain"] > 0).astype(int)
df["has_capital_loss"] = (df["capital-loss"] > 0).astype(int)
df["capital_gain_log"] = np.log1p(df["capital-gain"])
df["capital_loss_log"] = np.log1p(df["capital-loss"])
df["hours_per_age"] = df["hours-per-week"] / (df["age"] + 1e-5)
df["education_per_age"] = df["education-num"] / (df["age"] + 1e-5)
df["age_hours_prod"] = df["age"] * df["hours-per-week"]

# Categorical interactions
df["marital_sex"] = df["marital-status"].astype(str) + "_" + df["sex"].astype(str)
df["work_occupation"] = df["workclass"].astype(str) + "_" + df["occupation"].astype(str)
df["relationship_marital"] = (
    df["relationship"].astype(str) + "_" + df["marital-status"].astype(str)
)
df["education_occupation"] = (
    df["education"].astype(str) + "_" + df["occupation"].astype(str)
)
df["education_marital"] = (
    df["education"].astype(str) + "_" + df["marital-status"].astype(str)
)

cat_cols = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    "marital_sex",
    "work_occupation",
    "relationship_marital",
    "education_occupation",
    "education_marital",
]

# Frequency encodings
for col in cat_cols:
    freq = df[col].value_counts(normalize=True)
    df[f"{col}_freq"] = df[col].map(freq)

# Convert categoricals to ordinal integer codes
for col in cat_cols:
    df[col] = pd.factorize(df[col])[0]

# Prepare LightGBM version with category dtypes
df_lgb = df.copy()
for col in cat_cols:
    df_lgb[col] = df_lgb[col].astype("category")

# Separate train and test
X = df.iloc[: len(train)].copy()
X_test = df.iloc[len(train) :].copy()

X_lgb = df_lgb.iloc[: len(train)].copy()
X_lgb_test = df_lgb.iloc[len(train) :].copy()

# Stratified 5-Fold Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds_lgb1 = np.zeros(len(train))
test_preds_lgb1 = np.zeros(len(test))

oof_preds_lgb2 = np.zeros(len(train))
test_preds_lgb2 = np.zeros(len(test))

oof_preds_hgb = np.zeros(len(train))
test_preds_hgb = np.zeros(len(test))

oof_preds_other = np.zeros(len(train))
test_preds_other = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    # Data split
    X_tr_num, y_tr = X.iloc[train_idx], y[train_idx]
    X_va_num, y_va = X.iloc[val_idx], y[val_idx]

    X_tr_lgb = X_lgb.iloc[train_idx]
    X_va_lgb = X_lgb.iloc[val_idx]

    # Model 1: LGBM Deep
    model_lgb1 = LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    model_lgb1.fit(
        X_tr_lgb,
        y_tr,
        eval_set=[(X_va_lgb, y_va)],
        callbacks=[early_stopping(stopping_rounds=50, verbose=False)],
    )
    oof_preds_lgb1[val_idx] = model_lgb1.predict_proba(X_va_lgb)[:, 1]
    test_preds_lgb1 += model_lgb1.predict_proba(X_lgb_test)[:, 1] / skf.n_splits

    # Model 2: LGBM Shallow
    model_lgb2 = LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=4,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=100 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    model_lgb2.fit(
        X_tr_lgb,
        y_tr,
        eval_set=[(X_va_lgb, y_va)],
        callbacks=[early_stopping(stopping_rounds=50, verbose=False)],
    )
    oof_preds_lgb2[val_idx] = model_lgb2.predict_proba(X_va_lgb)[:, 1]
    test_preds_lgb2 += model_lgb2.predict_proba(X_lgb_test)[:, 1] / skf.n_splits

    # Model 3: HistGradientBoosting
    model_hgb = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.03, max_leaf_nodes=31, random_state=42 + fold
    )
    model_hgb.fit(X_tr_num, y_tr)
    oof_preds_hgb[val_idx] = model_hgb.predict_proba(X_va_num)[:, 1]
    test_preds_hgb += model_hgb.predict_proba(X_test)[:, 1] / skf.n_splits

    # Model 4: CatBoost or XGBoost or second HistGB
    if HAS_CAT:
        model_other = CatBoostClassifier(
            iterations=800,
            learning_rate=0.04,
            depth=6,
            random_seed=42 + fold,
            verbose=0,
        )
        model_other.fit(
            X_tr_num, y_tr, eval_set=(X_va_num, y_va), early_stopping_rounds=50
        )
    elif HAS_XGB:
        model_other = XGBClassifier(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            eval_metric="logloss",
            early_stopping_rounds=50,
            n_jobs=-1,
        )
        model_other.fit(X_tr_num, y_tr, eval_set=[(X_va_num, y_va)], verbose=False)
    else:
        model_other = HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.05, max_leaf_nodes=15, random_state=200 + fold
        )
        model_other.fit(X_tr_num, y_tr)

    oof_preds_other[val_idx] = model_other.predict_proba(X_va_num)[:, 1]
    test_preds_other += model_other.predict_proba(X_test)[:, 1] / skf.n_splits

# Blend OOF predictions
oof_blend = (
    0.35 * oof_preds_lgb1
    + 0.25 * oof_preds_lgb2
    + 0.20 * oof_preds_hgb
    + 0.20 * oof_preds_other
)

test_blend = (
    0.35 * test_preds_lgb1
    + 0.25 * test_preds_lgb2
    + 0.20 * test_preds_hgb
    + 0.20 * test_preds_other
)

# Threshold Optimization on OOF
best_threshold = 0.5
best_acc = 0.0

for t in np.linspace(0.45, 0.55, 101):
    acc = accuracy_score(y, (oof_blend > t).astype(int))
    if acc > best_acc:
        best_acc = acc
        best_threshold = t

print(f"Optimal Threshold: {best_threshold:.4f}")
print(f"Validation Accuracy: {best_acc:.5f}")

# Generate and save test submission
test_labels = np.where(test_blend > best_threshold, ">50K", "<=50K")
submission = pd.DataFrame({"id": np.arange(len(test_labels)), "income": test_labels})
submission.to_csv("./working/submission.csv", index=False)
