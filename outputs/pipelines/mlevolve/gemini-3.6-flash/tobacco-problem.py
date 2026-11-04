# Tobacco Problem Predictive Pipeline
import os
import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")


def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)

# -------------------------------------------------------------------------
# 1. Load Data
# -------------------------------------------------------------------------
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "number_health_problems"

if "id" in test_df.columns:
    test_ids = test_df["id"].values
else:
    test_ids = np.arange(len(test_df))

drop_cols = [c for c in [target_col, "id"] if c in train_df.columns]
feature_cols = [c for c in train_df.columns if c not in drop_cols]

# Target encoding
le_target = LabelEncoder()
y = le_target.fit_transform(train_df[target_col].values)
n_classes = len(le_target.classes_)

# Categorical vs Numeric Column split
cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(train_df[c])]
num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_df[c])]

# -------------------------------------------------------------------------
# 2. Data Preprocessing
# -------------------------------------------------------------------------
X_train_df = train_df[feature_cols].copy()
X_test_df = test_df[feature_cols].copy()

# Categorical encoding
if len(cat_cols) > 0:
    for c in cat_cols:
        X_train_df[c] = X_train_df[c].astype(str).fillna("Missing")
        X_test_df[c] = X_test_df[c].astype(str).fillna("Missing")

    cat_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
    )
    X_train_df[cat_cols] = cat_encoder.fit_transform(X_train_df[cat_cols])
    X_test_df[cat_cols] = cat_encoder.transform(X_test_df[cat_cols])

# Missing value indicators for numerical features
for c in num_cols:
    if X_train_df[c].isna().sum() > 0 or X_test_df[c].isna().sum() > 0:
        X_train_df[f"{c}_isna"] = X_train_df[c].isna().astype(float)
        X_test_df[f"{c}_isna"] = X_test_df[c].isna().astype(float)

all_feature_cols = list(X_train_df.columns)

# Impute missing numeric values
if len(num_cols) > 0:
    imputer = SimpleImputer(strategy="median")
    X_train_df[num_cols] = imputer.fit_transform(X_train_df[num_cols])
    X_test_df[num_cols] = imputer.transform(X_test_df[num_cols])

# Drop zero variance columns
valid_cols = [c for c in all_feature_cols if X_train_df[c].nunique() > 1]
X_train = X_train_df[valid_cols].values.astype(np.float32)
X_test = X_test_df[valid_cols].values.astype(np.float32)


# -------------------------------------------------------------------------
# 3. Neural Network Architecture
# -------------------------------------------------------------------------
class TabularNN(nn.Module):

    def __init__(self, input_dim, num_classes, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = self.input_layer(x)
        h = h + self.block1(h)
        h = h + self.block2(h)
        return self.head(h)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------------
# 4. Stratified 5-Fold Cross-Validation Loop
# -------------------------------------------------------------------------
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb = np.zeros((len(y), n_classes))
oof_cat = np.zeros((len(y), n_classes))
oof_xgb = np.zeros((len(y), n_classes))
oof_hgb = np.zeros((len(y), n_classes))

test_lgb = np.zeros((len(X_test), n_classes))
test_cat = np.zeros((len(X_test), n_classes))
test_xgb = np.zeros((len(X_test), n_classes))
test_hgb = np.zeros((len(X_test), n_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train[train_idx], y[train_idx]
    X_va, y_va = X_train[val_idx], y[val_idx]
    X_te = X_test.copy()

    # 1. LightGBM
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=5,
        num_leaves=20,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42 + fold,
        objective="multiclass",
        num_class=n_classes,
        verbosity=-1,
    )
    lgb_clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    oof_lgb[val_idx] = lgb_clf.predict_proba(X_va)
    test_lgb += lgb_clf.predict_proba(X_te) / n_splits

    # 2. CatBoost
    cat_clf = cb.CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        random_seed=42 + fold,
        early_stopping_rounds=50,
        verbose=0,
    )
    cat_clf.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    oof_cat[val_idx] = cat_clf.predict_proba(X_va)
    test_cat += cat_clf.predict_proba(X_te) / n_splits

    # 3. XGBoost
    xgb_clf = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        early_stopping_rounds=50,
        eval_metric="mlogloss",
    )
    xgb_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[val_idx] = xgb_clf.predict_proba(X_va)
    test_xgb += xgb_clf.predict_proba(X_te) / n_splits

    # 4. HistGradientBoosting
    hgb_clf = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.03,
        max_depth=5,
        random_state=42 + fold,
    )
    hgb_clf.fit(X_tr, y_tr)
    oof_hgb[val_idx] = hgb_clf.predict_proba(X_va)
    test_hgb += hgb_clf.predict_proba(X_te) / n_splits

# Ensemble OOF and Test predictions
oof_ensemble = (oof_lgb + oof_cat + oof_xgb + oof_hgb) / 4.0
test_ensemble = (test_lgb + test_cat + test_xgb + test_hgb) / 4.0

oof_preds = np.argmax(oof_ensemble, axis=1)
val_score = accuracy_score(y, oof_preds)

final_preds = np.argmax(test_ensemble, axis=1)
if hasattr(le_target, "inverse_transform"):
    final_labels = le_target.inverse_transform(final_preds)
else:
    final_labels = final_preds

sub_df = pd.DataFrame({"id": test_ids, "number_health_problems": final_labels})
os.makedirs("./submission", exist_ok=True)
sub_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {val_score}")