import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from xgboost import XGBRegressor

train_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/house-prices-advanced-regression-techniques/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/kaggle/house-prices-advanced-regression-techniques/data/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

if "GrLivArea" in train.columns and "SalePrice" in train.columns:
    train = train[~((train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000))].reset_index(drop=True)

y = np.log1p(train["SalePrice"].values)
train_len = len(train)
test_len = len(test)

df = pd.concat([train.drop(columns=["SalePrice"]), test], axis=0).reset_index(drop=True)
if "Id" in df.columns:
    df = df.drop(columns=["Id"])

for col in [
    "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "FullBath", "HalfBath",
    "BsmtFullBath", "BsmtHalfBath", "YrSold", "YearBuilt", "YearRemodAdd",
    "GarageYrBlt", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
    "3SsnPorch", "ScreenPorch", "GrLivArea", "OverallQual", "OverallCond"
]:
    if col in df.columns:
        df[col] = df[col].fillna(0)

df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
df["TotalBaths"] = df["FullBath"] + 0.5 * df["HalfBath"] + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
df["Age"] = df["YrSold"] - df["YearBuilt"]
df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
df["GarageAge"] = df["YrSold"] - df["GarageYrBlt"]
df["TotalPorchSF"] = df["WoodDeckSF"] + df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"]
df["TotalQual"] = df["OverallQual"] + df["OverallCond"]

cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].fillna("Missing").astype(str)

oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
df[cat_cols] = oe.fit_transform(df[cat_cols])

X_train = df.iloc[:train_len].copy()
X_test = df.iloc[train_len:].copy()

kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_preds = np.zeros(test_len)
xgb_preds = np.zeros(test_len)
cat_preds = np.zeros(test_len)
ridge_preds = np.zeros(test_len)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

for train_idx, val_idx in kf.split(X_train):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    
    lgb = LGBMRegressor(n_estimators=1000, learning_rate=0.03, max_depth=6, num_leaves=31, random_state=42, verbose=-1)
    lgb.fit(X_tr, y_tr)
    lgb_preds += lgb.predict(X_test) / kf.n_splits
    
    xgb = XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    xgb.fit(X_tr, y_tr)
    xgb_preds += xgb.predict(X_test) / kf.n_splits
    
    cat = CatBoostRegressor(iterations=1000, learning_rate=0.03, depth=6, random_seed=42, verbose=0)
    cat.fit(X_tr, y_tr)
    cat_preds += cat.predict(X_test) / kf.n_splits
    
    X_tr_s = X_train_scaled[train_idx]
    ridge = RidgeCV(alphas=np.logspace(-2, 3, 20))
    ridge.fit(X_tr_s, y_tr)
    ridge_preds += ridge.predict(X_test_scaled) / kf.n_splits

final_log_preds = 0.3 * lgb_preds + 0.3 * xgb_preds + 0.3 * cat_preds + 0.1 * ridge_preds
final_preds = np.expm1(final_log_preds)

sub = pd.DataFrame({"id": np.arange(test_len), "SalePrice": final_preds})
sub.to_csv("predictions.csv", index=False)