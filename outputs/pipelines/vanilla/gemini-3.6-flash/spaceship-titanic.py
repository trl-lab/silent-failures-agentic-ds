import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

train_path = (
    "/path/to/user/workspace/ds_failures/datasets/kaggle/spaceship-titanic/data/train.parquet"
)
test_path = (
    "/path/to/user/workspace/ds_failures/datasets/kaggle/spaceship-titanic/data/test.parquet"
)

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

y = train["Transported"].astype(bool).values

df = pd.concat([train.drop(columns=["Transported"]), test], ignore_index=True)

df["Group"] = df["PassengerId"].str.split("_").str[0]
df["GroupNum"] = pd.to_numeric(
    df["PassengerId"].str.split("_").str[1], errors="coerce"
)
group_size = df["Group"].value_counts()
df["GroupSize"] = df["Group"].map(group_size)
df["IsAlone"] = (df["GroupSize"] == 1).astype(float)

cabin_split = df["Cabin"].str.split("/", expand=True)
df["CabinDeck"] = cabin_split[0]
df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
df["CabinSide"] = cabin_split[2]
df["CabinDeckSide"] = df["CabinDeck"].fillna("") + "_" + df["CabinSide"].fillna("")
cabin_size = df["Cabin"].value_counts()
df["CabinGroupSize"] = df["Cabin"].map(cabin_size)

exp_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
df["TotalSpent"] = df[exp_cols].sum(axis=1, skipna=True)
df["NoSpending"] = (df["TotalSpent"] == 0).astype(float)
df["NumZeroExp"] = (df[exp_cols] == 0).sum(axis=1).astype(float)
df["NumMissingExp"] = df[exp_cols].isna().sum(axis=1).astype(float)

for col in exp_cols:
  df[f"{col}_log"] = np.log1p(df[col].fillna(0))

df["TotalSpent_log"] = np.log1p(df["TotalSpent"])

group_exp = df.groupby("Group")["TotalSpent"].transform("mean")
df["GroupMeanSpent"] = group_exp

df["Surname"] = df["Name"].str.split().str[-1]
surname_size = df["Surname"].value_counts()
df["SurnameSize"] = df["Surname"].map(surname_size)

df["IsChild"] = (df["Age"] < 13).astype(float)

df["CryoSleep"] = df["CryoSleep"].astype(float)
df["VIP"] = df["VIP"].astype(float)

df = df.drop(columns=["PassengerId", "Cabin", "Name", "Group", "Surname"])

cat_cols = ["HomePlanet", "Destination", "CabinDeck", "CabinSide", "CabinDeckSide"]
for col in cat_cols:
  df[col] = df[col].astype("category").cat.codes.replace(-1, np.nan)

for col in df.columns:
  df[col] = df[col].astype(float)

train_len = len(train)
X = df.iloc[:train_len].copy()
X_test = df.iloc[train_len:].copy()

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

oof_preds = np.zeros(train_len)
test_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
  X_tr, y_tr = X.iloc[train_idx], y[train_idx]
  X_va, y_va = X.iloc[val_idx], y[val_idx]