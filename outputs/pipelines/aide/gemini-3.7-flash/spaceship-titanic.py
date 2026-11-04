import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

train_len = len(train_df)
combined = pd.concat(
    [train_df.drop(columns=["Transported"], errors="ignore"), test_df],
    axis=0,
    ignore_index=True,
)

# 2. Extract Basic Structural Features
combined["Group"] = combined["PassengerId"].apply(lambda x: x.split("_")[0])
combined["Group_Num"] = (
    combined["PassengerId"].apply(lambda x: x.split("_")[1]).astype(int)
)
group_sizes = combined["Group"].value_counts()
combined["GroupSize"] = combined["Group"].map(group_sizes)
combined["IsAlone"] = (combined["GroupSize"] == 1).astype(int)

# Extract Surname
combined["Surname"] = combined["Name"].apply(
    lambda x: x.split()[-1] if pd.notna(x) else np.nan
)
surname_sizes = combined["Surname"].value_counts()
combined["FamilySize"] = combined["Surname"].map(surname_sizes).fillna(1)

# Extract Cabin Information
combined["CabinDeck"] = combined["Cabin"].apply(
    lambda x: x.split("/")[0] if pd.notna(x) else np.nan
)
combined["CabinNum"] = combined["Cabin"].apply(
    lambda x: float(x.split("/")[1]) if pd.notna(x) else np.nan
)
combined["CabinSide"] = combined["Cabin"].apply(
    lambda x: x.split("/")[2] if pd.notna(x) else np.nan
)

# Cabin Deck Region Feature
combined["CabinRegion"] = combined["CabinNum"] // 300

# 3. Domain-Specific Imputation Rules
expenditure_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

# If in CryoSleep, luxury expenditures must be 0
for col in expenditure_cols:
    combined.loc[combined["CryoSleep"] == True, col] = combined.loc[
        combined["CryoSleep"] == True, col
    ].fillna(0.0)

# If total spending > 0, CryoSleep is False
spent_sum = combined[expenditure_cols].sum(axis=1, skipna=True)
combined.loc[(combined["CryoSleep"].isna()) & (spent_sum > 0), "CryoSleep"] = False

# Fill remaining expenditures with 0
for col in expenditure_cols:
    combined[col] = combined[col].fillna(0.0)

# Infer HomePlanet from Group
group_homeplanet = combined.groupby("Group")["HomePlanet"].first()
combined["HomePlanet"] = combined["HomePlanet"].fillna(
    combined["Group"].map(group_homeplanet)
)

# Infer HomePlanet from Surname
surname_homeplanet = combined.groupby("Surname")["HomePlanet"].first()
combined["HomePlanet"] = combined["HomePlanet"].fillna(
    combined["Surname"].map(surname_homeplanet)
)

# Infer HomePlanet from CabinDeck
combined.loc[
    combined["HomePlanet"].isna() & combined["CabinDeck"].isin(["A", "B", "C", "T"]),
    "HomePlanet",
] = "Europa"
combined.loc[
    combined["HomePlanet"].isna() & combined["CabinDeck"].isin(["G"]), "HomePlanet"
] = "Earth"
combined["HomePlanet"] = combined["HomePlanet"].fillna("Earth")

# Infer CabinSide from Group
group_side = combined.groupby("Group")["CabinSide"].first()
combined["CabinSide"] = (
    combined["CabinSide"].fillna(combined["Group"].map(group_side)).fillna("Missing")
)

# Infer CabinDeck from Group
group_deck = combined.groupby("Group")["CabinDeck"].first()
combined["CabinDeck"] = (
    combined["CabinDeck"].fillna(combined["Group"].map(group_deck)).fillna("Missing")
)

# Fill missing categorical & boolean values
combined["CryoSleep"] = combined["CryoSleep"].fillna(False).astype(int)
combined["VIP"] = combined["VIP"].fillna(False).astype(int)
combined["Destination"] = combined["Destination"].fillna("Missing")

# Fill missing Age with median by HomePlanet and CryoSleep
median_ages = combined.groupby(["HomePlanet", "CryoSleep"])["Age"].transform("median")
combined["Age"] = combined["Age"].fillna(median_ages).fillna(combined["Age"].median())

# 4. Expenditure Aggregation and Ratio Features
combined["TotalExpenditure"] = combined[expenditure_cols].sum(axis=1)
combined["NoSpending"] = (combined["TotalExpenditure"] == 0).astype(int)
combined["LuxurySpending"] = (
    combined["Spa"] + combined["VRDeck"] + combined["RoomService"]
)
combined["AmenitiesSpending"] = combined["FoodCourt"] + combined["ShoppingMall"]

for col in expenditure_cols:
    combined[f"{col}_log"] = np.log1p(combined[col])
    combined[f"{col}_ratio"] = (
        combined[col] / (combined["TotalExpenditure"] + 1.0)
    ).astype(np.float32)

combined["TotalExpenditure_log"] = np.log1p(combined["TotalExpenditure"])
combined["LuxurySpending_log"] = np.log1p(combined["LuxurySpending"])

# Age Group Features
combined["IsChild"] = (combined["Age"] < 13).astype(int)
combined["IsSenior"] = (combined["Age"] >= 60).astype(int)

# 5. Prepare Modeling Data
features = (
    [
        "HomePlanet",
        "CryoSleep",
        "CabinDeck",
        "CabinNum",
        "CabinSide",
        "CabinRegion",
        "Destination",
        "Age",
        "VIP",
        "GroupSize",
        "IsAlone",
        "FamilySize",
        "TotalExpenditure",
        "NoSpending",
        "LuxurySpending",
        "AmenitiesSpending",
        "TotalExpenditure_log",
        "LuxurySpending_log",
        "IsChild",
        "IsSenior",
    ]
    + expenditure_cols
    + [f"{col}_log" for col in expenditure_cols]
    + [f"{col}_ratio" for col in expenditure_cols]
)

cat_features = ["HomePlanet", "CabinDeck", "CabinSide", "Destination"]

# Convert categorical columns to string
for col in cat_features:
    combined[col] = combined[col].astype(str)

X_train = combined.iloc[:train_len][features].copy()
y_train = train_df["Transported"].astype(int).values
X_test = combined.iloc[train_len:][features].copy()

# 6. Stratified 5-Fold Cross-Validation with CatBoost
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

oof_preds = np.zeros(train_len)
test_preds = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

    train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
    val_pool = Pool(X_va, y_va, cat_features=cat_features)
    test_pool = Pool(X_test, cat_features=cat_features)

    model = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        eval_metric="Accuracy",
        random_seed=SEED + fold,
        verbose=False,
    )

    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100, verbose=False)

    oof_preds[val_idx] = model.predict_proba(val_pool)[:, 1]
    test_preds += model.predict_proba(test_pool)[:, 1] / n_splits

# 7. Evaluate Out-of-Fold Accuracy
oof_binary = (oof_preds > 0.5).astype(int)
cv_accuracy = accuracy_score(y_train, oof_binary)
print(f"5-Fold Cross-Validation Accuracy: {cv_accuracy:.5f}")

# 8. Generate Submission File
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_df)), "Transported": (test_preds > 0.5)}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully.")
