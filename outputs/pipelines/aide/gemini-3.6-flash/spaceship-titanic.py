import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

y_train = train_df["Transported"].astype(bool).values
train_len = len(train_df)

df = pd.concat(
    [train_df.drop(columns=["Transported"], errors="ignore"), test_df],
    ignore_index=True,
)

# Passenger ID parsing
df["Group"] = df["PassengerId"].str.split("_").str[0]
df["GroupMember"] = df["PassengerId"].str.split("_").str[1].astype(int)

# Group size and isolation status
group_counts = df["Group"].value_counts()
df["GroupSize"] = df["Group"].map(group_counts)
df["IsAlone"] = (df["GroupSize"] == 1).astype(int)

# Surname extraction and family size
df["Surname"] = df["Name"].str.split(" ").str[1]
family_counts = df["Surname"].value_counts()
df["FamilySize"] = df["Surname"].map(family_counts).fillna(1)

# Cabin parsing
cabin_split = df["Cabin"].str.split("/", expand=True)
df["Deck"] = cabin_split[0]
df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
df["Side"] = cabin_split[2]
df["CabinRegion"] = (df["CabinNum"] // 300).fillna(-1)

# Amenity spending columns
spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

# Domain logic rule imputations
child_mask = df["Age"] < 13
for col in spend_cols:
    df.loc[child_mask, col] = 0

cryo_true_mask = df["CryoSleep"] == True
for col in spend_cols:
    df.loc[cryo_true_mask, col] = 0

df["TotalSpend"] = df[spend_cols].sum(axis=1)
df.loc[df["TotalSpend"] > 0, "CryoSleep"] = False

# Group metadata sharing
for col in ["HomePlanet", "Destination", "Deck", "Side", "Surname"]:
    group_mode = df.groupby("Group")[col].first()
    df[col] = df[col].fillna(df["Group"].map(group_mode))

# Surname metadata sharing
surname_home = df.groupby("Surname")["HomePlanet"].first()
df["HomePlanet"] = df["HomePlanet"].fillna(df["Surname"].map(surname_home))

# Deck constraints
df.loc[df["Deck"].isin(["A", "B", "C", "T"]), "HomePlanet"] = df.loc[
    df["Deck"].isin(["A", "B", "C", "T"]), "HomePlanet"
].fillna("Europa")
df.loc[df["Deck"] == "G", "HomePlanet"] = df.loc[
    df["Deck"] == "G", "HomePlanet"
].fillna("Earth")

# Detailed expenditure aggregations
df["TotalSpend"] = df[spend_cols].sum(axis=1)
df["HasSpent"] = (df["TotalSpend"] > 0).astype(int)
df["LuxurySpend"] = df["Spa"] + df["VRDeck"] + df["RoomService"]
df["BasicSpend"] = df["FoodCourt"] + df["ShoppingMall"]

for col in spend_cols + ["TotalSpend", "LuxurySpend", "BasicSpend"]:
    df[f"{col}_log"] = np.log1p(df[col].fillna(0))

for col in spend_cols:
    df[f"{col}_ratio"] = df[col] / (df["TotalSpend"] + 1)

# Group-level aggregate statistics
df["Group_Cryo_Ratio"] = df.groupby("Group")["CryoSleep"].transform(
    lambda x: x.astype(float).mean()
)
df["Group_Spend_Mean"] = df.groupby("Group")["TotalSpend"].transform("mean")

# Frequency encoding
for col in ["Group", "Surname", "CabinNum"]:
    df[f"{col}_Freq"] = df[col].map(df[col].value_counts())

# Fill categorical missing values explicitly
cat_cols = [
    "HomePlanet",
    "CryoSleep",
    "Destination",
    "VIP",
    "Deck",
    "Side",
    "Surname",
]
for col in cat_cols:
    df[col] = df[col].astype(str).fillna("Unknown")

# Feature set cleanup
drop_cols = ["PassengerId", "Name", "Cabin", "Group"]
df_feat = df.drop(columns=drop_cols)

for c in df_feat.select_dtypes(include=["object", "category"]).columns:
    df_feat[c] = df_feat[c].astype(str)

cat_features = list(df_feat.select_dtypes(include=["object", "string"]).columns)

X_train = df_feat.iloc[:train_len].copy()
X_test = df_feat.iloc[train_len:].copy()

# Stratified K-Fold CV with CatBoost
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(train_len)
test_preds = np.zeros(len(test_df))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.04,
        depth=6,
        eval_metric="Accuracy",
        random_seed=42 + fold,
        verbose=0,
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        cat_features=cat_features,
        early_stopping_rounds=100,
        use_best_model=True,
    )

    oof_preds[val_idx] = model.predict_proba(X_va)[:, 1]
    test_preds += model.predict_proba(X_test)[:, 1] / 5.0

# Optimal threshold selection
best_thresh = 0.5
best_acc = accuracy_score(y_train, oof_preds >= best_thresh)
for thresh in np.arange(0.40, 0.60, 0.005):
    acc = accuracy_score(y_train, oof_preds >= thresh)
    if acc > best_acc:
        best_acc = acc
        best_thresh = thresh

print(f"Validation Accuracy: {best_acc:.5f}")

# Save formatted test predictions
test_binary = test_preds >= best_thresh
submission = pd.DataFrame({"id": np.arange(len(test_df)), "Transported": test_binary})
submission.to_csv("./working/submission.csv", index=False)
