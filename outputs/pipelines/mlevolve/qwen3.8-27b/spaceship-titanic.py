import pandas as pd
import numpy as np
import os
import json
import lightgbm as lgb
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# ============================================================
# SETUP
# ============================================================
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Store test IDs (task requires integer row index 0-based)
test_ids = np.arange(len(test))

# Mark source
train["is_train"] = 1
test["is_train"] = 0
test["Transported"] = np.nan

# ============================================================
# ROW-LEVEL FEATURE ENGINEERING (no cross-row aggregation, safe on combined)
# ============================================================
df_all = pd.concat([train, test], axis=0, ignore_index=True)

# 1. Extract Group ID from PassengerId
df_all["GroupID"] = df_all["PassengerId"].str.split("_").str[0]
df_all["GroupNum"] = df_all["PassengerId"].str.split("_").str[1].astype(int)

# 2. Decompose Cabin
df_all["Cabin_Deck"] = df_all["Cabin"].str.split("/").str[0]
df_all["Cabin_Number"] = df_all["Cabin"].str.split("/").str[1].astype(float)
df_all["Cabin_Side"] = df_all["Cabin"].str.split("/").str[2]

# 3. Spending features
spend_cols = ["FoodCourt", "RoomService", "Spa", "VRDeck", "ShoppingMall"]

df_all["TotalSpend"] = df_all[spend_cols].sum(axis=1)
df_all["SpendMean"] = df_all[spend_cols].mean(axis=1)
df_all["SpendMax"] = df_all[spend_cols].max(axis=1)
df_all["SpendCount"] = (df_all[spend_cols] > 0).sum(axis=1)
for col in spend_cols:
    df_all[f"{col}_log"] = np.log1p(df_all[col])

# 4. Binary column encoding
for col in ["CryoSleep", "VIP"]:
    df_all[col] = df_all[col].map({True: 1, False: 0})

# 7. Interaction features (row-level only)
df_all["Age_Spend"] = df_all["Age"] * df_all["TotalSpend"]
df_all["Age_Cryo"] = df_all["Age"] * df_all["CryoSleep"]
df_all["SpendPerAge"] = df_all["TotalSpend"] / (df_all["Age"] + 1)

# 8. HomePlanet x Destination interaction
df_all["Planet_Dest"] = df_all["HomePlanet"].astype(str) + "_" + df_all["Destination"].astype(str)

# 10. Cabin number features
df_all["Cabin_Number_log"] = np.log1p(df_all["Cabin_Number"])

# ============================================================
# SPLIT DATA
# ============================================================
train_full = df_all[df_all["is_train"] == 1].copy()
test_full = df_all[df_all["is_train"] == 0].copy()

train_data, val_data = train_test_split(
    train_full, test_size=0.15, random_state=42, stratify=train_full["Transported"]
)

print(
    f"Train split: {train_data.shape}, Val split: {val_data.shape}, Test: {test_full.shape}"
)

# ============================================================
# GROUP/DECK-LEVEL FEATURES (fit on train_data only, transform val/test)
# ============================================================
# Group-level features using train_data as reference
train_group_stats = train_data.groupby("GroupID").agg(
    GroupSize=("GroupID", "count"),
    GroupTotalSpend=("TotalSpend", "mean"),
    GroupSpendMax=("TotalSpend", "max"),
).reset_index()

for name in [train_data, val_data, test_full]:
    name = name.merge(train_group_stats, on="GroupID", how="left")
    name["GroupSize"] = name["GroupSize"].fillna(1)
    name["IsAlone"] = (name["GroupSize"] == 1).astype(int)
    name["GroupTotalSpend"] = name["GroupTotalSpend"].fillna(name["TotalSpend"])
    name["GroupSpendMax"] = name["GroupSpendMax"].fillna(name["TotalSpend"])

# Deck-level features using train_data as reference
train_deck_stats = train_data.groupby("Cabin_Deck").agg(
    DeckMeanAge=("Age", "mean"),
    DeckCount=("Cabin_Deck", "count"),
).reset_index()

for name in [train_data, val_data, test_full]:
    name = name.merge(train_deck_stats, on="Cabin_Deck", how="left")
    name["DeckMeanAge"] = name["DeckMeanAge"].fillna(train_data["Age"].mean())
    name["DeckCount"] = name["DeckCount"].fillna(0)

# ============================================================
# PREPARE FEATURES
# ============================================================
exclude_cols = [
    "PassengerId",
    "Name",
    "Transported",
    "is_train",
    "GroupID",
    "Cabin",
    "Planet_Dest",
]
feature_cols = [c for c in train_data.columns if c not in exclude_cols]
# Ensure consistent column order across all splits
for name in [val_data, test_full]:
    for c in feature_cols:
        if c not in name.columns:
            name[c] = 0.0

# Encode categorical features (fit on train only)
cat_cols = [c for c in feature_cols if train_data[c].dtype == "object"]
print(f"Categorical features: {cat_cols}")

le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    train_data[col] = train_data[col].fillna("MISSING")
    val_data[col] = val_data[col].fillna("MISSING")
    test_full[col] = test_full[col].fillna("MISSING")
    # Fit on train only to avoid leaking test/val category info
    le.fit(sorted(train_data[col].unique()))
    train_data[col] = le.transform(train_data[col])
    val_data[col] = val_data[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    test_full[col] = test_full[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    le_dict[col] = le

# Handle remaining NaNs in numerical features
num_cols = [c for c in feature_cols if train_data[c].dtype != "object"]
for col in num_cols:
    if train_data[col].isna().any():
        median_val = train_data[col].median()
        train_data[col] = train_data[col].fillna(median_val)
        val_data[col] = val_data[col].fillna(median_val)
        test_full[col] = test_full[col].fillna(median_val)

# Ensure all features are numeric
for col in feature_cols:
    train_data[col] = train_data[col].astype(float)
    val_data[col] = val_data[col].astype(float)
    test_full[col] = test_full[col].astype(float)

# Encode target
train_data["Transported"] = train_data["Transported"].astype(int)
val_data["Transported"] = val_data["Transported"].astype(int)

print(f"Number of features: {len(feature_cols)}")

# ============================================================
# PREPARE ARRAYS
# ============================================================
X_train = train_data[feature_cols].values.astype(np.float32)
y_train = train_data["Transported"].values.astype(np.int64)
X_val = val_data[feature_cols].values.astype(np.float32)
y_val = val_data["Transported"].values.astype(np.int64)
X_test = test_full[feature_cols].values.astype(np.float32)

# ============================================================
# TRAIN LIGHTGBM
# ============================================================
lgb_params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "min_child_weight": 0.01,
    "max_depth": -1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "n_estimators": 2000,
    "random_state": 42,
}

train_ds = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
val_ds = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=train_ds)

model_lgb = lgb.train(
    lgb_params,
    train_ds,
    valid_sets=[val_ds],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=500),
    ],
)

print(f"LightGBM best iteration: {model_lgb.best_iteration}")


# ============================================================
# TRAIN NEURAL NETWORK
# ============================================================
class TabularNet(nn.Module):
    def __init__(self, n_features):
        super(TabularNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


# Scale features for NN (fit on train only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_val_scaled = scaler.transform(X_val).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

# Prepare tensors
X_train_t = torch.FloatTensor(X_train_scaled)
y_train_t = torch.FloatTensor(y_train)
X_val_t = torch.FloatTensor(X_val_scaled)
X_test_t = torch.FloatTensor(X_test_scaled)

criterion = nn.BCEWithLogitsLoss()
model_nn = TabularNet(n_features=len(feature_cols))
optimizer = torch.optim.AdamW(model_nn.parameters(), lr=3e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=500, eta_min=1e-5
)

# Mini-batch training with shuffling
train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)

best_val_acc_nn = 0.0
best_model_state = None
patience_counter = 0
patience = 30
eval_interval = 5

for epoch in range(500):
    model_nn.train()
    epoch_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        logits = model_nn(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()
    epoch_loss /= len(train_loader)

    if (epoch + 1) % eval_interval == 0:
        model_nn.eval()
        with torch.no_grad():
            val_logits = model_nn(X_val_t)
            val_preds_nn = (torch.sigmoid(val_logits) >= 0.5).int().numpy()
            val_acc_nn = accuracy_score(y_val, val_preds_nn)

        print(f"Epoch {epoch+1:3d} | loss={epoch_loss:.4f} | val_acc={val_acc_nn:.4f}")

        if val_acc_nn > best_val_acc_nn:
            best_val_acc_nn = val_acc_nn
            best_model_state = {k: v.clone() for k, v in model_nn.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Load best model state
model_nn.load_state_dict(best_model_state)
model_nn.eval()

# ============================================================
# VALIDATION PREDICTIONS
# ============================================================
val_probs_lgb = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration)

with torch.no_grad():
    val_probs_nn = torch.sigmoid(model_nn(X_val_t)).numpy()

# Weighted ensemble: favor LightGBM (stronger model)
val_probs_combined = 0.7 * val_probs_lgb + 0.3 * val_probs_nn
val_preds_combined = (val_probs_combined >= 0.5).astype(int)
val_acc_combined = accuracy_score(y_val, val_preds_combined)

# Also check LightGBM alone
val_acc_lgb = accuracy_score(y_val, (val_probs_lgb >= 0.5).astype(int))
val_acc_nn = accuracy_score(y_val, (val_probs_nn >= 0.5).astype(int))

print(f"\n--- Validation Results ---")
print(f"LightGBM accuracy: {val_acc_lgb:.4f}")
print(f"Neural Net accuracy: {val_acc_nn:.4f}")
print(f"Combined (0.7*LGB + 0.3*NN) accuracy: {val_acc_combined:.4f}")

# Use the best performing approach
if val_acc_lgb > val_acc_combined:
    print("Using LightGBM alone (better validation performance)")
    val_probs_combined = val_probs_lgb
    val_preds_combined = (val_probs_combined >= 0.5).astype(int)
    val_acc_combined = val_acc_lgb
    ensemble_weight_lgb = 1.0
else:
    ensemble_weight_lgb = 0.7

# ============================================================
# TEST INFERENCE
# ============================================================
test_probs_lgb = model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration)

with torch.no_grad():
    test_probs_nn = torch.sigmoid(model_nn(X_test_t)).numpy()

test_probs_combined = ensemble_weight_lgb * test_probs_lgb + (1 - ensemble_weight_lgb) * test_probs_nn
test_preds_combined = (test_probs_combined >= 0.5).astype(int)

# ============================================================
# SAVE SUBMISSION
# ============================================================
submission = pd.DataFrame(
    {"id": test_ids, "Transported": test_preds_combined}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(f"Test prediction distribution: {test_preds_combined.mean():.4f}")

# ============================================================
# SAVE MODELS
# ============================================================
model_lgb.save_model("./working/model_lgb.txt")
torch.save(best_model_state, "./working/model_nn_best.pt")
print("Models saved to ./working/")

# ============================================================
# FINAL SCORE
# ============================================================
score = val_acc_combined
print(f"Final Validation Score: {score}")