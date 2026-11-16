import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
import os
import re

# ============================================================
# 1. LOAD DATA
# ============================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================


def extract_title(name):
    if pd.isna(name):
        return "Unknown"
    title = re.search(r", (\w+)\.", name)
    if title:
        return title.group(1)
    return "Unknown"


def get_deck(cabin):
    if pd.isna(cabin):
        return "U"
    return str(cabin)[0]


def get_cabin_count(cabin):
    if pd.isna(cabin):
        return 0
    return len(str(cabin).split())


def engineer_features(df):
    df = df.copy()

    # Title features
    df["Title"] = df["Name"].apply(extract_title)
    title_map = {
        "Mr": "Mr",
        "Miss": "Miss",
        "Mrs": "Mrs",
        "Master": "Master",
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
        "Don": "Other",
        "Dona": "Other",
        "Rev": "Other",
        "Dr": "Other",
        "Major": "Other",
        "Lady": "Other",
        "Countess": "Other",
        "Col": "Other",
        "Capt": "Other",
        "Sir": "Other",
        "Jonkheer": "Other",
    }
    df["Title"] = df["Title"].map(lambda x: title_map.get(x, "Other"))

    # Family features
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["FamilySizeBin"] = pd.cut(
        df["FamilySize"], bins=[0, 1, 3, 5, 20], labels=[0, 1, 2, 3]
    ).astype(int)

    # Ticket-based group features
    ticket_counts = df["Ticket"].value_counts()
    df["TicketGroupSize"] = df["Ticket"].map(ticket_counts)

    # Cabin features
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    df["Deck"] = df["Cabin"].apply(get_deck)
    df["CabinCount"] = df["Cabin"].apply(get_cabin_count)

    # Age features
    df["AgeBin"] = pd.cut(
        df["Age"], bins=[0, 12, 18, 35, 55, 100], labels=[0, 1, 2, 3, 4]
    ).fillna(0).astype(int)
    df["IsChild"] = (df["Age"] <= 12).fillna(False).astype(int)
    df["IsSenior"] = (df["Age"] > 60).fillna(False).astype(int)

    # Fare features
    df["FareLog"] = np.log1p(df["Fare"].fillna(0))
    df["FareBin"] = pd.qcut(
        df["Fare"].fillna(0), q=4, labels=[0, 1, 2, 3], duplicates="drop"
    ).fillna(0).astype(int)

    # Interaction features
    df["Pclass_Sex"] = df["Pclass"].astype(str) + "_" + df["Sex"]
    df["Age_Pclass"] = df["Age"] * df["Pclass"]
    df["Fare_per_Pclass"] = df["Fare"] / df["Pclass"]
    df["Title_Pclass"] = df["Title"].astype(str) + "_" + df["Pclass"].astype(str)

    # Name/Ticket features
    df["NameLength"] = df["Name"].apply(lambda x: len(str(x)))
    df["TicketLength"] = df["Ticket"].apply(lambda x: len(str(x)))
    df["TicketIsNumeric"] = df["Ticket"].apply(lambda x: str(x).isdigit()).astype(int)

    # Sex encoding
    df["SexEncoded"] = (df["Sex"] == "male").astype(int)

    return df


train_df = engineer_features(train_df)
test_df = engineer_features(test_df)

# ============================================================
# 3. HANDLE MISSING VALUES (fit on train only)
# ============================================================
train_age_medians = train_df.groupby("Pclass")["Age"].median()
train_df["Age"] = train_df["Age"].fillna(train_df["Pclass"].map(train_age_medians))
test_df["Age"] = test_df["Age"].fillna(test_df["Pclass"].map(train_age_medians))

embarked_mode = train_df["Embarked"].mode()[0]
train_df["Embarked"] = train_df["Embarked"].fillna(embarked_mode)
test_df["Embarked"] = test_df["Embarked"].fillna(embarked_mode)

train_df["Fare"] = train_df["Fare"].fillna(0)
test_df["Fare"] = test_df["Fare"].fillna(0)

# Re-compute derived features after imputation
for df in [train_df, test_df]:
    df["IsChild"] = (df["Age"] <= 12).fillna(False).astype(int)
    df["IsSenior"] = (df["Age"] > 60).fillna(False).astype(int)
    df["AgeBin"] = pd.cut(
        df["Age"], bins=[0, 12, 18, 35, 55, 100], labels=[0, 1, 2, 3, 4]
    ).fillna(0).astype(int)

# ============================================================
# 4. ENCODING
# ============================================================
categorical_cols = ["Sex", "Embarked", "Title", "Deck", "Pclass_Sex", "Title_Pclass"]
numerical_cols = [
    "Age",
    "Fare",
    "FareLog",
    "Fare_per_Pclass",
    "SibSp",
    "Parch",
    "FamilySize",
    "IsAlone",
    "FamilySizeBin",
    "TicketGroupSize",
    "HasCabin",
    "CabinCount",
    "AgeBin",
    "IsChild",
    "IsSenior",
    "FareBin",
    "Age_Pclass",
    "NameLength",
    "TicketLength",
    "TicketIsNumeric",
    "SexEncoded",
    "Pclass",
]

train_encoded = pd.get_dummies(
    train_df[categorical_cols], columns=categorical_cols, drop_first=False
)
test_encoded = pd.get_dummies(
    test_df[categorical_cols], columns=categorical_cols, drop_first=False
)

# Align columns
all_cat_cols = sorted(set(train_encoded.columns) | set(test_encoded.columns))
train_encoded = train_encoded.reindex(columns=all_cat_cols, fill_value=0)
test_encoded = test_encoded.reindex(columns=all_cat_cols, fill_value=0)

X_train_full = pd.concat(
    [train_df[numerical_cols].reset_index(drop=True), train_encoded], axis=1
)
X_test_full = pd.concat(
    [test_df[numerical_cols].reset_index(drop=True), test_encoded], axis=1
)

X_train_full = X_train_full.fillna(0)
X_test_full = X_test_full.fillna(0)

# ============================================================
# 5. SPLIT (Stratified 80/20)
# ============================================================
y_train = train_df["Survived"].values

X_train, X_val, y_train_split, y_val = train_test_split(
    X_train_full, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test_full.shape}")

# ============================================================
# 6. SCALING (fit on train only)
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test_full)

# ============================================================
# 7. ENSEMBLE TRAINING WITH 5-FOLD CV
# ============================================================
def make_lgbm():
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=8,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

def make_xgb():
    return xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        early_stopping_rounds=50,
    )

def make_xgb_no_es():
    return xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )

def make_logreg():
    return LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=1000,
        random_state=42,
    )

def make_extra_trees():
    return ExtraTreesClassifier(
        n_estimators=500,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

model_factories = [make_lgbm, make_xgb, make_logreg, make_extra_trees]
model_names = ["LightGBM", "XGBoost", "LogReg", "ExtraTrees"]

# 5-fold CV to get OOF predictions on the training split
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(X_train_scaled), len(model_factories)))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train_split)):
    X_tr = X_train_scaled[train_idx]
    y_tr = y_train_split[train_idx]
    X_vl = X_train_scaled[val_idx]

    for m_idx, factory in enumerate(model_factories):
        model = factory()
        if m_idx == 0:  # LightGBM with early stopping
            callbacks = [
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=200),
            ]
            model.fit(X_tr, y_tr, eval_set=[(X_vl, y_train_split[val_idx])],
                      eval_metric="accuracy", callbacks=callbacks)
        elif m_idx == 1:  # XGBoost with early stopping (early_stopping_rounds in constructor)
            model.fit(X_tr, y_tr, eval_set=[(X_vl, y_train_split[val_idx])],
                      verbose=False)
        else:  # Logistic Regression or Extra Trees
            model.fit(X_tr, y_tr)
        oof_preds[val_idx, m_idx] = model.predict_proba(X_vl)[:, 1]

    print(f"Fold {fold+1} complete")

# ============================================================
# 8. STACKING META-LEARNER (Level 2)
# ============================================================
# Train Logistic Regression meta-learner on OOF predictions
meta_learner = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42)
meta_learner.fit(oof_preds, y_train_split)

# OOF ensemble prediction via meta-learner
oof_ensemble_proba = meta_learner.predict_proba(oof_preds)[:, 1]
oof_ensemble_pred = (oof_ensemble_proba >= 0.5).astype(int)

# ============================================================
# 9. VALIDATION (Official Metric: Accuracy)
# ============================================================
# Use OOF predictions from the training split as validation score
val_accuracy = accuracy_score(y_train_split, oof_ensemble_pred)
print(f"Validation Accuracy (5-fold OOF, meta-learner): {val_accuracy:.4f}")

# Also evaluate on held-out validation set for reference
val_level1_preds = np.zeros((len(X_val_scaled), len(model_factories)))
for m_idx, factory in enumerate(model_factories):
    if m_idx == 1:  # XGBoost without early stopping (no eval_set available)
        model = make_xgb_no_es()
    else:
        model = factory()
    model.fit(X_train_scaled, y_train_split)
    val_level1_preds[:, m_idx] = model.predict_proba(X_val_scaled)[:, 1]

val_ensemble_proba = meta_learner.predict_proba(val_level1_preds)[:, 1]
val_pred = (val_ensemble_proba >= 0.5).astype(int)
print(f"Held-out Validation Accuracy: {accuracy_score(y_val, val_pred):.4f}")

# ============================================================
# 10. TEST PREDICTIONS & SUBMISSION (retrain on ALL training data)
# ============================================================
scaler_full = StandardScaler()
X_train_full_scaled = scaler_full.fit_transform(X_train_full)
X_test_full_scaled = scaler_full.transform(X_test_full)

test_level1_preds = np.zeros((len(X_test_full_scaled), len(model_factories)))
for m_idx, factory in enumerate(model_factories):
    if m_idx == 1:  # XGBoost without early stopping (no eval_set available)
        model = make_xgb_no_es()
    else:
        model = factory()
    model.fit(X_train_full_scaled, y_train)
    test_level1_preds[:, m_idx] = model.predict_proba(X_test_full_scaled)[:, 1]

# Retrain meta-learner on full training data for consistency with base models
full_train_level1_preds = np.zeros((len(X_train_full_scaled), len(model_factories)))
for m_idx, factory in enumerate(model_factories):
    if m_idx == 1:  # XGBoost without early stopping
        model = make_xgb_no_es()
    else:
        model = factory()
    model.fit(X_train_full_scaled, y_train)
    full_train_level1_preds[:, m_idx] = model.predict_proba(X_train_full_scaled)[:, 1]

meta_learner_full = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42)
meta_learner_full.fit(full_train_level1_preds, y_train)

# Pass test predictions through retrained meta-learner
test_pred_proba = meta_learner_full.predict_proba(test_level1_preds)[:, 1]
test_pred = (test_pred_proba >= 0.5).astype(int)

os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_df)), "Survived": test_pred.astype(int)}
)
submission.to_csv("./submission/submission.csv", index=False)
print(
    f"Submission saved: {submission.shape[0]} rows, {submission['Survived'].mean():.3f} predicted survivors"
)

# ============================================================
# 11. FEATURE IMPORTANCE (diagnostic)
# ============================================================
lgbm_final = make_lgbm()
lgbm_final.fit(X_train_full_scaled, y_train)
fi = pd.Series(lgbm_final.feature_importances_, index=list(X_train_full.columns))
print(f"Top 5 features: {fi.nlargest(5).to_dict()}")

# ============================================================
# 12. FINAL OUTPUT
# ============================================================
score = val_accuracy
print(f"Final Validation Score: {score}")