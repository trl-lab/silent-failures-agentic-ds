import os
import re
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from xgboost import XGBClassifier

# Safe import for CatBoost
try:
    from catboost import CatBoostClassifier

    has_catboost = True
except ImportError:
    has_catboost = False

# Ensure output directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify ID column
if "PassengerId" in train_df.columns:
    id_col = "PassengerId"
elif "id" in train_df.columns:
    id_col = "id"
else:
    train_df["id"] = np.arange(len(train_df))
    test_df["id"] = np.arange(len(test_df))
    id_col = "id"

# Extract target from train
target = train_df["Survived"].values


# 2. Feature Engineering Pipeline
def build_features(train, test):
    train_copy = train.copy()
    test_copy = test.copy()

    # Combine train and test (excluding target) for unsupervised manifest features
    full_df = pd.concat(
        [
            train_copy.drop(columns=["Survived"], errors="ignore"),
            test_copy.drop(columns=["Survived"], errors="ignore"),
        ],
        ignore_index=True,
    )

    # Title & Name Length Extraction
    def extract_title(name):
        if pd.isna(name):
            return "Mr"
        match = re.search(r" ([A-Za-z]+)\.", str(name))
        return match.group(1) if match else "Mr"

    full_df["Title_Raw"] = full_df["Name"].apply(extract_title)
    full_df["Name_Len"] = full_df["Name"].apply(lambda x: len(str(x)))

    title_mapping = {
        "Mr": "Mr",
        "Miss": "Miss",
        "Mrs": "Mrs",
        "Master": "Master",
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
        "Lady": "Noble",
        "Countess": "Noble",
        "Dona": "Noble",
        "Don": "Noble",
        "Sir": "Noble",
        "Jonkheer": "Noble",
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Dr": "Officer",
        "Rev": "Officer",
    }
    full_df["Title"] = full_df["Title_Raw"].map(title_mapping).fillna("Rare")

    # Family & Group Dynamics
    full_df["FamilySize"] = full_df["SibSp"] + full_df["Parch"] + 1
    full_df["IsAlone"] = (full_df["FamilySize"] == 1).astype(int)

    def get_family_type(size):
        if size == 1:
            return "Alone"
        elif 2 <= size <= 4:
            return "Small"
        else:
            return "Large"

    full_df["Family_Type"] = full_df["FamilySize"].apply(get_family_type)

    # Ticket Grouping (Unsupervised Group Count across manifest)
    ticket_counts = full_df["Ticket"].value_counts()
    full_df["Ticket_GroupSize"] = full_df["Ticket"].map(ticket_counts).fillna(1)

    # Surname Extraction & Passenger Group Identification
    full_df["Surname"] = full_df["Name"].apply(
        lambda x: str(x).split(",")[0].strip() if pd.notna(x) and "," in str(x) else ""
    )
    full_df["Group_Id"] = full_df["Surname"] + "_" + full_df["Ticket"].astype(str)

    # Cabin & Macro-Deck Features
    def extract_deck(cabin):
        if pd.isna(cabin) or str(cabin).strip() == "":
            return "U"
        deck = str(cabin).strip()[0].upper()
        if deck in ["A", "B", "C"]:
            return "ABC"
        elif deck in ["D", "E"]:
            return "DE"
        elif deck in ["F", "G"]:
            return "FG"
        else:
            return "U"

    full_df["Deck"] = full_df["Cabin"].apply(extract_deck)
    full_df["Has_Cabin"] = full_df["Cabin"].notna().astype(int)
    full_df["Cabin_Count"] = full_df["Cabin"].apply(
        lambda x: len(str(x).split()) if pd.notna(x) else 0
    )

    # Ticket Prefix
    def extract_ticket_prefix(ticket):
        t = str(ticket).replace(".", "").replace("/", "").strip().upper()
        parts = t.split()
        if len(parts) > 1:
            return parts[0]
        elif not parts[0].isdigit():
            return parts[0]
        else:
            return "NUMERIC"

    full_df["Ticket_Prefix"] = full_df["Ticket"].apply(extract_ticket_prefix)

    # Split back to Train and Test
    n_train = len(train_copy)
    train_feat = full_df.iloc[:n_train].copy()
    test_feat = full_df.iloc[n_train:].copy()
    train_feat["Survived"] = target

    # Stateful Imputations (Fitted STRICTLY on Train split to prevent leakage)
    title_pclass_age = train_feat.groupby(["Title", "Pclass"])["Age"].median().to_dict()
    global_age_median = train_feat["Age"].median()
    pclass_fare_median = train_feat.groupby("Pclass")["Fare"].median().to_dict()
    global_fare_median = train_feat["Fare"].median()
    embarked_mode = (
        train_feat["Embarked"].mode()[0]
        if not train_feat["Embarked"].mode().empty
        else "S"
    )

    for df in [train_feat, test_feat]:
        # Age Imputation
        age_map = df.set_index(["Title", "Pclass"]).index.map(title_pclass_age)
        df["Age"] = (
            df["Age"]
            .fillna(pd.Series(age_map, index=df.index))
            .fillna(global_age_median)
        )

        # Fare Imputation
        df["Fare"] = (
            df["Fare"]
            .fillna(df["Pclass"].map(pclass_fare_median))
            .fillna(global_fare_median)
        )
        df["Fare_Per_Person"] = df["Fare"] / df["Ticket_GroupSize"]
        df["Fare_Per_Family"] = df["Fare"] / df["FamilySize"]

        # Embarked Imputation
        df["Embarked"] = df["Embarked"].fillna(embarked_mode)

        # Domain & Demographic Interactions
        df["Is_Child"] = (df["Age"] < 12).astype(int)
        df["Is_Elderly"] = (df["Age"] >= 60).astype(int)
        df["Is_Female_Or_Child"] = ((df["Sex"] == "female") | (df["Age"] < 12)).astype(
            int
        )
        df["Age_Class"] = df["Age"] * df["Pclass"]
        df["Fare_Log"] = np.log1p(df["Fare"])

    # Categorical Label Encodings (Fit strictly on Train)
    cat_cols = ["Sex", "Embarked", "Title", "Deck", "Ticket_Prefix", "Family_Type"]
    for col in cat_cols:
        unique_vals = list(train_feat[col].astype(str).unique())
        val_map = {val: idx for idx, val in enumerate(unique_vals)}
        train_feat[col + "_Code"] = (
            train_feat[col].astype(str).map(val_map).fillna(-1).astype(int)
        )
        test_feat[col + "_Code"] = (
            test_feat[col].astype(str).map(val_map).fillna(-1).astype(int)
        )

    # Note: Target encoding removed from global preprocessing to prevent CV leakage

    return train_feat, test_feat


# Run feature processing
train_processed, test_processed = build_features(train_df, test_df)

# Feature selection for modeling (base features existing in processed DataFrames)
feature_cols = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "IsAlone",
    "Name_Len",
    "Ticket_GroupSize",
    "Fare_Per_Person",
    "Fare_Per_Family",
    "Has_Cabin",
    "Cabin_Count",
    "Is_Child",
    "Is_Elderly",
    "Is_Female_Or_Child",
    "Age_Class",
    "Fare_Log",
    "Sex_Code",
    "Embarked_Code",
    "Title_Code",
    "Deck_Code",
    "Ticket_Prefix_Code",
    "Family_Type_Code",
]

X_full = train_processed[feature_cols].copy()
y_full = train_processed["Survived"].values
X_test = test_processed[feature_cols].copy()

# 3. Repeated Stratified K-Fold Cross-Validation Setup
n_splits = 10
n_repeats = 2
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=42
)
total_folds = n_splits * n_repeats

model_names = ["lgb", "xgb", "hgb", "rf", "et", "lr"]
if has_catboost:
    model_names.append("cat")

oof_probs = {m: np.zeros(len(X_full)) for m in model_names}
test_probs = {m: np.zeros(len(X_test)) for m in model_names}

# 4. Cross-Validation Training Loop
te_cols = ["Title", "Pclass", "Deck", "Ticket_Prefix", "Family_Type"]
smoothing = 10.0

for fold, (train_idx, val_idx) in enumerate(rskf.split(X_full, y_full)):
    X_train_f = X_full.iloc[train_idx].copy()
    y_train_f = y_full[train_idx]
    X_val_f = X_full.iloc[val_idx].copy()
    y_val_f = y_full[val_idx]
    X_test_f = X_test.copy()

    global_target_mean = y_train_f.mean()
    for col in te_cols:
        stats = train_processed.iloc[train_idx].groupby(col)["Survived"].agg(["count", "mean"])
        smooth = (stats["count"] * stats["mean"] + smoothing * global_target_mean) / (stats["count"] + smoothing)
        X_train_f[col + "_TE"] = train_processed.iloc[train_idx][col].map(smooth).fillna(global_target_mean).values
        X_val_f[col + "_TE"] = train_processed.iloc[val_idx][col].map(smooth).fillna(global_target_mean).values
        X_test_f[col + "_TE"] = test_processed[col].map(smooth).fillna(global_target_mean).values

    # OOF Group Survival Rate & Is_Group_Survived Features (WCG Logic)
    train_fold_df = train_processed.iloc[train_idx]
    val_fold_df = train_processed.iloc[val_idx]
    test_fold_df = test_processed

    # Filter train fold strictly for Woman-Child passengers to calculate WCG group stats
    wcg_train_fold = train_fold_df[train_fold_df["Is_Female_Or_Child"] == 1]
    wcg_stats = wcg_train_fold.groupby("Group_Id")["Survived"].agg(["sum", "count"])

    # Train fold (excluding own outcome for WCG passengers)
    is_wcg_train = (train_fold_df["Is_Female_Or_Child"] == 1).values.astype(int)
    train_grp_sum = train_fold_df["Group_Id"].map(wcg_stats["sum"]).fillna(0).values
    train_grp_cnt = train_fold_df["Group_Id"].map(wcg_stats["count"]).fillna(0).values

    train_other_cnt = train_grp_cnt - is_wcg_train
    train_other_sum = train_grp_sum - (y_train_f * is_wcg_train)

    train_surv_rate = np.divide(
        train_other_sum,
        train_other_cnt,
        out=np.full(len(train_fold_df), 0.5, dtype=float),
        where=train_other_cnt > 0,
    )
    train_is_survived = np.where(
        train_other_cnt > 0, (train_surv_rate > 0.5).astype(float), 0.5
    )

    X_train_f["Group_Survival_Rate"] = train_surv_rate
    X_train_f["Is_Group_Survived"] = train_is_survived

    # Validation fold
    val_grp_sum = val_fold_df["Group_Id"].map(wcg_stats["sum"]).fillna(0).values
    val_grp_cnt = val_fold_df["Group_Id"].map(wcg_stats["count"]).fillna(0).values

    val_surv_rate = np.divide(
        val_grp_sum,
        val_grp_cnt,
        out=np.full(len(val_fold_df), 0.5, dtype=float),
        where=val_grp_cnt > 0,
    )
    val_is_survived = np.where(
        val_grp_cnt > 0, (val_surv_rate > 0.5).astype(float), 0.5
    )

    X_val_f["Group_Survival_Rate"] = val_surv_rate
    X_val_f["Is_Group_Survived"] = val_is_survived

    # Test fold
    test_grp_sum = test_fold_df["Group_Id"].map(wcg_stats["sum"]).fillna(0).values
    test_grp_cnt = test_fold_df["Group_Id"].map(wcg_stats["count"]).fillna(0).values

    test_surv_rate = np.divide(
        test_grp_sum,
        test_grp_cnt,
        out=np.full(len(test_fold_df), 0.5, dtype=float),
        where=test_grp_cnt > 0,
    )
    test_is_survived = np.where(
        test_grp_cnt > 0, (test_surv_rate > 0.5).astype(float), 0.5
    )

    X_test_f["Group_Survival_Rate"] = test_surv_rate
    X_test_f["Is_Group_Survived"] = test_is_survived

    models_fold = {
        "lgb": lgb.LGBMClassifier(
            n_estimators=350,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            min_child_samples=15,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42 + fold,
            verbose=-1,
        ),
        "xgb": XGBClassifier(
            n_estimators=350,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42 + fold,
            eval_metric="logloss",
            early_stopping_rounds=35,
        ),
        "hgb": HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.03,
            max_depth=4,
            min_samples_leaf=15,
            l2_regularization=1.0,
            random_state=42 + fold,
        ),
        "rf": RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42 + fold,
        ),
        "et": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42 + fold,
        ),
        "lr": CalibratedClassifierCV(
            make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(C=0.5, max_iter=1000, random_state=42 + fold),
            ),
            cv=3,
        ),
    }

    if has_catboost:
        models_fold["cat"] = CatBoostClassifier(
            iterations=350,
            learning_rate=0.03,
            depth=4,
            l2_leaf_reg=3.0,
            random_seed=42 + fold,
            verbose=0,
            early_stopping_rounds=35,
        )

    for name, m in models_fold.items():
        if name == "lgb":
            m.fit(
                X_train_f,
                y_train_f,
                eval_set=[(X_val_f, y_val_f)],
                callbacks=[
                    lgb.early_stopping(35, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
        elif name == "xgb":
            m.fit(X_train_f, y_train_f, eval_set=[(X_val_f, y_val_f)], verbose=False)
        elif name == "cat":
            m.fit(X_train_f, y_train_f, eval_set=(X_val_f, y_val_f), verbose=False)
        else:
            m.fit(X_train_f, y_train_f)

        oof_probs[name][val_idx] += m.predict_proba(X_val_f)[:, 1] / n_repeats
        test_probs[name] += m.predict_proba(X_test_f)[:, 1] / total_folds


# 5. Performance-Weighted Probability Blending
model_accs = {
    m: accuracy_score(y_full, (oof_probs[m] >= 0.5).astype(int))
    for m in model_names
}
total_weight = sum(model_accs.values())

oof_ensemble = sum(oof_probs[m] * model_accs[m] for m in model_names) / total_weight
test_ensemble = sum(test_probs[m] * model_accs[m] for m in model_names) / total_weight

# 6. Threshold Calibration on Full OOF Predictions
best_threshold = 0.5
best_oof_score = 0.0

thresholds = np.linspace(0.3, 0.7, 401)
for t in thresholds:
    score_t = accuracy_score(y_full, (oof_ensemble >= t).astype(int))
    if score_t > best_oof_score:
        best_oof_score = score_t
        best_threshold = t

final_oof_preds = (oof_ensemble >= best_threshold).astype(int)
score = accuracy_score(y_full, final_oof_preds)

# 7. Test Inference & Submission Generation
final_test_preds = (test_ensemble >= best_threshold).astype(int)

submission_df = pd.DataFrame(
    {"id": np.arange(len(final_test_preds)), "Survived": final_test_preds}
)
submission_df.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {score}")