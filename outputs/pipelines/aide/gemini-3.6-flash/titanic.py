import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Store target and lengths
y = train_df["Survived"].values.astype(int)
n_train = len(train_df)
n_test = len(test_df)

# Combine for unified feature engineering
df = pd.concat(
    [train_df.drop(columns=["Survived"], errors="ignore"), test_df],
    ignore_index=True,
)

# 1. Title Extraction
df["Title"] = df["Name"].str.extract(r",\s*([A-Za-z]+)\.", expand=False)
title_mapping = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Mrs": "Mrs",
    "Master": "Master",
    "Mlle": "Miss",
    "MS": "Miss",
    "Mme": "Mrs",
    "Dr": "Officer",
    "Rev": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Capt": "Officer",
    "Don": "Royalty",
    "Sir": "Royalty",
    "Lady": "Royalty",
    "Countess": "Royalty",
    "Jonkheer": "Royalty",
    "Dona": "Royalty",
}
df["Title"] = df["Title"].map(title_mapping).fillna("Other")

# 2. Deck & Cabin Features
df["Deck"] = df["Cabin"].apply(lambda x: str(x)[0].upper() if pd.notnull(x) else "U")
df["Cabin_Count"] = df["Cabin"].apply(
    lambda x: len(str(x).split()) if pd.notnull(x) else 0
)
df["Has_Cabin"] = df["Cabin"].notnull().astype(int)

# 3. Family & Group Size Features
df["Family_Size"] = df["SibSp"] + df["Parch"] + 1
df["Is_Alone"] = (df["Family_Size"] == 1).astype(int)
df["Ticket_Freq"] = df.groupby("Ticket")["Ticket"].transform("count")
df["Group_Size"] = df[["Family_Size", "Ticket_Freq"]].max(axis=1)

# 4. Fare & Embarked Imputation & Transformations
df["Fare"] = df["Fare"].fillna(df.groupby("Pclass")["Fare"].transform("median"))
df["Fare_Per_Person"] = df["Fare"] / df["Group_Size"]
df["Log_Fare_Per_Person"] = np.log1p(df["Fare_Per_Person"])
df["Embarked"] = df["Embarked"].fillna("S")

# 5. Age Imputation
median_ages = df.groupby(["Title", "Pclass"])["Age"].transform("median")
df["Age"] = df["Age"].fillna(median_ages).fillna(df["Age"].median())

# 6. Domain Indicator Features
df["Is_Child"] = ((df["Age"] < 12) | (df["Title"] == "Master")).astype(int)
df["Is_Mother"] = (
    (df["Sex"] == "female")
    & (df["Parch"] > 0)
    & (df["Age"] > 18)
    & (df["Title"] != "Miss")
).astype(int)
df["Is_WCG"] = ((df["Sex"] == "female") | (df["Is_Child"] == 1)).astype(int)

# 7. Group Identifier for Target Encoding
df["Surname"] = df["Name"].apply(lambda x: str(x).split(",")[0].strip())
df["Group_ID"] = np.where(
    df["Ticket_Freq"] > 1,
    "T_" + df["Ticket"].astype(str),
    np.where(
        df["Family_Size"] > 1,
        "F_"
        + df["Surname"]
        + "_"
        + df["Pclass"].astype(str)
        + "_"
        + df["Fare"].astype(str),
        "A_" + df.index.astype(str),
    ),
)

# Convert categorical columns to category and numeric code types
cat_cols = ["Sex", "Embarked", "Title", "Deck", "Pclass"]
for c in cat_cols:
    df[c] = df[c].astype("category")
    df[c + "_code"] = df[c].cat.codes

cat_cols_num = [c + "_code" for c in cat_cols]

num_cols = [
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Family_Size",
    "Is_Alone",
    "Ticket_Freq",
    "Group_Size",
    "Fare_Per_Person",
    "Log_Fare_Per_Person",
    "Cabin_Count",
    "Has_Cabin",
    "Is_Child",
    "Is_Mother",
    "Is_WCG",
]

# Split back to train and test sets
X_train_full = df.iloc[:n_train].copy()
X_test_full = df.iloc[n_train:].copy()

model_names = ["CatBoost", "LightGBM", "XGBoost", "RandomForest", "ExtraTrees"]
n_models = len(model_names)
seeds = [42, 100, 2024]
n_seeds = len(seeds)

oof_preds_matrix = np.zeros((n_train, n_models))
test_preds_matrix = np.zeros((n_test, n_models))

for seed_idx, seed in enumerate(seeds):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y)):
        tr_df = X_train_full.iloc[train_idx].copy()
        val_df = X_train_full.iloc[val_idx].copy()
        te_df = X_test_full.copy()
        tr_y = y[train_idx]

        # Stratified Priors calculated strictly on training fold
        prior_map = (
            tr_df.assign(y=tr_y)
            .groupby(["Pclass", "Sex"], observed=False)["y"]
            .mean()
            .to_dict()
        )
        overall_prior = float(np.mean(tr_y))

        # Overall Group statistics
        group_sums = (
            tr_df.assign(y=tr_y)
            .groupby("Group_ID", observed=False)["y"]
            .sum()
            .to_dict()
        )
        group_counts = tr_df.groupby("Group_ID", observed=False).size().to_dict()

        # Woman-Child Group (WCG) statistics
        wcg_tr_df = tr_df[tr_df["Is_WCG"] == 1].assign(y=tr_y[tr_df["Is_WCG"] == 1])
        wcg_group_sums = (
            wcg_tr_df.groupby("Group_ID", observed=False)["y"].sum().to_dict()
        )
        wcg_group_counts = (
            wcg_tr_df.groupby("Group_ID", observed=False).size().to_dict()
        )

        # Calculate train fold group survival (excluding self) & explicitly mark fallback priors
        tr_surv, tr_wcg_surv = [], []
        tr_has_grp, tr_has_wcg = [], []

        for g_id, y_val, pc, s, wcg_flag in zip(
            tr_df["Group_ID"], tr_y, tr_df["Pclass"], tr_df["Sex"], tr_df["Is_WCG"]
        ):
            cnt = group_counts.get(g_id, 0)
            if cnt > 1:
                tr_surv.append((group_sums[g_id] - y_val) / (cnt - 1))
                tr_has_grp.append(1)
            else:
                tr_surv.append(prior_map.get((pc, s), overall_prior))
                tr_has_grp.append(0)

            wcg_cnt = wcg_group_counts.get(g_id, 0)
            wcg_sum = wcg_group_sums.get(g_id, 0)
            if wcg_flag == 1:
                if wcg_cnt > 1:
                    tr_wcg_surv.append((wcg_sum - y_val) / (wcg_cnt - 1))
                    tr_has_wcg.append(1)
                else:
                    tr_wcg_surv.append(prior_map.get((pc, s), overall_prior))
                    tr_has_wcg.append(0)
            else:
                if wcg_cnt > 0:
                    tr_wcg_surv.append(wcg_sum / wcg_cnt)
                    tr_has_wcg.append(1)
                else:
                    tr_wcg_surv.append(prior_map.get((pc, s), overall_prior))
                    tr_has_wcg.append(0)

        tr_df["Group_Survival"] = tr_surv
        tr_df["WCG_Group_Survival"] = tr_wcg_surv
        tr_df["Has_Group_Info"] = tr_has_grp
        tr_df["Has_WCG_Info"] = tr_has_wcg

        # Calculate validation and test group survival metrics & indicator flags
        for target_df in [val_df, te_df]:
            v_surv, v_wcg_surv = [], []
            v_has_grp, v_has_wcg = [], []

            for g_id, pc, s in zip(
                target_df["Group_ID"], target_df["Pclass"], target_df["Sex"]
            ):
                cnt = group_counts.get(g_id, 0)
                if cnt > 0:
                    v_surv.append(group_sums[g_id] / cnt)
                    v_has_grp.append(1)
                else:
                    v_surv.append(prior_map.get((pc, s), overall_prior))
                    v_has_grp.append(0)

                wcg_cnt = wcg_group_counts.get(g_id, 0)
                wcg_sum = wcg_group_sums.get(g_id, 0)
                if wcg_cnt > 0:
                    v_wcg_surv.append(wcg_sum / wcg_cnt)
                    v_has_wcg.append(1)
                else:
                    v_wcg_surv.append(prior_map.get((pc, s), overall_prior))
                    v_has_wcg.append(0)

            target_df["Group_Survival"] = v_surv
            target_df["WCG_Group_Survival"] = v_wcg_surv
            target_df["Has_Group_Info"] = v_has_grp
            target_df["Has_WCG_Info"] = v_has_wcg

        feature_cols_cat = (
            cat_cols
            + num_cols
            + [
                "Group_Survival",
                "WCG_Group_Survival",
                "Has_Group_Info",
                "Has_WCG_Info",
            ]
        )
        feature_cols_num = (
            cat_cols_num
            + num_cols
            + [
                "Group_Survival",
                "WCG_Group_Survival",
                "Has_Group_Info",
                "Has_WCG_Info",
            ]
        )

        # 1. CatBoost
        m_cat = CatBoostClassifier(
            iterations=600,
            learning_rate=0.03,
            depth=4,
            eval_metric="Accuracy",
            cat_features=cat_cols,
            random_seed=seed + fold,
            verbose=0,
        )
        m_cat.fit(
            tr_df[feature_cols_cat],
            tr_y,
            eval_set=(val_df[feature_cols_cat], y[val_idx]),
            early_stopping_rounds=50,
            verbose=False,
        )

        # 2. LightGBM
        m_lgb = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            random_state=seed + fold,
            verbose=-1,
        )
        m_lgb.fit(
            tr_df[feature_cols_cat],
            tr_y,
            eval_set=[(val_df[feature_cols_cat], y[val_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        # 3. XGBoost
        m_xgb = XGBClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=4,
            random_state=seed + fold,
            enable_categorical=True,
            early_stopping_rounds=50,
            eval_metric="logloss",
        )
        m_xgb.fit(
            tr_df[feature_cols_cat],
            tr_y,
            eval_set=[(val_df[feature_cols_cat], y[val_idx])],
            verbose=False,
        )

        # 4. Random Forest
        m_rf = RandomForestClassifier(
            n_estimators=150, max_depth=6, random_state=seed + fold
        )
        m_rf.fit(tr_df[feature_cols_num], tr_y)

        # 5. Extra Trees
        m_et = ExtraTreesClassifier(
            n_estimators=150, max_depth=6, random_state=seed + fold
        )
        m_et.fit(tr_df[feature_cols_num], tr_y)

        # Predict OOF
        oof_preds_matrix[val_idx, 0] += (
            m_cat.predict_proba(val_df[feature_cols_cat])[:, 1] / n_seeds
        )
        oof_preds_matrix[val_idx, 1] += (
            m_lgb.predict_proba(val_df[feature_cols_cat])[:, 1] / n_seeds
        )
        oof_preds_matrix[val_idx, 2] += (
            m_xgb.predict_proba(val_df[feature_cols_cat])[:, 1] / n_seeds
        )
        oof_preds_matrix[val_idx, 3] += (
            m_rf.predict_proba(val_df[feature_cols_num])[:, 1] / n_seeds
        )
        oof_preds_matrix[val_idx, 4] += (
            m_et.predict_proba(val_df[feature_cols_num])[:, 1] / n_seeds
        )

        # Predict Test
        test_preds_matrix[:, 0] += m_cat.predict_proba(te_df[feature_cols_cat])[
            :, 1
        ] / (n_seeds * skf.n_splits)
        test_preds_matrix[:, 1] += m_lgb.predict_proba(te_df[feature_cols_cat])[
            :, 1
        ] / (n_seeds * skf.n_splits)
        test_preds_matrix[:, 2] += m_xgb.predict_proba(te_df[feature_cols_cat])[
            :, 1
        ] / (n_seeds * skf.n_splits)
        test_preds_matrix[:, 3] += m_rf.predict_proba(te_df[feature_cols_num])[:, 1] / (
            n_seeds * skf.n_splits
        )
        test_preds_matrix[:, 4] += m_et.predict_proba(te_df[feature_cols_num])[:, 1] / (
            n_seeds * skf.n_splits
        )


# Optimize Ensemble Weights on Multi-Seed OOF Predictions
def log_loss_objective(weights):
    weights = np.array(weights)
    weights = np.maximum(weights, 0)
    w_sum = np.sum(weights)
    if w_sum > 0:
        weights /= w_sum
    p_blend = np.dot(oof_preds_matrix, weights)
    eps = 1e-15
    p_blend = np.clip(p_blend, eps, 1 - eps)
    return -np.mean(y * np.log(p_blend) + (1 - y) * np.log(1 - p_blend))


init_weights = [1.0 / n_models] * n_models
bounds = [(0, 1)] * n_models
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

opt_res = minimize(
    log_loss_objective,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
best_weights = opt_res.x / np.sum(opt_res.x)

blended_oof = np.dot(oof_preds_matrix, best_weights)
blended_test = np.dot(test_preds_matrix, best_weights)

# Optimize Classification Decision Threshold
thresholds = np.linspace(0.35, 0.65, 301)
best_acc = 0.0
best_thresh = 0.5

for t in thresholds:
    acc = accuracy_score(y, (blended_oof >= t).astype(int))
    if acc > best_acc:
        best_acc = acc
        best_thresh = t

print(f"Validation Metric (Accuracy): {best_acc:.5f}")

# Save Test Predictions
submission = pd.DataFrame(
    {"id": np.arange(n_test), "Survived": (blended_test >= best_thresh).astype(int)}
)
submission.to_csv("./working/submission.csv", index=False)
