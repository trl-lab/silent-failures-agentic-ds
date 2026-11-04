import os
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

# Ensure required directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# 1. Load Raw Datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")


# 2. String Cleaning & Raw Target Standardization
def clean_data(df, is_train=True):
    df = df.copy()
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for c in str_cols:
        df[c] = df[c].astype(str).str.strip()
        df[c] = df[c].replace(["?", "nan", "None", ""], "Unknown")

    if is_train and "income" in df.columns:
        df["income"] = (
            df["income"]
            .astype(str)
            .str.strip()
            .str.rstrip(".")
            .isin([">50K", ">50K."])
            .astype(int)
        )
    return df


train_df = clean_data(train_df, is_train=True)
test_df = clean_data(test_df, is_train=False)

# 3. Stratified CV Data Preparation
X_full = train_df.drop(columns=["income"])
y_full = train_df["income"].values
X_test_raw = test_df.copy()


# 4. Domain Feature Engineering
def create_domain_features(df):
    df = df.copy()

    # Net Capital and Log Financial Transformations
    df["net_capital"] = df["capital-gain"] - df["capital-loss"]
    df["has_capital_gain"] = (df["capital-gain"] > 0).astype(int)
    df["has_capital_loss"] = (df["capital-loss"] > 0).astype(int)
    df["log_capital_gain"] = np.log1p(np.maximum(0, df["capital-gain"]))
    df["log_capital_loss"] = np.log1p(np.maximum(0, df["capital-loss"]))
    df["log_net_capital"] = np.sign(df["net_capital"]) * np.log1p(
        np.abs(df["net_capital"])
    )

    # Age & Work Productivity Indicators
    df["working_age"] = np.maximum(0, df["age"] - 17)
    df["hours_per_working_year"] = df["hours-per-week"] * 52
    df["capital_per_hour"] = df["net_capital"] / (df["hours-per-week"] + 1.0)
    df["age_education_ratio"] = df["age"] / (df["education-num"] + 1.0)

    # Demographic & Marital Interactions
    df["is_married"] = (
        df["marital-status"]
        .isin(["Married-civ-spouse", "Married-AF-spouse"])
        .astype(int)
    )
    df["married_education"] = df["is_married"] * df["education-num"]
    df["married_hours"] = df["is_married"] * df["hours-per-week"]

    # Combinatorial Categoricals & Relationship Interactions
    df["marital_sex"] = df["marital-status"] + "_" + df["sex"]
    df["workclass_occupation"] = df["workclass"] + "_" + df["occupation"]
    df["education_occupation"] = df["education"] + "_" + df["occupation"]
    if "relationship" in df.columns:
        if "sex" in df.columns:
            df["relationship_sex"] = df["relationship"] + "_" + df["sex"]
        if "education" in df.columns:
            df["relationship_education"] = (
                df["relationship"] + "_" + df["education"]
            )

    # Fnlwgt Features
    if "fnlwgt" in df.columns:
        df["log_fnlwgt"] = np.log1p(np.maximum(0, df["fnlwgt"]))
        df["relative_fnlwgt"] = df["fnlwgt"] / (df["fnlwgt"].mean() + 1.0)
        df["fnlwgt_per_age"] = df["fnlwgt"] / (df["age"] + 1.0)

    return df


class AccuracyThresholdOptimizer:
    """Optimizes probability classification threshold for holdout Accuracy."""

    def __init__(self, step: float = 0.01):
        self.step = step
        self.best_threshold = 0.5
        self.best_score = 0.0

    def fit(self, y_true: np.ndarray, y_probs: np.ndarray) -> float:
        thresholds = np.arange(0.1, 0.9, self.step)
        best_acc = -1.0
        best_thresh = 0.5

        for thresh in thresholds:
            preds = (y_probs >= thresh).astype(int)
            acc = accuracy_score(y_true, preds)
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh

        self.best_threshold = float(best_thresh)
        self.best_score = float(best_acc)
        return self.best_threshold

    def predict(self, y_probs: np.ndarray) -> np.ndarray:
        return (y_probs >= self.best_threshold).astype(int)


# Apply Domain Feature Engineering
X_full_fe = create_domain_features(X_full)
X_test_fe = create_domain_features(X_test_raw)


def calc_target_encoding(
    train_df, val_df, test_df, cat_cols, target_col, smoothing=10.0
):
    train_encoded = train_df.copy()
    val_encoded = val_df.copy()
    test_encoded = test_df.copy()
    global_mean = train_df[target_col].mean()

    for col in cat_cols:
        if col not in train_df.columns:
            continue
        stats = train_df.groupby(col)[target_col].agg(["count", "mean"])
        smooth_te = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
            stats["count"] + smoothing
        )
        te_map = smooth_te.to_dict()

        train_encoded[f"{col}_te"] = (
            train_encoded[col].map(te_map).fillna(global_mean).astype(float)
        )
        val_encoded[f"{col}_te"] = (
            val_encoded[col].map(te_map).fillna(global_mean).astype(float)
        )
        test_encoded[f"{col}_te"] = (
            test_encoded[col].map(te_map).fillna(global_mean).astype(float)
        )

    return train_encoded, val_encoded, test_encoded


# Setup 5-Fold Stratified Cross Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X_full))
oof_xgb = np.zeros(len(X_full))
oof_cat = np.zeros(len(X_full))

test_lgb = np.zeros(len(X_test_fe))
test_xgb = np.zeros(len(X_test_fe))
test_cat = np.zeros(len(X_test_fe))

te_cols = [
    "occupation",
    "workclass",
    "education",
    "marital-status",
    "native-country",
    "relationship",
]
group_keys = ["occupation", "workclass", "education", "relationship"]
num_target_agg_cols = [
    "hours-per-week",
    "age",
    "net_capital",
    "education-num",
    "fnlwgt",
]

for fold, (train_idx, val_idx) in enumerate(skf.split(X_full_fe, y_full)):
    X_tr = X_full_fe.iloc[train_idx].copy()
    y_tr = y_full[train_idx]
    X_va = X_full_fe.iloc[val_idx].copy()
    y_va = y_full[val_idx]
    X_te = X_test_fe.copy()

    X_tr["income_target"] = y_tr

    # Group aggregations fitted strictly on train fold
    for key in group_keys:
        if key in X_tr.columns:
            agg_df = (
                X_tr.groupby(key)[num_target_agg_cols]
                .agg(["mean", "std"])
                .reset_index()
            )
            agg_df.columns = [key] + [
                f"{key}_{col}_{stat}" for col, stat in agg_df.columns[1:]
            ]
            X_tr = X_tr.merge(agg_df, on=key, how="left")
            X_va = X_va.merge(agg_df, on=key, how="left")
            X_te = X_te.merge(agg_df, on=key, how="left")

    # Target Encoding with smoothing strictly on train fold
    X_tr, X_va, X_te = calc_target_encoding(
        X_tr, X_va, X_te, te_cols, "income_target", smoothing=10.0
    )
    X_tr = X_tr.drop(columns=["income_target"])

    # Frequency and Ordinal Encodings
    cat_cols = X_tr.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    for c in cat_cols:
        freq_map = X_tr[c].value_counts(normalize=True).to_dict()
        X_tr[f"{c}_freq"] = X_tr[c].map(freq_map).fillna(0).astype(float)
        X_va[f"{c}_freq"] = X_va[c].map(freq_map).fillna(0).astype(float)
        X_te[f"{c}_freq"] = X_te[c].map(freq_map).fillna(0).astype(float)

        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        oe.fit(X_tr[[c]])
        X_tr[c] = oe.transform(X_tr[[c]]).astype(int).ravel()
        X_va[c] = oe.transform(X_va[[c]]).astype(int).ravel()
        X_te[c] = oe.transform(X_te[[c]]).astype(int).ravel()

    feature_cols = [col for col in X_tr.columns if col not in ["income", "id"]]
    X_tr_final = X_tr[feature_cols].copy()
    X_va_final = X_va[feature_cols].copy()
    X_te_final = X_te[feature_cols].copy()

    medians = X_tr_final.median()
    X_tr_final = X_tr_final.fillna(medians)
    X_va_final = X_va_final.fillna(medians)
    X_te_final = X_te_final.fillna(medians)

    # Train LightGBM Classifier with Native Categorical Handling
    lgb_model = lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        boosting_type="gbdt",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=8,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_tr_final,
        y_tr,
        eval_set=[(X_va_final, y_va)],
        eval_metric="binary_logloss",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    # Train XGBoost Classifier
    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42 + fold,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    xgb_model.fit(
        X_tr_final,
        y_tr,
        eval_set=[(X_va_final, y_va)],
        verbose=False,
    )

    # Train CatBoost Classifier with Native Categorical Handling
    cat_model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        eval_metric="Logloss",
        random_seed=42 + fold,
        verbose=0,
        early_stopping_rounds=50,
    )
    cat_model.fit(
        X_tr_final,
        y_tr,
        eval_set=(X_va_final, y_va),
        cat_features=cat_cols,
        use_best_model=True,
    )

    oof_lgb[val_idx] = lgb_model.predict_proba(X_va_final)[:, 1]
    oof_xgb[val_idx] = xgb_model.predict_proba(X_va_final)[:, 1]
    oof_cat[val_idx] = cat_model.predict_proba(X_va_final)[:, 1]

    test_lgb += lgb_model.predict_proba(X_te_final)[:, 1] / n_splits
    test_xgb += xgb_model.predict_proba(X_te_final)[:, 1] / n_splits
    test_cat += cat_model.predict_proba(X_te_final)[:, 1] / n_splits

# Optimize Out-Of-Fold Ensemble Weights via SLSQP
def loss_func(weights):
    weights = weights / np.sum(weights)
    blend_oof = (
        weights[0] * oof_lgb + weights[1] * oof_xgb + weights[2] * oof_cat
    )
    blend_oof = np.clip(blend_oof, 1e-15, 1 - 1e-15)
    return -np.mean(
        y_full * np.log(blend_oof) + (1 - y_full) * np.log(1 - blend_oof)
    )

init_weights = [1.0 / 3, 1.0 / 3, 1.0 / 3]
bounds = [(0, 1)] * 3
constraints = ({"type": "eq", "fun": lambda w: 1 - sum(w)},)

res = minimize(
    loss_func,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
)
best_weights = res.x / np.sum(res.x)
print(f"Optimal Ensemble Weights (LGB, XGB, CAT): {best_weights}")

oof_probs = (
    best_weights[0] * oof_lgb
    + best_weights[1] * oof_xgb
    + best_weights[2] * oof_cat
)
test_probs = (
    best_weights[0] * test_lgb
    + best_weights[1] * test_xgb
    + best_weights[2] * test_cat
)

# Threshold Optimization on Full Out-Of-Fold Probabilities
optimizer = AccuracyThresholdOptimizer(step=0.01)
best_thresh = optimizer.fit(y_full, oof_probs)
oof_preds = optimizer.predict(oof_probs)
val_score = accuracy_score(y_full, oof_preds)

# Test Inference & Format Final Submission
test_binary_preds = (test_probs >= best_thresh).astype(int)

raw_train_input = pd.read_parquet("./input/train.parquet")
raw_income_col = raw_train_input["income"]

if pd.api.types.is_numeric_dtype(raw_income_col):
    test_final_preds = test_binary_preds
else:
    unique_vals = raw_income_col.astype(str).str.strip().unique()
    gt_50k_str = next((v for v in unique_vals if ">50K" in v), ">50K")
    lte_50k_str = next((v for v in unique_vals if "<=50K" in v), "<=50K")
    test_final_preds = np.where(test_binary_preds == 1, gt_50k_str, lte_50k_str)

submission = pd.DataFrame({"id": np.arange(len(X_test_fe)), "income": test_final_preds})
submission.to_csv("./submission/submission.csv", index=False)

print(f"Final Validation Score: {val_score}")
