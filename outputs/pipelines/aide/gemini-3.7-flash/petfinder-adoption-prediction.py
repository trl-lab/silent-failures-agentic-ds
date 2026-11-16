import os
import re
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")


# 2. Feature Engineering
def feature_engineering(df, is_train=True, tfidf=None, svd=None, rescuer_counts=None):
    data = df.copy()

    # Text length and basic features
    data["Desc_len"] = data["Description"].fillna("").apply(len)
    data["Desc_word_cnt"] = (
        data["Description"].fillna("").apply(lambda x: len(x.split()))
    )
    data["Has_Name"] = data["Name"].notna().astype(int)
    data["Name_len"] = data["Name"].fillna("").apply(len)

    # Domain specific interaction features
    data["Fee_per_pet"] = data["Fee"] / np.maximum(data["Quantity"], 1)
    data["Is_Free"] = (data["Fee"] == 0).astype(int)
    data["Is_Purebred"] = (
        (data["Breed2"] == 0) | (data["Breed1"] == data["Breed2"])
    ).astype(int)
    data["Color_Count"] = (
        (data["Color1"] > 0).astype(int)
        + (data["Color2"] > 0).astype(int)
        + (data["Color3"] > 0).astype(int)
    )
    data["Age_in_years"] = data["Age"] / 12.0

    # Rescuer count encoding
    if is_train:
        rescuer_counts = data["RescuerID"].value_counts()
    data["Rescuer_Count"] = data["RescuerID"].map(rescuer_counts).fillna(1)

    # TF-IDF + SVD on Description
    clean_desc = (
        data["Description"]
        .fillna("")
        .apply(lambda x: re.sub(r"[^a-zA-Z0-9\s]", " ", x.lower()))
    )
    if is_train:
        tfidf = TfidfVectorizer(
            max_features=5000, stop_words="english", ngram_range=(1, 2)
        )
        svd = TruncatedSVD(n_components=16, random_state=42)
        tfidf_mat = tfidf.fit_transform(clean_desc)
        svd_feats = svd.fit_transform(tfidf_mat)
    else:
        tfidf_mat = tfidf.transform(clean_desc)
        svd_feats = svd.transform(tfidf_mat)

    for i in range(16):
        data[f"svd_desc_{i}"] = svd_feats[:, i]

    # Categorical / numeric feature selection
    drop_cols = ["PetID", "Name", "Description", "RescuerID"]
    if "AdoptionSpeed" in data.columns:
        drop_cols.append("AdoptionSpeed")

    features = [col for col in data.columns if col not in drop_cols]

    cat_cols = [
        "Type",
        "Breed1",
        "Breed2",
        "Gender",
        "Color1",
        "Color2",
        "Color3",
        "MaturitySize",
        "FurLength",
        "Vaccinated",
        "Dewormed",
        "Sterilized",
        "Health",
        "State",
    ]
    for c in cat_cols:
        if c in data.columns:
            data[c] = data[c].astype("category")

    return data[features], tfidf, svd, rescuer_counts


X_train, tfidf, svd, rescuer_counts = feature_engineering(train_df, is_train=True)
y_train = train_df["AdoptionSpeed"].values
X_test, _, _, _ = feature_engineering(
    test_df, is_train=False, tfidf=tfidf, svd=svd, rescuer_counts=rescuer_counts
)

# 3. Stratified 5-Fold LightGBM Regressor
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 45,
    "max_depth": -1,
    "min_child_samples": 25,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)

    model = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=1500,
        valid_sets=[trn_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    oof_preds[val_idx] = model.predict(X_va, num_iteration=model.best_iteration)
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_splits


# 4. Threshold Optimization for Quadratic Weighted Kappa (QWK)
class OptimizedRounder:

    def __init__(self):
        self.coef_ = np.array([0.5, 1.5, 2.5, 3.5])

    def _loss(self, coef, X, y):
        coef_sorted = np.sort(coef)
        X_p = np.clip(np.digitize(X, coef_sorted), 0, 4)
        return -cohen_kappa_score(y, X_p, weights="quadratic")

    def fit(self, X, y):
        res = minimize(
            self._loss,
            self.coef_,
            args=(X, y),
            method="Nelder-Mead",
            options={"maxiter": 500},
        )
        self.coef_ = np.sort(res.x)

    def predict(self, X):
        return np.clip(np.digitize(X, self.coef_), 0, 4)


opt_rounder = OptimizedRounder()
opt_rounder.fit(oof_preds, y_train)

oof_binned = opt_rounder.predict(oof_preds)
oof_qwk = cohen_kappa_score(y_train, oof_binned, weights="quadratic")
print(f"Optimized Thresholds: {opt_rounder.coef_}")
print(f"Out-of-Fold Quadratic Weighted Kappa (QWK): {oof_qwk:.5f}")

# 5. Generate Submission
test_binned = opt_rounder.predict(test_preds)
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_binned)), "AdoptionSpeed": test_binned.astype(int)}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully with shape:", submission.shape)
