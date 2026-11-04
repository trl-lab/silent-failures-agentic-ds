import os
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# Load dataset
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "AdoptionSpeed"
y_train = train_df[target_col].values
train_df_raw = train_df.drop(columns=[target_col])
test_df_raw = test_df.copy()

# Text vectorization via TF-IDF + SVD
full_desc = pd.concat(
    [
        train_df_raw["Description"].fillna(""),
        test_df_raw["Description"].fillna(""),
    ],
    axis=0,
)
tfidf = TfidfVectorizer(
    max_features=10000, stop_words="english", ngram_range=(1, 2), min_df=3
)
tfidf_vecs = tfidf.fit_transform(full_desc)

svd = TruncatedSVD(n_components=32, random_state=42)
svd_feats = svd.fit_transform(tfidf_vecs)

svd_cols = [f"svd_desc_{i}" for i in range(32)]
svd_df = pd.DataFrame(svd_feats, columns=svd_cols)

# Combine datasets for unified tabular feature engineering (unsupervised feature transforms)
combined_df = pd.concat([train_df_raw, test_df_raw], axis=0).reset_index(drop=True)

# Text length and presence features
combined_df["Desc_Len"] = combined_df["Description"].fillna("").apply(len)
combined_df["Desc_Word_Count"] = (
    combined_df["Description"].fillna("").apply(lambda x: len(x.split()))
)
combined_df["Name_Len"] = combined_df["Name"].fillna("").apply(len)
combined_df["Has_Name"] = (
    ~combined_df["Name"].fillna("").str.lower().isin(["", "no name", "unnamed"])
).astype(int)

# Domain interaction features
combined_df["Fee_Per_Pet"] = combined_df["Fee"] / (combined_df["Quantity"] + 1e-5)
combined_df["Is_Free"] = (combined_df["Fee"] == 0).astype(int)
combined_df["Has_Photo"] = (combined_df["PhotoAmt"] > 0).astype(int)
combined_df["Has_Video"] = (combined_df["VideoAmt"] > 0).astype(int)
combined_df["Is_Purebred"] = (
    (combined_df["Breed2"] == 0) | (combined_df["Breed1"] == combined_df["Breed2"])
).astype(int)

# Age categorization features
combined_df["Age_In_Years"] = combined_df["Age"] / 12.0
combined_df["Is_Baby"] = (combined_df["Age"] <= 3).astype(int)
combined_df["Is_Young"] = (
    (combined_df["Age"] > 3) & (combined_df["Age"] <= 12)
).astype(int)
combined_df["Is_Senior"] = (combined_df["Age"] >= 60).astype(int)

# Color features
combined_df["Num_Colors"] = (
    (combined_df["Color1"] > 0).astype(int)
    + (combined_df["Color2"] > 0).astype(int)
    + (combined_df["Color3"] > 0).astype(int)
)

# Combined health score
combined_df["Health_Score"] = (
    (combined_df["Vaccinated"] == 1).astype(int)
    + (combined_df["Dewormed"] == 1).astype(int)
    + (combined_df["Sterilized"] == 1).astype(int)
)

# Frequency encoding for high-cardinality categorical variables
for col in ["Breed1", "Breed2", "State", "RescuerID"]:
    freq = combined_df[col].value_counts().to_dict()
    combined_df[f"{col}_Freq"] = combined_df[col].map(freq)

# Group aggregations for key numerical features by Breed1 and State
for group_col in ["Breed1", "State"]:
    for num_col in ["Age", "Fee"]:
        group_mean = combined_df.groupby(group_col)[num_col].transform("mean")
        group_std = combined_df.groupby(group_col)[num_col].transform("std")
        combined_df[f"{num_col}_mean_by_{group_col}"] = group_mean
        combined_df[f"{num_col}_std_by_{group_col}"] = group_std.fillna(0)

# Drop non-feature identifiers
features_df = combined_df.drop(columns=["Description", "Name", "PetID", "RescuerID"])

# Categorical column handling
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
for col in cat_cols:
    if col in features_df.columns:
        features_df[col] = features_df[col].astype("category")

# Concatenate TF-IDF SVD features
features_df = pd.concat([features_df, svd_df], axis=1)

# Split train and test feature sets
X_train = features_df.iloc[: len(train_df_raw)].reset_index(drop=True)
X_test = features_df.iloc[len(train_df_raw) :].reset_index(drop=True)


# Post-processing Optimizer for QWK with strictly sorted thresholds
class OptimizedRounder:

    def __init__(self):
        self.coef_ = [0.5, 1.5, 2.5, 3.5]

    def _loss(self, coef, X, y):
        coef_sorted = np.sort(coef)
        X_p = np.digitize(X, coef_sorted)
        return -cohen_kappa_score(y, X_p, weights="quadratic")

    def fit(self, X, y):
        res = minimize(self._loss, self.coef_, args=(X, y), method="Nelder-Mead")
        self.coef_ = np.sort(res.x)

    def predict(self, X):
        return np.digitize(X, np.sort(self.coef_))


# Cross-Validation setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(y_train))
test_preds = np.zeros(len(X_test))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "random_state": 42,
    "verbose": -1,
    "n_estimators": 1000,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    oof_preds[val_idx] = model.predict(X_va)
    test_preds += model.predict(X_test) / skf.n_splits

# Threshold optimization on out-of-fold continuous predictions
opt = OptimizedRounder()
opt.fit(oof_preds, y_train)

oof_discrete = opt.predict(oof_preds)
cv_qwk = cohen_kappa_score(y_train, oof_discrete, weights="quadratic")

print(f"Overall 5-Fold Cross-Validation QWK: {cv_qwk:.5f}")

# Predict discrete classes on test set using optimal threshold cutoffs
test_discrete = opt.predict(test_preds)

# Export submission file
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(X_test)), "AdoptionSpeed": test_discrete}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission successfully saved to ./working/submission.csv")
