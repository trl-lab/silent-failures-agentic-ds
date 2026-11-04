import itertools
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Set random seed for reproducibility
SEED = 42
SHIFT = 200.0
FAIR_C = 0.7

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

test_ids = test_df["id"].values
y = train_df["loss"].values
y_log = np.log(y + SHIFT)

# Separate features
cat_cols = [c for c in train_df.columns if c.startswith("cat")]
cont_cols = [c for c in train_df.columns if c.startswith("cont")]

# Combine for consistent categorical encoding and feature engineering
combined_df = pd.concat(
    [train_df[cat_cols + cont_cols], test_df[cat_cols + cont_cols]], axis=0
).reset_index(drop=True)

# 2. Feature Engineering
# Continuous aggregated features
combined_df["cont_mean"] = combined_df[cont_cols].mean(axis=1)
combined_df["cont_std"] = combined_df[cont_cols].std(axis=1)
combined_df["cont_min"] = combined_df[cont_cols].min(axis=1)
combined_df["cont_max"] = combined_df[cont_cols].max(axis=1)
combined_df["cont_skew"] = combined_df[cont_cols].skew(axis=1)

# Continuous interactions
combined_df["cont1_cont9_diff"] = combined_df["cont1"] - combined_df["cont9"]
combined_df["cont1_cont10_diff"] = combined_df["cont1"] - combined_df["cont10"]
combined_df["cont6_cont7_diff"] = combined_df["cont6"] - combined_df["cont7"]

# Pairwise categorical interactions on high-importance categorical features
top_cats = [
    "cat80",
    "cat87",
    "cat79",
    "cat12",
    "cat101",
    "cat111",
    "cat57",
    "cat103",
    "cat1",
    "cat114",
]
interaction_cols = []
for c1, c2 in itertools.combinations(top_cats, 2):
    col_name = f"{c1}_{c2}"
    combined_df[col_name] = (
        combined_df[c1].astype(str) + "_" + combined_df[c2].astype(str)
    )
    interaction_cols.append(col_name)

all_cat_cols = cat_cols + interaction_cols

# Encode categorical variables: Label Encoding and Frequency Encoding
for col in all_cat_cols:
    # Frequency encoding
    freq = combined_df[col].value_counts(normalize=True).to_dict()
    combined_df[f"{col}_freq"] = combined_df[col].map(freq).astype(np.float32)

    # Label encoding
    le = LabelEncoder()
    combined_df[col] = le.fit_transform(combined_df[col].astype(str))

n_train = len(train_df)
X_train = combined_df.iloc[:n_train].copy()
X_test = combined_df.iloc[n_train:].copy()


# 3. Custom Fair Loss Objective and MAE Metric
def fair_obj(preds, dtrain):
    labels = dtrain.get_label()
    x = preds - labels
    c = FAIR_C
    den = np.abs(x) + c
    grad = c * x / den
    hess = (c**2) / (den**2)
    return grad, hess


def eval_mae(preds, dtrain):
    labels = dtrain.get_label()
    loss_pred = np.exp(preds) - SHIFT
    loss_true = np.exp(labels) - SHIFT
    mae = np.mean(np.abs(loss_pred - loss_true))
    return "mae", float(mae)


# 4. Model Training with 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof_preds = np.zeros(n_train)
test_preds = np.zeros(len(test_df))

xgb_params = {
    "tree_method": "hist",
    "learning_rate": 0.03,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 100,
    "random_state": SEED,
    "n_jobs": -1,
    "disable_default_eval_metric": 1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_log)):
    X_tr, y_tr = X_train.iloc[train_idx], y_log[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_log[val_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_va, label=y_va)
    dtest = xgb.DMatrix(X_test)

    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=2500,
        evals=[(dtrain, "train"), (dval, "val")],
        obj=fair_obj,
        custom_metric=eval_mae,
        maximize=False,
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    val_pred_log = model.predict(dval)
    oof_preds[val_idx] = np.exp(val_pred_log) - SHIFT
    test_preds += (np.exp(model.predict(dtest)) - SHIFT) / kf.n_splits

# 5. Evaluate Validation Metric
cv_mae = mean_absolute_error(y, oof_preds)
print(f"Validation MAE: {cv_mae:.4f}")

# 6. Save Submission
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": test_ids, "loss": np.clip(test_preds, a_min=0, a_max=None)}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully to ./working/submission.csv")
