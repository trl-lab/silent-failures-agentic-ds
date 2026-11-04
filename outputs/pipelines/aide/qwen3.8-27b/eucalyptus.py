import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

os.makedirs("./working", exist_ok=True)

# Load data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

target = "Utility"
y = train[target].values

# Identify categorical and numerical columns
all_cols = [c for c in train.columns if c != target]
cat_cols = []
num_cols = []
for c in all_cols:
    if train[c].dtype == object or str(train[c].dtype).startswith("category"):
        cat_cols.append(c)
    else:
        num_cols.append(c)

print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")
print(f"Numerical columns ({len(num_cols)}): {num_cols}")
print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print(f"Target classes: {sorted(np.unique(y))}")

# Grouping column for site-level leakage control
group_col = "Abbrev" if "Abbrev" in train.columns else all_cols[0]
groups = train[group_col].values

# 5-fold grouped CV, with fallback if too few groups
n_splits = 5
if len(np.unique(groups)) >= n_splits:
    splitter = GroupKFold(n_splits=n_splits)
    print(f"\nUsing GroupKFold with {len(np.unique(groups))} unique groups")
else:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    print(
        f"\nUnique groups ({len(np.unique(groups))}) < {n_splits}, falling back to StratifiedKFold"
    )

# Atomic improvement: add min_samples_leaf=5 to regularise the Random Forest
cv_scores = []
for fold, (tr_idx, va_idx) in enumerate(splitter.split(train, y, groups)):
    # Fit one-hot encoder on training fold only
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat_tr = ohe.fit_transform(train[cat_cols].iloc[tr_idx])
    X_cat_va = ohe.transform(train[cat_cols].iloc[va_idx])

    X_tr = np.hstack([X_cat_tr, train[num_cols].iloc[tr_idx].values.astype(np.float64)])
    X_va = np.hstack([X_cat_va, train[num_cols].iloc[va_idx].values.astype(np.float64)])
    y_tr = y[tr_idx]
    y_va = y[va_idx]

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=5,  # <-- single atomic improvement
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)

    y_pred = rf.predict(X_va)
    acc = np.mean(y_pred == y_va)
    cv_scores.append(acc)
    print(f"  Fold {fold}: accuracy = {acc:.4f}")

mean_acc = np.mean(cv_scores)
std_acc = np.std(cv_scores)
print(f"\nCV Accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")

# Final model trained on full training set with the same regularisation
ohe_final = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_cat_tr_full = ohe_final.fit_transform(train[cat_cols])
X_tr_full = np.hstack([X_cat_tr_full, train[num_cols].values.astype(np.float64)])

rf_final = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=5,  # <-- same improvement in final model
    random_state=42,
    n_jobs=-1,
)
rf_final.fit(X_tr_full, y)

# Generate test predictions
X_cat_te = ohe_final.transform(test[cat_cols])
X_te = np.hstack([X_cat_te, test[num_cols].values.astype(np.float64)])
y_test_pred = rf_final.predict(X_te)

# Save submission
submission = pd.DataFrame({"id": range(len(y_test_pred)), "Utility": y_test_pred})
submission.to_csv("./working/submission.csv", index=False)
print(f"\nSubmission saved to ./working/submission.csv with {len(submission)} rows.")
print(f"Prediction distribution:\n{pd.Series(y_test_pred).value_counts().sort_index()}")
