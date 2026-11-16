import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# Load datasets
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "Damage Description (total)"

# Drop rows with missing target in train data if any
train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)

# Encode target
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df[target_col].astype(str))

# Prepare features
X_train = train_df.drop(columns=[target_col])
X_test = test_df.copy()

# Keep common features between train and test
features = [col for col in X_train.columns if col in X_test.columns]
X_train = X_train[features]
X_test = X_test[features]

# Convert categorical/object columns to category dtype for native LightGBM handling
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
for col in cat_cols:
    X_train[col] = X_train[col].astype("category")
    X_test[col] = X_test[col].astype("category")

num_classes = len(label_encoder.classes_)
n_splits = 5
seeds = [42, 2024, 777]

oof_probs_total = np.zeros((len(X_train), num_classes))
test_probs_total = np.zeros((len(X_test), num_classes))

# Train across multiple seeds and Stratified K-Folds
for seed in seeds:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=num_classes,
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            cat_smooth=10.0,
            random_state=seed + fold,
            n_jobs=-1,
            verbose=-1,
        )

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        val_probs = model.predict_proba(X_va)
        oof_probs_total[val_idx] += val_probs / len(seeds)
        test_probs_total += model.predict_proba(X_test) / (n_splits * len(seeds))

# Evaluate raw accuracy before post-processing
raw_oof_preds = oof_probs_total.argmax(axis=1)
raw_oof_acc = accuracy_score(y_train, raw_oof_preds)


# Optimize class probability weights for global multi-class accuracy metric
def loss_func(weights):
    scaled_probs = oof_probs_total * weights
    preds = scaled_probs.argmax(axis=1)
    return -accuracy_score(y_train, preds)


initial_weights = np.ones(num_classes)
res = minimize(
    loss_func,
    initial_weights,
    method="Nelder-Mead",
    options={"maxiter": 1000, "disp": False},
)

opt_weights = res.x
optimized_oof_preds = (oof_probs_total * opt_weights).argmax(axis=1)
opt_oof_acc = accuracy_score(y_train, optimized_oof_preds)

print(f"Raw OOF Accuracy: {raw_oof_acc:.5f}")
print(f"Optimized OOF Accuracy: {opt_oof_acc:.5f}")

# Generate optimized test set predictions
final_test_indices = (test_probs_total * opt_weights).argmax(axis=1)
final_test_labels = label_encoder.inverse_transform(final_test_indices)

# Save submission file
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {"id": np.arange(len(test_df)), target_col: final_test_labels}
)
submission.to_csv("./working/submission.csv", index=False)
