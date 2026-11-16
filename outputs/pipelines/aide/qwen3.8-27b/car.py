import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

RANDOM_STATE = 42
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

TARGET = "class"
FEATURES = [c for c in train.columns if c != TARGET]

ORDERS = {
    "buying": ["low", "med", "high", "vhigh"],
    "maint": ["low", "med", "high", "vhigh"],
    "doors": ["2", "3", "4", "5", "more"],
    "persons": ["2", "3", "4", "5", "more"],
    "lug_boot": ["small", "med", "big"],
    "safety": ["low", "med", "high"],
}


def normalize_factor_series(s: pd.Series) -> pd.Series:
    """Convert ordinal-like series to stable string categories."""
    if pd.api.types.is_numeric_dtype(s):
        non_null = s.dropna()
        if len(non_null) > 0 and (non_null % 1 == 0).all():
            return s.astype(int).astype(str)
    return s.astype(str)


def encode_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Encode known ordinal features and safely fallback for unexpected columns."""
    out = pd.DataFrame(index=df.index)

    for col in FEATURES:
        if col in ORDERS:
            mapping = {value: idx for idx, value in enumerate(ORDERS[col])}
            out[col] = normalize_factor_series(df[col]).map(mapping).astype(float)
            out[col] = out[col].fillna(-1.0)
        else:
            dummies = pd.get_dummies(df[col].astype(str), prefix=col, drop_first=False)
            dummies = dummies.astype(float)
            out = pd.concat([out, dummies], axis=1)

    return out


X_train = encode_frame(train.reindex(columns=FEATURES))
y_raw = train[TARGET]

valid = y_raw.notna()
X_train = X_train.loc[valid]
y_train = y_raw[valid].astype(str)

X_test = encode_frame(test.reindex(columns=FEATURES))


def make_model() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=5,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


# Stratified 5-fold cross-validation as the validation protocol.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
    model = make_model()
    model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
    val_pred = model.predict(X_train.iloc[val_idx])
    score = accuracy_score(y_train.iloc[val_idx], val_pred)
    cv_scores.append(score)
    print(f"Fold {fold} accuracy: {score:.6f}")

print(f"Mean 5-fold CV accuracy: {np.mean(cv_scores):.6f}")

# Retrain on the full training set for the final submission.
final_model = make_model()
final_model.fit(X_train, y_train)
test_pred = final_model.predict(X_test)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test_pred), dtype=int),
        "class": test_pred,
    }
)

submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
print(f"Saved submission with {len(submission)} rows to {WORK_DIR}/submission.csv")
