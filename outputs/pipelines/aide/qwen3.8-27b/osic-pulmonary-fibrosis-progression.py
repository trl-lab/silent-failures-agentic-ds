import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

os.makedirs("./working", exist_ok=True)

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

TARGET = "FVC"
GROUP = "Patient"

train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce")
train = train.dropna(subset=[TARGET]).copy()
train = train[np.isfinite(train[TARGET])].copy()

if GROUP not in train.columns:
    train[GROUP] = "single"
train = train.dropna(subset=[GROUP]).copy()


def make_features(df):
    X = df.copy()

    for col in ["Weeks", "Percent"]:
        if col not in X.columns:
            X[col] = np.nan
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X["Weeks_sq"] = X["Weeks"] ** 2
    X["Percent_sq"] = X["Percent"] ** 2
    X["Weeks_Pct"] = X["Weeks"] * X["Percent"]

    feature_cols = ["Weeks", "Percent", "Weeks_sq", "Percent_sq", "Weeks_Pct"]
    X[feature_cols] = X[feature_cols].replace([np.inf, -np.inf], np.nan)

    return X[feature_cols]


X_train = make_features(train)
y_train = train[TARGET].astype(float)
groups = train[GROUP]


def make_pipeline():
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )


scores = []
n_patients = groups.nunique()

if n_patients >= 2:
    n_splits = min(5, n_patients)
    gkf = GroupKFold(n_splits=n_splits)

    for tr_idx, val_idx in gkf.split(X_train, y_train, groups):
        pipe = make_pipeline()
        pipe.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        pred = pipe.predict(X_train.iloc[val_idx])
        scores.append(mean_absolute_error(y_train.iloc[val_idx], pred))

    cv_mae = float(np.mean(scores))
else:
    if len(X_train) >= 2:
        tr_idx, val_idx = train_test_split(
            np.arange(len(X_train)), test_size=0.2, random_state=42
        )
        pipe = make_pipeline()
        pipe.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        pred = pipe.predict(X_train.iloc[val_idx])
        cv_mae = float(mean_absolute_error(y_train.iloc[val_idx], pred))
    else:
        cv_mae = float("nan")


final_pipe = make_pipeline()
if len(X_train) > 0:
    final_pipe.fit(X_train, y_train)
    X_test = make_features(test)
    test_pred = final_pipe.predict(X_test)
else:
    test_pred = np.zeros(len(test), dtype=float)


submission = pd.DataFrame(
    {"id": np.arange(len(test), dtype=int), "FVC": test_pred.astype(float)}
)

submission.to_csv("./working/submission.csv", index=False)

print(f"Validation MAE: {cv_mae:.3f}")
