import os
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss

os.makedirs("./working", exist_ok=True)

UNKNOWN = "UNKNOWN"

train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

train["Category"] = (
    train["Category"]
    .astype(str)
    .str.strip()
    .replace({"nan": UNKNOWN, "None": UNKNOWN, "": UNKNOWN})
)
all_classes = sorted(train["Category"].unique())

FEATURE_COLS = [
    "Descript",
    "PdDistrict",
    "Resolution",
    "DayOfWeek",
    "hour",
    "month",
    "day",
    "year",
]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    for col in ["Descript", "PdDistrict", "Resolution", "DayOfWeek"]:
        if col in df.columns:
            out[col] = df[col].astype(str).str.strip()
        else:
            out[col] = UNKNOWN

    out = out.replace({"nan": UNKNOWN, "None": UNKNOWN, "": UNKNOWN})

    if "Dates" in df.columns:
        dates = pd.to_datetime(df["Dates"], errors="coerce")
    else:
        dates = pd.Series(pd.NaT, index=df.index)

    out["hour"] = dates.dt.hour.fillna(-1).astype(int).astype(str)
    out["month"] = dates.dt.month.fillna(-1).astype(int).astype(str)
    out["day"] = dates.dt.day.fillna(-1).astype(int).astype(str)
    out["year"] = dates.dt.year.fillna(-1).astype(int).astype(str)

    return out[FEATURE_COLS]


def make_model() -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-6,
        max_iter=50,
        tol=1e-3,
        random_state=42,
        n_jobs=-1,
    )


def fit_predict_proba(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: np.ndarray,
) -> np.ndarray:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    X_train_enc = encoder.fit_transform(X_train)
    X_val_enc = encoder.transform(X_val)

    model = make_model()
    model.fit(X_train_enc, y_train)

    proba = model.predict_proba(X_val_enc)

    proba_all = np.zeros((proba.shape[0], len(all_classes)), dtype=np.float64)
    class_to_idx = {cls: i for i, cls in enumerate(all_classes)}

    for cls, col in zip(model.classes_, range(proba.shape[1])):
        if cls in class_to_idx:
            proba_all[:, class_to_idx[cls]] = proba[:, col]

    proba_all = np.clip(proba_all, 1e-15, None)
    proba_all /= proba_all.sum(axis=1, keepdims=True)
    return proba_all


# Chronological expanding-window validation.
train = train.copy()
if "Dates" in train.columns:
    train["_date"] = pd.to_datetime(train["Dates"], errors="coerce")
else:
    train["_date"] = pd.Series(pd.NaT, index=train.index)

train = train.sort_values("_date", kind="mergesort").reset_index(drop=True)

train_features = prepare_features(train)
test_features = prepare_features(test)

# Six chronological chunks give five non-empty expanding-window validation folds.
folds = np.array_split(np.arange(len(train)), 6)
fold_losses = []

for i in range(1, 6):
    train_idx = np.concatenate(folds[:i])
    val_idx = folds[i]

    if len(train_idx) == 0 or len(val_idx) == 0:
        continue

    y_tr = train.loc[train_idx, "Category"].to_numpy()
    y_val = train.loc[val_idx, "Category"].to_numpy()

    proba_val = fit_predict_proba(
        train_features.loc[train_idx],
        train_features.loc[val_idx],
        y_tr,
    )

    loss = log_loss(y_val, proba_val, labels=all_classes)
    fold_losses.append(loss)
    print(f"Fold {i} log_loss: {loss:.6f}")

validation_metric = float(np.mean(fold_losses)) if fold_losses else float("nan")
print(f"Validation log_loss: {validation_metric:.6f}")

# Final model on all training data.
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
X_train_all_enc = encoder.fit_transform(train_features)
X_test_enc = encoder.transform(test_features)

final_model = make_model()
final_model.fit(X_train_all_enc, train["Category"].to_numpy())

test_pred = final_model.predict(X_test_enc)

submission = pd.DataFrame(
    {
        "id": np.arange(len(test), dtype=int),
        "Category": test_pred,
    }
)

submission.to_csv("./working/submission.csv", index=False)
print("Saved ./working/submission.csv")
