import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EXPENSE_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
NUMERIC_COLS = (
    ["Age", "CabinNum", "CabinNumLog", "TotalSpendLog"]
    + EXPENSE_COLS
    + ["Age_missing", "CabinNum_missing"]
    + [f"{col}_missing" for col in EXPENSE_COLS]
)
CAT_COLS = ["HomePlanet", "Destination", "CryoSleep", "VIP", "Deck", "Side"]


def parse_cabin(value):
    if pd.isna(value):
        return np.nan, np.nan, np.nan

    parts = str(value).strip().split("/")
    deck = parts[0] if len(parts) > 0 and parts[0] else np.nan
    num = pd.to_numeric(parts[1], errors="coerce") if len(parts) > 1 else np.nan
    side = parts[2] if len(parts) > 2 and parts[2] else np.nan
    return deck, num, side


def to_categorical(series):
    return series.where(series.notna(), "Unknown").astype(str)


def prepare(df):
    out = pd.DataFrame(index=df.index)

    out["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    out["Age_missing"] = out["Age"].isna().astype(int)

    parsed_cabin = df["Cabin"].apply(parse_cabin)
    cabin = pd.DataFrame(
        parsed_cabin.tolist(),
        index=df.index,
        columns=["Deck", "CabinNum", "Side"],
    )
    out = out.join(cabin)

    out["CabinNum"] = pd.to_numeric(out["CabinNum"], errors="coerce")
    out["CabinNum_missing"] = out["CabinNum"].isna().astype(int)
    out["CabinNumLog"] = np.log1p(out["CabinNum"].clip(lower=0))

    raw_expense_sum = pd.Series(0.0, index=df.index)
    for col in EXPENSE_COLS:
        vals = pd.to_numeric(df[col], errors="coerce")
        out[f"{col}_missing"] = vals.isna().astype(int)
        vals = vals.fillna(0.0)
        out[col] = np.log1p(vals)
        raw_expense_sum += vals

    out["TotalSpendLog"] = np.log1p(raw_expense_sum)

    for col in ["HomePlanet", "Destination", "CryoSleep", "VIP"]:
        out[col] = to_categorical(df[col])

    out["Deck"] = to_categorical(out["Deck"])
    out["Side"] = to_categorical(out["Side"])

    return out


def make_pipeline():
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLS),
            ("cat", categorical_transformer, CAT_COLS),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=5000, C=1.0, random_state=42),
            ),
        ]
    )


def main():
    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    X = prepare(train)
    y = train["Transported"].astype(int)
    X_test = prepare(test)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        model = make_pipeline()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        val_pred = model.predict(X.iloc[val_idx])
        acc = accuracy_score(y.iloc[val_idx], val_pred)
        scores.append(acc)
        print(f"Fold {fold} accuracy: {acc:.4f}")

    final_model = make_pipeline()
    final_model.fit(X, y)
    test_pred = final_model.predict(X_test)

    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame(
        {
            "id": np.arange(len(test)),
            "Transported": test_pred.astype(bool),
        }
    )
    submission.to_csv("./working/submission.csv", index=False)
    print(f"Saved submission with {len(submission)} rows.")
    print(f"CV accuracy: {np.mean(scores):.4f}")


if __name__ == "__main__":
    main()
