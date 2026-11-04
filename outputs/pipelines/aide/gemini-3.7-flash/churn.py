import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")


def load_and_preprocess_data():
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = "class"

    # Process target column
    y = train_df[target_col]
    if y.dtype == "object" or y.dtype.name == "category" or y.dtype == bool:
        unique_vals = y.unique()
        if set(unique_vals).issubset(
            {
                0,
                1,
                "0",
                "1",
                True,
                False,
                "True.",
                "False.",
                "True",
                "False",
                "yes",
                "no",
                "Yes",
                "No",
            }
        ):
            mapping = {
                "True.": 1,
                "True": 1,
                True: 1,
                "1": 1,
                1: 1,
                "yes": 1,
                "Yes": 1,
                "False.": 0,
                "False": 0,
                False: 0,
                "0": 0,
                0: 0,
                "no": 0,
                "No": 0,
            }
            y = y.map(mapping).astype(int)
        else:
            le = LabelEncoder()
            y = le.fit_transform(y)
    else:
        # If target has values like 1 and 2, map to 0 and 1
        if sorted(y.unique().tolist()) == [1, 2]:
            y = y - 1
        y = y.astype(int)

    X_train = train_df.drop(columns=[target_col]).copy()
    X_test = test_df.copy()

    # Drop identifier columns if present
    cols_to_drop = [
        c for c in ["phone number", "phone_number", "id", "ID"] if c in X_train.columns
    ]
    X_train = X_train.drop(columns=cols_to_drop)
    X_test = X_test.drop(columns=[c for c in cols_to_drop if c in X_test.columns])

    # Feature Engineering
    def add_features(df):
        df_feat = df.copy()

        # Standardize column names for easy lookup
        col_map = {c: c.lower().replace(" ", "_") for c in df_feat.columns}
        inv_map = {v: k for k, v in col_map.items()}

        day_mins = inv_map.get("total_day_minutes")
        eve_mins = inv_map.get("total_eve_minutes")
        night_mins = inv_map.get("total_night_minutes")
        intl_mins = inv_map.get("total_intl_minutes")

        day_calls = inv_map.get("total_day_calls")
        eve_calls = inv_map.get("total_eve_calls")
        night_calls = inv_map.get("total_night_calls")
        intl_calls = inv_map.get("total_intl_calls")

        day_charge = inv_map.get("total_day_charge")
        eve_charge = inv_map.get("total_eve_charge")
        night_charge = inv_map.get("total_night_charge")
        intl_charge = inv_map.get("total_intl_charge")

        cust_serv = inv_map.get("customer_service_calls") or inv_map.get(
            "number_customer_service_calls"
        )
        acc_len = inv_map.get("account_length")

        # Aggregations
        if day_mins and eve_mins and night_mins and intl_mins:
            df_feat["total_minutes"] = (
                df_feat[day_mins]
                + df_feat[eve_mins]
                + df_feat[night_mins]
                + df_feat[intl_mins]
            )
        if day_calls and eve_calls and night_calls and intl_calls:
            df_feat["total_calls"] = (
                df_feat[day_calls]
                + df_feat[eve_calls]
                + df_feat[night_calls]
                + df_feat[intl_calls]
            )
        if day_charge and eve_charge and night_charge and intl_charge:
            df_feat["total_charge"] = (
                df_feat[day_charge]
                + df_feat[eve_charge]
                + df_feat[night_charge]
                + df_feat[intl_charge]
            )

        if "total_minutes" in df_feat.columns and "total_calls" in df_feat.columns:
            df_feat["avg_minutes_per_call"] = df_feat["total_minutes"] / (
                df_feat["total_calls"] + 1e-5
            )
        if "total_charge" in df_feat.columns and "total_minutes" in df_feat.columns:
            df_feat["avg_charge_per_minute"] = df_feat["total_charge"] / (
                df_feat["total_minutes"] + 1e-5
            )

        if cust_serv and acc_len:
            df_feat["cust_serv_per_acc_len"] = df_feat[cust_serv] / (
                df_feat[acc_len] + 1
            )
        if cust_serv and "total_calls" in df_feat.columns:
            df_feat["cust_serv_call_ratio"] = df_feat[cust_serv] / (
                df_feat["total_calls"] + 1
            )

        return df_feat

    X_train = add_features(X_train)
    X_test = add_features(X_test)

    # Categorical encoding
    cat_cols = []
    for col in X_train.columns:
        if (
            X_train[col].dtype == "object"
            or X_train[col].dtype.name == "category"
            or X_train[col].dtype == bool
        ):
            cat_cols.append(col)
            le = LabelEncoder()
            full_series = pd.concat([X_train[col].astype(str), X_test[col].astype(str)])
            le.fit(full_series)
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))

    return X_train, y, X_test, cat_cols


def main():
    X_train, y_train, X_test, cat_cols = load_and_preprocess_data()

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 20,
        "random_state": 42,
        "n_estimators": 1500,
        "verbose": -1,
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

        trn_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
        val_data = lgb.Dataset(
            X_va, label=y_val, categorical_feature=cat_cols, reference=trn_data
        )

        model = lgb.train(
            params,
            trn_data,
            valid_sets=[trn_data, val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        oof_preds[val_idx] = model.predict(X_va, num_iteration=model.best_iteration)
        test_preds += (
            model.predict(X_test, num_iteration=model.best_iteration) / n_splits
        )

    cv_auc = roc_auc_score(y_train, oof_preds)
    print(f"5-Fold CV ROC-AUC: {cv_auc:.5f}")

    # Save submission
    os.makedirs("./working", exist_ok=True)
    submission = pd.DataFrame({"id": np.arange(len(test_preds)), "class": test_preds})
    submission.to_csv("./working/submission.csv", index=False)
    print("Submission saved successfully.")


if __name__ == "__main__":
    main()
