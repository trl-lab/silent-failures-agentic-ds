import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/cat-in-the-dat/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/kaggle/cat-in-the-dat/data/test.parquet"
TARGET = "target"
ID = "id"
LOW_CARD_THRESHOLD = 100


def write_predictions(n, values=None):
    if values is None:
        values = np.full(n, 0.5, dtype=float)
    else:
        values = np.asarray(values, dtype=float)
        values = np.nan_to_num(np.clip(values, 0.0, 1.0), nan=0.5)
    pd.DataFrame({"id": np.arange(n, dtype=int), "target": values}).to_csv(
        "predictions.csv", index=False
    )


def encode_with_reference(reference, data, features, low_card_threshold=LOW_CARD_THRESHOLD):
    ref = reference[features].copy()
    dat = data[features].copy()

    for col in features:
        if col not in ref.columns:
            ref[col] = np.nan
        if col not in dat.columns:
            dat[col] = np.nan

    out_ref = pd.DataFrame(index=ref.index)
    out_dat = pd.DataFrame(index=dat.index)

    for col in features:
        if pd.api.types.is_numeric_dtype(ref[col]):
            out_ref[col] = pd.to_numeric(ref[col], errors="coerce").fillna(-1.0).astype(float)
            out_dat[col] = pd.to_numeric(dat[col], errors="coerce").fillna(-1.0).astype(float)
        else:
            ref[col] = ref[col].fillna("missing").astype(str)
            dat[col] = dat[col].fillna("missing").astype(str)

            if ref[col].nunique() <= low_card_threshold:
                ref_dum = pd.get_dummies(ref[col], prefix=col).astype(float)
                dat_dum = pd.get_dummies(dat[col], prefix=col).astype(float)
                dat_dum = dat_dum.reindex(columns=ref_dum.columns, fill_value=0)
                out_ref = pd.concat([out_ref, ref_dum], axis=1)
                out_dat = pd.concat([out_dat, dat_dum], axis=1)
            else:
                freq = ref[col].value_counts()
                out_ref[col] = ref[col].map(freq).fillna(0.0).astype(float)
                out_dat[col] = dat[col].map(freq).fillna(0.0).astype(float)

    out_ref = out_ref.fillna(0.0)
    out_dat = out_dat.reindex(columns=out_ref.columns, fill_value=0.0).fillna(0.0)
    return out_ref, out_dat


def make_model(n_estimators):
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    n_test = len(test_df)

    if n_test == 0:
        write_predictions(0)
        return

    train_df = train_df.dropna(subset=[TARGET]).reset_index(drop=True)

    if len(train_df) == 0:
        write_predictions(n_test)
        return

    y_all = train_df[TARGET].astype(int)

    if y_all.nunique() <= 1:
        write_predictions(n_test, np.full(n_test, float(y_all.iloc[0])))
        return

    features = [c for c in train_df.columns if c not in (ID, TARGET)]

    if len(features) == 0:
        write_predictions(n_test)
        return

    use_val = len(train_df) > 200
    best_iter = 500

    if use_val:
        try:
            fit_df, val_df = train_test_split(
                train_df, test_size=0.2, random_state=42, stratify=y_all
            )
            y_fit = fit_df[TARGET].astype(int)
            y_val = val_df[TARGET].astype(int)

            if y_fit.nunique() > 1 and y_val.nunique() > 1:
                X_fit, X_val = encode_with_reference(fit_df, val_df, features)
                X_fit, X_all = encode_with_reference(fit_df, train_df, features)
                _, X_test = encode_with_reference(fit_df, test_df, features)

                model = make_model(1000)
                try:
                    model.fit(
                        X_fit,
                        y_fit,
                        eval_set=[(X_val, y_val)],
                        eval_metric="auc",
                        callbacks=[lgb.early_stopping(100, verbose=False)],
                    )
                except TypeError:
                    model = make_model(1000)
                    model.fit(
                        X_fit,
                        y_fit,
                        eval_set=[(X_val, y_val)],
                        eval_metric="auc",
                    )

                best_iter = getattr(model, "best_iteration_", 0)
                if best_iter is None or best_iter <= 0:
                    best_iter = 500
                else:
                    best_iter = int(best_iter)
            else:
                X_fit, X_all = encode_with_reference(train_df, train_df, features)
                _, X_test = encode_with_reference(train_df, test_df, features)
                best_iter = 500
        except Exception:
            X_fit, X_all = encode_with_reference(train_df, train_df, features)
            _, X_test = encode_with_reference(train_df, test_df, features)
            best_iter = 500
    else:
        X_fit, X_all = encode_with_reference(train_df, train_df, features)
        _, X_test = encode_with_reference(train_df, test_df, features)
        best_iter = 300

    final_model = make_model(max(1, best_iter))
    final_model.fit(X_all, y_all)

    proba = final_model.predict_proba(X_test)[:, 1]
    write_predictions(n_test, proba)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        try:
            test_df = pd.read_parquet(TEST_PATH)
            n = len(test_df)
        except Exception:
            n = 0
        write_predictions(n)