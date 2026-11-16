import copy
import json
import os
import pickle
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")


# ---------------------------------------------------------
# PyTorch Architecture
# ---------------------------------------------------------
class GatedTabularNet(nn.Module):

    def __init__(
        self, input_dim, num_classes, hidden_dims=[256, 128], dropout_rate=0.20
    ):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.feature_gate = nn.Sequential(nn.Linear(input_dim, input_dim), nn.Sigmoid())

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LayerNorm(h_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(h_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LayerNorm(h_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout_rate),
                )
            )
            in_dim = h_dim

        self.res_blocks = nn.ModuleList(layers)
        self.projections = nn.ModuleList(
            [
                nn.Linear(input_dim if i == 0 else hidden_dims[i - 1], hidden_dims[i])
                for i in range(len(hidden_dims))
            ]
        )
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x):
        x = self.input_bn(x)
        gate = self.feature_gate(x)
        x = x * gate

        current = x
        for block, proj in zip(self.res_blocks, self.projections):
            res = proj(current)
            current = block(current) + res

        logits = self.classifier(current)
        return logits


def train_nn_fold(
    X_tr, y_tr, X_va, y_va, X_te, num_classes, device, epochs=25, batch_size=64
):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    input_dim = X_tr_s.shape[1]
    model = GatedTabularNet(
        input_dim=input_dim, num_classes=num_classes, hidden_dims=[256, 128]
    ).to(device)

    train_ds = TensorDataset(torch.FloatTensor(X_tr_s), torch.LongTensor(y_tr))
    train_ld = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0
    )

    val_ds = TensorDataset(torch.FloatTensor(X_va_s), torch.LongTensor(y_va))
    val_ld = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    test_ds = TensorDataset(torch.FloatTensor(X_te_s))
    test_ld = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    best_loss = float("inf")
    best_weights = None

    for epoch in range(epochs):
        model.train()
        for bx, by in train_ld:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for bx, by in val_ld:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                loss = criterion(out, by)
                val_loss += loss.item() * bx.size(0)
                val_count += bx.size(0)
        val_loss = val_loss / max(val_count, 1)

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())

    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.eval()
    val_probs_list = []
    with torch.no_grad():
        for bx, _ in val_ld:
            bx = bx.to(device)
            out = model(bx)
            val_probs_list.append(F.softmax(out, dim=-1).cpu().numpy())
    val_probs = np.vstack(val_probs_list)

    test_probs_list = []
    with torch.no_grad():
        for (bx,) in test_ld:
            bx = bx.to(device)
            out = model(bx)
            test_probs_list.append(F.softmax(out, dim=-1).cpu().numpy())
    test_probs = np.vstack(test_probs_list)

    return val_probs, test_probs


# ---------------------------------------------------------
# Feature Engineering & Pipeline Execution
# ---------------------------------------------------------
def prepare_features():
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    print("Loading datasets...")
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    target_col = "Decision type"
    assert (
        target_col in train_df.columns
    ), f"Target column '{target_col}' missing from train data"

    y_raw = train_df[target_col].copy()
    train_df = train_df.drop(columns=[target_col])

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)

    with open("./working/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    n_train = len(train_df)
    combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    id_cols = [c for c in combined_df.columns if c.lower() in ["id", "index", "row_id"]]
    date_cols = []
    text_cols = []
    cat_cols = []
    num_cols = []

    for col in combined_df.columns:
        if col in id_cols:
            continue

        if pd.api.types.is_datetime64_any_dtype(combined_df[col]):
            date_cols.append(col)
            continue

        if combined_df[col].dtype == "object":
            sample_non_null = combined_df[col].dropna().head(20)
            if (
                len(sample_non_null) > 0
                and sample_non_null.astype(str).str.contains(r"\d{4}-\d{2}-\d{2}").all()
            ):
                date_cols.append(col)
                continue

        if pd.api.types.is_numeric_dtype(combined_df[col]):
            if combined_df[col].nunique() < 10 and not col.endswith("id"):
                cat_cols.append(col)
            else:
                num_cols.append(col)
            continue

        if (
            combined_df[col].dtype == "object"
            or combined_df[col].dtype.name == "category"
        ):
            avg_word_len = (
                combined_df[col].astype(str).apply(lambda x: len(x.split())).mean()
            )
            if avg_word_len > 3 or combined_df[col].nunique() > 100:
                text_cols.append(col)
            else:
                cat_cols.append(col)

    n_classes = len(label_encoder.classes_)
    processed_features = pd.DataFrame(index=combined_df.index)
    processed_features["row_null_count"] = combined_df.isnull().sum(axis=1).values

    for col in id_cols:
        processed_features[col] = combined_df[col]

    parsed_date_cols = []
    for col in date_cols:
        parsed_dt = pd.to_datetime(combined_df[col], errors="coerce")
        parsed_date_cols.append((col, parsed_dt))

        processed_features[f"{col}_year"] = parsed_dt.dt.year.fillna(-1).astype(int)
        processed_features[f"{col}_month"] = parsed_dt.dt.month.fillna(-1).astype(int)
        processed_features[f"{col}_day"] = parsed_dt.dt.day.fillna(-1).astype(int)
        processed_features[f"{col}_dayofweek"] = parsed_dt.dt.dayofweek.fillna(
            -1
        ).astype(int)
        processed_features[f"{col}_quarter"] = parsed_dt.dt.quarter.fillna(-1).astype(
            int
        )
        processed_features[f"{col}_days_epoch"] = (
            parsed_dt - pd.Timestamp("1970-01-01")
        ).dt.days.fillna(-1)

    for i in range(len(parsed_date_cols)):
        for j in range(i + 1, len(parsed_date_cols)):
            col1_name, col1_dt = parsed_date_cols[i]
            col2_name, col2_dt = parsed_date_cols[j]
            diff_col = f"diff_days_{col1_name}_vs_{col2_name}"
            processed_features[diff_col] = (col1_dt - col2_dt).dt.days.fillna(-9999)

    for col in num_cols:
        processed_features[col] = combined_df[col].astype(float)
        processed_features[f"{col}_is_null"] = combined_df[col].isnull().astype(int)
        median_val = combined_df[col].iloc[:n_train].median()
        processed_features[col] = processed_features[col].fillna(
            median_val if not np.isnan(median_val) else 0.0
        )

    for col in cat_cols:
        col_str = combined_df[col].astype(str).fillna("MISSING")
        freq_map = col_str.iloc[:n_train].value_counts().to_dict()
        processed_features[f"{col}_freq"] = (
            col_str.map(freq_map).fillna(0).astype(float)
        )

        cat_le = LabelEncoder()
        processed_features[f"{col}_encoded"] = cat_le.fit_transform(col_str)

    text_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for col in text_cols:
        text_series = combined_df[col].astype(str).fillna("")

        processed_features[f"{col}_char_len"] = text_series.apply(len)
        processed_features[f"{col}_word_count"] = text_series.apply(
            lambda x: len(x.split())
        )
        processed_features[f"{col}_digit_count"] = text_series.apply(
            lambda x: sum(c.isdigit() for c in x)
        )

        train_text = text_series.iloc[:n_train]
        test_text = text_series.iloc[n_train:]

        tfidf = TfidfVectorizer(
            max_features=2500, stop_words="english", ngram_range=(1, 2)
        )
        tfidf_train = tfidf.fit_transform(train_text)
        tfidf_test = tfidf.transform(test_text)

        oof_text_probs = np.zeros((n_train, n_classes))
        test_text_probs = np.zeros((len(test_df), n_classes))

        for tr_idx, val_idx in text_skf.split(train_text, y_encoded):
            lr_text = LogisticRegression(C=1.0, max_iter=200, random_state=42)
            lr_text.fit(tfidf_train[tr_idx], y_encoded[tr_idx])
            oof_text_probs[val_idx] = lr_text.predict_proba(tfidf_train[val_idx])
            test_text_probs += lr_text.predict_proba(tfidf_test) / 5.0

        for c_idx in range(n_classes):
            processed_features[f"{col}_oof_prob_class_{c_idx}"] = np.concatenate(
                [oof_text_probs[:, c_idx], test_text_probs[:, c_idx]]
            )

        n_comp = min(32, tfidf_train.shape[1] - 1) if tfidf_train.shape[1] > 1 else 1
        if n_comp > 0 and tfidf_train.shape[1] > 0:
            svd = TruncatedSVD(n_components=n_comp, random_state=42)
            svd_train = svd.fit_transform(tfidf_train)
            svd_test = svd.transform(tfidf_test)

            svd_cols = [f"{col}_svd_{k}" for k in range(n_comp)]
            svd_df = pd.DataFrame(
                np.vstack([svd_train, svd_test]),
                columns=svd_cols,
                index=combined_df.index,
            )
            processed_features = pd.concat([processed_features, svd_df], axis=1)

    X_train_full = processed_features.iloc[:n_train].copy()
    X_test_full = processed_features.iloc[n_train:].copy()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    n_classes = len(label_encoder.classes_)

    for col in cat_cols:
        cat_values_train = combined_df[col].iloc[:n_train].astype(str).fillna("MISSING")
        cat_values_test = combined_df[col].iloc[n_train:].astype(str).fillna("MISSING")

        global_priors = pd.Series(y_encoded).value_counts(normalize=True).to_dict()

        for c_idx in range(n_classes):
            binary_target = (y_encoded == c_idx).astype(float)
            prior = global_priors.get(c_idx, 0.0)

            te_col_name = f"{col}_te_class_{c_idx}"
            X_train_full[te_col_name] = np.nan

            for train_idx, val_idx in skf.split(X_train_full, y_encoded):
                fold_cats = cat_values_train.iloc[train_idx]
                fold_target = binary_target[train_idx]

                stats = (
                    pd.DataFrame({"cat": fold_cats, "target": fold_target})
                    .groupby("cat")["target"]
                    .agg(["count", "mean"])
                )
                smooth_weight = 10.0
                smoothed_te = (
                    stats["count"] * stats["mean"] + smooth_weight * prior
                ) / (stats["count"] + smooth_weight)

                val_cats = cat_values_train.iloc[val_idx]
                X_train_full.iloc[
                    val_idx, X_train_full.columns.get_loc(te_col_name)
                ] = val_cats.map(smoothed_te).fillna(prior)

            full_stats = (
                pd.DataFrame({"cat": cat_values_train, "target": binary_target})
                .groupby("cat")["target"]
                .agg(["count", "mean"])
            )
            full_smoothed_te = (
                full_stats["count"] * full_stats["mean"] + smooth_weight * prior
            ) / (full_stats["count"] + smooth_weight)
            X_test_full[te_col_name] = cat_values_test.map(full_smoothed_te).fillna(
                prior
            )

    fold_assignments = np.zeros(n_train, dtype=int)
    for fold_idx, (_, val_idx) in enumerate(skf.split(X_train_full, y_encoded)):
        fold_assignments[val_idx] = fold_idx

    X_train_full["fold"] = fold_assignments
    X_train_full["target"] = y_encoded

    return X_train_full, X_test_full, label_encoder


def main():
    X_train_df, X_test_df, label_encoder = prepare_features()

    y_train = X_train_df["target"].values.astype(int)
    folds = X_train_df["fold"].values.astype(int)

    feature_cols = [
        c
        for c in X_train_df.columns
        if c not in ["fold", "target"] and not c.lower() in ["id", "index", "row_id"]
    ]
    numeric_feature_cols = [
        c for c in feature_cols if pd.api.types.is_numeric_dtype(X_train_df[c])
    ]

    X_tr_proc = (
        X_train_df[numeric_feature_cols].replace([np.inf, -np.inf], np.nan).copy()
    )
    X_te_proc = (
        X_test_df[numeric_feature_cols].replace([np.inf, -np.inf], np.nan).copy()
    )

    train_medians = X_tr_proc.median()
    X_train_mat = X_tr_proc.fillna(train_medians).fillna(0.0).values
    X_test_mat = X_te_proc.fillna(train_medians).fillna(0.0).values

    num_classes = len(label_encoder.classes_)
    num_samples = len(X_train_mat)
    num_test = len(X_test_mat)
    num_folds = len(np.unique(folds))

    oof_lgb = np.zeros((num_samples, num_classes))
    oof_xgb = np.zeros((num_samples, num_classes))
    oof_cb = np.zeros((num_samples, num_classes))
    oof_nn = np.zeros((num_samples, num_classes))

    test_lgb = np.zeros((num_test, num_classes))
    test_xgb = np.zeros((num_test, num_classes))
    test_cb = np.zeros((num_test, num_classes))
    test_nn = np.zeros((num_test, num_classes))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold_idx in range(num_folds):
        train_mask = folds != fold_idx
        val_mask = folds == fold_idx

        X_tr, y_tr = X_train_mat[train_mask], y_train[train_mask]
        X_va, y_va = X_train_mat[val_mask], y_train[val_mask]

        # Model 1: LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_samples=15,
            random_state=42 + fold_idx,
            n_jobs=-1,
        )
        lgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        oof_lgb[val_mask] = lgb_model.predict_proba(X_va)
        test_lgb += lgb_model.predict_proba(X_test_mat) / num_folds

        # Model 2: XGBoost or HistGradientBoosting Fallback
        try:
            import xgboost as xgb

            xgb_model = xgb.XGBClassifier(
                n_estimators=1000,
                learning_rate=0.03,
                max_depth=5,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=0.1,
                reg_lambda=1.0,
                min_child_weight=3,
                eval_metric="mlogloss",
                early_stopping_rounds=50,
                random_state=42 + fold_idx,
                n_jobs=-1,
            )
            xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            oof_xgb[val_mask] = xgb_model.predict_proba(X_va)
            test_xgb += xgb_model.predict_proba(X_test_mat) / num_folds
        except Exception:
            hgb_model = HistGradientBoostingClassifier(
                max_iter=500,
                learning_rate=0.03,
                max_depth=5,
                l2_regularization=1.0,
                min_samples_leaf=15,
                random_state=42 + fold_idx,
            )
            hgb_model.fit(X_tr, y_tr)
            oof_xgb[val_mask] = hgb_model.predict_proba(X_va)
            test_xgb += hgb_model.predict_proba(X_test_mat) / num_folds

        # Model 3: CatBoost or ExtraTrees Fallback
        try:
            from catboost import CatBoostClassifier

            cb_model = CatBoostClassifier(
                iterations=1000,
                learning_rate=0.04,
                depth=5,
                l2_leaf_reg=3.0,
                subsample=0.8,
                eval_metric="MultiClass",
                early_stopping_rounds=50,
                random_state=42 + fold_idx,
                verbose=0,
            )
            cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)
            oof_cb[val_mask] = cb_model.predict_proba(X_va)
            test_cb += cb_model.predict_proba(X_test_mat) / num_folds
        except Exception:
            et_model = ExtraTreesClassifier(
                n_estimators=300, max_depth=12, min_samples_leaf=3, random_state=42 + fold_idx, n_jobs=-1
            )
            et_model.fit(X_tr, y_tr)
            oof_cb[val_mask] = et_model.predict_proba(X_va)
            test_cb += et_model.predict_proba(X_test_mat) / num_folds

        # Model 4: PyTorch Gated Tabular Net
        nn_val_probs, nn_test_probs = train_nn_fold(
            X_tr,
            y_tr,
            X_va,
            y_va,
            X_test_mat,
            num_classes,
            device,
            epochs=25,
            batch_size=64,
        )
        oof_nn[val_mask] = nn_val_probs
        test_nn += nn_test_probs / num_folds

        fold_acc = accuracy_score(y_va, np.argmax(oof_lgb[val_mask], axis=1))
        print(f"Fold {fold_idx + 1}/{num_folds} LightGBM Accuracy: {fold_acc:.5f}")

    # Multi-Model Stacking
    X_meta_oof = np.hstack([oof_lgb, oof_xgb, oof_cb, oof_nn])
    X_meta_test = np.hstack([test_lgb, test_xgb, test_cb, test_nn])

    meta_model = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    meta_model.fit(X_meta_oof, y_train)

    oof_meta_preds = meta_model.predict(X_meta_oof)
    final_val_score = accuracy_score(y_train, oof_meta_preds)

    test_meta_preds = meta_model.predict(X_meta_test)
    test_decoded = label_encoder.inverse_transform(test_meta_preds)

    # Export Submission
    submission_df = pd.DataFrame()
    if "id" in X_test_df.columns:
        submission_df["id"] = X_test_df["id"].astype(int)
    elif "ID" in X_test_df.columns:
        submission_df["id"] = X_test_df["ID"].astype(int)
    else:
        submission_df["id"] = np.arange(len(X_test_df))

    submission_df["Decision type"] = test_decoded
    os.makedirs("./submission", exist_ok=True)
    submission_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_val_score}")


if __name__ == "__main__":
    main()
