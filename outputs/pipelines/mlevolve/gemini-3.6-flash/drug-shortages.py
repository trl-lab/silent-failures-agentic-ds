import os
import random
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import lightgbm as lgb
import xgboost as xgb

try:
    from catboost import CatBoostClassifier

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

warnings.filterwarnings("ignore")


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class TabularResNet(nn.Module):

    def __init__(self, input_dim, num_classes, hidden_dim=256, dropout=0.25):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Residual Block 1
        self.block1_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.block1_norm1 = nn.BatchNorm1d(hidden_dim)
        self.block1_act1 = nn.SiLU()
        self.block1_drop1 = nn.Dropout(dropout)
        self.block1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.block1_norm2 = nn.BatchNorm1d(hidden_dim)
        self.block1_act2 = nn.SiLU()

        # Residual Block 2
        self.block2_fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.block2_norm1 = nn.BatchNorm1d(hidden_dim // 2)
        self.block2_act1 = nn.SiLU()
        self.block2_drop1 = nn.Dropout(dropout)
        self.block2_fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 2)
        self.block2_norm2 = nn.BatchNorm1d(hidden_dim // 2)
        self.block2_act2 = nn.SiLU()

        self.shortcut_b2 = nn.Linear(hidden_dim, hidden_dim // 2)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes),
        )

    def forward(self, x):
        x = self.input_layer(x)

        res1 = x
        out1 = self.block1_fc1(x)
        out1 = self.block1_norm1(out1)
        out1 = self.block1_act1(out1)
        out1 = self.block1_drop1(out1)
        out1 = self.block1_fc2(out1)
        out1 = self.block1_norm2(out1)
        x = self.block1_act2(out1 + res1)

        res2 = self.shortcut_b2(x)
        out2 = self.block2_fc1(x)
        out2 = self.block2_norm1(out2)
        out2 = self.block2_act1(out2)
        out2 = self.block2_drop1(out2)
        out2 = self.block2_fc2(out2)
        out2 = self.block2_norm2(out2)
        x = self.block2_act2(out2 + res2)

        return self.head(x)


def extract_date_features(df, cols):
    date_feats = pd.DataFrame(index=df.index)
    for col in cols:
        dt_s = pd.to_datetime(df[col], errors="coerce")
        date_feats[f"{col}_year"] = dt_s.dt.year
        date_feats[f"{col}_month"] = dt_s.dt.month
        date_feats[f"{col}_day"] = dt_s.dt.day
        date_feats[f"{col}_dayofweek"] = dt_s.dt.dayofweek
        date_feats[f"{col}_dayofyear"] = dt_s.dt.dayofyear
        date_feats[f"{col}_quarter"] = dt_s.dt.quarter
        date_feats[f"{col}_elapsed"] = (dt_s - pd.Timestamp("2000-01-01")).dt.days
    return date_feats


def concat_array_list(arr_list, num_rows):
    if not arr_list:
        return np.empty((num_rows, 0))
    return np.hstack(arr_list)


def main():
    seed_everything(42)

    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    # 1. Load raw data
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    target_col = "availability"
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in train data.")

    if "id" in test_df.columns:
        test_ids = test_df["id"].values
    else:
        test_ids = np.arange(len(test_df))
        test_df["id"] = test_ids

    # Target Encoding setup
    y_raw = train_df[target_col].copy()
    train_df = train_df.drop(columns=[target_col])

    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y_raw.astype(str))
    num_classes = len(target_encoder.classes_)

    # Save target encoder classes
    np.save("./working/target_classes.npy", target_encoder.classes_)

    # 2. Train / Validation Split (80/20 Stratified)
    train_idx, val_idx = train_test_split(
        np.arange(len(train_df)), test_size=0.20, random_state=42, stratify=y_encoded
    )

    X_tr_raw = train_df.iloc[train_idx].copy().reset_index(drop=True)
    y_tr = y_encoded[train_idx]

    X_va_raw = train_df.iloc[val_idx].copy().reset_index(drop=True)
    y_va = y_encoded[val_idx]

    X_te_raw = test_df.copy().reset_index(drop=True)

    # 3. Dynamic Column Type Categorization
    all_cols = list(train_df.columns)
    if "id" in all_cols:
        all_cols.remove("id")

    date_cols, text_cols, cat_cols, num_cols = [], [], [], []

    for col in all_cols:
        col_type = train_df[col].dtype
        if (
            pd.api.types.is_datetime64_any_dtype(col_type)
            or "date" in col.lower()
            or "time" in col.lower()
        ):
            date_cols.append(col)
        elif pd.api.types.is_numeric_dtype(col_type):
            num_cols.append(col)
        else:
            non_nulls = train_df[col].dropna().astype(str)
            if len(non_nulls) > 0 and non_nulls.str.len().mean() > 30:
                text_cols.append(col)
            else:
                cat_cols.append(col)

    # 4. Feature Extraction - Dates
    tr_date_df = extract_date_features(X_tr_raw, date_cols)
    va_date_df = extract_date_features(X_va_raw, date_cols)
    te_date_df = extract_date_features(X_te_raw, date_cols)

    # 5. Feature Extraction - Text (TF-IDF + SVD)
    tr_text_feats, va_text_feats, te_text_feats = [], [], []

    for col in text_cols:
        tfidf = TfidfVectorizer(
            max_features=1000, stop_words="english", ngram_range=(1, 2)
        )
        svd = TruncatedSVD(n_components=10, random_state=42)

        tr_text_str = X_tr_raw[col].fillna("").astype(str)
        va_text_str = X_va_raw[col].fillna("").astype(str)
        te_text_str = X_te_raw[col].fillna("").astype(str)

        for df_str, feat_list in zip(
            [tr_text_str, va_text_str, te_text_str],
            [tr_text_feats, va_text_feats, te_text_feats],
        ):
            char_len = df_str.str.len().values.reshape(-1, 1)
            word_cnt = df_str.str.split().str.len().values.reshape(-1, 1)
            feat_list.extend([char_len, word_cnt])

        tr_tfidf = tfidf.fit_transform(tr_text_str)
        va_tfidf = tfidf.transform(va_text_str)
        te_tfidf = tfidf.transform(te_text_str)

        tr_svd = svd.fit_transform(tr_tfidf)
        va_svd = svd.transform(va_tfidf)
        te_svd = svd.transform(te_tfidf)

        tr_text_feats.append(tr_svd)
        va_text_feats.append(va_svd)
        te_text_feats.append(te_svd)

    tr_text_mat = concat_array_list(tr_text_feats, len(X_tr_raw))
    va_text_mat = concat_array_list(va_text_feats, len(X_va_raw))
    te_text_mat = concat_array_list(te_text_feats, len(X_te_raw))

    # 6. Feature Extraction - Categoricals (Freq + OOF Target Encoding + Label Encoding)
    tr_cat_feats = pd.DataFrame(index=X_tr_raw.index)
    va_cat_feats = pd.DataFrame(index=X_va_raw.index)
    te_cat_feats = pd.DataFrame(index=X_te_raw.index)

    for col in cat_cols:
        freq_map = X_tr_raw[col].value_counts(normalize=True).to_dict()
        tr_cat_feats[f"{col}_freq"] = X_tr_raw[col].map(freq_map).fillna(0)
        va_cat_feats[f"{col}_freq"] = X_va_raw[col].map(freq_map).fillna(0)
        te_cat_feats[f"{col}_freq"] = X_te_raw[col].map(freq_map).fillna(0)

        le = LabelEncoder()
        all_vals = (
            pd.concat([X_tr_raw[col], X_va_raw[col], X_te_raw[col]])
            .fillna("__missing__")
            .astype(str)
        )
        le.fit(all_vals)
        tr_cat_feats[f"{col}_le"] = le.transform(
            X_tr_raw[col].fillna("__missing__").astype(str)
        )
        va_cat_feats[f"{col}_le"] = le.transform(
            X_va_raw[col].fillna("__missing__").astype(str)
        )
        te_cat_feats[f"{col}_le"] = le.transform(
            X_te_raw[col].fillna("__missing__").astype(str)
        )

    # OOF Target Encoding
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    smoothing = 10.0

    for col in cat_cols:
        for c in range(num_classes):
            global_mean_c = np.mean(y_tr == c)
            binary_y = (y_tr == c).astype(float)
            col_te_tr = np.zeros(len(X_tr_raw))

            for fold_tr, fold_va in skf.split(X_tr_raw, y_tr):
                tr_sub = X_tr_raw.iloc[fold_tr]
                y_sub = pd.Series(binary_y[fold_tr], index=tr_sub.index)
                val_sub = X_tr_raw.iloc[fold_va]

                stats = y_sub.groupby(tr_sub[col]).agg(["count", "mean"])
                smooth_te = (
                    stats["count"] * stats["mean"] + smoothing * global_mean_c
                ) / (stats["count"] + smoothing)
                te_dict = smooth_te.to_dict()

                col_te_tr[fold_va] = val_sub[col].map(te_dict).fillna(global_mean_c)

            full_y_series = pd.Series(binary_y, index=X_tr_raw.index)
            full_stats = full_y_series.groupby(X_tr_raw[col]).agg(["count", "mean"])
            smooth_full = (
                full_stats["count"] * full_stats["mean"] + smoothing * global_mean_c
            ) / (full_stats["count"] + smoothing)
            full_te_dict = smooth_full.to_dict()

            tr_cat_feats[f"{col}_te_cls_{c}"] = col_te_tr
            va_cat_feats[f"{col}_te_cls_{c}"] = (
                X_va_raw[col].map(full_te_dict).fillna(global_mean_c)
            )
            te_cat_feats[f"{col}_te_cls_{c}"] = (
                X_te_raw[col].map(full_te_dict).fillna(global_mean_c)
            )

    # 7. Feature Extraction - Numerical & Imputation
    tr_num_df = X_tr_raw[num_cols].copy()
    va_num_df = X_va_raw[num_cols].copy()
    te_num_df = X_te_raw[num_cols].copy()

    for col in num_cols:
        if (
            tr_num_df[col].isna().sum() > 0
            or va_num_df[col].isna().sum() > 0
            or te_num_df[col].isna().sum() > 0
        ):
            tr_num_df[f"{col}_isna"] = tr_num_df[col].isna().astype(float)
            va_num_df[f"{col}_isna"] = va_num_df[col].isna().astype(float)
            te_num_df[f"{col}_isna"] = te_num_df[col].isna().astype(float)

    # Combine all feature DataFrames
    X_tr_full = pd.concat([tr_num_df, tr_date_df, tr_cat_feats], axis=1)
    X_va_full = pd.concat([va_num_df, va_date_df, va_cat_feats], axis=1)
    X_te_full = pd.concat([te_num_df, te_date_df, te_cat_feats], axis=1)

    if tr_text_mat.shape[1] > 0:
        text_cols_names = [f"text_feat_{i}" for i in range(tr_text_mat.shape[1])]
        X_tr_full = pd.concat(
            [
                X_tr_full,
                pd.DataFrame(
                    tr_text_mat, columns=text_cols_names, index=X_tr_full.index
                ),
            ],
            axis=1,
        )
        X_va_full = pd.concat(
            [
                X_va_full,
                pd.DataFrame(
                    va_text_mat, columns=text_cols_names, index=X_va_full.index
                ),
            ],
            axis=1,
        )
        X_te_full = pd.concat(
            [
                X_te_full,
                pd.DataFrame(
                    te_text_mat, columns=text_cols_names, index=X_te_full.index
                ),
            ],
            axis=1,
        )

    # Impute and Scale
    imputer = SimpleImputer(strategy="median")
    X_tr_array = imputer.fit_transform(X_tr_full)
    X_va_array = imputer.transform(X_va_full)
    X_te_array = imputer.transform(X_te_full)

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_array)
    X_va_scaled = scaler.transform(X_va_array)
    X_te_scaled = scaler.transform(X_te_array)

    feature_names = list(X_tr_full.columns)
    X_tr = pd.DataFrame(X_tr_scaled, columns=feature_names)
    X_va = pd.DataFrame(X_va_scaled, columns=feature_names)
    X_te = pd.DataFrame(X_te_scaled, columns=feature_names)

    # Save processed features to working directory
    X_tr.to_parquet("./working/X_train.parquet")
    X_va.to_parquet("./working/X_val.parquet")
    X_te.to_parquet("./working/X_test.parquet")
    pd.DataFrame({"availability": y_tr}).to_parquet("./working/y_train.parquet")
    pd.DataFrame({"availability": y_va}).to_parquet("./working/y_val.parquet")

    input_dim = X_tr.shape[1]

    # 8. Train LightGBM Classifier
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )
    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    joblib.dump(lgb_model, "./working/lgb_model.joblib")
    p_lgb_va = lgb_model.predict_proba(X_va)
    p_lgb_te = lgb_model.predict_proba(X_te)

    # 9. Train XGBoost Classifier
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=50,
        eval_metric="mlogloss" if num_classes > 2 else "logloss",
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    joblib.dump(xgb_model, "./working/xgb_model.joblib")
    p_xgb_va = xgb_model.predict_proba(X_va)
    p_xgb_te = xgb_model.predict_proba(X_te)

    # 10. Train CatBoost / HistGradientBoosting Classifier
    if HAS_CATBOOST:
        cb_model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.03,
            depth=6,
            random_seed=42,
            early_stopping_rounds=50,
            verbose=False,
        )
        cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        joblib.dump(cb_model, "./working/cb_model.joblib")
        p_cb_va = cb_model.predict_proba(X_va)
        p_cb_te = cb_model.predict_proba(X_te)
    else:
        hgb_model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=6, random_state=42
        )
        hgb_model.fit(X_tr, y_tr)
        joblib.dump(hgb_model, "./working/hgb_model.joblib")
        p_cb_va = hgb_model.predict_proba(X_va)
        p_cb_te = hgb_model.predict_proba(X_te)

    # 11. Train PyTorch Deep Tabular ResNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr_t = torch.tensor(X_tr.values, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    X_va_t = torch.tensor(X_va.values, dtype=torch.float32)
    y_va_t = torch.tensor(y_va, dtype=torch.long)
    X_te_t = torch.tensor(X_te.values, dtype=torch.float32)

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=False)

    nn_model = TabularResNet(
        input_dim=input_dim, num_classes=num_classes, hidden_dim=256, dropout=0.25
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35)

    best_val_loss = float("inf")
    best_weights_path = "./working/nn_best_model.pt"

    for epoch in range(35):
        nn_model.train()
        running_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = nn_model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(by)
        scheduler.step()

        nn_model.eval()
        with torch.no_grad():
            val_out = nn_model(X_va_t.to(device))
            val_loss = criterion(val_out, y_va_t.to(device)).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(nn_model.state_dict(), best_weights_path)

        if (epoch + 1) % 10 == 0 or epoch == 34:
            tr_epoch_loss = running_loss / len(train_ds)
            print(
                f"Epoch {epoch+1:02d}/35 - Train Loss: {tr_epoch_loss:.4f} - Val Loss: {val_loss:.4f}"
            )

    nn_model.load_state_dict(torch.load(best_weights_path))
    nn_model.eval()
    with torch.no_grad():
        p_nn_va = torch.softmax(nn_model(X_va_t.to(device)), dim=1).cpu().numpy()
        p_nn_te = torch.softmax(nn_model(X_te_t.to(device)), dim=1).cpu().numpy()

    # 12. Soft-Voting Ensemble Optimization (Nelder-Mead)
    models_va_probs = [p_lgb_va, p_xgb_va, p_cb_va, p_nn_va]
    models_te_probs = [p_lgb_te, p_xgb_te, p_cb_te, p_nn_te]

    def loss_func(weights):
        w = np.maximum(np.array(weights), 0)
        s = w.sum()
        w = w / s if s > 0 else np.ones_like(w) / len(w)
        blend_va = sum(wi * pi for wi, pi in zip(w, models_va_probs))
        preds = np.argmax(blend_va, axis=1)
        return -accuracy_score(y_va, preds)

    init_weights = [0.25] * len(models_va_probs)
    res = minimize(
        loss_func, init_weights, method="Nelder-Mead", options={"maxiter": 300}
    )

    best_w = np.maximum(res.x, 0)
    s = best_w.sum()
    best_w = best_w / s if s > 0 else np.ones_like(best_w) / len(best_w)

    blend_va = sum(w * p for w, p in zip(best_w, models_va_probs))
    final_val_preds = np.argmax(blend_va, axis=1)
    score = accuracy_score(y_va, final_val_preds)

    blend_te = sum(w * p for w, p in zip(best_w, models_te_probs))
    final_test_preds = np.argmax(blend_te, axis=1)

    # 13. Save Submission File
    target_classes = target_encoder.classes_
    final_test_labels = target_classes[final_test_preds]
    sub_df = pd.DataFrame({"id": test_ids, "availability": final_test_labels})
    sub_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()
