import json
import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# -------------------------------------------------------------------
# Reproducibility Setup
# -------------------------------------------------------------------
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


seed_everything(42)


# -------------------------------------------------------------------
# Data Processing and Feature Engineering
# -------------------------------------------------------------------
def process_data():
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = "class"
    if target_col not in train_df.columns:
        for col in train_df.columns:
            if col.lower() in ["class", "target", "label"]:
                target_col = col
                break

    train_targets_raw = train_df[target_col].copy()

    id_col = None
    for possible_id in ["id", "ID", "index"]:
        if possible_id in test_df.columns:
            id_col = possible_id
            break

    train_features_raw = train_df.drop(
        columns=[target_col]
        + ([id_col] if id_col and id_col in train_df.columns else [])
    )
    test_features_raw = test_df.drop(
        columns=([id_col] if id_col and id_col in test_df.columns else [])
    )

    column_mapping = {
        col: col.strip().lower().replace(" ", "_").replace("-", "_")
        for col in train_features_raw.columns
    }
    train_features_raw = train_features_raw.rename(columns=column_mapping)
    test_features_raw = test_features_raw.rename(columns=column_mapping)

    feature_cols = list(train_features_raw.columns)

    def add_geometric_features(df):
        d = df.copy()
        eps = 1e-6

        if "pr_axis_aspect_ratio" in d and "max_length_aspect_ratio" in d:
            norm_aspect = d["pr_axis_aspect_ratio"] / (
                d["max_length_aspect_ratio"] + eps
            )
            d["eccentricity"] = np.sqrt(np.maximum(0.0, 1.0 - norm_aspect**2))
            d["aspect_ratio_diff"] = (
                d["pr_axis_aspect_ratio"] - d["max_length_aspect_ratio"]
            )

        if "max_length_rectangularity" in d and "hollows_ratio" in d:
            d["bounding_poly_ratio"] = d["max_length_rectangularity"] / (
                1.0 - d["hollows_ratio"] + eps
            )

        if (
            "scaled_variance_along_major_axis" in d
            and "scaled_variance_along_minor_axis" in d
        ):
            d["moment_variance_ratio"] = d["scaled_variance_along_major_axis"] / (
                d["scaled_variance_along_minor_axis"] + eps
            )
            d["variance_sum"] = (
                d["scaled_variance_along_major_axis"]
                + d["scaled_variance_along_minor_axis"]
            )
            d["variance_diff"] = (
                d["scaled_variance_along_major_axis"]
                - d["scaled_variance_along_minor_axis"]
            )

        if (
            "skewness_about_major_axis" in d
            and "skewness_about_minor_axis" in d
        ):
            d["skewness_norm"] = np.sqrt(
                d["skewness_about_major_axis"] ** 2
                + d["skewness_about_minor_axis"] ** 2
            )
            d["skewness_prod"] = (
                d["skewness_about_major_axis"] * d["skewness_about_minor_axis"]
            )

        if (
            "kurtosis_about_major_axis" in d
            and "kurtosis_about_minor_axis" in d
        ):
            d["kurtosis_norm"] = np.sqrt(
                d["kurtosis_about_major_axis"] ** 2
                + d["kurtosis_about_minor_axis"] ** 2
            )
            d["kurtosis_sum"] = (
                d["kurtosis_about_major_axis"] + d["kurtosis_about_minor_axis"]
            )

        if "compactness" in d and "circularity" in d:
            d["compactness_circularity_ratio"] = d["compactness"] / (
                d["circularity"] + eps
            )

        if "pr_axis_rectangularity" in d and "max_length_rectangularity" in d:
            d["rectangularity_diff"] = (
                d["pr_axis_rectangularity"] - d["max_length_rectangularity"]
            )

        if "hollows_ratio" in d:
            d["solidness_ratio"] = 1.0 - d["hollows_ratio"]

        if "elongatedness" in d and "scatter_ratio" in d:
            d["elongatedness_scatter_ratio"] = d["elongatedness"] / (
                d["scatter_ratio"] + eps
            )

        return d

    train_eng = add_geometric_features(train_features_raw)
    test_eng = add_geometric_features(test_features_raw)

    train_eng = train_eng.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    test_eng = test_eng.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    le = LabelEncoder()
    train_y = le.fit_transform(train_targets_raw)
    class_mapping = {int(i): str(cls_) for i, cls_ in enumerate(le.classes_)}

    train_processed = train_eng.copy()
    test_processed = test_eng.copy()

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    train_processed["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_processed, train_y)):
        train_processed.loc[val_idx, "fold"] = fold

    train_processed["target"] = train_y

    if id_col and id_col in test_df.columns:
        test_processed["id"] = test_df[id_col].values
    else:
        test_processed["id"] = np.arange(len(test_processed))

    final_features = list(train_eng.columns)

    return train_processed, test_processed, final_features, class_mapping


# -------------------------------------------------------------------
# Model Architecture: PyTorch Tabular ResNet with SE-Attention & MSD
# -------------------------------------------------------------------
class SEBlock(nn.Module):

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 8), bias=False),
            nn.SiLU(),
            nn.Linear(max(channels // reduction, 8), channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x)
        return x * w


class TabularResBlock(nn.Module):

    def __init__(self, in_features, hidden_dim, dropout_rate=0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, in_features),
            nn.LayerNorm(in_features),
        )
        self.se = SEBlock(in_features)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out = self.se(out)
        return F.silu(residual + out)


class MultiSampleDropout(nn.Module):

    def __init__(self, in_features, out_features, num_dropouts=4, dropout_rate=0.2):
        super().__init__()
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(num_dropouts)]
        )
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return torch.mean(
            torch.stack([self.fc(drop(x)) for drop in self.dropouts], dim=0), dim=0
        )


class TabularResNet(nn.Module):

    def __init__(self, input_dim, num_classes, hidden_dim=128, num_blocks=3):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()
        )
        self.blocks = nn.ModuleList(
            [TabularResBlock(hidden_dim, hidden_dim) for _ in range(num_blocks)]
        )
        self.head = MultiSampleDropout(
            hidden_dim, num_classes, num_dropouts=4, dropout_rate=0.2
        )

    def forward(self, x):
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


# -------------------------------------------------------------------
# PyTorch Training Loop per Fold
# -------------------------------------------------------------------
def train_pytorch_fold(
    X_tr, y_tr, X_va, y_va, X_te, num_classes, epochs=80, batch_size=32
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.long),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_va, dtype=torch.float32),
        torch.tensor(y_va, dtype=torch.long),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_tensor = torch.tensor(X_te, dtype=torch.float32).to(device)

    model = TabularResNet(input_dim=X_tr.shape[1], num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    best_val_loss = float("inf")
    best_val_probs = None
    best_test_probs = None

    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_preds = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                loss = criterion(out, by)
                val_loss += loss.item() * len(by)
                probs = F.softmax(out, dim=1).cpu().numpy()
                val_preds.append(probs)

        val_loss /= len(y_va)
        val_probs = np.vstack(val_preds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_probs = val_probs

            with torch.no_grad():
                test_out = model(test_tensor)
                best_test_probs = F.softmax(test_out, dim=1).cpu().numpy()

    return best_val_probs, best_test_probs


# -------------------------------------------------------------------
# Execution Pipeline
# -------------------------------------------------------------------
def main():
    sub_dir = "./submission"
    os.makedirs(sub_dir, exist_ok=True)

    train_df, test_df, feature_cols, class_mapping = process_data()

    num_classes = len(class_mapping)

    X_all = train_df[feature_cols].values.astype(np.float32)
    y_all = train_df["target"].values.astype(int)
    folds = train_df["fold"].values.astype(int)

    X_test = test_df[feature_cols].values.astype(np.float32)

    num_samples = len(train_df)
    num_test_samples = len(test_df)

    oof_nn = np.zeros((num_samples, num_classes))
    oof_lgb = np.zeros((num_samples, num_classes))
    oof_xgb = np.zeros((num_samples, num_classes))
    oof_cat = np.zeros((num_samples, num_classes))
    oof_et = np.zeros((num_samples, num_classes))
    oof_rf = np.zeros((num_samples, num_classes))

    test_nn = np.zeros((num_test_samples, num_classes))
    test_lgb = np.zeros((num_test_samples, num_classes))
    test_xgb = np.zeros((num_test_samples, num_classes))
    test_cat = np.zeros((num_test_samples, num_classes))
    test_et = np.zeros((num_test_samples, num_classes))
    test_rf = np.zeros((num_test_samples, num_classes))

    n_folds = len(np.unique(folds))

    for f in range(n_folds):
        tr_idx = np.where(folds != f)[0]
        va_idx = np.where(folds == f)[0]

        scaler = RobustScaler()

        X_tr_raw = X_all[tr_idx]
        X_va_raw = X_all[va_idx]
        X_te_raw = X_test

        X_tr = scaler.fit_transform(X_tr_raw)
        X_va = scaler.transform(X_va_raw)
        X_te_fold = scaler.transform(X_te_raw)

        y_tr = y_all[tr_idx]
        y_va = y_all[va_idx]

        val_probs_nn, test_probs_nn = train_pytorch_fold(
            X_tr, y_tr, X_va, y_va, X_te_fold, num_classes
        )
        oof_nn[va_idx] = val_probs_nn
        test_nn += test_probs_nn / n_folds

        lgb_model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=num_classes,
            metric="multi_logloss",
            boosting_type="gbdt",
            num_leaves=15,
            learning_rate=0.03,
            feature_fraction=0.7,
            min_child_samples=10,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_estimators=300,
            random_state=42 + f,
            verbose=-1,
        )
        lgb_model.fit(X_tr, y_tr)
        oof_lgb[va_idx] = lgb_model.predict_proba(X_va)
        test_lgb += lgb_model.predict_proba(X_te_fold) / n_folds

        xgb_model = XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_estimators=300,
            random_state=42 + f,
            verbosity=0,
        )
        xgb_model.fit(X_tr, y_tr)
        oof_xgb[va_idx] = xgb_model.predict_proba(X_va)
        test_xgb += xgb_model.predict_proba(X_te_fold) / n_folds

        cat_model = CatBoostClassifier(
            loss_function="MultiClass",
            depth=5,
            learning_rate=0.03,
            l2_leaf_reg=3.0,
            iterations=300,
            random_seed=42 + f,
            verbose=False,
        )
        cat_model.fit(X_tr, y_tr)
        oof_cat[va_idx] = cat_model.predict_proba(X_va)
        test_cat += cat_model.predict_proba(X_te_fold) / n_folds

        et_model = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=4,
            random_state=42 + f,
            n_jobs=-1,
        )
        et_model.fit(X_tr, y_tr)
        oof_et[va_idx] = et_model.predict_proba(X_va)
        test_et += et_model.predict_proba(X_te_fold) / n_folds

        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=4,
            random_state=42 + f,
            n_jobs=-1,
        )
        rf_model.fit(X_tr, y_tr)
        oof_rf[va_idx] = rf_model.predict_proba(X_va)
        test_rf += rf_model.predict_proba(X_te_fold) / n_folds

    X_meta_oof = np.hstack([oof_nn, oof_lgb, oof_xgb, oof_cat, oof_et, oof_rf])
    X_meta_test = np.hstack([test_nn, test_lgb, test_xgb, test_cat, test_et, test_rf])

    meta_model = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
    meta_model.fit(X_meta_oof, y_all)

    oof_preds = meta_model.predict(X_meta_oof)
    final_score = accuracy_score(y_all, oof_preds)

    test_probs = meta_model.predict_proba(X_meta_test)
    test_pred_indices = np.argmax(test_probs, axis=1)
    test_pred_labels = [
        class_mapping.get(str(idx), class_mapping.get(int(idx)))
        for idx in test_pred_indices
    ]

    submission_df = pd.DataFrame({"id": test_df["id"], "class": test_pred_labels})
    sub_path = os.path.join(sub_dir, "submission.csv")
    submission_df.to_csv(sub_path, index=False)

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()