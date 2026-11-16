import os
import json
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

SEED = 42
INPUT_DIR = "./input"
WORK_DIR = "./working"
SUBMISSION_DIR = "./submission"

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Data processing and feature engineering
# ---------------------------------------------------------------------------
def _find_target_column(df: pd.DataFrame) -> str:
    priority = ["class", "target", "label", "outcome"]
    lower_to_col = {str(c).lower(): c for c in df.columns}
    for name in priority:
        if name in lower_to_col:
            return lower_to_col[name]
    raise ValueError(
        "Could not identify target column. Expected one of: class, target, label, outcome."
    )


def _find_id_columns(df: pd.DataFrame) -> set:
    id_names = {
        "id",
        "index",
        "row_id",
        "sample_id",
        "identifier",
        "uid",
        "uid_1",
        "uid_2",
    }
    return {c for c in df.columns if str(c).lower() in id_names}


def _find_raw_feature_columns(df: pd.DataFrame, target_col: str) -> list:
    id_cols = _find_id_columns(df)
    feature_cols = []
    for col in df.columns:
        if col == target_col or col in id_cols:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any():
            feature_cols.append(col)
    return feature_cols


def _locate_named_column(df: pd.DataFrame, candidates: list):
    lower_to_col = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        if name in lower_to_col:
            return lower_to_col[name]
    return None


def _safe_divide(
    numerator: pd.Series, denominator: pd.Series, fill: float = 0.0
) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return (numerator / denominator).fillna(fill)


def _make_quantile_binned_feature(
    train_values: pd.Series, value: pd.Series, name: str, q: int = 5
) -> pd.Series:
    clean_train = train_values.dropna()
    if clean_train.nunique() < 2 or len(clean_train) < max(10, q):
        return pd.Series(0, index=value.index, name=name, dtype="int64")

    try:
        _, edges = pd.qcut(clean_train, q=q, duplicates="drop", retbins=True)
    except Exception:
        return pd.Series(0, index=value.index, name=name, dtype="int64")

    if len(edges) < 2:
        return pd.Series(0, index=value.index, name=name, dtype="int64")

    codes = np.searchsorted(edges, value.to_numpy(dtype="float64"), side="right") - 1
    codes = np.clip(codes, 0, len(edges) - 1)
    return pd.Series(codes, index=value.index, name=name, dtype="int64")


def _engineer_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
) -> tuple:
    """
    Fit stateful helpers on train_df only, then transform val_df and test_df.
    """

    def _col(frame: pd.DataFrame, candidates: list):
        return _locate_named_column(frame, candidates)

    recency_train = _col(train_df, ["recency", "v1"])
    frequency_train = _col(train_df, ["frequency", "v2"])
    monetary_train = _col(train_df, ["monetary", "v3"])
    time_train = _col(train_df, ["time", "v4", "tenure"])

    if recency_train is None and len(feature_cols) >= 1:
        recency_train = feature_cols[0]
    if frequency_train is None and len(feature_cols) >= 2:
        frequency_train = feature_cols[1]
    if monetary_train is None and len(feature_cols) >= 3:
        monetary_train = feature_cols[2]
    if time_train is None and len(feature_cols) >= 4:
        time_train = feature_cols[3]

    def _as_float(frame: pd.DataFrame, train_col):
        if train_col is None or train_col not in frame.columns:
            return pd.Series(np.nan, index=frame.index, name=train_col or "missing")
        return pd.to_numeric(frame[train_col], errors="coerce")

    bin_specs = []
    for train_col in [recency_train, frequency_train, monetary_train, time_train]:
        if train_col is not None and train_col in train_df.columns:
            bin_specs.append((train_col, f"{train_col}_quantile_bin"))

    def _transform(frame: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)

        for col in feature_cols:
            if col in frame.columns:
                out[col] = pd.to_numeric(frame[col], errors="coerce")

        recency = _as_float(frame, recency_train)
        frequency = _as_float(frame, frequency_train)
        monetary = _as_float(frame, monetary_train)
        time = _as_float(frame, time_train)

        out["recency_log1p"] = np.log1p(recency.clip(lower=0.0))
        out["recency_inv"] = 1.0 / (1.0 + recency.clip(lower=0.0))
        out["recency_decay_3m"] = np.exp(-recency.clip(lower=0.0) / 3.0)
        out["recency_decay_6m"] = np.exp(-recency.clip(lower=0.0) / 6.0)
        out["recency_mod_3"] = (recency.clip(lower=0.0) % 3).fillna(0.0)
        out["is_recent_1m"] = (recency <= 1.0).astype(float).fillna(0.0)
        out["is_recent_3m"] = (recency <= 3.0).astype(float).fillna(0.0)

        out["frequency_log1p"] = np.log1p(frequency.clip(lower=0.0))
        out["frequency_sqrt"] = np.sqrt(frequency.clip(lower=0.0))
        out["is_single_donation"] = (frequency == 1.0).astype(float).fillna(0.0)
        out["is_multi_donation"] = (frequency > 1.0).astype(float).fillna(0.0)

        out["time_log1p"] = np.log1p(time.clip(lower=0.0))
        out["time_sqrt"] = np.sqrt(time.clip(lower=0.0))

        out["tenure_span"] = (time - recency).clip(lower=0.0)
        out["donation_rate_per_month"] = _safe_divide(frequency, time + 1.0, fill=0.0)
        out["avg_interval_between_donations"] = _safe_divide(
            out["tenure_span"],
            (frequency - 1.0).clip(lower=0.0),
            fill=0.0,
        )
        out["time_per_donation"] = _safe_divide(time, frequency, fill=0.0)

        out["monetary_log1p"] = np.log1p(monetary.clip(lower=0.0))
        out["monetary_per_donation"] = _safe_divide(monetary, frequency, fill=0.0)
        out["monetary_per_month"] = _safe_divide(monetary, time + 1.0, fill=0.0)
        out["expected_monetary_250cc"] = frequency * 250.0
        out["monetary_residual_250cc"] = monetary - out["expected_monetary_250cc"]

        out["freq_over_recency"] = _safe_divide(frequency, 1.0 + recency, fill=0.0)
        out["log_freq_times_recency_inv"] = out["frequency_log1p"] * out["recency_inv"]
        out["log_time_minus_recency"] = np.log1p(out["tenure_span"])
        out["rate_times_recency_inv"] = (
            out["donation_rate_per_month"] * out["recency_inv"]
        )
        out["recency_rank_proxy"] = out["recency_inv"] * (1.0 + out["frequency_log1p"])

        for train_col, bin_name in bin_specs:
            train_values = _as_float(train_df, train_col)
            values = _as_float(frame, train_col)
            out[bin_name] = _make_quantile_binned_feature(
                train_values, values, bin_name
            )

        return out

    X_train = _transform(train_df)
    X_val = _transform(val_df)
    X_test = _transform(test_df)

    common_cols = [
        c for c in X_train.columns if c in X_val.columns and c in X_test.columns
    ]
    X_train = X_train[common_cols]
    X_val = X_val[common_cols]
    X_test = X_test[common_cols]

    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_val = pd.DataFrame(
        imputer.transform(X_val),
        columns=X_val.columns,
        index=X_val.index,
    )
    X_test = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    keep_cols = []
    for col in X_train.columns:
        values = X_train[col].to_numpy(dtype="float64")
        if np.nanstd(values) > 1e-12 and len(np.unique(values)) > 1:
            keep_cols.append(col)

    X_train = X_train[keep_cols]
    X_val = X_val[keep_cols]
    X_test = X_test[keep_cols]

    if "Monetary" in X_train.columns and "Frequency" in X_train.columns:
        corr = np.corrcoef(
            X_train["Monetary"].to_numpy(), X_train["Frequency"].to_numpy()
        )[0, 1]
        if np.isfinite(corr) and abs(corr) > 0.9995:
            drop_cols = [
                c for c in ["Monetary", "monetary_per_donation"] if c in X_train.columns
            ]
            X_train = X_train.drop(columns=drop_cols)
            X_val = X_val.drop(columns=drop_cols)
            X_test = X_test.drop(columns=drop_cols)

    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_val = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index,
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return X_train, X_val, X_test


def prepare_data():
    train_path = os.path.join(INPUT_DIR, "train.parquet")
    test_path = os.path.join(INPUT_DIR, "test.parquet")

    train_raw = pd.read_parquet(train_path)
    test_raw = pd.read_parquet(test_path)

    target_col = _find_target_column(train_raw)
    if target_col in test_raw.columns:
        test_raw = test_raw.drop(columns=[target_col])

    feature_cols = _find_raw_feature_columns(train_raw, target_col)
    if len(feature_cols) == 0:
        raise ValueError("No numeric feature columns found in training data.")

    feature_cols = [c for c in feature_cols if c in test_raw.columns]
    if len(feature_cols) == 0:
        raise ValueError("No shared numeric feature columns between train and test.")

    target_values = pd.to_numeric(train_raw[target_col], errors="coerce")
    if target_values.isna().any():
        raise ValueError("Target column contains missing or non-numeric values.")

    unique_target = np.sort(target_values.dropna().unique())
    if len(unique_target) < 2:
        raise ValueError("Target must be binary with at least two observed classes.")

    if set(unique_target).issubset({0, 1}):
        positive_label = 1
    elif set(unique_target).issubset({1, 2}):
        positive_label = 2
    else:
        positive_label = int(unique_target[-1])

    y_raw = (target_values == positive_label).astype(int)

    stratify = y_raw if y_raw.nunique() == 2 else None
    train_idx, val_idx = train_test_split(
        train_raw.index,
        test_size=0.20,
        random_state=SEED,
        stratify=stratify,
    )

    train_split = train_raw.loc[train_idx, feature_cols].reset_index(drop=True)
    val_split = train_raw.loc[val_idx, feature_cols].reset_index(drop=True)
    test_split = test_raw[feature_cols].reset_index(drop=True)

    X_train, X_val, X_test = _engineer_features(
        train_split,
        val_split,
        test_split,
        feature_cols,
    )

    y_train = y_raw.loc[train_idx].reset_index(drop=True)
    y_val = y_raw.loc[val_idx].reset_index(drop=True)

    X_all = pd.concat([X_train, X_val], axis=0, ignore_index=True)
    y_all = pd.concat([y_train, y_val], axis=0, ignore_index=True)

    metadata = {
        "seed": SEED,
        "target_column": target_col,
        "positive_original_label": int(positive_label),
        "negative_original_label": (
            int(1 - positive_label)
            if positive_label in {0, 1}
            else int(unique_target[0])
        ),
        "internal_target_encoding": "1 = positive/donated, 0 = negative/not donated",
        "feature_columns": list(X_train.columns),
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "test_size": int(len(X_test)),
        "split_strategy": "stratified 80/20 on labeled train data before feature fitting",
        "stateful_transforms_fitted_on": "train split only",
    }
    with open(os.path.join(WORK_DIR, "data_processing_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return X_train, y_train, X_val, y_val, X_test, X_all, y_all


# ---------------------------------------------------------------------------
# Model and training utilities
# ---------------------------------------------------------------------------
def _set_deterministic():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _as_float_tensor(x, is_target=False):
    if x is None:
        return None

    if isinstance(x, torch.Tensor):
        t = x.detach().clone()
    elif isinstance(x, pd.DataFrame):
        t = torch.tensor(x.to_numpy(dtype=np.float32))
    elif isinstance(x, pd.Series):
        t = torch.tensor(x.to_numpy(dtype=np.float32).reshape(-1, 1))
    else:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1 and not is_target:
            arr = arr.reshape(1, -1)
        t = torch.tensor(arr)

    t = t.float()
    t[~torch.isfinite(t)] = 0.0

    if is_target:
        t = t.reshape(-1, 1)
    return t


def _finite_tensor(x):
    if x is None:
        return None
    x = x.clone()
    x[~torch.isfinite(x)] = 0.0
    return x


def _binary_target(y):
    y = y.reshape(-1).float()
    vals = np.unique(y.detach().cpu().numpy())
    if vals.size == 0:
        return y
    if set(vals.tolist()).issubset({0.0, 1.0}):
        return y
    if set(vals.tolist()).issubset({1.0, 2.0}):
        return (y == 2.0).float()
    positive = float(vals.max())
    return (y == positive).float()


def _align_features(reference, x):
    if x is None:
        return None
    if x.shape[1] == reference.shape[1]:
        return x
    if x.shape[1] > reference.shape[1]:
        return x[:, : reference.shape[1]]
    pad = torch.zeros((x.shape[0], reference.shape[1] - x.shape[1]), dtype=x.dtype)
    return torch.cat([x, pad], dim=1)


def _safe_roc_auc(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)
    y_prob = np.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0)
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_prob))


def _postprocess_probs(p):
    p = np.asarray(p, dtype=float).reshape(-1)
    p = np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(p, 0.0, 1.0)


class DonorMLP(nn.Module):
    """
    Small regularized MLP for the tiny imbalanced RFMTC donor-response task.
    """

    def __init__(self, input_dim: int, hidden_width: int = 48, dropout_p: float = 0.25):
        super().__init__()
        h = max(8, int(hidden_width))
        self.net = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.BatchNorm1d(h),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(h, max(8, h // 2)),
            nn.BatchNorm1d(max(8, h // 2)),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(max(8, h // 2), 12),
            nn.GELU(),
            nn.Dropout(dropout_p * 0.5),
            nn.Linear(12, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _make_model(input_dim: int):
    return DonorMLP(input_dim=input_dim, hidden_width=48, dropout_p=0.25)


def _make_criterion(y: torch.Tensor):
    y = y.reshape(-1)
    pos = float(y.sum().clamp(min=1.0))
    neg = float((1.0 - y).sum().clamp(min=1.0))
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def _make_optimizer(model: nn.Module):
    return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)


def _make_scheduler(optimizer, max_epochs):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(max_epochs)),
        eta_min=1e-5,
    )


def _train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for xb, yb in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(1, n_batches)


def _predict_proba(model, x):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
    return _postprocess_probs(probs)


def _train_and_select(
    X_train,
    y_train,
    X_val,
    y_val,
    max_epochs=350,
    patience=35,
    batch_size=64,
):
    model = _make_model(X_train.shape[1])
    criterion = _make_criterion(y_train)
    optimizer = _make_optimizer(model)
    scheduler = _make_scheduler(optimizer, max_epochs)

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    y_val_np = y_val.detach().cpu().numpy().reshape(-1).astype(int)

    best_auc = -1.0
    best_epoch = 1
    best_state = None
    bad_epochs = 0
    history = []

    for epoch in range(1, int(max_epochs) + 1):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer)
        scheduler.step()

        val_prob = _predict_proba(model, X_val)
        val_auc = _safe_roc_auc(y_val_np, val_prob)

        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_auc": float(val_auc),
            }
        )

        print(f"epoch={epoch} loss={train_loss:.4f} val_auc={val_auc:.4f}")

        if val_auc > best_auc + 1e-6:
            best_auc = val_auc
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is None:
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        best_auc = _safe_roc_auc(y_val_np, _predict_proba(model, X_val))

    model.load_state_dict(best_state)
    return model, best_state, best_epoch, best_auc, history


def _refit_on_all(
    X_all,
    y_all,
    fit_epochs,
    batch_size=64,
):
    fit_epochs = max(1, int(fit_epochs))

    model = _make_model(X_all.shape[1])
    criterion = _make_criterion(y_all)
    optimizer = _make_optimizer(model)
    scheduler = _make_scheduler(optimizer, fit_epochs)

    train_ds = TensorDataset(X_all, y_all)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    for _ in range(fit_epochs):
        _train_one_epoch(model, train_loader, criterion, optimizer)
        scheduler.step()

    return model


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    _set_deterministic()

    (
        X_train_df,
        y_train_series,
        X_val_df,
        y_val_series,
        X_test_df,
        X_all_df,
        y_all_series,
    ) = prepare_data()

    X_train = _as_float_tensor(X_train_df)
    y_train = _as_float_tensor(y_train_series, is_target=True)
    X_val = _as_float_tensor(X_val_df)
    y_val = _as_float_tensor(y_val_series, is_target=True)
    X_test = _as_float_tensor(X_test_df)
    X_all = _as_float_tensor(X_all_df)
    y_all = _as_float_tensor(y_all_series, is_target=True)

    X_train = _finite_tensor(X_train)
    X_val = _finite_tensor(X_val)
    X_test = _finite_tensor(X_test)
    X_all = _finite_tensor(X_all)

    if X_train is None or y_train is None or X_val is None or y_val is None:
        raise FileNotFoundError(
            "Required processed train/validation tensors are missing."
        )
    if X_test is None:
        raise FileNotFoundError("Required processed test tensor is missing.")

    y_train = _binary_target(y_train)
    y_val = _binary_target(y_val)
    y_all = _binary_target(y_all)

    X_val = _align_features(X_train, X_val)
    X_test = _align_features(X_train, X_test)
    X_all = _align_features(X_train, X_all)

    if X_all.shape[0] != y_all.shape[0]:
        X_all = torch.cat([X_train, X_val], dim=0)
        y_all = torch.cat([y_train, y_val], dim=0)
        y_all = _binary_target(y_all)

    val_model, best_state, best_epoch, best_val_auc, history = _train_and_select(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        max_epochs=350,
        patience=35,
        batch_size=64,
    )

    final_val_prob = _predict_proba(val_model, X_val)
    final_val_true = y_val.detach().cpu().numpy().reshape(-1).astype(int)
    score = _safe_roc_auc(final_val_true, final_val_prob)

    final_model = _refit_on_all(
        X_all=X_all,
        y_all=y_all,
        fit_epochs=best_epoch,
        batch_size=64,
    )

    test_prob = _predict_proba(final_model, X_test)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_prob), dtype=int),
            "Class": test_prob.astype(float),
        }
    )
    submission.to_csv(os.path.join(SUBMISSION_DIR, "submission.csv"), index=False)

    try:
        audit = {
            "model": "DonorMLP",
            "loss": "class-weighted BCEWithLogitsLoss",
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "selected_epoch": int(best_epoch),
            "validation_metric": "roc_auc",
            "validation_score": float(score),
            "test_prediction_source": "refit model forward pass on X_test",
            "history": history,
        }
        with open(os.path.join(WORK_DIR, "training_evaluation_audit.json"), "w") as f:
            json.dump(audit, f, indent=2)
    except Exception:
        pass

    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()
