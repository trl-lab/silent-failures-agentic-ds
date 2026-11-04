import os
import json
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

INPUT_DIR = "./input"
WORK_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

TARGET = "Weekly_Sales"
ID_COL = "id"

SEED = 42
BATCH_SIZE = 8192
MAX_EPOCHS = 30
PATIENCE = 6
LR = 3e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 5.0

HIDDEN = 128
DROPOUT = 0.12
EMBED_DIM_SMALL = 8
EMBED_DIM_MEDIUM = 16
EMBED_DIM_LARGE = 24


def _read_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df


def _clean_raw(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    df = df.copy()

    if "Date" not in df.columns:
        raise ValueError("Missing required column: Date")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Store"] = pd.to_numeric(df.get("Store"), errors="coerce")
    df["Dept"] = pd.to_numeric(df.get("Dept"), errors="coerce")

    if "IsHoliday" in df.columns:
        df["IsHoliday"] = df["IsHoliday"].fillna(False).astype(bool).astype(int)
    else:
        df["IsHoliday"] = 0

    if is_train:
        df[TARGET] = pd.to_numeric(df.get(TARGET), errors="coerce")
        df = df.dropna(subset=["Date", "Store", "Dept", TARGET])
    else:
        df = df.dropna(subset=["Date", "Store", "Dept"])

    df["Store"] = df["Store"].astype(int)
    df["Dept"] = df["Dept"].astype(int)
    df["IsHoliday"] = df["IsHoliday"].astype(int)

    if is_train:
        df[TARGET] = df[TARGET].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["Store_Dept"] = df["Store"] * 100 + df["Dept"]
    if is_train:
        df = df.sort_values(["Store_Dept", "Date"], ignore_index=True)
    return df


def _safe_div(num: pd.Series, den: pd.Series, eps: float = 1e-8) -> pd.Series:
    return num / (den.abs() + eps)


def _add_time_features(df: pd.DataFrame, min_train_date: pd.Timestamp) -> pd.DataFrame:
    df = df.copy()

    df["year"] = df["Date"].dt.year.astype(int)
    df["month"] = df["Date"].dt.month.astype(int)
    df["weekofyear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["dayofweek"] = df["Date"].dt.dayofweek.astype(int)
    df["quarter"] = df["Date"].dt.quarter.astype(int)

    df["days_since_min_train"] = (df["Date"] - min_train_date).dt.days.astype(int)
    df["weeks_since_min_train"] = np.floor(df["days_since_min_train"] / 7.0).astype(int)

    df["is_month_start"] = df["Date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["Date"].dt.is_month_end.astype(int)
    df["is_quarter_start"] = df["Date"].dt.is_quarter_start.astype(int)
    df["is_quarter_end"] = df["Date"].dt.is_quarter_end.astype(int)
    df["is_year_start"] = df["Date"].dt.is_year_start.astype(int)
    df["is_year_end"] = df["Date"].dt.is_year_end.astype(int)

    week = 2 * np.pi * df["weekofyear"] / 52.0
    month = 2 * np.pi * df["month"] / 12.0
    dow = 2 * np.pi * df["dayofweek"] / 7.0

    df["week_sin"] = np.sin(week)
    df["week_cos"] = np.cos(week)
    df["month_sin"] = np.sin(month)
    df["month_cos"] = np.cos(month)
    df["dow_sin"] = np.sin(dow)
    df["dow_cos"] = np.cos(dow)

    df["holiday_week"] = df["IsHoliday"].astype(int)
    df["holiday_month"] = (df["holiday_week"] * df["month"]).astype(int)

    return df


def _fit_and_apply_stats(
    train: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    min_train_date: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    train = train.copy()

    global_mean = float(train[TARGET].mean())
    global_median = float(train[TARGET].median())
    global_std = float(train[TARGET].std(ddof=0))
    global_mean_abs = float(train[TARGET].abs().mean())
    global_min = float(train[TARGET].min())
    global_max = float(train[TARGET].max())

    stats = {
        "global_mean": global_mean,
        "global_median": global_median,
        "global_std": global_std,
        "global_mean_abs": global_mean_abs,
        "global_min": global_min,
        "global_max": global_max,
        "min_train_date": str(min_train_date.date()),
        "max_train_date": str(train["Date"].max().date()),
    }

    def _add_group_stats(key: str, suffix: str):
        g = (
            train.groupby(key, observed=True)[TARGET]
            .agg(
                **{
                    f"{suffix}_mean": "mean",
                    f"{suffix}_median": "median",
                    f"{suffix}_std": lambda x: x.std(ddof=0),
                    f"{suffix}_min": "min",
                    f"{suffix}_max": "max",
                    f"{suffix}_count": "count",
                }
            )
            .reset_index()
        )

        for frame_name, frame in frames.items():
            frames[frame_name] = frame.merge(g, on=key, how="left")

        for col in g.columns:
            if col == key:
                continue
            fill = (
                global_mean
                if "mean" in col or "median" in col
                else (global_std if "std" in col else 0.0)
            )
            for frame_name in frames:
                frames[frame_name][col] = frames[frame_name][col].fillna(fill)

    _add_group_stats("Store", "store")
    _add_group_stats("Dept", "dept")
    _add_group_stats("Store_Dept", "sd")

    train_h = train.assign(holiday_flag=train["IsHoliday"].astype(int))
    holiday_stats = (
        train_h.groupby(["Store_Dept", "holiday_flag"], observed=True)[TARGET]
        .agg(
            holiday_mean="mean",
            holiday_std=lambda x: x.std(ddof=0),
            holiday_count="count",
        )
        .reset_index()
    )

    pivot_mean = holiday_stats.pivot(
        index="Store_Dept", columns="holiday_flag", values="holiday_mean"
    ).reset_index()
    pivot_std = holiday_stats.pivot(
        index="Store_Dept", columns="holiday_flag", values="holiday_std"
    ).reset_index()
    pivot_count = holiday_stats.pivot(
        index="Store_Dept", columns="holiday_flag", values="holiday_count"
    ).reset_index()

    for col in [0, 1]:
        if col not in pivot_mean.columns:
            pivot_mean[col] = np.nan
        if col not in pivot_std.columns:
            pivot_std[col] = np.nan
        if col not in pivot_count.columns:
            pivot_count[col] = np.nan

    pivot_mean = pivot_mean.rename(
        columns={0: "sd_nonholiday_mean", 1: "sd_holiday_mean"}
    )
    pivot_std = pivot_std.rename(columns={0: "sd_nonholiday_std", 1: "sd_holiday_std"})
    pivot_count = pivot_count.rename(
        columns={0: "sd_nonholiday_count", 1: "sd_holiday_count"}
    )

    pivot_mean["sd_holiday_diff"] = (
        pivot_mean["sd_holiday_mean"] - pivot_mean["sd_nonholiday_mean"]
    )
    pivot_mean["sd_holiday_ratio"] = _safe_div(
        pivot_mean["sd_holiday_mean"], pivot_mean["sd_nonholiday_mean"]
    )

    for frame_name, frame in frames.items():
        frames[frame_name] = frame.merge(pivot_mean, on="Store_Dept", how="left")
        frames[frame_name] = frame.merge(pivot_std, on="Store_Dept", how="left")
        frames[frame_name] = frame.merge(pivot_count, on="Store_Dept", how="left")

    holiday_fill = {
        "sd_nonholiday_mean": global_mean,
        "sd_holiday_mean": global_mean,
        "sd_holiday_diff": 0.0,
        "sd_holiday_ratio": 1.0,
        "sd_nonholiday_std": global_std,
        "sd_holiday_std": global_std,
        "sd_nonholiday_count": 0.0,
        "sd_holiday_count": 0.0,
    }
    for frame_name in frames:
        frames[frame_name] = frames[frame_name].fillna(holiday_fill)

    seasonal_keys = [
        ("Store_Dept", "weekofyear", "sd_week"),
        ("Store_Dept", "month", "sd_month"),
        ("Dept", "weekofyear", "dept_week"),
        ("Store", "weekofyear", "store_week"),
        ("Store_Dept", "IsHoliday", "sd_holiday_flag"),
    ]

    for key1, key2, prefix in seasonal_keys:
        key = [key1, key2]
        g = (
            train.groupby(key, observed=True)[TARGET]
            .agg(
                **{
                    f"{prefix}_mean": "mean",
                    f"{prefix}_median": "median",
                    f"{prefix}_std": lambda x: x.std(ddof=0),
                    f"{prefix}_count": "count",
                }
            )
            .reset_index()
        )

        for frame_name, frame in frames.items():
            frames[frame_name] = frame.merge(g, on=key, how="left")

        for col in g.columns:
            if col in key:
                continue
            fill = (
                global_mean
                if "mean" in col or "median" in col
                else (global_std if "std" in col else 0.0)
            )
            for frame_name in frames:
                frames[frame_name][col] = frames[frame_name][col].fillna(fill)

    # These residual features would require target values at inference time.
    # Keep them as zero constants so validation/test remain leakage-free.
    for frame_name, frame in frames.items():
        frame["target_vs_sd_mean"] = 0.0
        frame["target_vs_store_mean"] = 0.0
        frame["target_vs_dept_mean"] = 0.0

    with open(os.path.join(WORK_DIR, "feature_metadata.json"), "w") as f:
        json.dump(stats, f, indent=2)

    return frames


def _select_features(train: pd.DataFrame) -> list[str]:
    exclude = {"Date", TARGET, ID_COL}
    feature_cols = [c for c in train.columns if c not in exclude]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train[c])]
    feature_cols = [c for c in feature_cols if train[c].nunique(dropna=True) > 1]
    return feature_cols


def _wmae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = float(np.sum(np.abs(y_true)))
    if denom <= 1e-8:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def _set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WMaeLoss(nn.Module):
    """
    Task-faithful WMAE loss.

    Official metric:
        WMAE = sum(|y - yhat|) / sum(|y|)

    For a fixed training set, the denominator is constant, so minimizing this
    loss is equivalent to minimizing the official numerator while preserving
    the official metric semantics.
    """

    def __init__(self, denominator: float | None = None):
        super().__init__()
        self.denominator = denominator

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        abs_error_sum = torch.abs(pred - target).sum()
        if self.denominator is not None and self.denominator > 1e-8:
            denom = torch.tensor(
                float(self.denominator), device=pred.device, dtype=pred.dtype
            )
        else:
            denom = target.abs().sum().clamp_min(1e-8)
        return abs_error_sum / denom


class BaseResidualTabularMLP(nn.Module):
    """
    Hierarchical base + residual architecture.

    - Base term: learnable scale/offset applied to a leakage-safe store-dept
      baseline feature, usually `sd_mean`.
    - Residual term: MLP with categorical embeddings and scaled numeric features.
    - The residual head is zero-initialized, so the model starts as the
      store-dept baseline and learns corrections during training.
    """

    def __init__(
        self,
        numeric_feature_names: list[str],
        categorical_feature_names: list[str],
        cat_cardinalities: dict[str, int],
        base_feature_name: str | None,
        hidden: int = HIDDEN,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        self.numeric_feature_names = list(numeric_feature_names)
        self.categorical_feature_names = list(categorical_feature_names)
        self.base_feature_name = base_feature_name

        n_numeric = len(self.numeric_feature_names)
        if n_numeric <= 0:
            raise ValueError("At least one numeric feature is required.")

        if (
            base_feature_name is not None
            and base_feature_name not in self.numeric_feature_names
        ):
            raise ValueError(
                f"Base feature {base_feature_name} not found in numeric features."
            )

        self.base_idx = (
            self.numeric_feature_names.index(base_feature_name)
            if base_feature_name is not None
            else None
        )

        self.base_scale = nn.Parameter(torch.tensor(1.0))
        self.base_offset = nn.Parameter(torch.tensor(0.0))

        self.embeddings = nn.ModuleDict()
        total_embed_dim = 0
        for name in self.categorical_feature_names:
            card = int(cat_cardinalities.get(name, 2))
            if name == "Store_Dept":
                dim = EMBED_DIM_LARGE
            elif name in {"Store", "Dept"}:
                dim = EMBED_DIM_MEDIUM
            else:
                dim = EMBED_DIM_SMALL
            emb = nn.Embedding(max(2, card), dim)
            nn.init.normal_(emb.weight, mean=0.0, std=0.05)
            self.embeddings[name] = emb
            total_embed_dim += dim

        if total_embed_dim == 0:
            total_embed_dim = 1
            self.embeddings["none"] = nn.Embedding(2, 1)

        self.num_encoder = nn.Sequential(
            nn.Linear(n_numeric, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.cat_encoder = nn.Sequential(
            nn.Linear(total_embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self, x_numeric: torch.Tensor, x_categorical: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if self.base_idx is not None:
            base = self.base_scale * x_numeric[:, self.base_idx] + self.base_offset
        else:
            base = self.base_offset.expand(x_numeric.size(0))

        num_hidden = self.num_encoder(x_numeric)

        cat_tensors = [
            self.embeddings[name](x_categorical[name])
            for name in self.categorical_feature_names
        ]
        if cat_tensors:
            cat_vec = torch.cat(cat_tensors, dim=1)
        else:
            cat_vec = torch.zeros(x_numeric.size(0), 1, device=x_numeric.device)
        cat_hidden = self.cat_encoder(cat_vec)

        residual = self.head(torch.cat([num_hidden, cat_hidden], dim=1)).squeeze(-1)
        return base + residual


def _build_scaler_fit_on_train(train_df: pd.DataFrame, numeric_features: list[str]):
    X = train_df[numeric_features].astype(float)
    mean = X.mean(axis=0).to_numpy(dtype=np.float32)
    std = X.std(axis=0, ddof=0).to_numpy(dtype=np.float32)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def _scale_features(
    df: pd.DataFrame, numeric_features: list[str], mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    X = df[numeric_features].astype(float).to_numpy(dtype=np.float32)
    return ((X - mean) / std).astype(np.float32)


def _prepare_tensors(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    base_feature: str | None,
):
    mean, std = _build_scaler_fit_on_train(train_df, numeric_features)

    X_train_num = _scale_features(train_df, numeric_features, mean, std)
    X_val_num = _scale_features(val_df, numeric_features, mean, std)
    X_test_num = _scale_features(test_df, numeric_features, mean, std)

    if base_feature is None:
        base_train = np.zeros(len(train_df), dtype=np.float32)
        base_val = np.zeros(len(val_df), dtype=np.float32)
        base_test = np.zeros(len(test_df), dtype=np.float32)
    else:
        base_train = train_df[base_feature].astype(float).to_numpy(dtype=np.float32)
        base_val = val_df[base_feature].astype(float).to_numpy(dtype=np.float32)
        base_test = test_df[base_feature].astype(float).to_numpy(dtype=np.float32)

    if base_feature is not None and base_feature in numeric_features:
        idx = numeric_features.index(base_feature)
        X_train_num[:, idx] = base_train
        X_val_num[:, idx] = base_val
        X_test_num[:, idx] = base_test

    y_train = train_df[TARGET].astype(float).to_numpy(dtype=np.float32)
    y_val = val_df[TARGET].astype(float).to_numpy(dtype=np.float32)

    def _cat_dict(df: pd.DataFrame) -> dict[str, torch.Tensor]:
        out = {}
        for col in categorical_features:
            out[col] = torch.from_numpy(
                df[col].astype(np.int64).to_numpy(dtype=np.int64)
            )
        return out

    cat_train = _cat_dict(train_df)
    cat_val = _cat_dict(val_df)
    cat_test = _cat_dict(test_df)

    return (
        torch.from_numpy(X_train_num),
        torch.from_numpy(X_val_num),
        torch.from_numpy(X_test_num),
        torch.from_numpy(y_train),
        torch.from_numpy(y_val),
        cat_train,
        cat_val,
        cat_test,
        mean,
        std,
    )


def _make_dataloader(
    x_num: torch.Tensor,
    y: torch.Tensor,
    cat: dict[str, torch.Tensor],
    shuffle: bool,
    batch_size: int,
):
    dataset = TensorDataset(x_num, y, *cat.values())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _predict(
    model: nn.Module,
    x_num: torch.Tensor,
    cat: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int = 8192,
) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, x_num.size(0), batch_size):
            end = min(start + batch_size, x_num.size(0))
            xb = x_num[start:end].to(device, non_blocking=True)
            cb = {k: v[start:end].to(device, non_blocking=True) for k, v in cat.items()}
            preds.append(model(xb, cb).cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.zeros(0, dtype=np.float32)


def _train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for x_num, y, *cat_list in loader:
        x_num = x_num.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        cat = {
            name: t.to(device, non_blocking=True)
            for name, t in zip(model.categorical_feature_names, cat_list)
        }

        optimizer.zero_grad(set_to_none=True)
        pred = model(x_num, cat)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()

        total_loss += float(loss.item()) * y.size(0)
        n += y.size(0)
    return total_loss / max(1, n)


def main():
    _set_seed()

    train_raw = _read_parquet(os.path.join(INPUT_DIR, "train.parquet"))
    test_raw = _read_parquet(os.path.join(INPUT_DIR, "test.parquet"))

    train_raw = _clean_raw(train_raw, is_train=True)
    test_raw = _clean_raw(test_raw, is_train=False)

    # The required submission id is the 0-based integer row index of test.
    test_raw[ID_COL] = np.arange(len(test_raw), dtype=int)

    unique_train_dates = np.sort(train_raw["Date"].unique())
    n_unique_dates = len(unique_train_dates)
    val_weeks = max(4, int(round(0.10 * n_unique_dates)))
    cutoff = pd.Timestamp(
        unique_train_dates[-val_weeks]
        if n_unique_dates >= val_weeks
        else unique_train_dates[0]
    )

    train_mask = train_raw["Date"] < cutoff
    val_mask = train_raw["Date"] >= cutoff

    train_df = train_raw[train_mask].copy().reset_index(drop=True)
    val_df = train_raw[val_mask].copy().reset_index(drop=True)
    test_df = test_raw.copy().reset_index(drop=True)

    if len(train_df) == 0 or len(val_df) == 0:
        fallback_cutoff = unique_train_dates[int(0.95 * n_unique_dates)]
        train_mask = train_raw["Date"] < fallback_cutoff
        val_mask = train_raw["Date"] >= fallback_cutoff
        train_df = train_raw[train_mask].copy().reset_index(drop=True)
        val_df = train_raw[val_mask].copy().reset_index(drop=True)

    min_train_date = train_df["Date"].min()

    train_df = _add_time_features(train_df, min_train_date)
    val_df = _add_time_features(val_df, min_train_date)
    test_df = _add_time_features(test_df, min_train_date)

    frames = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }
    frames = _fit_and_apply_stats(train_df, frames, min_train_date)

    feature_cols = _select_features(frames["train"])
    categorical_features = [
        c for c in ["Store", "Dept", "Store_Dept", "IsHoliday"] if c in feature_cols
    ]
    numeric_features = [c for c in feature_cols if c not in categorical_features]

    if len(numeric_features) == 0:
        raise ValueError("No numeric features available for modeling.")

    preferred_base = ["sd_mean", "store_mean", "dept_mean", "global_mean"]
    base_feature = None
    for name in preferred_base:
        if name in numeric_features:
            base_feature = name
            break
    if base_feature is None:
        base_feature = numeric_features[0]

    cat_cardinalities = {}
    for col in categorical_features:
        max_id = int(max(train_df[col].max(), val_df[col].max(), test_df[col].max()))
        cat_cardinalities[col] = max_id + 2

    (
        X_train_num,
        X_val_num,
        X_test_num,
        y_train,
        y_val,
        cat_train,
        cat_val,
        cat_test,
        scaler_mean,
        scaler_std,
    ) = _prepare_tensors(
        train_df=frames["train"],
        val_df=frames["val"],
        test_df=frames["test"],
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        base_feature=base_feature,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BaseResidualTabularMLP(
        numeric_feature_names=numeric_features,
        categorical_feature_names=categorical_features,
        cat_cardinalities=cat_cardinalities,
        base_feature_name=base_feature,
        hidden=HIDDEN,
        dropout=DROPOUT,
    ).to(device)

    train_abs_sum = float(y_train.abs().sum().item())
    criterion = WMaeLoss(denominator=train_abs_sum).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, MAX_EPOCHS),
        eta_min=1e-5,
    )

    train_loader = _make_dataloader(
        X_train_num,
        y_train,
        cat_train,
        shuffle=True,
        batch_size=BATCH_SIZE,
    )

    best_val_score = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        val_pred = _predict(model, X_val_num, cat_val, device)
        val_pred = np.nan_to_num(val_pred, nan=0.0, posinf=0.0, neginf=0.0)
        val_score = _wmae(y_val.numpy(), val_pred)

        print(
            f"epoch {epoch}/{MAX_EPOCHS} train_wmae_loss={train_loss:.6f} val_wmae={val_score:.6f}"
        )

        if val_score < best_val_score - 1e-7:
            best_val_score = val_score
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break

    model.load_state_dict(best_state)

    val_pred = _predict(model, X_val_num, cat_val, device)
    test_pred = _predict(model, X_test_num, cat_test, device)

    val_pred = np.asarray(val_pred, dtype=float)
    test_pred = np.asarray(test_pred, dtype=float)

    val_pred = np.nan_to_num(val_pred, nan=0.0, posinf=0.0, neginf=0.0)
    test_pred = np.nan_to_num(test_pred, nan=0.0, posinf=0.0, neginf=0.0)

    score = _wmae(y_val.numpy(), val_pred)

    submission = pd.DataFrame(
        {
            ID_COL: frames["test"][ID_COL].astype(int).to_numpy(),
            TARGET: test_pred,
        }
    )
    submission.to_csv(os.path.join(SUBMISSION_DIR, "submission.csv"), index=False)

    with open(os.path.join(WORK_DIR, "split_info.json"), "w") as f:
        json.dump(
            {
                "split": "time-based holdout from training dates",
                "cutoff": str(cutoff.date()),
                "val_weeks": int(val_weeks),
                "n_train": int(len(frames["train"])),
                "n_val": int(len(frames["val"])),
                "n_test": int(len(frames["test"])),
                "target": TARGET,
                "id_col": ID_COL,
            },
            f,
            indent=2,
        )

    with open(os.path.join(WORK_DIR, "model_info.json"), "w") as f:
        json.dump(
            {
                "architecture": "BaseResidualTabularMLP",
                "base_feature": base_feature,
                "numeric_features": numeric_features,
                "categorical_features": categorical_features,
                "cat_cardinalities": cat_cardinalities,
                "loss": "WMaeLoss",
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingLR",
                "hidden": HIDDEN,
                "dropout": DROPOUT,
                "batch_size": BATCH_SIZE,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "best_val_wmae": float(best_val_score),
                "scaler_mean": scaler_mean.tolist(),
                "scaler_std": scaler_std.tolist(),
            },
            f,
            indent=2,
        )

    torch.save(
        {
            "state_dict": best_state,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "cat_cardinalities": cat_cardinalities,
            "base_feature": base_feature,
            "hidden": HIDDEN,
            "dropout": DROPOUT,
            "scaler_mean": scaler_mean,
            "scaler_std": scaler_std,
            "best_val_wmae": float(best_val_score),
            "target": TARGET,
            "id_col": ID_COL,
        },
        os.path.join(WORK_DIR, "best_model.pt"),
    )

    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()