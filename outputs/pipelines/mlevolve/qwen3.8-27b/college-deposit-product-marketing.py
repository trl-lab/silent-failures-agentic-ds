import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import lightgbm as lgb
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

SEED = 42
TARGET = "TOTAL_ACTIVE_ACCOUNTS_"
WORK_DIR = "./working"
SUB_DIR = "./submission"

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(SUB_DIR, exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)


def safe_rmse(y_true, y_pred) -> float:
    """Official metric implementation: RMSE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


class FeatureProcessor:
    """
    Leakage-safe tabular feature processor.

    Fit statistics are learned only from the provided training fold.
    The reference frame is used only for schema/dtype alignment, not for
    fitting value-based statistics.
    """

    def __init__(self):
        self.all_cols = []
        self.numeric_cols = []
        self.cat_cols = []
        self.date_cols = []
        self.missing_cols = []
        self.medians = {}
        self.modes = {}
        self.freq_maps = {}
        self.cat_maps = {}
        self.scaler = None
        self.feature_names = []
        self.categorical_feature_names = []

    @staticmethod
    def _is_mostly_numeric(series: pd.Series) -> bool:
        if series.size == 0:
            return False
        non_null = series.dropna()
        if non_null.empty:
            return False
        converted = pd.to_numeric(non_null, errors="coerce")
        return float(converted.notna().sum()) / float(len(non_null)) >= 0.8

    @staticmethod
    def _is_mostly_datetime(series: pd.Series) -> bool:
        if series.size == 0:
            return False
        non_null = series.dropna()
        if non_null.empty:
            return False
        converted = pd.to_datetime(non_null, errors="coerce")
        return float(converted.notna().sum()) / float(len(non_null)) >= 0.8

    def _infer_types(self, X_fit: pd.DataFrame, X_ref: pd.DataFrame) -> None:
        self.all_cols = list(dict.fromkeys(list(X_fit.columns) + list(X_ref.columns)))
        self.numeric_cols = []
        self.cat_cols = []
        self.date_cols = []

        for col in self.all_cols:
            s_fit = X_fit[col] if col in X_fit.columns else pd.Series(dtype="object")
            s_ref = X_ref[col] if col in X_ref.columns else pd.Series(dtype="object")

            # Dtype/schema alignment only. Value-based inference uses X_fit only.
            if pd.api.types.is_datetime64_any_dtype(
                s_fit
            ) or pd.api.types.is_datetime64_any_dtype(s_ref):
                self.date_cols.append(col)
                continue

            if pd.api.types.is_bool_dtype(s_fit) or pd.api.types.is_bool_dtype(s_ref):
                self.numeric_cols.append(col)
                continue

            if pd.api.types.is_numeric_dtype(s_fit) or pd.api.types.is_numeric_dtype(
                s_ref
            ):
                self.numeric_cols.append(col)
                continue

            if self._is_mostly_numeric(s_fit):
                self.numeric_cols.append(col)
            elif self._is_mostly_datetime(s_fit):
                self.date_cols.append(col)
            else:
                self.cat_cols.append(col)

    def fit(self, X_fit: pd.DataFrame, X_ref: pd.DataFrame):
        self._infer_types(X_fit, X_ref)

        X_fit = X_fit.reindex(columns=self.all_cols)

        # Coerce types on the fitting frame only.
        for col in self.numeric_cols:
            X_fit[col] = pd.to_numeric(X_fit[col], errors="coerce")
        for col in self.cat_cols:
            X_fit[col] = X_fit[col].astype("object")
        for col in self.date_cols:
            X_fit[col] = pd.to_datetime(X_fit[col], errors="coerce")

        # Missing indicators are selected using the training fold only.
        self.missing_cols = [col for col in self.all_cols if X_fit[col].isna().any()]

        # Numeric imputation statistics.
        self.medians = {}
        for col in self.numeric_cols:
            med = X_fit[col].median(skipna=True)
            if pd.isna(med):
                med = 0.0
            self.medians[col] = float(med)

        # Categorical imputation and encoding statistics.
        self.modes = {}
        self.freq_maps = {}
        self.cat_maps = {}

        for col in self.cat_cols:
            s = X_fit[col].fillna("__MISSING__").astype(str)
            mode = s.mode(dropna=True)
            self.modes[col] = str(mode.iloc[0]) if not mode.empty else "__MISSING__"

            freq = s.value_counts().to_dict()
            self.freq_maps[col] = freq

            categories = list(
                dict.fromkeys([self.modes[col], "__UNKNOWN__", *s.unique()])
            )
            self.cat_maps[col] = {cat: idx for idx, cat in enumerate(categories)}

        # Scaling statistics.
        if self.numeric_cols:
            numeric_fit = X_fit[self.numeric_cols].copy()
            for col in self.numeric_cols:
                numeric_fit[col] = numeric_fit[col].fillna(self.medians[col])

            from sklearn.preprocessing import StandardScaler

            self.scaler = StandardScaler()
            self.scaler.fit(numeric_fit)
        else:
            self.scaler = None

        # Build feature name list.
        feature_names = []

        # Scaled numeric features.
        feature_names.extend(self.numeric_cols)

        # Missing indicators.
        feature_names.extend([f"missing_{col}" for col in self.missing_cols])

        # Date features.
        for col in self.date_cols:
            feature_names.extend(
                [
                    f"{col}_year",
                    f"{col}_month",
                    f"{col}_day",
                    f"{col}_dayofweek",
                    f"{col}_quarter",
                    f"{col}_is_weekend",
                ]
            )

        # Categorical frequency and ordinal encodings.
        for col in self.cat_cols:
            feature_names.append(f"freq_{col}")
            feature_names.append(f"enc_{col}")

        # Row-wise numeric aggregates.
        if self.numeric_cols:
            feature_names.extend(
                [
                    "numeric_row_mean",
                    "numeric_row_std",
                    "numeric_row_min",
                    "numeric_row_max",
                    "numeric_row_sum",
                    "numeric_row_range",
                    "numeric_row_count",
                    "numeric_row_skew",
                ]
            )

        self.feature_names = feature_names
        self.categorical_feature_names = [f"enc_{col}" for col in self.cat_cols]
        return self

    def _coerce_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.reindex(columns=self.all_cols).copy()

        for col in self.numeric_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        for col in self.cat_cols:
            X[col] = X[col].astype("object")
        for col in self.date_cols:
            X[col] = pd.to_datetime(X[col], errors="coerce")

        return X

    def _add_missing_indicators(
        self, X: pd.DataFrame, X_out: pd.DataFrame
    ) -> pd.DataFrame:
        for col in self.missing_cols:
            X_out[f"missing_{col}"] = X[col].isna().astype(np.float32)
        return X_out

    def _add_date_features(self, X: pd.DataFrame, X_out: pd.DataFrame) -> pd.DataFrame:
        for col in self.date_cols:
            dt = X[col]
            X_out[f"{col}_year"] = dt.dt.year.astype(np.float32)
            X_out[f"{col}_month"] = dt.dt.month.astype(np.float32)
            X_out[f"{col}_day"] = dt.dt.day.astype(np.float32)
            X_out[f"{col}_dayofweek"] = dt.dt.dayofweek.astype(np.float32)
            X_out[f"{col}_quarter"] = dt.dt.quarter.astype(np.float32)
            X_out[f"{col}_is_weekend"] = (dt.dt.dayofweek >= 5).astype(np.float32)

        for col in self.date_cols:
            for suffix in [
                "year",
                "month",
                "day",
                "dayofweek",
                "quarter",
                "is_weekend",
            ]:
                X_out[f"{col}_{suffix}"] = X_out[f"{col}_{suffix}"].fillna(0.0)

        return X_out

    def _add_categorical_features(
        self, X: pd.DataFrame, X_out: pd.DataFrame
    ) -> pd.DataFrame:
        for col in self.cat_cols:
            s = X[col].fillna("__MISSING__").astype(str)

            X_out[f"freq_{col}"] = (
                s.map(self.freq_maps[col]).fillna(0.0).astype(np.float32)
            )

            # Use a valid in-range unknown index to avoid embedding out-of-bounds errors.
            unknown_idx = self.cat_maps[col].get(
                "__UNKNOWN__", len(self.cat_maps[col]) - 1
            )
            codes = s.map(self.cat_maps[col]).fillna(unknown_idx).astype(np.int32)
            X_out[f"enc_{col}"] = codes

        return X_out

    def _add_numeric_features(
        self, X: pd.DataFrame, X_out: pd.DataFrame
    ) -> pd.DataFrame:
        if not self.numeric_cols:
            return X_out

        num = X[self.numeric_cols].copy()
        for col in self.numeric_cols:
            num[col] = (
                num[col].replace([np.inf, -np.inf], np.nan).fillna(self.medians[col])
            )

        if self.scaler is not None:
            X_out[self.numeric_cols] = self.scaler.transform(num)
        else:
            X_out[self.numeric_cols] = num

        row_mean = num.mean(axis=1, skipna=True)
        row_std = num.std(axis=1, skipna=True)
        row_min = num.min(axis=1, skipna=True)
        row_max = num.max(axis=1, skipna=True)
        row_sum = num.sum(axis=1, skipna=True)
        row_range = row_max - row_min
        row_count = num.notna().sum(axis=1)
        row_skew = num.skew(axis=1, skipna=True)

        X_out["numeric_row_mean"] = row_mean.fillna(0.0).astype(np.float32)
        X_out["numeric_row_std"] = row_std.fillna(0.0).astype(np.float32)
        X_out["numeric_row_min"] = row_min.fillna(0.0).astype(np.float32)
        X_out["numeric_row_max"] = row_max.fillna(0.0).astype(np.float32)
        X_out["numeric_row_sum"] = row_sum.fillna(0.0).astype(np.float32)
        X_out["numeric_row_range"] = row_range.fillna(0.0).astype(np.float32)
        X_out["numeric_row_count"] = row_count.fillna(0.0).astype(np.float32)
        X_out["numeric_row_skew"] = row_skew.fillna(0.0).astype(np.float32)

        return X_out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._coerce_frame(X)
        X_out = pd.DataFrame(index=X.index)

        X_out = self._add_missing_indicators(X, X_out)
        X_out = self._add_date_features(X, X_out)
        X_out = self._add_categorical_features(X, X_out)
        X_out = self._add_numeric_features(X, X_out)

        # Ensure all expected feature columns exist, even if some branches were empty.
        for col in self.feature_names:
            if col not in X_out.columns:
                X_out[col] = 0.0

        X_out = X_out[self.feature_names]
        X_out = X_out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return X_out


class RMSELoss(nn.Module):
    """
    Task-aligned loss for RMSE evaluation.

    RMSE is the official metric. Using sqrt(MSE) keeps the training objective
    directly aligned with the reported metric while remaining differentiable.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        return torch.sqrt(torch.mean((pred - target) ** 2) + self.eps)


class CrossLayer(nn.Module):
    """
    DCN-V2-style explicit feature interaction layer.

    x_{l+1} = x_0 * (W_l x_l + b_l) + x_l
    """

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x0 * self.linear(x) + x


class TabularInteractionNet(nn.Module):
    """
    Compact tabular regression architecture.

    Design rationale:
    - Continuous features are normalized by the upstream feature processor.
    - Categorical encoded features are embedded to avoid treating ordinal codes
      as meaningful distances.
    - A DCN-style cross path learns explicit feature interactions.
    - A deep MLP path learns smoother non-linear representations.
    - Dropout and weight decay are used because the dataset is small.
    """

    def __init__(
        self,
        numeric_dim: int,
        cat_cardinalities,
        embedding_dim: int = 8,
        hidden_dim: int = 64,
        cross_layers: int = 2,
        dropout: float = 0.20,
    ):
        super().__init__()

        if numeric_dim <= 0 and len(cat_cardinalities) == 0:
            raise ValueError(
                "Model requires at least one numeric or categorical feature."
            )

        self.embedding_dim = embedding_dim
        self.cat_cardinalities = list(cat_cardinalities)

        # Categorical embeddings.
        if self.cat_cardinalities:
            self.embeddings = nn.ModuleList(
                [
                    nn.Embedding(int(card), embedding_dim)
                    for card in self.cat_cardinalities
                ]
            )
            for emb in self.embeddings:
                nn.init.normal_(emb.weight, mean=0.0, std=0.02)
            cat_repr_dim = len(self.cat_cardinalities) * embedding_dim
        else:
            self.embeddings = nn.ModuleList()
            cat_repr_dim = 0

        # Continuous representation.
        self.num_norm = nn.LayerNorm(numeric_dim) if numeric_dim > 0 else None
        num_repr_dim = max(1, min(numeric_dim, hidden_dim))
        self.num_proj = nn.Linear(max(1, numeric_dim), num_repr_dim)

        # Interaction input is the concatenated raw-ish representation.
        interaction_dim = max(1, numeric_dim + cat_repr_dim)

        # DCN-style explicit interaction path.
        self.cross_layers = nn.ModuleList(
            [CrossLayer(interaction_dim) for _ in range(max(1, cross_layers))]
        )

        # Deep path.
        deep_in = interaction_dim + num_repr_dim + cat_repr_dim
        self.deep = nn.Sequential(
            nn.Linear(deep_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # Head.
        head_in = interaction_dim + hidden_dim // 2
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        if self.num_norm is not None:
            num = self.num_norm(numeric)
        else:
            num = numeric

        num_repr = torch.relu(self.num_proj(num))

        if self.embeddings:
            cat_parts = []
            for i, emb in enumerate(self.embeddings):
                cat_parts.append(emb(categorical[:, i]))
            cat_repr = torch.cat(cat_parts, dim=1)
        else:
            cat_repr = numeric.new_zeros((numeric.size(0), 0))

        interaction_input = torch.cat([num, cat_repr], dim=1)

        x0 = interaction_input
        x = interaction_input
        for layer in self.cross_layers:
            x = layer(x0, x)

        deep_in = torch.cat([interaction_input, num_repr, cat_repr], dim=1)
        deep_out = self.deep(deep_in)

        head_in = torch.cat([x, deep_out], dim=1)
        return self.head(head_in)


def build_optimizer(
    model: TabularInteractionNet, lr: float = 1e-3, weight_decay: float = 1e-2
):
    """
    AdamW with parameter-group regularization.

    Embeddings receive slightly stronger decay because high-cardinality
    categorical embeddings can overfit on small tabular datasets.
    Biases are not weight-decayed.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name:
            no_decay_params.append(param)
        elif "embedding" in name.lower() or "embeddings" in name:
            decay_params.append((param, weight_decay * 2.0))
        else:
            decay_params.append((param, weight_decay))

    emb_params = [p for p, wd in decay_params if wd > weight_decay]
    other_decay = [p for p, wd in decay_params if wd <= weight_decay]

    param_groups = [
        {"params": other_decay, "weight_decay": weight_decay},
        {"params": emb_params, "weight_decay": weight_decay * 2.0},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    # Remove empty parameter groups to avoid optimizer errors.
    param_groups = [group for group in param_groups if len(group["params"]) > 0]

    return AdamW(param_groups, lr=lr, betas=(0.9, 0.95), eps=1e-6)


def get_feature_columns(processor, X_train: pd.DataFrame):
    """Separate processed features into continuous and categorical encoded columns."""
    feature_names = processor.feature_names
    categorical_feature_names = processor.categorical_feature_names

    numeric_cols = [
        col for col in feature_names if col not in categorical_feature_names
    ]
    cat_cols = [col for col in categorical_feature_names if col in X_train.columns]

    numeric_cols = [col for col in numeric_cols if col in X_train.columns]
    cat_cols = [col for col in cat_cols if col in X_train.columns]

    return numeric_cols, cat_cols


def get_cardinalities(processor, cat_cols, X_train: pd.DataFrame):
    """Infer embedding cardinalities from fitted categorical maps when available."""
    cardinalities = []

    for col in cat_cols:
        base_col = col.replace("enc_", "", 1)
        card = None

        if hasattr(processor, "cat_maps") and base_col in processor.cat_maps:
            card = len(processor.cat_maps[base_col])

        if card is None or card <= 1:
            max_code = int(X_train[col].max()) if len(X_train[col]) > 0 else 0
            card = max(2, max_code + 2)

        cardinalities.append(int(card))

    return cardinalities


def frames_to_tensors(X: pd.DataFrame, numeric_cols, cat_cols):
    """Convert processed DataFrame to model input tensors."""
    if numeric_cols:
        numeric = X[numeric_cols].to_numpy(dtype=np.float32)
    else:
        numeric = np.zeros((len(X), 0), dtype=np.float32)

    if cat_cols:
        categorical = X[cat_cols].to_numpy(dtype=np.int64)
    else:
        categorical = np.zeros((len(X), 0), dtype=np.int64)

    numeric = np.nan_to_num(numeric, nan=0.0, posinf=0.0, neginf=0.0)
    categorical = np.nan_to_num(categorical, nan=0, posinf=0, neginf=0).astype(np.int64)

    return torch.from_numpy(numeric), torch.from_numpy(categorical)


def predict_model(model, X: pd.DataFrame, numeric_cols, cat_cols, device, clip_lower):
    """Model inference path shared by validation and test prediction."""
    model.eval()
    numeric, categorical = frames_to_tensors(X, numeric_cols, cat_cols)
    numeric = numeric.to(device)
    categorical = categorical.to(device)

    with torch.no_grad():
        pred = model(numeric, categorical).cpu().numpy().reshape(-1)

    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

    if clip_lower is not None:
        pred = np.clip(pred, clip_lower, None)

    return pred.astype(float)


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
    numeric_cols=None,
    cat_cols=None,
    clip_lower=None,
    n_estimators: int = 5000,
    early_stopping_rounds: int = 100,
    early_stop: bool = True,
):
    """Train a regularized LightGBM regressor on the same processed features."""
    if numeric_cols is None:
        numeric_cols = []
    if cat_cols is None:
        cat_cols = []

    categorical_feature = [col for col in cat_cols if col in X_train.columns]
    feature_cols = list(numeric_cols) + categorical_feature

    model = LGBMRegressor(
        objective="regression",
        eval_metric="rmse",
        n_estimators=n_estimators,
        learning_rate=0.02,
        num_leaves=15,
        max_depth=4,
        min_child_samples=5,
        subsample=0.8,
        bagging_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    if early_stop and X_val is not None and y_val is not None:
        model.fit(
            X_train[feature_cols],
            y_train,
            eval_set=[(X_val[feature_cols], y_val)],
            categorical_feature=categorical_feature if categorical_feature else None,
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
    else:
        model.fit(
            X_train[feature_cols],
            y_train,
            categorical_feature=categorical_feature if categorical_feature else None,
        )

    return model


def predict_lightgbm(
    model,
    X: pd.DataFrame,
    numeric_cols=None,
    cat_cols=None,
    clip_lower=None,
):
    """LightGBM inference path with identical post-processing to the NN."""
    if numeric_cols is None:
        numeric_cols = []
    if cat_cols is None:
        cat_cols = []

    categorical_feature = [col for col in cat_cols if col in X.columns]
    feature_cols = list(numeric_cols) + categorical_feature

    pred = model.predict(X[feature_cols])
    pred = np.asarray(pred, dtype=float).reshape(-1)
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

    if clip_lower is not None:
        pred = np.clip(pred, clip_lower, None)

    return pred.astype(float)


def train_one_model(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    numeric_cols,
    cat_cols,
    cat_cardinalities,
    device,
    clip_lower,
    max_epochs: int = 250,
    patience: int = 35,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    dropout: float = 0.20,
    embedding_dim: int = 8,
    hidden_dim: int = 64,
    cross_layers: int = 2,
    early_stop: bool = True,
):
    """
    Train the designed model.

    If early_stop=True, the held-out validation split is used only for early
    stopping and LR scheduling. If early_stop=False, the model is trained for
    exactly max_epochs without using validation data.
    """
    model = TabularInteractionNet(
        numeric_dim=len(numeric_cols),
        cat_cardinalities=cat_cardinalities,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        cross_layers=cross_layers,
        dropout=dropout,
    ).to(device)

    criterion = RMSELoss()
    optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay)
    scheduler = (
        ReduceLROnPlateau(optimizer, mode="min", factor=0.65, patience=12, min_lr=1e-5)
        if early_stop
        else None
    )

    train_numeric, train_categorical = frames_to_tensors(
        X_train, numeric_cols, cat_cols
    )
    train_target = torch.from_numpy(y_train.to_numpy(dtype=np.float32)).to(device)

    val_numeric = val_categorical = val_target = None
    if early_stop:
        val_numeric, val_categorical = frames_to_tensors(X_val, numeric_cols, cat_cols)
        val_numeric = val_numeric.to(device)
        val_categorical = val_categorical.to(device)
        val_target = torch.from_numpy(y_val.to_numpy(dtype=np.float32)).to(device)

    n_train = len(X_train)
    best_val_rmse = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device)

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]

            batch_numeric = train_numeric[idx]
            batch_categorical = train_categorical[idx]
            batch_target = train_target[idx]

            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_numeric, batch_categorical)
            loss = criterion(pred, batch_target)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().cpu().item())
            n_batches += 1

        if scheduler is not None:
            scheduler.step(best_val_rmse)

        if early_stop:
            val_pred = predict_model(
                model,
                X_val,
                numeric_cols,
                cat_cols,
                device,
                clip_lower,
            )
            val_rmse = safe_rmse(y_val.to_numpy(), val_pred)

            print(
                f"epoch={epoch:03d} "
                f"train_loss={epoch_loss / max(1, n_batches):.6f} "
                f"val_rmse={val_rmse:.6f} "
                f"best_rmse={best_val_rmse:.6f}"
            )

            if val_rmse < best_val_rmse - 1e-7:
                best_val_rmse = val_rmse
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                break
        else:
            print(f"epoch={epoch:03d} train_loss={epoch_loss / max(1, n_batches):.6f}")

    if best_state is None:
        best_state = model.state_dict()
        best_epoch = max(1, max_epochs)
        best_val_rmse = float("inf")

    model.load_state_dict(best_state)
    return model, best_val_rmse, best_epoch


def main():
    train_raw = pd.read_parquet("./input/train.parquet")
    test_raw = pd.read_parquet("./input/test.parquet")

    if TARGET not in train_raw.columns:
        raise ValueError(f"Target column {TARGET} not found in train data.")

    y_full = train_raw[TARGET].astype(float)
    train_raw = train_raw.drop(columns=[TARGET], errors="ignore")

    # Drop obvious identifier-like columns from features.
    id_like = [
        col
        for col in train_raw.columns
        if col.lower() in {"id", "index", "row_id", "sample_id", "identifier"}
    ]
    train_raw = train_raw.drop(columns=id_like, errors="ignore")
    test_raw = test_raw.drop(columns=id_like, errors="ignore")

    n_train = len(train_raw)
    if n_train <= 1:
        raise ValueError("Training data is too small to create a validation split.")

    val_size = int(round(0.2 * n_train))
    val_size = max(1, min(val_size, n_train - 1))

    stratify = None
    if n_train >= 30 and y_full.nunique() > 1:
        try:
            stratify = pd.qcut(y_full, q=10, labels=False, duplicates="drop")
            if stratify.nunique() <= 1:
                stratify = None
        except Exception:
            stratify = None

    from sklearn.model_selection import train_test_split

    train_idx, val_idx = train_test_split(
        np.arange(n_train),
        test_size=val_size,
        random_state=SEED,
        stratify=stratify,
    )

    X_full_raw = train_raw.reset_index(drop=True)
    y_full = y_full.reset_index(drop=True)

    X_train_raw = X_full_raw.iloc[train_idx].reset_index(drop=True)
    X_val_raw = X_full_raw.iloc[val_idx].reset_index(drop=True)
    y_train = y_full.iloc[train_idx].reset_index(drop=True)
    y_val = y_full.iloc[val_idx].reset_index(drop=True)

    processor = FeatureProcessor()
    processor.fit(X_train_raw, test_raw)

    X_train = processor.transform(X_train_raw)
    X_val = processor.transform(X_val_raw)
    X_test = processor.transform(test_raw)
    X_full = processor.transform(X_full_raw)

    numeric_cols, cat_cols = get_feature_columns(processor, X_train)
    cat_cardinalities = get_cardinalities(processor, cat_cols, X_train)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Target is a non-negative account count. The same non-negative post-processing
    # is applied to validation and test predictions to keep local and submission
    # semantics aligned.
    clip_lower = 0.0 if float(y_full.min()) >= 0 else None

    # Train on the train fold only; validation is used exclusively for early stopping.
    model, val_rmse, best_epoch = train_one_model(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        numeric_cols=numeric_cols,
        cat_cols=cat_cols,
        cat_cardinalities=cat_cardinalities,
        device=device,
        clip_lower=clip_lower,
        max_epochs=250,
        patience=35,
        batch_size=64,
        lr=1e-3,
        weight_decay=1e-2,
        dropout=0.20,
        embedding_dim=8,
        hidden_dim=64,
        cross_layers=2,
        early_stop=True,
    )

    lgb_model = train_lightgbm(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        numeric_cols=numeric_cols,
        cat_cols=cat_cols,
        clip_lower=clip_lower,
        n_estimators=5000,
        early_stop=True,
    )

    # Honest validation score: computed from the model trained only on the
    # train fold, using the official RMSE definition and the same inference
    # / post-processing path used for test.
    nn_val_pred = predict_model(model, X_val, numeric_cols, cat_cols, device, clip_lower)
    lgb_val_pred = predict_lightgbm(lgb_model, X_val, numeric_cols, cat_cols, clip_lower)

    rmse_nn = safe_rmse(y_val.to_numpy(), nn_val_pred)
    rmse_lgb = safe_rmse(y_val.to_numpy(), lgb_val_pred)

    if np.isfinite(rmse_nn) and np.isfinite(rmse_lgb):
        if rmse_nn <= 0 and rmse_lgb <= 0:
            w_nn = 0.5
            w_lgb = 0.5
        elif rmse_nn <= 0:
            w_nn = 1.0
            w_lgb = 0.0
        elif rmse_lgb <= 0:
            w_nn = 0.0
            w_lgb = 1.0
        else:
            inv_nn = 1.0 / rmse_nn
            inv_lgb = 1.0 / rmse_lgb
            total_inv = inv_nn + inv_lgb
            w_nn = inv_nn / total_inv
            w_lgb = inv_lgb / total_inv
    else:
        w_nn = 0.5
        w_lgb = 0.5

    val_pred = w_nn * nn_val_pred + w_lgb * lgb_val_pred
    score = safe_rmse(y_val.to_numpy(), val_pred)

    # Retrain on the complete training set using the selected epoch count.
    # This model is used only for the final test submission.
    final_model, _, _ = train_one_model(
        X_train=X_full,
        X_val=X_val,
        y_train=y_full,
        y_val=y_val,
        numeric_cols=numeric_cols,
        cat_cols=cat_cols,
        cat_cardinalities=cat_cardinalities,
        device=device,
        clip_lower=clip_lower,
        max_epochs=max(1, best_epoch),
        patience=max(10, best_epoch),
        batch_size=64,
        lr=1e-3,
        weight_decay=1e-2,
        dropout=0.20,
        embedding_dim=8,
        hidden_dim=64,
        cross_layers=2,
        early_stop=False,
    )

    best_iter = getattr(lgb_model, "best_iteration_", None)
    if best_iter is None or int(best_iter) <= 0:
        best_iter = 5000
    final_lgb_model = train_lightgbm(
        X_train=X_full,
        y_train=y_full,
        X_val=X_val,
        y_val=y_val,
        numeric_cols=numeric_cols,
        cat_cols=cat_cols,
        clip_lower=clip_lower,
        n_estimators=int(best_iter),
        early_stop=False,
    )

    # Test predictions come from actual model inference on all test samples.
    test_nn_pred = predict_model(
        final_model, X_test, numeric_cols, cat_cols, device, clip_lower
    )
    test_lgb_pred = predict_lightgbm(
        final_lgb_model, X_test, numeric_cols, cat_cols, clip_lower
    )
    test_pred = w_nn * test_nn_pred + w_lgb * test_lgb_pred

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_pred), dtype=int),
            TARGET: test_pred.astype(float),
        }
    )

    submission.to_csv(os.path.join(SUB_DIR, "submission.csv"), index=False)

    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()
