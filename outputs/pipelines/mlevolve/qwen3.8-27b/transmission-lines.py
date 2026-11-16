# Removed: this import must be the first statement, but generated runfiles may add lines before it.

import os
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer


TARGET_COL = "VOLT_CLASS"
ID_COLS = {
    "id",
    "index",
    "idx",
    "row_id",
    "sample_id",
    "line_id",
    "id_",
}


@dataclass
class ModelConfig:
    """Configuration for the voltage-class MLP.

    The training step should fill `input_dim`, `num_classes`, and optionally
    pass `y_train` to `build_model_bundle` to compute class weights from the
    training split only.
    """

    input_dim: int = 0
    num_classes: int = 0

    hidden_dims: Tuple[int, int] = (256, 128)
    dropout: float = 0.20

    label_smoothing: float = 0.05
    use_class_weights: bool = True
    max_class_weight: float = 10.0

    lr: float = 1e-3
    weight_decay: float = 1e-2
    epochs: int = 100


class VoltageClassMLP(nn.Module):
    """Compact MLP for tabular multi-class voltage classification.

    Design choices:
    - LayerNorm after the input and hidden layers for stable tabular training.
    - GELU activations and dropout for nonlinear interaction learning and
      regularization.
    - A small two-hidden-layer backbone is intentionally simple for the first
      model design step, while still being expressive enough for tabular
      interactions.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.20,
    ):
        super().__init__()

        if input_dim <= 0:
            raise ValueError("`input_dim` must be positive.")
        if num_classes <= 1:
            raise ValueError("`num_classes` must be greater than 1.")
        if len(hidden_dims) == 0:
            raise ValueError("`hidden_dims` must contain at least one width.")

        self.input_norm = nn.LayerNorm(input_dim)

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                raise ValueError("All hidden dimensions must be positive.")

            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Float tensor of shape `(batch_size, input_dim)`.

        Returns:
            Logits tensor of shape `(batch_size, num_classes)`.
        """
        x = self.input_norm(x)
        x = self.backbone(x)
        return self.head(x)


class WeightedSmoothCrossEntropyLoss(nn.Module):
    """Class-weighted cross-entropy with optional label smoothing.

    This loss is compatible with standard multi-class accuracy evaluation:
    the model still outputs class logits, and the predicted class is obtained
    by argmax over the logits.
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()

        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("`label_smoothing` must be in [0, 1).")

        self.label_smoothing = float(label_smoothing)

        if class_weights is not None:
            class_weights = class_weights.detach().float()
            if torch.any(class_weights < 0):
                raise ValueError("`class_weights` must be non-negative.")
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.long()

        if logits.dim() != 2:
            raise ValueError("`logits` must have shape `(batch_size, num_classes)`.")

        if target.dim() != 1:
            raise ValueError("`target` must have shape `(batch_size,)`.")

        if logits.size(0) != target.size(0):
            raise ValueError("Batch size mismatch between `logits` and `target`.")

        num_classes = logits.size(1)
        if target.min() < 0 or target.max() >= num_classes:
            raise ValueError("`target` contains class indices outside the valid range.")

        log_probs = F.log_softmax(logits, dim=1)

        nll_loss = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=1)

        loss = (
            1.0 - self.label_smoothing
        ) * nll_loss + self.label_smoothing * smooth_loss

        if self.class_weights is not None:
            if self.class_weights.numel() != num_classes:
                raise ValueError(
                    "`class_weights` length must match the number of classes."
                )
            loss = loss * self.class_weights[target]

        return loss.mean()


def _optimizer_param_groups(model: nn.Module, weight_decay: float) -> List[dict]:
    """Separate decay and no-decay parameters for AdamW.

    Biases and normalization parameters are kept at zero weight decay, which
    is a standard and stable choice for MLPs with LayerNorm.
    """
    decay_params: List[torch.Tensor] = []
    no_decay_params: List[torch.Tensor] = []

    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": float(weight_decay)},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def build_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
) -> torch.optim.AdamW:
    """Build an AdamW optimizer with sensible parameter groups."""
    return torch.optim.AdamW(
        _optimizer_param_groups(model, weight_decay),
        lr=float(lr),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int = 100,
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    """Build a cosine LR schedule for stable training."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(epochs)),
        eta_min=1e-6,
    )


def _compute_class_weights(
    y_train: np.ndarray,
    num_classes: int,
    max_class_weight: float = 10.0,
) -> torch.Tensor:
    """Compute capped inverse-frequency class weights from the train split only.

    The weights are normalized so their mean is approximately 1, then capped
    to avoid excessive emphasis on rare classes.
    """
    y_train = np.asarray(y_train).ravel()

    if y_train.size == 0:
        return torch.ones(num_classes, dtype=torch.float32)

    counts = np.bincount(y_train.astype(np.int64), minlength=num_classes).astype(
        np.float64
    )

    # Avoid division by zero for absent classes.
    counts = np.where(counts > 0, counts, 1.0)

    weights = y_train.size / counts
    weights = np.clip(weights, 1.0, float(max_class_weight))
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


def build_model_bundle(
    config: Optional[ModelConfig] = None,
    y_train: Optional[np.ndarray] = None,
) -> Tuple[
    VoltageClassMLP,
    WeightedSmoothCrossEntropyLoss,
    torch.optim.AdamW,
    torch.optim.lr_scheduler.CosineAnnealingLR,
    ModelConfig,
]:
    """Instantiate model, loss, optimizer, and scheduler.

    This function is intended to be called by the training step after the
    validation split has been created and only `y_train` from the training
    portion is supplied.
    """
    if config is None:
        config = ModelConfig()

    if config.input_dim <= 0:
        raise ValueError("`config.input_dim` must be set before building the model.")
    if config.num_classes <= 1:
        raise ValueError("`config.num_classes` must be greater than 1.")

    model = VoltageClassMLP(
        input_dim=config.input_dim,
        num_classes=config.num_classes,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    )

    class_weights = None
    if config.use_class_weights and y_train is not None:
        class_weights = _compute_class_weights(
            y_train=y_train,
            num_classes=config.num_classes,
            max_class_weight=config.max_class_weight,
        )

    criterion = WeightedSmoothCrossEntropyLoss(
        class_weights=class_weights,
        label_smoothing=config.label_smoothing,
    )

    optimizer = build_optimizer(
        model=model,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    scheduler = build_scheduler(
        optimizer=optimizer,
        epochs=config.epochs,
    )

    return model, criterion, optimizer, scheduler, config


def _set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _as_array(X):
    if isinstance(X, torch.Tensor):
        return X.detach().cpu().numpy()
    return np.asarray(X)


def _n_rows(X) -> int:
    if isinstance(X, pd.DataFrame):
        return len(X)
    if isinstance(X, torch.Tensor):
        return int(X.shape[0])
    if hasattr(X, "shape"):
        return int(X.shape[0])
    return len(X)


def _drop_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        return df

    drop_cols = []
    for col in df.columns:
        if col == TARGET_COL:
            drop_cols.append(col)
        elif str(col).strip().lower() in ID_COLS:
            drop_cols.append(col)

    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    return df


def _get_raw_data():
    """Use prepared variables if they exist; otherwise load the raw parquet files."""
    globals_dict = globals()

    candidate_sets = [
        ("X_train", "y_train", "X_test"),
        ("Xtr", "ytr", "Xte"),
        ("X_train", "y", "X_test"),
        ("train_features", "train_target", "test_features"),
    ]

    for x_name, y_name, t_name in candidate_sets:
        if x_name in globals_dict and y_name in globals_dict and t_name in globals_dict:
            X = globals_dict[x_name]
            y = globals_dict[y_name]
            T = globals_dict[t_name]

            if isinstance(X, pd.DataFrame):
                X = _drop_unwanted_columns(X.copy())
            if isinstance(T, pd.DataFrame):
                T = _drop_unwanted_columns(T.copy())

            return X, y, T

    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    if TARGET_COL not in train.columns:
        raise ValueError(f"Target column {TARGET_COL} not found in train.parquet")

    y = train[TARGET_COL].reset_index(drop=True)
    X = _drop_unwanted_columns(train.copy())
    T = _drop_unwanted_columns(test.copy())

    return X, y, T


def _row_select(X, idx):
    if isinstance(X, pd.DataFrame):
        return X.iloc[idx].reset_index(drop=True)

    arr = _as_array(X)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    return arr[idx]


def _to_float_df(X, feature_names):
    if isinstance(X, pd.DataFrame):
        df = X.copy()
    else:
        arr = _as_array(X)

        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        if arr.shape[1] < len(feature_names):
            pad = np.zeros(
                (arr.shape[0], len(feature_names) - arr.shape[1]), dtype=np.float32
            )
            arr = np.concatenate([arr, pad], axis=1)
        elif arr.shape[1] > len(feature_names):
            arr = arr[:, : len(feature_names)]

        df = pd.DataFrame(arr, columns=feature_names)

    df = df.reindex(columns=feature_names)
    return df


def _prepare_features(X_train, X_val, X_test, feature_names):
    """Leakage-safe feature preparation: fit imputers/encoders/scaler on train fold only."""
    if len(feature_names) == 0:
        n_tr = _n_rows(X_train)
        n_va = _n_rows(X_val)
        n_te = _n_rows(X_test)
        return (
            np.zeros((n_tr, 1), dtype=np.float32),
            np.zeros((n_va, 1), dtype=np.float32),
            np.zeros((n_te, 1), dtype=np.float32),
        )

    Xtr = _to_float_df(X_train, feature_names)
    Xval = _to_float_df(X_val, feature_names)
    Xte = _to_float_df(X_test, feature_names)

    numeric_cols = []
    cat_cols = []

    for col in feature_names:
        s = Xtr[col]

        if pd.api.types.is_numeric_dtype(s):
            numeric_cols.append(col)
            continue

        converted = pd.to_numeric(s, errors="coerce")
        if s.size > 0 and converted.notna().mean() > 0.8:
            Xtr[col] = converted
            Xval[col] = pd.to_numeric(Xval[col], errors="coerce")
            Xte[col] = pd.to_numeric(Xte[col], errors="coerce")
            numeric_cols.append(col)
        else:
            cat_cols.append(col)

    if numeric_cols:
        imputer = SimpleImputer(strategy="median").fit(Xtr[numeric_cols])
        Xtr[numeric_cols] = imputer.transform(Xtr[numeric_cols])
        Xval[numeric_cols] = imputer.transform(Xval[numeric_cols])
        Xte[numeric_cols] = imputer.transform(Xte[numeric_cols])

    if cat_cols:
        Xtr[cat_cols] = Xtr[cat_cols].fillna("__missing__")
        Xval[cat_cols] = Xval[cat_cols].fillna("__missing__")
        Xte[cat_cols] = Xte[cat_cols].fillna("__missing__")

        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        Xtr[cat_cols] = encoder.fit_transform(Xtr[cat_cols])
        Xval[cat_cols] = encoder.transform(Xval[cat_cols])
        Xte[cat_cols] = encoder.transform(Xte[cat_cols])

    Xtr = Xtr[feature_names].to_numpy(dtype=np.float32)
    Xval = Xval[feature_names].to_numpy(dtype=np.float32)
    Xte = Xte[feature_names].to_numpy(dtype=np.float32)

    Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    Xval = np.nan_to_num(Xval, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    Xte = np.nan_to_num(Xte, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if Xtr.shape[0] > 1:
        scaler = StandardScaler().fit(Xtr)
        Xtr = scaler.transform(Xtr)
        Xval = scaler.transform(Xval)
        Xte = scaler.transform(Xte)
    else:
        mean = Xtr.mean(axis=0, keepdims=True)
        Xtr = Xtr - mean
        Xval = Xval - mean
        Xte = Xte - mean

    Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    Xval = np.nan_to_num(Xval, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    Xte = np.nan_to_num(Xte, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return Xtr, Xval, Xte


def _predict_classes(model, X_np, device, batch_size: int = 8192) -> np.ndarray:
    if X_np is None or X_np.shape[0] == 0:
        return np.empty(0, dtype=np.int64)

    model.eval()
    preds = []

    with torch.no_grad():
        for start in range(0, X_np.shape[0], batch_size):
            xb = torch.from_numpy(X_np[start : start + batch_size]).to(
                device, non_blocking=True
            )
            logits = model(xb)
            preds.append(logits.argmax(dim=1).cpu().numpy())

    return np.concatenate(preds).astype(np.int64)


def _format_labels(labels) -> np.ndarray:
    arr = np.asarray(labels)

    if arr.size == 0:
        return arr

    if np.issubdtype(arr.dtype, np.floating):
        if np.all(np.isfinite(arr)) and np.allclose(arr, np.round(arr)):
            return arr.astype(np.int64)

    if arr.dtype == object:
        try:
            numeric = arr.astype(np.float64)
            if np.all(np.isfinite(numeric)) and np.allclose(numeric, np.round(numeric)):
                return numeric.astype(np.int64)
            return numeric
        except (TypeError, ValueError):
            return arr

    return arr


def main() -> float:
    _set_seed(42)
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    os.makedirs("./submission", exist_ok=True)
    os.makedirs("./working", exist_ok=True)

    X_train_raw, y_train_raw, X_test_raw = _get_raw_data()

    # Normalize target to a 1-D array and drop missing targets.
    if isinstance(y_train_raw, pd.Series):
        y_series = y_train_raw.reset_index(drop=True)
    else:
        y_series = pd.Series(_as_array(y_train_raw).ravel())

    valid = y_series.notna().values
    if pd.api.types.is_numeric_dtype(y_series):
        y_float = y_series.to_numpy(dtype=np.float64, na_value=np.nan)
        valid = valid & np.isfinite(y_float)

    n_train = len(y_series)

    if n_train == 0:
        test_len = _n_rows(X_test_raw)
        submission = pd.DataFrame(
            {
                "id": np.arange(test_len, dtype=int),
                "VOLT_CLASS": np.zeros(test_len, dtype=int),
            }
        )
        submission.to_csv("./submission/submission.csv", index=False)
        return 0.0

    if isinstance(X_train_raw, pd.DataFrame):
        X_train_raw = X_train_raw.iloc[valid].reset_index(drop=True)
    else:
        X_train_raw = _as_array(X_train_raw)
        if X_train_raw.ndim == 0:
            X_train_raw = X_train_raw.reshape(1, 1)
        elif X_train_raw.ndim == 1:
            X_train_raw = X_train_raw.reshape(-1, 1)
        X_train_raw = X_train_raw[valid]

    y_raw = _format_labels(y_series[valid].to_numpy().ravel())

    # Feature names.
    if isinstance(X_train_raw, pd.DataFrame):
        feature_names = list(X_train_raw.columns)
    else:
        feature_names = [f"feature_{i}" for i in range(X_train_raw.shape[1])]

    # Ensure at least one feature for the MLP.
    if len(feature_names) == 0:
        X_train_raw = np.zeros((len(y_raw), 1), dtype=np.float32)
        X_test_raw = np.zeros((_n_rows(X_test_raw), 1), dtype=np.float32)
        feature_names = ["constant"]

    # Label mapping. This only defines the class index space and does not use feature statistics.
    label_encoder = LabelEncoder()
    y_all_enc = label_encoder.fit_transform(y_raw)
    num_classes = len(label_encoder.classes_)

    # Edge case: single-class target.
    if num_classes <= 1:
        only_label = label_encoder.classes_[0] if num_classes == 1 else 0
        test_len = _n_rows(X_test_raw)

        submission = pd.DataFrame(
            {
                "id": np.arange(test_len, dtype=int),
                "VOLT_CLASS": np.full(test_len, only_label),
            }
        )
        submission.to_csv("./submission/submission.csv", index=False)
        return 1.0

    # Stratified hold-out validation split.
    n = len(y_all_enc)
    val_size = max(1, int(round(n * 0.15)))

    if val_size >= n:
        val_size = max(1, n - 1) if n > 1 else 0

    if n > 1 and val_size > 0:
        stratify = None
        class_counts = np.bincount(y_all_enc, minlength=num_classes)

        if np.all(class_counts >= 2):
            stratify = y_all_enc

        try:
            train_idx, val_idx = train_test_split(
                np.arange(n),
                test_size=val_size,
                random_state=42,
                stratify=stratify,
            )
        except ValueError:
            train_idx, val_idx = train_test_split(
                np.arange(n),
                test_size=val_size,
                random_state=42,
            )
    else:
        train_idx = np.arange(n)
        val_idx = np.empty(0, dtype=int)

    X_train_split = _row_select(X_train_raw, train_idx)
    X_val_split = _row_select(X_train_raw, val_idx)

    y_train_enc = y_all_enc[train_idx]
    y_val_enc = y_all_enc[val_idx]
    y_val_raw = y_raw[val_idx]

    # Leakage-safe feature preparation: fit on training fold only.
    X_train_np, X_val_np, X_test_np = _prepare_features(
        X_train_split,
        X_val_split,
        X_test_raw,
        feature_names,
    )

    input_dim = X_train_np.shape[1]
    if input_dim <= 0:
        input_dim = 1
        X_train_np = np.ones((X_train_np.shape[0], 1), dtype=np.float32)
        X_val_np = np.ones((X_val_np.shape[0], 1), dtype=np.float32)
        X_test_np = np.ones((X_test_np.shape[0], 1), dtype=np.float32)

    # Build the model bundle defined in the previous model_design step.
    if "ModelConfig" not in globals() or "build_model_bundle" not in globals():
        raise RuntimeError("Model design step variables are not available.")

    config = ModelConfig(
        input_dim=input_dim,
        num_classes=num_classes,
        epochs=100,
    )

    model, criterion, optimizer, scheduler, config = build_model_bundle(
        config=config,
        y_train=y_train_enc,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion.to(device)

    train_dataset = TensorDataset(
        torch.from_numpy(X_train_np),
        torch.from_numpy(y_train_enc).long(),
    )

    batch_size = 1024
    if X_train_np.shape[0] < batch_size:
        batch_size = max(16, X_train_np.shape[0])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=False,
    )

    best_state = None
    best_val_acc = -1.0
    patience = 15
    bad_epochs = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(xb)
            loss = criterion(logits, yb)

            if bool(torch.isfinite(loss)):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += float(loss.item()) * xb.size(0)
                seen += xb.size(0)

        scheduler.step()

        train_loss = total_loss / max(1, seen)

        val_pred_enc = _predict_classes(model, X_val_np, device)
        if val_pred_enc.size > 0:
            val_acc = float(accuracy_score(y_val_enc, val_pred_enc))
        else:
            val_acc = 0.0

        print(
            f"Epoch {epoch:03d} | train_loss {train_loss:.4f} | val_acc {val_acc:.4f}"
        )

        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is None:
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

    torch.save(best_state, "./working/best_model.pt")

    model.load_state_dict(best_state)
    model.to(device)

    # Final validation using the exact same inference path as test.
    val_pred_enc = _predict_classes(model, X_val_np, device)

    if val_pred_enc.size > 0:
        val_pred_labels = np.asarray(
            label_encoder.inverse_transform(val_pred_enc)
        ).ravel()
        val_pred_labels = _format_labels(val_pred_labels)

        y_val_raw_formatted = _format_labels(np.asarray(y_val_raw).ravel())

        score = float(accuracy_score(y_val_raw_formatted, val_pred_labels))
    else:
        score = 0.0

    # Test inference and submission.
    test_pred_enc = _predict_classes(model, X_test_np, device)

    if test_pred_enc.size > 0:
        test_pred_labels = np.asarray(
            label_encoder.inverse_transform(test_pred_enc)
        ).ravel()
    else:
        test_pred_labels = np.empty(0, dtype=object)

    test_pred_labels = _format_labels(test_pred_labels)

    submission = pd.DataFrame(
        {
            "id": np.arange(len(test_pred_labels), dtype=int),
            "VOLT_CLASS": test_pred_labels,
        }
    )
    submission.to_csv("./submission/submission.csv", index=False)

    return score


score = main()
print(f"Final Validation Score: {score}")
