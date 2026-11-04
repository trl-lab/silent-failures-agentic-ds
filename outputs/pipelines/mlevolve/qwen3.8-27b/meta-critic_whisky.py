import os
import re
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    import xgboost as xgb
except Exception:
    xgb = None

TARGET_NAME = "Meta Critic"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"


def find_target_column(df):
    for c in df.columns:
        if c == TARGET_NAME:
            return c

    normalized = {c: re.sub(r"[^0-9a-z]+", "", str(c).lower()) for c in df.columns}
    for c in df.columns:
        if normalized[c] == "metacritic":
            return c

    for c in df.columns:
        if "meta" in normalized[c] and "critic" in normalized[c]:
            return c

    raise ValueError("Target column not found.")


def find_id_column(df):
    for c in ["id", "Id", "ID", "index", "Index", "row_id", "sample_id", "sample"]:
        if c in df.columns:
            return c
    return None


def sanitize(name):
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", str(name)).strip("_")
    return s if s else "feature"


def smooth_target_encoding_map(cat, y, smoothing=10.0):
    y = np.asarray(y, dtype=float)
    global_mean = float(np.mean(y)) if len(y) else 0.0

    g = pd.DataFrame({"cat": cat.values, "y": y})
    stats = g.groupby("cat")["y"].agg(["mean", "count"])
    stats["te"] = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
        stats["count"] + smoothing
    )
    return stats["te"], global_mean


def loo_target_encoding(cat, y, smoothing=10.0):
    y = np.asarray(y, dtype=float)
    global_mean = float(np.mean(y)) if len(y) else 0.0

    g = pd.DataFrame({"cat": cat.values, "y": y})
    group_sum = g.groupby("cat")["y"].transform("sum").to_numpy(dtype=float)
    group_count = g.groupby("cat")["y"].transform("count").to_numpy(dtype=int)

    denom = np.where(group_count > 1, group_count - 1, 1)
    loot = np.where(group_count > 1, (group_sum - y) / denom, global_mean)
    te = (group_count * loot + smoothing * global_mean) / (group_count + smoothing)
    return pd.Series(te, index=cat.index, dtype=float)


def build_features(X_fit, y_fit, X_apply_list):
    fit = X_fit.copy()

    if fit.shape[1] == 0:
        Xs = [pd.DataFrame(index=df.index) for df in X_apply_list]
        for i, X in enumerate(Xs):
            X["fallback_feature"] = np.arange(len(X), dtype=float)
        return Xs

    fit = fit.loc[:, fit.notna().any()]

    if fit.shape[1] == 0:
        Xs = [pd.DataFrame(index=df.index) for df in X_apply_list]
        for i, X in enumerate(Xs):
            X["fallback_feature"] = np.arange(len(X), dtype=float)
        return Xs

    apply_frames = [df.reindex(columns=fit.columns) for df in X_apply_list]
    results = [pd.DataFrame(index=df.index) for df in apply_frames]

    datetime_cols = []
    numeric_cols = []
    text_cols = []
    categorical_cols = []

    for c in fit.columns:
        s = fit[c]

        if pd.api.types.is_bool_dtype(s):
            numeric_cols.append(c)
            continue

        if pd.api.types.is_numeric_dtype(s):
            numeric_cols.append(c)
            continue

        if (
            pd.api.types.is_object_dtype(s)
            or pd.api.types.is_string_dtype(s)
            or s.dtype == "category"
        ):
            sample = s.dropna().astype(str).head(20)
            if len(sample) > 0:
                dt_mask = sample.str.match(
                    r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}"
                ) | sample.str.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{4}")
                if dt_mask.mean() > 0.5:
                    datetime_cols.append(c)
                    continue

            try:
                num = pd.to_numeric(s, errors="coerce")
            except Exception:
                num = pd.Series(np.nan, index=s.index)

            if num.notna().mean() > 0.5:
                numeric_cols.append(c)
                continue

            non_null = s.dropna().astype(str)
            if len(non_null) > 0:
                mean_len = float(non_null.str.len().mean())
                mean_words = float(non_null.str.split().str.len().mean())
                if mean_len > 30 or mean_words > 5:
                    text_cols.append(c)
                    continue

            categorical_cols.append(c)
        else:
            categorical_cols.append(c)

    numeric_scaled_names = {}

    for c in numeric_cols:
        s_fit = pd.to_numeric(fit[c], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        median = s_fit.median()
        if pd.isna(median):
            median = 0.0

        s_fit_imp = s_fit.fillna(median)
        scaler = StandardScaler()
        scaler.fit(s_fit_imp.values.reshape(-1, 1))

        base = sanitize(c)
        scaled_name = sanitize(f"{c}_scaled")
        numeric_scaled_names[c] = scaled_name

        for idx, apply in enumerate(apply_frames):
            s = pd.to_numeric(apply[c], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            s_imp = s.fillna(median)

            results[idx][base] = s_imp.astype(float)
            results[idx][sanitize(f"{c}_missing")] = apply[c].isna().astype(float)
            results[idx][scaled_name] = (
                scaler.transform(s_imp.values.reshape(-1, 1)).ravel().astype(float)
            )

            if float((s_fit_imp >= 0).mean()) > 0.8 and float(s_fit_imp.std()) > 0:
                results[idx][sanitize(f"{c}_log")] = np.log1p(
                    s_imp.clip(lower=0)
                ).astype(float)

    for c in datetime_cols:
        base = sanitize(c)
        for idx, apply in enumerate(apply_frames):
            s = pd.to_datetime(apply[c], errors="coerce")
            results[idx][f"{base}_year"] = s.dt.year.astype(float)
            results[idx][f"{base}_month"] = s.dt.month.astype(float)
            results[idx][f"{base}_day"] = s.dt.day.astype(float)
            results[idx][f"{base}_dayofweek"] = s.dt.dayofweek.astype(float)
            results[idx][f"{base}_missing"] = apply[c].isna().astype(float)

    for c in categorical_cols:
        fit_str = fit[c].astype("string").fillna("missing").astype(str)
        base = sanitize(c)
        freq = fit_str.value_counts(normalize=True)

        for idx, apply in enumerate(apply_frames):
            apply_str = apply[c].astype("string").fillna("missing").astype(str)
            results[idx][f"{base}_missing"] = apply[c].isna().astype(float)
            results[idx][f"{base}_freq"] = apply_str.map(freq).fillna(0.0).astype(float)

        if fit_str.nunique() > 1:
            te_map, global_mean = smooth_target_encoding_map(
                fit_str, y_fit, smoothing=10.0
            )
            for idx, apply in enumerate(apply_frames):
                apply_str = apply[c].astype("string").fillna("missing").astype(str)
                if idx == 0:
                    te_vals = loo_target_encoding(fit_str, y_fit, smoothing=10.0)
                else:
                    te_vals = apply_str.map(te_map).fillna(global_mean)
                results[idx][f"{base}_target"] = te_vals.astype(float)
        else:
            global_mean = float(np.mean(y_fit)) if len(y_fit) else 0.0
            for idx in range(len(apply_frames)):
                results[idx][f"{base}_target"] = global_mean

        if fit_str.nunique() <= 20:
            cats = fit_str.value_counts().index[:20]
            for cat in cats:
                col = sanitize(f"{c}_{cat}")
                for idx, apply in enumerate(apply_frames):
                    apply_str = apply[c].astype("string").fillna("missing").astype(str)
                    results[idx][col] = (apply_str == cat).astype(float)

    for c in text_cols:
        fit_str = fit[c].astype("string").fillna("").astype(str)
        base = sanitize(c)

        for idx, apply in enumerate(apply_frames):
            s = apply[c].astype("string").fillna("").astype(str)
            words = s.str.split()

            results[idx][f"{base}_len"] = s.str.len().astype(float)
            results[idx][f"{base}_words"] = words.str.len().astype(float)

            def avg_word_len(x):
                return float(np.mean([len(w) for w in x])) if x else 0.0

            results[idx][f"{base}_avg_word_len"] = words.apply(avg_word_len).astype(
                float
            )
            results[idx][f"{base}_excl"] = s.str.count("!").astype(float)
            results[idx][f"{base}_quest"] = s.str.count(r"\?").astype(float)
            results[idx][f"{base}_digits"] = s.str.count(r"\d").astype(float)

        try:
            max_features = 100 if len(fit) < 100 else 300
            min_df = 1 if len(fit) < 50 else 2
            vec = TfidfVectorizer(
                max_features=max_features,
                min_df=min_df,
                stop_words="english",
                ngram_range=(1, 2),
                dtype=np.float32,
                lowercase=True,
            )
            vec.fit(fit_str)

            if len(vec.vocabulary_) > 0:
                for idx, apply in enumerate(apply_frames):
                    s = apply[c].astype("string").fillna("").astype(str)
                    tf = vec.transform(s).toarray().astype(np.float32)
                    cols = [sanitize(f"{c}_tf_{k}") for k in range(tf.shape[1])]
                    results[idx][cols] = pd.DataFrame(
                        tf, index=apply.index, columns=cols
                    )
        except Exception:
            pass

    num_items = list(numeric_scaled_names.items())
    if 2 <= len(num_items) <= 8:
        for i in range(len(num_items)):
            for j in range(i + 1, len(num_items)):
                a, a_scaled = num_items[i]
                b, b_scaled = num_items[j]

                prod_col = sanitize(f"inter_{a}_{b}_prod")
                ratio_col = sanitize(f"inter_{a}_{b}_ratio")

                for idx in range(len(apply_frames)):
                    a_vals = results[idx][a_scaled]
                    b_vals = results[idx][b_scaled]

                    results[idx][prod_col] = (a_vals * b_vals).astype(float)
                    results[idx][ratio_col] = np.where(
                        b_vals != 0, a_vals / b_vals, 0.0
                    ).astype(float)

    Xs = []
    for res in results:
        X = res.copy()
        X = X.loc[:, ~X.columns.duplicated()]
        X = X.select_dtypes(include=[np.number])
        X = X.fillna(0.0)
        Xs.append(X)

    if Xs[0].shape[1] == 0:
        for i, X in enumerate(Xs):
            X["fallback_feature"] = np.arange(len(X), dtype=float)
    else:
        base_cols = Xs[0].columns
        for i in range(1, len(Xs)):
            Xs[i] = Xs[i].reindex(columns=base_cols)

        mask = Xs[0].notna().any()
        Xs = [X.loc[:, mask] for X in Xs]

        var_mask = Xs[0].std(ddof=0) > 0
        Xs = [X.loc[:, var_mask] for X in Xs]

        if Xs[0].shape[1] == 0:
            for i, X in enumerate(Xs):
                X["fallback_feature"] = np.arange(len(X), dtype=float)

    Xs = [X.astype(np.float32) for X in Xs]
    return Xs


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) == 0:
        return 0.0

    result = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    if not np.isfinite(result):
        return float("inf")
    return result


def prepare_data():
    os.makedirs(WORKING_DIR, exist_ok=True)

    train = pd.read_parquet("./input/train.parquet")
    test = pd.read_parquet("./input/test.parquet")

    target_col = find_target_column(train)
    train[target_col] = pd.to_numeric(train[target_col], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    train = train.dropna(subset=[target_col]).reset_index(drop=True)

    if len(train) == 0:
        raise ValueError("No valid training rows with target.")

    test = test.copy()
    if target_col in test.columns:
        test = test.drop(columns=[target_col])
    test = test.reset_index(drop=True)

    train_id_col = find_id_column(train)
    test_ids = np.arange(len(test), dtype=int)

    feature_cols = [c for c in train.columns if c not in [target_col, train_id_col]]

    X_full = (
        train[feature_cols].copy() if feature_cols else pd.DataFrame(index=train.index)
    )
    y_full = train[target_col].values.astype(float)

    n = len(y_full)

    if n <= 1:
        X_train, X_val = X_full.copy(), X_full.copy()
        y_train, y_val = y_full.copy(), y_full.copy()
    else:
        val_size = min(max(1, int(round(0.2 * n))), n - 1)
        stratify = None

        if n > 10 and len(np.unique(y_full)) > 4:
            try:
                stratify = pd.qcut(y_full, q=4, labels=False, duplicates="drop")
            except Exception:
                stratify = None

        X_train, X_val, y_train, y_val = train_test_split(
            X_full,
            y_full,
            test_size=val_size,
            random_state=42,
            stratify=stratify,
        )

    X_test = (
        test.reindex(columns=feature_cols).copy()
        if feature_cols
        else pd.DataFrame(index=test.index)
    )

    X_train, X_val, X_test = build_features(X_train, y_train, [X_train, X_val, X_test])

    X_train.to_parquet(os.path.join(WORKING_DIR, "X_train.parquet"))
    X_val.to_parquet(os.path.join(WORKING_DIR, "X_val.parquet"))
    X_test.to_parquet(os.path.join(WORKING_DIR, "X_test.parquet"))

    np.save(os.path.join(WORKING_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(WORKING_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(WORKING_DIR, "test_ids.npy"), test_ids)

    with open(os.path.join(WORKING_DIR, "feature_names.json"), "w") as f:
        json.dump(list(X_train.columns), f)

    return X_train, X_val, X_test, y_train, y_val, test_ids


def load_artifacts():
    X_train = pd.read_parquet(os.path.join(WORKING_DIR, "X_train.parquet"))
    X_val = pd.read_parquet(os.path.join(WORKING_DIR, "X_val.parquet"))
    X_test = pd.read_parquet(os.path.join(WORKING_DIR, "X_test.parquet"))
    y_train = np.load(os.path.join(WORKING_DIR, "y_train.npy")).astype(float)
    y_val = np.load(os.path.join(WORKING_DIR, "y_val.npy")).astype(float)
    test_ids = np.load(os.path.join(WORKING_DIR, "test_ids.npy")).astype(int)

    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)

    if X_train.shape[1] != X_val.shape[1]:
        X_val = X_val.reindex(columns=X_train.columns).fillna(0.0)
    if X_train.shape[1] != X_test.shape[1]:
        X_test = X_test.reindex(columns=X_train.columns).fillna(0.0)

    if X_train.shape[1] == 0:
        X_train["fallback_feature"] = np.arange(len(X_train), dtype=np.float32)
        X_val["fallback_feature"] = np.arange(len(X_val), dtype=np.float32)
        X_test["fallback_feature"] = np.arange(len(X_test), dtype=np.float32)

    return X_train, X_val, X_test, y_train, y_val, test_ids


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.15):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        h = self.norm(x)
        h = self.linear1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.linear2(h)
        h = self.dropout(h)
        return residual + h


class HybridTabularRegressor(nn.Module):
    """
    Model architecture:
    - Linear branch: captures stable additive effects.
    - Nonlinear residual MLP branch: captures interactions among engineered features.
    - Learned scalar gate: balances linear and nonlinear contributions.
    """

    def __init__(self, input_dim, hidden_dim=64, n_blocks=2, dropout=0.15):
        super().__init__()

        if input_dim <= 0:
            input_dim = 1

        self.input_dim = input_dim
        self.hidden_dim = max(16, hidden_dim)

        # Linear branch.
        self.linear_head = nn.Linear(self.input_dim, 1)

        # Nonlinear branch.
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(self.hidden_dim, dropout=dropout) for _ in range(n_blocks)]
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1),
        )

        # Learned gate in [0, 1].
        self.gate_logit = nn.Parameter(torch.tensor(0.0))

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        if x.shape[1] != self.input_dim:
            if x.shape[1] < self.input_dim:
                x = nn.functional.pad(x, (0, self.input_dim - x.shape[1]))
            else:
                x = x[:, : self.input_dim]

        linear_out = self.linear_head(x)

        h = self.input_norm(x)
        h = self.encoder(h)
        for block in self.blocks:
            h = block(h)
        mlp_out = self.mlp_head(h)

        gate = torch.sigmoid(self.gate_logit)
        return gate * mlp_out + (1.0 - gate) * linear_out


def build_model(input_dim: int) -> nn.Module:
    return HybridTabularRegressor(
        input_dim=input_dim,
        hidden_dim=64,
        n_blocks=2,
        dropout=0.15,
    )


def build_criterion() -> nn.Module:
    # Official metric is RMSE; MSE is its differentiable surrogate.
    return nn.MSELoss()


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=1e-3,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, max_epochs: int, warmup_epochs: int = 10):
    base_lr = float(optimizer.param_groups[0]["lr"])
    eta_min = 1e-5

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        remaining = max(1, max_epochs - warmup_epochs)
        progress = (epoch - warmup_epochs) / float(remaining)
        progress = min(1.0, max(0.0, progress))
        return (eta_min / base_lr) + 0.5 * (1.0 - eta_min / base_lr) * (
            1.0 + np.cos(np.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def to_tensor(df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    arr = np.asarray(df.values, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.from_numpy(arr).to(device)


@torch.no_grad()
def predict_model(
    model: nn.Module, X: pd.DataFrame, device: torch.device
) -> np.ndarray:
    model.eval()
    X_t = to_tensor(X, device)
    preds = []
    batch_size = 512
    for i in range(0, len(X_t), batch_size):
        xb = X_t[i : i + batch_size]
        preds.append(model(xb).squeeze(-1).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=float)


def main():
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    prepare_data()
    X_train, X_val, X_test, y_train, y_val, test_ids = load_artifacts()

    if len(y_train) == 0:
        raise ValueError("No training target values available.")

    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train))
    if not np.isfinite(y_std) or y_std < 1e-8:
        y_std = 1.0

    y_train_std = (y_train - y_mean) / y_std

    lower = float(np.min(y_train))
    upper = float(np.max(y_train))
    if not np.isfinite(lower):
        lower = 0.0
    if not np.isfinite(upper):
        upper = 1.0
    if lower > upper:
        lower, upper = upper, lower
    if upper - lower < 1e-8:
        lower -= 1.0
        upper += 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    input_dim = int(X_train.shape[1])
    best_epoch = 0

    try:
        model = build_model(input_dim=input_dim).to(device)
        criterion = build_criterion()
        optimizer = build_optimizer(model)

        max_epochs = 200
        patience = 30
        scheduler = build_scheduler(optimizer, max_epochs=max_epochs, warmup_epochs=10)

        train_ds = TensorDataset(
            to_tensor(X_train, device),
            torch.from_numpy(y_train_std.astype(np.float32)).to(device),
        )
        num_workers = 0 if len(X_train) < 1000 else 2
        train_loader = DataLoader(
            train_ds,
            batch_size=32,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
        )

        best_val_rmse = float("inf")
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
        epochs_without_improvement = 0

        for epoch in range(1, max_epochs + 1):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for xb, yb in train_loader:
                optimizer.zero_grad(set_to_none=True)
                pred = model(xb).squeeze(-1)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                epoch_loss += float(loss.item())
                n_batches += 1

            scheduler.step()

            val_pred_std = predict_model(model, X_val, device)
            val_pred = val_pred_std * y_std + y_mean
            val_pred = np.clip(val_pred, lower, upper)
            val_rmse = rmse(y_val, val_pred)

            if epoch == 1 or epoch % 10 == 0:
                print(
                    f"epoch={epoch} "
                    f"train_mse={epoch_loss / max(1, n_batches):.6f} "
                    f"val_rmse={val_rmse:.6f}"
                )

            if val_rmse < best_val_rmse - 1e-6:
                best_val_rmse = val_rmse
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break

        model.load_state_dict(best_state)
        model.to(device)

        val_pred = predict_model(model, X_val, device)
        val_pred = val_pred * y_std + y_mean
        val_pred = np.clip(val_pred, lower, upper)

        test_pred = predict_model(model, X_test, device)
        test_pred = test_pred * y_std + y_mean
        test_pred = np.clip(test_pred, lower, upper)

        score = rmse(y_val, val_pred)
        model_name = "HybridTabularRegressor"

    except Exception:
        model_name = "RidgeFallback"
        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(X_train, y_train)

        val_pred = np.clip(ridge.predict(X_val), lower, upper)
        test_pred = np.clip(ridge.predict(X_test), lower, upper)
        score = rmse(y_val, val_pred)

    if not np.isfinite(score):
        model_name = "RidgeFallback"
        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(X_train, y_train)

        val_pred = np.clip(ridge.predict(X_val), lower, upper)
        test_pred = np.clip(ridge.predict(X_test), lower, upper)
        score = rmse(y_val, val_pred)

    if len(test_pred) != len(test_ids):
        test_ids = np.arange(len(test_pred), dtype=int)

    submission = pd.DataFrame(
        {
            "id": np.asarray(test_ids, dtype=int),
            TARGET_NAME: np.asarray(test_pred, dtype=float),
        }
    )
    submission.to_csv(os.path.join(SUBMISSION_DIR, "submission.csv"), index=False)

    with open(os.path.join(WORKING_DIR, "model_design_info.json"), "w") as f:
        json.dump(
            {
                "model": model_name,
                "input_dim": input_dim,
                "loss": "MSELoss",
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingLR",
                "best_epoch": best_epoch,
                "validation_rmse": float(score) if np.isfinite(score) else None,
                "postprocess": "inverse_target_standardization_and_clip_to_train_range",
            },
            f,
            indent=2,
        )

    print(f"Final Validation Score: {score}")


if __name__ == "__main__":
    main()
