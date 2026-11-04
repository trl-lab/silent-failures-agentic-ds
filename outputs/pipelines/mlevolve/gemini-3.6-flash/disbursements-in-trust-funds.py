import os
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 1. Data Processing & Feature Engineering
# ==========================================
def process_and_engineer_features():
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"
    working_dir = "./working"
    os.makedirs(working_dir, exist_ok=True)

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    target_col = "Cash Disbursements (US$)"
    assert (
        target_col in train_df.columns
    ), f"Target column '{target_col}' missing in train data!"

    if "id" not in test_df.columns:
        test_df["id"] = np.arange(len(test_df))
    if "id" not in train_df.columns:
        train_df["id"] = np.arange(len(train_df))

    train_target = train_df[target_col].values
    train_df = train_df.drop(columns=[target_col])

    n_train = len(train_df)
    combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    combined_df.columns = [str(col).strip() for col in combined_df.columns]

    # Date/Time Feature Engineering
    date_cols = []
    for col in combined_df.columns:
        if col.lower() in ["id", target_col.lower()]:
            continue
        if (
            "date" in col.lower()
            or "year" in col.lower()
            or "month" in col.lower()
            or "period" in col.lower()
        ):
            date_cols.append(col)

    for col in date_cols:
        try:
            parsed_dt = pd.to_datetime(combined_df[col], errors="coerce")
            if parsed_dt.notna().sum() > 0.1 * len(combined_df):
                combined_df[f"{col}_year"] = parsed_dt.dt.year
                combined_df[f"{col}_month"] = parsed_dt.dt.month
                combined_df[f"{col}_quarter"] = parsed_dt.dt.quarter
                combined_df[f"{col}_dayofweek"] = parsed_dt.dt.dayofweek
                combined_df[f"{col}_is_year_end"] = parsed_dt.dt.is_year_end.astype(
                    float
                )
        except Exception:
            pass

    # Numeric & Monetary Features
    num_cols = [
        col
        for col in combined_df.columns
        if col != "id"
        and combined_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]
    ]

    monetary_cols = [
        col
        for col in num_cols
        if any(
            k in col.lower()
            for k in [
                "us$",
                "amount",
                "cash",
                "balance",
                "commitment",
                "contribution",
                "paid",
                "transfer",
            ]
        )
    ]
    for col in monetary_cols:
        combined_df[f"{col}_log1p"] = np.log1p(
            np.maximum(0, combined_df[col].fillna(0))
        )

    if len(monetary_cols) >= 2:
        for i in range(len(monetary_cols)):
            for j in range(i + 1, min(i + 4, len(monetary_cols))):
                c1, c2 = monetary_cols[i], monetary_cols[j]
                combined_df[f"{c1}_minus_{c2}"] = combined_df[c1].fillna(
                    0
                ) - combined_df[c2].fillna(0)
                combined_df[f"{c1}_ratio_{c2}"] = (combined_df[c1].fillna(0) + 1.0) / (
                    combined_df[c2].fillna(0) + 1.0
                )

    # Missing Value Indicators
    cols_with_nan = [
        col for col in combined_df.columns if col != "id" and combined_df[col].isna().sum() > 0
    ]
    for col in cols_with_nan:
        combined_df[f"{col}_isna"] = combined_df[col].isna().astype(float)

    # Categorical Identification & Frequency Encoding
    cat_cols = [
        col
        for col in combined_df.columns
        if col != "id" and combined_df[col].dtype in [object, "category"]
    ]

    for col in cat_cols:
        freq = combined_df[col].astype(str).value_counts(normalize=True).to_dict()
        combined_df[f"{col}_freq"] = combined_df[col].astype(str).map(freq).fillna(0)

    # TF-IDF Vectorization with TruncatedSVD on concatenated string/text fields
    if len(cat_cols) > 0:
        text_data = combined_df[cat_cols].astype(str).fillna("").agg(" ".join, axis=1)
        tfidf = TfidfVectorizer(max_features=5000, stop_words="english", token_pattern=r"(?u)\b\w+\b")
        tfidf_mat = tfidf.fit_transform(text_data)
        n_components = min(10, tfidf_mat.shape[1] - 1) if tfidf_mat.shape[1] > 1 else 1
        if n_components > 0:
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            svd_feats = svd.fit_transform(tfidf_mat)
            for k in range(n_components):
                combined_df[f"text_svd_{k}"] = svd_feats[:, k]

    # Split combined back into Train and Test
    train_proc = combined_df.iloc[:n_train].copy().reset_index(drop=True)
    test_proc = combined_df.iloc[n_train:].copy().reset_index(drop=True)

    train_proc[target_col] = train_target
    train_proc["target_log1p"] = np.log1p(np.maximum(0, train_target))

    # K-Fold split
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_proc["fold"] = -1
    for fold, (_, val_idx) in enumerate(kf.split(train_proc)):
        train_proc.loc[val_idx, "fold"] = fold

    # Composite Categorical Pairs
    cat_for_pairs = [
        col for col in cat_cols
        if 2 <= train_proc[col].nunique() <= 500
    ][:4]

    composite_cats = []
    for i in range(len(cat_for_pairs)):
        for j in range(i + 1, len(cat_for_pairs)):
            c1, c2 = cat_for_pairs[i], cat_for_pairs[j]
            comp_col = f"{c1}__x__{c2}"
            train_proc[comp_col] = train_proc[c1].astype(str) + "_" + train_proc[c2].astype(str)
            test_proc[comp_col] = test_proc[c1].astype(str) + "_" + test_proc[c2].astype(str)
            composite_cats.append(comp_col)

    # Expanded Out-of-Fold Target Encoding (Mean + Std on single & composite categoricals)
    top_cats = [
        col
        for col in cat_cols
        if train_proc[col].nunique() > 1 and train_proc[col].nunique() < 1000
    ] + composite_cats

    global_mean = train_proc["target_log1p"].mean()
    global_std = train_proc["target_log1p"].std()

    for col in top_cats:
        train_proc[f"{col}_te_mean"] = np.nan
        train_proc[f"{col}_te_std"] = np.nan
        test_proc[f"{col}_te_mean"] = np.nan
        test_proc[f"{col}_te_std"] = np.nan

        for fold in range(5):
            val_idx = train_proc["fold"] == fold
            tr_idx = ~val_idx
            fold_tr = train_proc.loc[tr_idx]
            fold_mean = fold_tr["target_log1p"].mean()
            fold_std = fold_tr["target_log1p"].std()

            te_mean_map = fold_tr.groupby(col)["target_log1p"].mean().to_dict()
            te_std_map = fold_tr.groupby(col)["target_log1p"].std().to_dict()

            train_proc.loc[val_idx, f"{col}_te_mean"] = (
                train_proc.loc[val_idx, col].map(te_mean_map).fillna(fold_mean)
            )
            train_proc.loc[val_idx, f"{col}_te_std"] = (
                train_proc.loc[val_idx, col].map(te_std_map).fillna(fold_std).fillna(0)
            )

        full_mean_map = train_proc.groupby(col)["target_log1p"].mean().to_dict()
        full_std_map = train_proc.groupby(col)["target_log1p"].std().to_dict()

        test_proc[f"{col}_te_mean"] = test_proc[col].map(full_mean_map).fillna(global_mean)
        test_proc[f"{col}_te_std"] = test_proc[col].map(full_std_map).fillna(global_std).fillna(0)

    # Group Aggregations based on key categoricals
    group_cats = [col for col in cat_cols if 2 <= train_proc[col].nunique() <= 200]
    agg_num_cols = [c for c in monetary_cols if c in train_proc.columns][:3]

    if group_cats and agg_num_cols:
        for g_col in group_cats[:3]:
            for a_col in agg_num_cols:
                grp_stats = (
                    train_proc.groupby(g_col)[a_col].agg(["mean", "std"]).reset_index()
                )
                grp_stats.columns = [
                    g_col,
                    f"{g_col}_{a_col}_grp_mean",
                    f"{g_col}_{a_col}_grp_std",
                ]
                train_proc = train_proc.merge(grp_stats, on=g_col, how="left")
                test_proc = test_proc.merge(grp_stats, on=g_col, how="left")

    # Clean object types for GBDT compatibility
    final_cat_cols = [
        col
        for col in train_proc.columns
        if train_proc[col].dtype in [object, "category"]
        and col not in ["id", target_col]
    ]
    for col in final_cat_cols:
        train_proc[col] = train_proc[col].astype(str).astype("category")
        test_proc[col] = test_proc[col].astype(str).astype("category")

    # Impute numeric features with training median
    all_feature_cols = [
        col
        for col in train_proc.columns
        if col not in ["id", target_col, "target_log1p", "fold"]
    ]
    num_feature_cols = [
        col
        for col in all_feature_cols
        if train_proc[col].dtype not in [object, "category"]
    ]

    train_medians = train_proc[num_feature_cols].median()
    train_proc[num_feature_cols] = train_proc[num_feature_cols].fillna(train_medians)
    test_proc[num_feature_cols] = test_proc[num_feature_cols].fillna(train_medians)

    return train_proc, test_proc


# ==========================================
# 2. PyTorch Tabular Architecture
# ==========================================
class ResBlock(nn.Module):

    def __init__(self, in_features, dropout_rate=0.25):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class TabularResNet(nn.Module):

    def __init__(self, input_dim, hidden_dim=256, num_blocks=3, dropout_rate=0.25):
        super(TabularResNet, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.blocks = nn.ModuleList(
            [ResBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        out = self.head(h)
        return out.squeeze(-1)


def train_pytorch_model(
    X_train, y_train, X_val, y_val, X_test, epochs=20, batch_size=256, lr=1e-3
):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    model = TabularResNet(
        input_dim=X_train.shape[1], hidden_dim=256, num_blocks=3, dropout_rate=0.25
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    best_val_loss = float("inf")
    best_val_preds = None

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
        val_preds_list = []
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                val_loss_sum += criterion(out, by).item() * len(by)
                val_count += len(by)
                val_preds_list.append(out.cpu().numpy())

        val_loss = val_loss_sum / val_count
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_preds = np.concatenate(val_preds_list)
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        test_preds = model(test_tensor).cpu().numpy()

    return best_val_preds, test_preds


# ==========================================
# 3. Model Training & Ensemble Evaluation
# ==========================================
def main():
    target_col = "Cash Disbursements (US$)"
    train_df, test_df = process_and_engineer_features()

    y_true_orig = train_df[target_col].values
    y_target_log1p = train_df["target_log1p"].values
    folds = train_df["fold"].values

    feature_cols = [
        c
        for c in train_df.columns
        if c not in ["id", target_col, "target_log1p", "fold"]
    ]

    # Convert categorical/object columns to integer encoding
    for col in feature_cols:
        if train_df[col].dtype.name in ["category", "object"]:
            cat_series_tr = train_df[col].astype(str)
            cat_series_te = test_df[col].astype(str)
            all_categories = sorted(
                list(set(cat_series_tr.unique()) | set(cat_series_te.unique()))
            )
            cat_map = {cat: i for i, cat in enumerate(all_categories)}
            train_df[col] = cat_series_tr.map(cat_map).fillna(-1).astype(int)
            test_df[col] = cat_series_te.map(cat_map).fillna(-1).astype(int)

    X_all = train_df[feature_cols].values.astype(np.float32)
    X_test_all = test_df[feature_cols].values.astype(np.float32)

    # Impute NaNs with training median if any remain
    train_medians = np.nanmedian(X_all, axis=0)
    train_medians[np.isnan(train_medians)] = 0.0

    inds = np.where(np.isnan(X_all))
    X_all[inds] = np.take(train_medians, inds[1])
    inds_te = np.where(np.isnan(X_test_all))
    X_test_all[inds_te] = np.take(train_medians, inds_te[1])

    n_samples = len(train_df)
    n_test = len(test_df)

    oof_lgb = np.zeros(n_samples)
    oof_xgb = np.zeros(n_samples)
    oof_cat = np.zeros(n_samples)
    oof_nn = np.zeros(n_samples)

    test_lgb = np.zeros(n_test)
    test_xgb = np.zeros(n_test)
    test_cat = np.zeros(n_test)
    test_nn = np.zeros(n_test)

    for fold in range(5):
        tr_idx = np.where(folds != fold)[0]
        va_idx = np.where(folds == fold)[0]

        X_tr, y_tr_log = X_all[tr_idx], y_target_log1p[tr_idx]
        X_va, y_va_log = X_all[va_idx], y_target_log1p[va_idx]

        # 1. LightGBM
        model_lgb = lgb.LGBMRegressor(
            objective="regression",
            metric="rmse",
            n_estimators=1500,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42 + fold,
            n_jobs=-1,
            verbose=-1,
        )
        model_lgb.fit(
            X_tr,
            y_tr_log,
            eval_set=[(X_va, y_va_log)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        oof_lgb[va_idx] = model_lgb.predict(X_va)
        test_lgb += model_lgb.predict(X_test_all) / 5.0

        # 2. XGBoost
        model_xgb = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42 + fold,
            n_jobs=-1,
            tree_method="hist",
            early_stopping_rounds=50,
        )
        model_xgb.fit(X_tr, y_tr_log, eval_set=[(X_va, y_va_log)], verbose=False)
        oof_xgb[va_idx] = model_xgb.predict(X_va)
        test_xgb += model_xgb.predict(X_test_all) / 5.0

        # 3. CatBoost
        model_cat = cb.CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=1500,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=42 + fold,
            verbose=0,
            thread_count=-1,
            early_stopping_rounds=50,
        )
        model_cat.fit(X_tr, y_tr_log, eval_set=(X_va, y_va_log), verbose=0)
        oof_cat[va_idx] = model_cat.predict(X_va)
        test_cat += model_cat.predict(X_test_all) / 5.0

        # 4. PyTorch TabularResNet
        val_nn_log, te_nn_log = train_pytorch_model(
            X_tr, y_tr_log, X_va, y_va_log, X_test_all, epochs=20, batch_size=256
        )
        oof_nn[va_idx] = val_nn_log
        test_nn += te_nn_log / 5.0

    # Back-transform all predictions from log1p to original target scale
    oof_lgb_orig = np.expm1(np.maximum(0, oof_lgb))
    oof_xgb_orig = np.expm1(np.maximum(0, oof_xgb))
    oof_cat_orig = np.expm1(np.maximum(0, oof_cat))
    oof_nn_orig = np.expm1(np.maximum(0, oof_nn))

    test_lgb_orig = np.expm1(np.maximum(0, test_lgb))
    test_xgb_orig = np.expm1(np.maximum(0, test_xgb))
    test_cat_orig = np.expm1(np.maximum(0, test_cat))
    test_nn_orig = np.expm1(np.maximum(0, test_nn))

    # Post-processing scale calibration (multiplicative and additive shift) per model
    calibrated_oofs = []
    calibrated_tests = []
    raw_oofs = [oof_lgb_orig, oof_xgb_orig, oof_cat_orig, oof_nn_orig]
    raw_tests = [test_lgb_orig, test_xgb_orig, test_cat_orig, test_nn_orig]

    for oof_p, test_p in zip(raw_oofs, raw_tests):
        def scale_obj(params):
            a, b = params
            pred = np.maximum(0, a * oof_p + b)
            return np.sqrt(mean_squared_error(y_true_orig, pred))

        res = minimize(scale_obj, [1.0, 0.0], method="L-BFGS-B", bounds=[(0, None), (None, None)])
        a_opt, b_opt = res.x

        calibrated_oofs.append(np.maximum(0, a_opt * oof_p + b_opt))
        calibrated_tests.append(np.maximum(0, a_opt * test_p + b_opt))

    # Constrained Ensemble Optimization (SLSQP)
    oof_matrix = np.column_stack(calibrated_oofs)
    test_matrix = np.column_stack(calibrated_tests)

    def blend_objective(weights):
        w = np.array(weights)
        w = w / np.sum(w)
        blend_oof = oof_matrix @ w
        return np.sqrt(mean_squared_error(y_true_orig, blend_oof))

    init_weights = np.ones(4) / 4.0
    bounds = [(0, 1)] * 4
    constraints = {"type": "eq", "fun": lambda w: 1.0 - sum(w)}

    opt_res = minimize(
        blend_objective,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    best_weights = opt_res.x / np.sum(opt_res.x)

    final_oof_preds = oof_matrix @ best_weights
    final_oof_rmse = np.sqrt(mean_squared_error(y_true_orig, final_oof_preds))

    # Generate final predictions for submission
    final_test_preds = np.maximum(0, test_matrix @ best_weights)

    os.makedirs("./submission", exist_ok=True)
    submission_path = "./submission/submission.csv"

    sub = pd.DataFrame({"id": test_df["id"].astype(int), target_col: final_test_preds})
    sub.to_csv(submission_path, index=False)

    print(f"Final Validation Score: {final_oof_rmse}")


if __name__ == "__main__":
    main()