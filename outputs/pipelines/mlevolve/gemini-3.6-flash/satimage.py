import os
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ==========================================
# 1. MODEL ARCHITECTURE DEFINITIONS
# ==========================================
class SpatialSpectralCNN(nn.Module):
    """
    2D Spatial-Spectral Convolutional Neural Network operating on (4, 3, 3) Landsat spatial patches
    alongside a tabular feature branch.
    """

    def __init__(
        self,
        tabular_dim: int,
        num_classes: int = 6,
        cnn_channels: int = 32,
        tabular_hidden_dim: int = 128,
        dropout_rate: float = 0.25,
    ):
        super().__init__()

        # Spatial-Spectral 2D Conv Branch on (4, 3, 3) patches
        self.conv_branch = nn.Sequential(
            nn.Conv2d(4, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.GELU(),
            nn.Conv2d(cnn_channels, cnn_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels * 2),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # Tabular Feature Branch
        self.tabular_branch = nn.Sequential(
            nn.Linear(tabular_dim, tabular_hidden_dim),
            nn.BatchNorm1d(tabular_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(tabular_hidden_dim, tabular_hidden_dim),
            nn.BatchNorm1d(tabular_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # Combined Fusion Head
        combined_dim = (cnn_channels * 2) + tabular_hidden_dim
        self.head = nn.Sequential(
            nn.Linear(combined_dim, tabular_hidden_dim),
            nn.LayerNorm(tabular_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(tabular_hidden_dim, num_classes),
        )

    def forward(self, x_patch: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        spatial_feats = self.conv_branch(x_patch)
        tabular_feats = self.tabular_branch(x_tab)
        combined = torch.cat([spatial_feats, tabular_feats], dim=1)
        logits = self.head(combined)
        return logits


def get_lgb_model_params(num_classes: int = 6, seed: int = 42) -> dict:
    """
    Returns LightGBM hyperparameter configuration tuned for multi-spectral tabular classification.
    """
    return {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": 7,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 20,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }


def create_torch_components(
    input_dim: int,
    num_classes: int = 6,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 100,
    device: str = "cpu",
):
    """
    Instantiates PyTorch Spatial-Spectral CNN Model, Cross Entropy Loss with Label Smoothing,
    AdamW Optimizer, and Cosine Annealing Learning Rate Scheduler.
    """
    model = SpatialSpectralCNN(
        tabular_dim=input_dim,
        num_classes=num_classes,
        cnn_channels=32,
        tabular_hidden_dim=128,
        dropout_rate=0.25,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    return model, criterion, optimizer, scheduler


# ==========================================
# 2. FEATURE ENGINEERING FUNCTION
# ==========================================
def engineer_landsat_features(df_in: pd.DataFrame, cols: list) -> np.ndarray:
    """
    Engineers remote sensing spectral indices and 3x3 spatial context features
    for 4-band multi-spectral Landsat MSS data.
    """
    df_cols = list(cols)
    assert (
        len(df_cols) >= 36
    ), f"Expected at least 36 spatial-spectral features, found {len(df_cols)}"
    spectral_cols = df_cols[:36]

    X_mat = df_in[spectral_cols].values.astype(np.float32)
    N = X_mat.shape[0]

    # Reshape to (N, 9 pixels, 4 bands)
    pixels_bands = X_mat.reshape(N, 9, 4)

    # Separate bands: Band 0 (Green), Band 1 (Red), Band 2 (NIR1), Band 3 (NIR2)
    g_pixels = pixels_bands[:, :, 0]
    r_pixels = pixels_bands[:, :, 1]
    nir1_pixels = pixels_bands[:, :, 2]
    nir2_pixels = pixels_bands[:, :, 3]

    feats = [X_mat]  # Raw spectral attributes

    # 2D Filter Kernels
    k_sobel_x = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=np.float32)
    k_sobel_y = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=np.float32)
    k_gauss = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=np.float32) / 16.0
    k_laplacian_2d = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    # A. Spatial Statistics & 2D Texture Filters per Spectral Band across 3x3 Grid
    for b_mat in [g_pixels, r_pixels, nir1_pixels, nir2_pixels]:
        mean_b = np.mean(b_mat, axis=1, keepdims=True)
        std_b = np.std(b_mat, axis=1, keepdims=True)
        min_b = np.min(b_mat, axis=1, keepdims=True)
        max_b = np.max(b_mat, axis=1, keepdims=True)
        range_b = max_b - min_b

        central_b = b_mat[:, 4:5]  # Index 4 is the central pixel
        diff_mean_b = central_b - mean_b
        ratio_mean_b = central_b / (mean_b + 1e-5)

        # Spatial Gradients around central pixel
        horiz_grad = np.abs(b_mat[:, 5:6] - b_mat[:, 3:4])  # Right - Left
        vert_grad = np.abs(b_mat[:, 7:8] - b_mat[:, 1:2])  # Bottom - Top
        diag1_grad = np.abs(b_mat[:, 8:9] - b_mat[:, 0:1])  # BotRight - TopLeft
        diag2_grad = np.abs(b_mat[:, 6:7] - b_mat[:, 2:3])  # BotLeft - TopRight
        laplacian = 4.0 * central_b - (
            b_mat[:, 1:2] + b_mat[:, 3:4] + b_mat[:, 5:6] + b_mat[:, 7:8]
        )
        edge_mag = np.sqrt(horiz_grad**2 + vert_grad**2)

        # 2D Spatial Texture Operators across 3x3 Grid
        b_grid = b_mat.reshape(N, 3, 3)
        sobel_x_resp = np.sum(b_grid * k_sobel_x, axis=(1, 2), keepdims=True).reshape(-1, 1)
        sobel_y_resp = np.sum(b_grid * k_sobel_y, axis=(1, 2), keepdims=True).reshape(-1, 1)
        gauss_resp = np.sum(b_grid * k_gauss, axis=(1, 2), keepdims=True).reshape(-1, 1)
        laplace_2d_resp = np.sum(b_grid * k_laplacian_2d, axis=(1, 2), keepdims=True).reshape(-1, 1)
        sobel_mag_2d = np.sqrt(sobel_x_resp**2 + sobel_y_resp**2)

        feats.extend(
            [
                mean_b,
                std_b,
                min_b,
                max_b,
                range_b,
                diff_mean_b,
                ratio_mean_b,
                horiz_grad,
                vert_grad,
                diag1_grad,
                diag2_grad,
                laplacian,
                edge_mag,
                sobel_x_resp,
                sobel_y_resp,
                gauss_resp,
                laplace_2d_resp,
                sobel_mag_2d,
            ]
        )

    # B. Compute 3x3 Spatial Distribution Maps of Spectral Indices
    ndvi1_map = (nir1_pixels - r_pixels) / (nir1_pixels + r_pixels + 1e-5)
    ndvi2_map = (nir2_pixels - r_pixels) / (nir2_pixels + r_pixels + 1e-5)
    ndwi_map = (g_pixels - nir1_pixels) / (g_pixels + nir1_pixels + 1e-5)
    savi1_map = 1.5 * (nir1_pixels - r_pixels) / (nir1_pixels + r_pixels + 0.5)
    savi2_map = 1.5 * (nir2_pixels - r_pixels) / (nir2_pixels + r_pixels + 0.5)
    bright_map = 0.35 * g_pixels + 0.50 * r_pixels + 0.60 * nir1_pixels + 0.50 * nir2_pixels
    green_map = -0.28 * g_pixels - 0.66 * r_pixels + 0.58 * nir1_pixels + 0.39 * nir2_pixels

    # Additional Spectral Indices: EVI, RVI, DVI, Soil Brightness Index
    evi1_map = 2.5 * (nir1_pixels - r_pixels) / (nir1_pixels + 6.0 * r_pixels - 7.5 * g_pixels + 1.0 + 1e-5)
    evi2_map = 2.5 * (nir2_pixels - r_pixels) / (nir2_pixels + 6.0 * r_pixels - 7.5 * g_pixels + 1.0 + 1e-5)
    rvi1_map = (nir1_pixels + 1e-5) / (r_pixels + 1e-5)
    rvi2_map = (nir2_pixels + 1e-5) / (r_pixels + 1e-5)
    dvi1_map = nir1_pixels - r_pixels
    dvi2_map = nir2_pixels - r_pixels
    sbi_map = np.sqrt(r_pixels**2 + nir1_pixels**2 + 1e-5)

    index_maps = [
        ndvi1_map, ndvi2_map, ndwi_map, savi1_map, savi2_map,
        bright_map, green_map, evi1_map, evi2_map, rvi1_map, rvi2_map,
        dvi1_map, dvi2_map, sbi_map
    ]

    # Central-only spectral ratios
    g_c = g_pixels[:, 4:5]
    r_c = r_pixels[:, 4:5]
    nir1_c = nir1_pixels[:, 4:5]
    nir2_c = nir2_pixels[:, 4:5]
    nir_ratio_c = (nir1_c + 1e-5) / (nir2_c + 1e-5)
    rg_ratio_c = (r_c + 1e-5) / (g_c + 1e-5)
    feats.extend([nir_ratio_c, rg_ratio_c])

    surround_indices = [0, 1, 2, 3, 5, 6, 7, 8]
    corner_indices = [0, 2, 6, 8]
    edge_indices = [1, 3, 5, 7]

    for idx_map in index_maps:
        # Full 9-pixel index map
        feats.append(idx_map)

        # Spatial distribution features per index map
        s_mean = np.mean(idx_map, axis=1, keepdims=True)
        s_std = np.std(idx_map, axis=1, keepdims=True)
        s_min = np.min(idx_map, axis=1, keepdims=True)
        s_max = np.max(idx_map, axis=1, keepdims=True)

        center_val = idx_map[:, 4:5]
        surround_mean = np.mean(idx_map[:, surround_indices], axis=1, keepdims=True)
        center_vs_surround = center_val - surround_mean

        corner_mean = np.mean(idx_map[:, corner_indices], axis=1, keepdims=True)
        edge_mean = np.mean(idx_map[:, edge_indices], axis=1, keepdims=True)
        corner_vs_edge = corner_mean - edge_mean

        feats.extend([
            s_mean,
            s_std,
            s_min,
            s_max,
            center_vs_surround,
            corner_vs_edge,
        ])

    X_engineered = np.hstack(
        [f if len(f.shape) == 2 else f.reshape(-1, 1) for f in feats]
    )
    return X_engineered.astype(np.float32)


# ==========================================
# 3. MAIN PIPELINE EXECUTION
# ==========================================
def main():
    # Directories and Reproducibility setup
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load raw data
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"

    df_train_raw = pd.read_parquet(train_path)
    df_test_raw = pd.read_parquet(test_path)

    target_col = "class"
    if target_col not in df_train_raw.columns:
        possible_targets = [
            c
            for c in df_train_raw.columns
            if "class" in c.lower() or "target" in c.lower()
        ]
        target_col = possible_targets[0]

    feature_cols = [
        c for c in df_train_raw.columns if c != target_col and c.lower() != "id"
    ]

    if "id" in df_test_raw.columns:
        test_ids = df_test_raw["id"].values
        test_feature_cols = [
            c for c in df_test_raw.columns if c != "id" and c != target_col
        ]
    elif "ID" in df_test_raw.columns:
        test_ids = df_test_raw["ID"].values
        test_feature_cols = [
            c for c in df_test_raw.columns if c != "ID" and c != target_col
        ]
    else:
        test_ids = np.arange(len(df_test_raw))
        test_feature_cols = [c for c in df_test_raw.columns if c != target_col]

    X_raw = df_train_raw[feature_cols].copy()
    y_raw = df_train_raw[target_col].values
    X_test_raw = df_test_raw[test_feature_cols].copy()

    # Target Label Encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    classes = label_encoder.classes_
    num_classes = len(classes)

    # Feature Engineering
    X = engineer_landsat_features(X_raw, feature_cols)
    X_test = engineer_landsat_features(X_test_raw, test_feature_cols)

    # 5-Fold Stratified Cross-Validation Setup
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    N = len(X)
    N_test = len(X_test)

    oof_lgb = np.zeros((N, num_classes))
    oof_xgb = np.zeros((N, num_classes))
    oof_cb = np.zeros((N, num_classes))
    oof_nn = np.zeros((N, num_classes))

    test_lgb_probs = np.zeros((N_test, num_classes))
    test_xgb_probs = np.zeros((N_test, num_classes))
    test_cb_probs = np.zeros((N_test, num_classes))
    test_nn_probs = np.zeros((N_test, num_classes))

    lgb_base_params = get_lgb_model_params(num_classes=num_classes, seed=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        # 1. LightGBM
        lgb_params = lgb_base_params.copy()
        lgb_params["random_state"] = 42 + fold
        train_lgb_ds = lgb.Dataset(X_tr, label=y_tr)
        val_lgb_ds = lgb.Dataset(X_va, label=y_va, reference=train_lgb_ds)

        lgb_model = lgb.train(
            lgb_params,
            train_lgb_ds,
            num_boost_round=1000,
            valid_sets=[val_lgb_ds],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )

        oof_lgb[val_idx] = lgb_model.predict(X_va, num_iteration=lgb_model.best_iteration)
        test_lgb_probs += lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration) / n_splits

        # 2. XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=2,
            objective="multi:softprob",
            num_class=num_classes,
            random_state=42 + fold,
            n_jobs=-1,
            eval_metric="mlogloss",
            early_stopping_rounds=50,
        )
        xgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )

        oof_xgb[val_idx] = xgb_model.predict_proba(X_va)
        test_xgb_probs += xgb_model.predict_proba(X_test) / n_splits

        # 3. CatBoost
        cb_model = cb.CatBoostClassifier(
            iterations=1000,
            learning_rate=0.03,
            depth=6,
            loss_function="MultiClass",
            eval_metric="MultiClass",
            random_seed=42 + fold,
            thread_count=-1,
            early_stopping_rounds=50,
            verbose=False,
        )
        cb_model.fit(
            X_tr,
            y_tr,
            eval_set=(X_va, y_va),
            verbose=False,
        )

        oof_cb[val_idx] = cb_model.predict_proba(X_va)
        test_cb_probs += cb_model.predict_proba(X_test) / n_splits

        # 4. PyTorch 2D Spatial-Spectral CNN
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)
        X_test_scaled = scaler.transform(X_test)

        def make_patch_tensor(x_scaled):
            return torch.tensor(
                x_scaled[:, :36].reshape(-1, 3, 3, 4).transpose(0, 3, 1, 2),
                dtype=torch.float32,
            )

        train_patch = make_patch_tensor(X_tr_scaled)
        val_patch = make_patch_tensor(X_va_scaled)
        test_patch = make_patch_tensor(X_test_scaled)

        train_dataset = TensorDataset(
            train_patch,
            torch.tensor(X_tr_scaled, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.long),
        )
        val_dataset = TensorDataset(
            val_patch,
            torch.tensor(X_va_scaled, dtype=torch.float32),
            torch.tensor(y_va, dtype=torch.long),
        )
        test_dataset = TensorDataset(
            test_patch,
            torch.tensor(X_test_scaled, dtype=torch.float32),
        )

        train_loader = DataLoader(
            train_dataset, batch_size=128, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=256, shuffle=False, drop_last=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=256, shuffle=False, drop_last=False
        )

        input_dim = X_tr_scaled.shape[1]
        epochs = 80

        torch.manual_seed(42 + fold)
        torch_model, criterion, optimizer, scheduler = create_torch_components(
            input_dim=input_dim,
            num_classes=num_classes,
            learning_rate=1e-3,
            weight_decay=1e-4,
            epochs=epochs,
            device=device,
        )

        best_val_loss = float("inf")
        best_model_path = f"./working/best_spatial_cnn_fold{fold}.pt"

        for epoch in range(1, epochs + 1):
            torch_model.train()
            train_loss = 0.0
            for b_patch, b_tab, by in train_loader:
                b_patch, b_tab, by = b_patch.to(device), b_tab.to(device), by.to(device)
                optimizer.zero_grad()
                out = torch_model(b_patch, b_tab)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * b_patch.size(0)

            scheduler.step()

            torch_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for b_patch, b_tab, by in val_loader:
                    b_patch, b_tab, by = b_patch.to(device), b_tab.to(device), by.to(device)
                    out = torch_model(b_patch, b_tab)
                    loss = criterion(out, by)
                    val_loss += loss.item() * b_patch.size(0)

            val_loss /= len(val_dataset)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(torch_model.state_dict(), best_model_path)

        torch_model.load_state_dict(torch.load(best_model_path))
        torch_model.eval()

        def predict_probs(model, loader):
            probs_list = []
            with torch.no_grad():
                for batch in loader:
                    b_patch = batch[0].to(device)
                    b_tab = batch[1].to(device)
                    logits = model(b_patch, b_tab)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    probs_list.append(probs)
            return np.vstack(probs_list)

        nn_val_probs = predict_probs(torch_model, val_loader)
        nn_test_probs = predict_probs(torch_model, test_loader)

        oof_nn[val_idx] = nn_val_probs
        test_nn_probs += nn_test_probs / n_splits

    # Logistic Regression Stacking Meta-Learner on OOF Probability Distributions
    oof_meta_features = np.hstack([oof_lgb, oof_xgb, oof_cb, oof_nn])
    test_meta_features = np.hstack([test_lgb_probs, test_xgb_probs, test_cb_probs, test_nn_probs])

    meta_model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42,
        multi_class="multinomial",
        solver="lbfgs",
    )
    meta_model.fit(oof_meta_features, y)

    oof_final_probs = meta_model.predict_proba(oof_meta_features)
    oof_final_preds = np.argmax(oof_final_probs, axis=1)
    final_val_acc = accuracy_score(y, oof_final_preds)

    test_final_probs = meta_model.predict_proba(test_meta_features)
    final_test_preds_encoded = np.argmax(test_final_probs, axis=1)

    # Map target indices back to original satellite class categories
    final_test_preds_class = classes[final_test_preds_encoded]

    # Save final submission file
    submission_path = "./submission/submission.csv"
    sub_df = pd.DataFrame({"id": test_ids, "class": final_test_preds_class})
    sub_df.to_csv(submission_path, index=False)

    # Print Final Validation Score
    print(f"Final Validation Score: {final_val_acc}")


if __name__ == "__main__":
    main()