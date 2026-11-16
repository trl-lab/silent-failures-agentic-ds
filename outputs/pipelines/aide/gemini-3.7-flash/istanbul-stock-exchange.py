import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNetCV,
    HuberRegressor,
    LassoCV,
    RidgeCV,
)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Create working directory if it doesn't exist
os.makedirs("./working", exist_ok=True)

# Load train and test data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Identify target column
target_col = "ISE"
if target_col not in train_df.columns:
    for col in train_df.columns:
        if "ISE" in col:
            target_col = col
            break

# Separate target
y = train_df[target_col].values


# Feature engineering function
def extract_features(df):
    data = df.copy()

    # Process date column
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"])
        data["dayofweek"] = data["date"].dt.dayofweek
        data["month"] = data["date"].dt.month
        data["day"] = data["date"].dt.day
        data["dayofyear"] = data["date"].dt.dayofyear

        # Cyclical calendar encodings
        data["sin_dow"] = np.sin(2 * np.pi * data["dayofweek"] / 5.0)
        data["cos_dow"] = np.cos(2 * np.pi * data["dayofweek"] / 5.0)
        data["sin_month"] = np.sin(2 * np.pi * data["month"] / 12.0)
        data["cos_month"] = np.cos(2 * np.pi * data["month"] / 12.0)
        data = data.drop(columns=["date"])

    # Drop target column if present
    cols_to_drop = [c for c in data.columns if c == target_col]
    if cols_to_drop:
        data = data.drop(columns=cols_to_drop)

    # Core market columns
    market_cols = [
        c
        for c in [
            "SP",
            "DAX",
            "FTSE",
            "NIKKEI",
            "BOVESPA",
            "EU",
            "EM",
            "MSCE_EU",
            "MSCI_EM",
        ]
        if c in data.columns
    ]

    if market_cols:
        # Cross-market summary statistics
        data["market_mean"] = data[market_cols].mean(axis=1)
        data["market_std"] = data[market_cols].std(axis=1)
        data["market_max"] = data[market_cols].max(axis=1)
        data["market_min"] = data[market_cols].min(axis=1)
        data["market_median"] = data[market_cols].median(axis=1)
        data["market_skew"] = data[market_cols].skew(axis=1)
        data["market_range"] = data["market_max"] - data["market_min"]

        # Full pairwise market spreads and interactions
        for i in range(len(market_cols)):
            for j in range(i + 1, len(market_cols)):
                col_i = market_cols[i]
                col_j = market_cols[j]
                data[f"spread_{col_i}_{col_j}"] = data[col_i] - data[col_j]
                data[f"inter_{col_i}_{col_j}"] = data[col_i] * data[col_j]

    return data


# Extract features
X_train = extract_features(train_df)
X_test = extract_features(test_df)

# Align columns
all_cols = [c for c in X_train.columns if c in X_test.columns]
X_train = X_train[all_cols].fillna(0)
X_test = X_test[all_cols].fillna(0)

# Setup 5-Fold Cross Validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

models_list = [
    "RidgeCV",
    "LassoCV",
    "ElasticNetCV",
    "BayesianRidge",
    "Huber",
    "XGBoost",
]
num_models = len(models_list)
oof_matrix = np.zeros((len(X_train), num_models))
test_matrix = np.zeros((len(X_test), num_models))

# Fine-grained alpha grids for regularized linear models
ridge_alphas = np.logspace(-6, 3, 100)
lasso_alphas = np.logspace(-7, 1, 100)
l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]

# XGBoost Parameters
xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.03,
    "max_depth": 3,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.01,
    "reg_lambda": 0.1,
    "random_state": SEED,
    "n_estimators": 350,
}

market_cols = [
    c
    for c in ["SP", "DAX", "FTSE", "NIKKEI", "BOVESPA", "EU", "EM"]
    if c in X_train.columns
]

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y)):
    X_tr = X_train.iloc[train_idx].copy()
    y_tr = y[train_idx]
    X_va = X_train.iloc[val_idx].copy()
    y_va = y[val_idx]
    X_te = X_test.copy()

    # Fold-specific PCA features to avoid leakage
    if len(market_cols) >= 3:
        pca = PCA(n_components=3, random_state=SEED)
        tr_pca = pca.fit_transform(X_tr[market_cols])
        va_pca = pca.transform(X_va[market_cols])
        te_pca = pca.transform(X_te[market_cols])
        for i in range(3):
            X_tr[f"market_pca_{i}"] = tr_pca[:, i]
            X_va[f"market_pca_{i}"] = va_pca[:, i]
            X_te[f"market_pca_{i}"] = te_pca[:, i]

    # Standard scale features
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)
    X_te_scaled = scaler.transform(X_te)

    # Model 0: RidgeCV
    m_ridge = RidgeCV(alphas=ridge_alphas)
    m_ridge.fit(X_tr_scaled, y_tr)
    oof_matrix[val_idx, 0] = m_ridge.predict(X_va_scaled)
    test_matrix[:, 0] += m_ridge.predict(X_te_scaled) / n_splits

    # Model 1: LassoCV
    m_lasso = LassoCV(alphas=lasso_alphas, max_iter=5000, random_state=SEED)
    m_lasso.fit(X_tr_scaled, y_tr)
    oof_matrix[val_idx, 1] = m_lasso.predict(X_va_scaled)
    test_matrix[:, 1] += m_lasso.predict(X_te_scaled) / n_splits

    # Model 2: ElasticNetCV
    m_enet = ElasticNetCV(
        l1_ratio=l1_ratios, alphas=lasso_alphas, max_iter=5000, random_state=SEED
    )
    m_enet.fit(X_tr_scaled, y_tr)
    oof_matrix[val_idx, 2] = m_enet.predict(X_va_scaled)
    test_matrix[:, 2] += m_enet.predict(X_te_scaled) / n_splits

    # Model 3: Bayesian Ridge
    m_bridge = BayesianRidge(max_iter=1000)
    m_bridge.fit(X_tr_scaled, y_tr)
    oof_matrix[val_idx, 3] = m_bridge.predict(X_va_scaled)
    test_matrix[:, 3] += m_bridge.predict(X_te_scaled) / n_splits

    # Model 4: Huber Regressor with low alpha
    m_huber = HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=2000)
    m_huber.fit(X_tr_scaled, y_tr)
    oof_matrix[val_idx, 4] = m_huber.predict(X_va_scaled)
    test_matrix[:, 4] += m_huber.predict(X_te_scaled) / n_splits

    # Model 5: XGBoost
    m_xgb = xgb.XGBRegressor(**xgb_params)
    m_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_matrix[val_idx, 5] = m_xgb.predict(X_va)
    test_matrix[:, 5] += m_xgb.predict(X_te) / n_splits


# Optimize ensemble weights using bounded SLSQP
def loss_func(weights):
    w = weights / np.sum(weights)
    pred = np.dot(oof_matrix, w)
    return np.sqrt(mean_squared_error(y, pred))


init_weights = np.ones(num_models) / num_models
bounds = [(0, 1) for _ in range(num_models)]
res = minimize(loss_func, init_weights, bounds=bounds, method="SLSQP")
opt_weights = res.x / np.sum(res.x)

final_oof_preds = np.dot(oof_matrix, opt_weights)
final_test_preds = np.dot(test_matrix, opt_weights)

cv_rmse = np.sqrt(mean_squared_error(y, final_oof_preds))
for name, w in zip(models_list, opt_weights):
    print(f"Weight {name}: {w:.4f}")
print(f"5-Fold CV RMSE: {cv_rmse:.6f}")

# Prepare and save submission
submission = pd.DataFrame(
    {"id": np.arange(len(final_test_preds)), "ISE": final_test_preds}
)
submission.to_csv("./working/submission.csv", index=False)
print("Submission successfully saved to ./working/submission.csv")
