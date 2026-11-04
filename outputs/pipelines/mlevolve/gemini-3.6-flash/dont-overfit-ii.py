import os
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, rankdata, skew
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.optim as optim

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# PyTorch regularized linear model architecture with high input dropout
class SparseLinearNet(nn.Module):
    """PyTorch regularized linear neural model with heavy input dropout

    for high-dimensional low-sample-size binary classification.
    """

    def __init__(self, input_dim: int, dropout_rate: float = 0.5):
        super(SparseLinearNet, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return self.fc(x)


class ElasticNetLoss(nn.Module):
    """Binary Cross Entropy loss augmented with explicit L1 and L2 weight penalty terms."""

    def __init__(self, l1_lambda: float = 0.005, l2_lambda: float = 0.01):
        super(ElasticNetLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, model: nn.Module
    ) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        l1_reg = sum(p.abs().sum() for p in model.parameters())
        l2_reg = sum((p**2).sum() for p in model.parameters())
        return bce_loss + self.l1_lambda * l1_reg + self.l2_lambda * l2_reg


def get_sklearn_models(seed: int = 42):
    """Returns a dictionary of strongly regularized scikit-learn models specifically tuned for high-dimensional low-sample-size datasets."""
    return {
        "logreg_l1": LogisticRegression(
            penalty="l1", solver="liblinear", C=0.1, random_state=seed
        ),
        "logreg_elasticnet": LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=0.1,
            max_iter=2000,
            random_state=seed,
        ),
        "linear_svc": SVC(kernel="linear", C=0.05, probability=True, random_state=seed),
        "ridge_cls": RidgeClassifier(alpha=100.0, random_state=seed),
        "sgd_elasticnet": SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            l1_ratio=0.7,
            alpha=0.1,
            random_state=seed,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=200,
            max_depth=3,
            min_samples_leaf=10,
            random_state=seed,
        ),
    }


def predict_sklearn_model(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        return model.decision_function(X)
    else:
        return model.predict(X)


def train_pytorch_fold(X_tr, y_tr, X_val, y_val, X_te, device):
    input_dim = X_tr.shape[1]
    model = SparseLinearNet(input_dim=input_dim, dropout_rate=0.5).to(device)
    criterion = ElasticNetLoss(l1_lambda=0.005, l2_lambda=0.01)
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.01)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
    X_te_t = torch.tensor(X_te, dtype=torch.float32).to(device)

    best_val_loss = float("inf")
    best_model_weights = None
    patience = 25
    patience_counter = 0

    for epoch in range(150):
        model.train()
        optimizer.zero_grad()
        logits = model(X_tr_t)
        loss = criterion(logits, y_tr_t, model)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t, model).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
    model.eval()

    with torch.no_grad():
        val_preds = torch.sigmoid(model(X_val_t)).cpu().numpy().ravel()
        test_preds = torch.sigmoid(model(X_te_t)).cpu().numpy().ravel()

    return val_preds, test_preds


def to_rank_percentile(arr):
    return rankdata(arr) / len(arr)


def process_data(train_df, test_df, target_col="target", id_col="id"):
    y = train_df[target_col].values.astype(int)
    train_ids = train_df[id_col].values
    test_ids = test_df[id_col].values

    feature_cols = [col for col in train_df.columns if col not in [id_col, target_col]]

    X_train_raw = train_df[feature_cols].values.astype(np.float32)
    X_test_raw = test_df[feature_cols].values.astype(np.float32)

    # 1. Extract row-wise statistical summary features across the feature space
    def extract_row_stats(X):
        stats = []
        stats.append(np.mean(X, axis=1, keepdims=True))
        stats.append(np.std(X, axis=1, keepdims=True))
        stats.append(np.min(X, axis=1, keepdims=True))
        stats.append(np.max(X, axis=1, keepdims=True))
        stats.append(np.median(X, axis=1, keepdims=True))
        stats.append(np.quantile(X, 0.25, axis=1, keepdims=True))
        stats.append(np.quantile(X, 0.75, axis=1, keepdims=True))
        stats.append(skew(X, axis=1, bias=False)[:, None])
        stats.append(kurtosis(X, axis=1, bias=False)[:, None])
        stats.append(np.sum(X > 0, axis=1, keepdims=True))
        stats.append(np.sum(X < 0, axis=1, keepdims=True))
        stats.append(np.linalg.norm(X, ord=2, axis=1, keepdims=True))
        return np.hstack(stats)

    row_stats_train = extract_row_stats(X_train_raw)
    row_stats_test = extract_row_stats(X_test_raw)

    # 2. Identify top univariate features based on target correlation in training set
    correlations = np.abs(
        [np.corrcoef(X_train_raw[:, i], y)[0, 1] for i in range(X_train_raw.shape[1])]
    )
    top_feature_indices = np.argsort(correlations)[::-1][:15]

    # 3. Create key pairwise interaction features (products and differences)
    def extract_top_interactions(X, indices):
        X_top = X[:, indices]
        n_samples, n_top = X_top.shape
        interactions = []

        for i in range(n_top):
            for j in range(i + 1, min(i + 5, n_top)):
                prod = (X_top[:, i] * X_top[:, j])[:, None]
                diff = (X_top[:, i] - X_top[:, j])[:, None]
                interactions.extend([prod, diff])

        return np.hstack(interactions)

    train_inter = extract_top_interactions(X_train_raw, top_feature_indices)
    test_inter = extract_top_interactions(X_test_raw, top_feature_indices)

    # 4. Fit PCA on training features
    pca = PCA(n_components=10, random_state=42)
    pca_train = pca.fit_transform(X_train_raw)
    pca_test = pca.transform(X_test_raw)

    # 5. Quantile Transformation for normal feature scaling
    qt = QuantileTransformer(
        output_distribution="normal", random_state=42, n_quantiles=min(len(y), 100)
    )
    X_train_qt = qt.fit_transform(X_train_raw)
    X_test_qt = qt.transform(X_test_raw)

    # Combine engineered features
    X_train_full = np.hstack(
        [X_train_raw, X_train_qt, row_stats_train, train_inter, pca_train]
    )
    X_test_full = np.hstack(
        [X_test_raw, X_test_qt, row_stats_test, test_inter, pca_test]
    )

    # Standardize all engineered features strictly using training set statistics
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test_full)

    return X_train_scaled, y, X_test_scaled, test_ids


def main():
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    # Load raw data
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    X_train, y_train, X_test, test_ids = process_data(train_df, test_df)

    # Stratified 5-Fold split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(X_train, y_train))

    n_samples = len(y_train)
    n_test = len(X_test)
    n_folds = len(folds)

    sklearn_models = get_sklearn_models(seed=42)
    model_names = list(sklearn_models.keys()) + ["pytorch_linearnet"]

    oof_preds = {m: np.zeros(n_samples) for m in model_names}
    test_preds = {m: np.zeros(n_test) for m in model_names}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 5-Fold Cross-Validation loop
    for fold, (tr_idx, val_idx) in enumerate(folds):
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]

        # 1. Train Scikit-Learn models
        fold_sklearn_models = get_sklearn_models(seed=42 + fold)
        for m_name, model in fold_sklearn_models.items():
            model.fit(X_tr, y_tr)
            val_p = predict_sklearn_model(model, X_val)
            test_p = predict_sklearn_model(model, X_test)

            oof_preds[m_name][val_idx] = val_p
            test_preds[m_name] += test_p / n_folds

        # 2. Train PyTorch SparseLinearNet
        py_val_p, py_test_p = train_pytorch_fold(
            X_tr, y_tr, X_val, y_val, X_test, device
        )
        oof_preds["pytorch_linearnet"][val_idx] = py_val_p
        test_preds["pytorch_linearnet"] += py_test_p / n_folds

    # Compute individual model OOF AUC scores and rank percentiles
    model_scores = {}
    oof_ranks = {}
    test_ranks = {}

    for m in model_names:
        score = roc_auc_score(y_train, oof_preds[m])
        model_scores[m] = score
        oof_ranks[m] = to_rank_percentile(oof_preds[m])
        test_ranks[m] = to_rank_percentile(test_preds[m])

    # Rank-averaged dynamic weighting based on performance above baseline
    weights = {}
    for m, score in model_scores.items():
        weights[m] = max(0.001, (score - 0.5) ** 2)

    total_weight = sum(weights.values())
    for m in weights:
        weights[m] /= total_weight

    # Blended ensemble predictions
    blended_oof_rank = np.zeros(n_samples)
    blended_test_rank = np.zeros(n_test)

    for m in model_names:
        blended_oof_rank += weights[m] * oof_ranks[m]
        blended_test_rank += weights[m] * test_ranks[m]

    final_score = roc_auc_score(y_train, blended_oof_rank)

    # Save submission file
    submission_df = pd.DataFrame(
        {"id": test_ids, "target": to_rank_percentile(blended_test_rank)}
    )
    submission_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()
