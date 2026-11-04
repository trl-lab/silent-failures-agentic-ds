import copy
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from catboost import CatBoostClassifier
import lightgbm as lgb
from scipy.optimize import minimize
from xgboost import XGBClassifier

# Ensure output directories exist
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working/models", exist_ok=True)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================================
# 1. Feature Engineering and Data Processing
# =========================================================================
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

base_features = [
    "follower_count",
    "following_count",
    "listed_count",
    "mentions_received",
    "mentions_sent",
    "network_feature_1",
    "network_feature_2",
    "network_feature_3",
    "posts",
    "retweets_received",
    "retweets_sent",
]

# Precompute empirical percentile pools across all entities in train and test
pooled_distributions = {}
for feat in base_features:
    all_vals = np.concatenate([
        train_df[f"A_{feat}"].values.astype(np.float64),
        train_df[f"B_{feat}"].values.astype(np.float64),
        test_df[f"A_{feat}"].values.astype(np.float64),
        test_df[f"B_{feat}"].values.astype(np.float64),
    ])
    pooled_distributions[feat] = np.sort(all_vals)


def get_percentile_rank(feat: str, values: np.ndarray) -> np.ndarray:
    """Computes empirical percentile rank [0, 1] relative to the pooled distribution."""
    sorted_ref = pooled_distributions[feat]
    n = len(sorted_ref)
    return np.searchsorted(sorted_ref, values, side="right") / float(n)


def compute_derived_metrics(df, prefix):
    """Computes engagement, authority, activity, and social graph ratio features for an entity."""
    metrics = pd.DataFrame(index=df.index)

    follower = df[f"{prefix}_follower_count"].clip(lower=0).astype(np.float64)
    following = df[f"{prefix}_following_count"].clip(lower=0).astype(np.float64)
    listed = df[f"{prefix}_listed_count"].clip(lower=0).astype(np.float64)
    mentions_rec = df[f"{prefix}_mentions_received"].clip(lower=0).astype(np.float64)
    mentions_sent = df[f"{prefix}_mentions_sent"].clip(lower=0).astype(np.float64)
    retweets_rec = df[f"{prefix}_retweets_received"].clip(lower=0).astype(np.float64)
    retweets_sent = df[f"{prefix}_retweets_sent"].clip(lower=0).astype(np.float64)
    posts = df[f"{prefix}_posts"].clip(lower=0).astype(np.float64)
    net1 = df[f"{prefix}_network_feature_1"].clip(lower=0).astype(np.float64)
    net2 = df[f"{prefix}_network_feature_2"].clip(lower=0).astype(np.float64)
    net3 = df[f"{prefix}_network_feature_3"].clip(lower=0).astype(np.float64)

    # Basic ratios
    metrics[f"{prefix}_follower_following_ratio"] = follower / (following + 1.0)
    metrics[f"{prefix}_listed_follower_ratio"] = listed / (follower + 1.0)
    metrics[f"{prefix}_retweets_per_post"] = retweets_rec / (posts + 0.1)
    metrics[f"{prefix}_mentions_per_post"] = mentions_rec / (posts + 0.1)
    metrics[f"{prefix}_retweets_sent_per_post"] = retweets_sent / (posts + 0.1)
    metrics[f"{prefix}_mentions_sent_per_post"] = mentions_sent / (posts + 0.1)
    metrics[f"{prefix}_retweet_impact_ratio"] = retweets_rec / (retweets_sent + 0.1)
    metrics[f"{prefix}_mention_impact_ratio"] = mentions_rec / (mentions_sent + 0.1)

    # High-signal social influence features:
    # 1. Follower engagement rate ((retweets + mentions) / follower)
    metrics[f"{prefix}_follower_engagement_rate"] = (retweets_rec + mentions_rec) / (follower + 1.0)
    # 2. Virality quotient (retweets / mentions)
    metrics[f"{prefix}_virality_quotient"] = (retweets_rec + 0.1) / (mentions_rec + 0.1)
    # 3. Network centrality interaction ratios
    metrics[f"{prefix}_network_density_approx"] = (net1 + 1.0) / (net3 + 1.0)
    metrics[f"{prefix}_network_interaction"] = np.log1p(net1) * np.log1p(net2)
    metrics[f"{prefix}_network_ratio_1_2"] = (net1 + 1.0) / (net2 + 1.0)
    metrics[f"{prefix}_network_ratio_2_3"] = (net2 + 1.0) / (net3 + 1.0)
    metrics[f"{prefix}_network_hub_score"] = np.log1p(net1) * np.log1p(net3) / (np.log1p(net2) + 1.0)
    # 4. In-out interaction reciprocity metrics
    total_received = retweets_rec + mentions_rec
    total_sent = retweets_sent + mentions_sent
    metrics[f"{prefix}_total_engagement"] = total_received
    metrics[f"{prefix}_interaction_reciprocity"] = (total_received + 1.0) / (total_sent + 1.0)
    metrics[f"{prefix}_broadcast_authority"] = np.log1p(total_received) - np.log1p(total_sent)
    metrics[f"{prefix}_list_prestige"] = np.log1p(listed) / (np.log1p(follower) + 1e-5)
    metrics[f"{prefix}_retweet_out_in_ratio"] = (retweets_sent + 0.1) / (retweets_rec + 0.1)
    metrics[f"{prefix}_mention_out_in_ratio"] = (mentions_sent + 0.1) / (mentions_rec + 0.1)

    return metrics


def extract_features(df):
    """Builds comprehensive pairwise comparative features between Person A and Person B."""
    feat_df = pd.DataFrame(index=df.index)

    for col in [f"A_{f}" for f in base_features] + [f"B_{f}" for f in base_features]:
        if col in df.columns:
            feat_df[col] = df[col].astype(np.float64)

    # Global percentile ranks for base features
    for feat in base_features:
        feat_df[f"A_pct_rank_{feat}"] = get_percentile_rank(feat, df[f"A_{feat}"].values.astype(np.float64))
        feat_df[f"B_pct_rank_{feat}"] = get_percentile_rank(feat, df[f"B_{feat}"].values.astype(np.float64))
        feat_df[f"diff_pct_rank_{feat}"] = feat_df[f"A_pct_rank_{feat}"] - feat_df[f"B_pct_rank_{feat}"]

    a_derived = compute_derived_metrics(df, "A")
    b_derived = compute_derived_metrics(df, "B")

    feat_df = pd.concat([feat_df, a_derived, b_derived], axis=1)
    all_metrics = base_features + [c.replace("A_", "") for c in a_derived.columns]

    dominance_scores = np.zeros(len(df), dtype=np.float32)
    for metric in all_metrics:
        a_val = feat_df[f"A_{metric}"]
        b_val = feat_df[f"B_{metric}"]

        feat_df[f"diff_{metric}"] = a_val - b_val
        log_a = np.log1p(np.maximum(a_val, 0))
        log_b = np.log1p(np.maximum(b_val, 0))
        feat_df[f"log_diff_{metric}"] = log_a - log_b
        feat_df[f"norm_diff_{metric}"] = (a_val - b_val) / (
            np.abs(a_val) + np.abs(b_val) + 1e-5
        )
        is_greater = (a_val > b_val).astype(np.float32)
        feat_df[f"is_greater_{metric}"] = is_greater
        dominance_scores += is_greater.values

    feat_df["pairwise_dominance_score"] = dominance_scores / float(len(all_metrics))

    return feat_df


def augment_pairwise_symmetry(df):
    """Doubles the training set by swapping A and B while flipping Choice."""
    df_swapped = df.copy()
    a_cols = [c for c in df.columns if c.startswith("A_")]
    b_cols = [c for c in df.columns if c.startswith("B_")]

    rename_dict = {}
    for a_col, b_col in zip(sorted(a_cols), sorted(b_cols)):
        rename_dict[a_col] = b_col
        rename_dict[b_col] = a_col

    df_swapped = df_swapped.rename(columns=rename_dict)
    if "Choice" in df_swapped.columns:
        df_swapped["Choice"] = 1 - df_swapped["Choice"]

    augmented_df = pd.concat([df, df_swapped], ignore_index=True)
    return augmented_df


# Stratified K-Fold on original dataset to prevent fold-leakage across pairs
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
original_folds = np.zeros(len(train_df), dtype=np.int64)
for fold, (train_idx, val_idx) in enumerate(
    skf.split(train_df, train_df["Choice"].values)
):
    original_folds[val_idx] = fold

# Symmetrically augment training data
train_augmented = augment_pairwise_symmetry(train_df)
folds = np.concatenate([original_folds, original_folds])

X_train = extract_features(train_augmented)
y_train_all = train_augmented["Choice"].values.astype(np.int64)
X_test = extract_features(test_df)

feature_cols = [c for c in X_train.columns]
n_features = len(feature_cols)
n_folds = 5

X_train_df = X_train[feature_cols]
X_test_df = X_test[feature_cols]


# =========================================================================
# 2. Model Definitions
# =========================================================================
class PairwiseInfluenceNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128, 64],
        dropout_rate: float = 0.25,
    ):
        super(PairwiseInfluenceNet, self).__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            in_dim = h_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.skip_proj = (
            nn.Linear(input_dim, hidden_dims[-1])
            if input_dim != hidden_dims[-1]
            else nn.Identity()
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2.0),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature_extractor(x)
        skip = self.skip_proj(x)
        out = self.head(feat + skip)
        return out.squeeze(-1)


def build_neural_model(
    input_dim: int, lr: float = 1e-3, weight_decay: float = 1e-4, epochs: int = 35
):
    model = PairwiseInfluenceNet(
        input_dim=input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.25
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    return model, criterion, optimizer, scheduler


def build_lgbm_classifier(seed: int = 42) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        boosting_type="gbdt",
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        min_child_samples=20,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        n_estimators=1500,
        n_jobs=-1,
        verbose=-1,
    )


def build_xgb_classifier(seed: int = 42) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        n_estimators=1500,
        n_jobs=-1,
        tree_method="hist",
        early_stopping_rounds=50,
    )


def build_catboost_classifier(seed: int = 42) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        subsample=0.85,
        random_seed=seed,
        early_stopping_rounds=50,
        verbose=False,
    )


# =========================================================================
# 3. 5-Fold Training and Evaluation Loop
# =========================================================================
oof_preds_lgb = np.zeros(len(X_train), dtype=np.float64)
oof_preds_xgb = np.zeros(len(X_train), dtype=np.float64)
oof_preds_nn = np.zeros(len(X_train), dtype=np.float64)

test_preds_lgb = np.zeros(len(X_test), dtype=np.float64)
test_preds_xgb = np.zeros(len(X_test), dtype=np.float64)
test_preds_nn = np.zeros(len(X_test), dtype=np.float64)

for fold in range(n_folds):
    train_mask = folds != fold
    val_mask = folds == fold

    X_tr = X_train_df.iloc[train_mask].values
    y_tr = y_train_all[train_mask]
    X_va = X_train_df.iloc[val_mask].values
    y_va = y_train_all[val_mask]
    X_te = X_test_df.values

    # 1. LightGBM
    lgb_clf = build_lgbm_classifier(seed=42 + fold)
    lgb_callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=-1),
    ]
    lgb_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=lgb_callbacks)

    val_prob_lgb = lgb_clf.predict_proba(X_va)[:, 1]
    oof_preds_lgb[val_mask] = val_prob_lgb
    test_preds_lgb += lgb_clf.predict_proba(X_te)[:, 1] / n_folds
    joblib.dump(lgb_clf, f"./working/models/lgb_fold_{fold}.joblib")

    # 2. XGBoost
    xgb_clf = build_xgb_classifier(seed=42 + fold)
    xgb_clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    val_prob_xgb = xgb_clf.predict_proba(X_va)[:, 1]
    oof_preds_xgb[val_mask] = val_prob_xgb
    test_preds_xgb += xgb_clf.predict_proba(X_te)[:, 1] / n_folds
    joblib.dump(xgb_clf, f"./working/models/xgb_fold_{fold}.joblib")

    # 3. PyTorch PairwiseInfluenceNet
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)
    X_te_scaled = scaler.transform(X_te)
    joblib.dump(scaler, f"./working/models/scaler_fold_{fold}.joblib")

    train_dataset = TensorDataset(
        torch.tensor(X_tr_scaled, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, drop_last=False
    )
    val_tensor_x = torch.tensor(X_va_scaled, dtype=torch.float32).to(device)
    test_tensor_x = torch.tensor(X_te_scaled, dtype=torch.float32).to(device)

    nn_model, criterion, optimizer, scheduler = build_neural_model(
        input_dim=n_features, lr=1e-3, weight_decay=1e-4, epochs=35
    )
    nn_model.to(device)

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(nn_model.state_dict())

    for epoch in range(35):
        nn_model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = nn_model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

        nn_model.eval()
        with torch.no_grad():
            val_logits = nn_model(val_tensor_x)
            val_loss = criterion(
                val_logits, torch.tensor(y_va).float().to(device)
            ).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(nn_model.state_dict())

    nn_model.load_state_dict(best_weights)
    nn_model.eval()
    with torch.no_grad():
        val_prob_nn = torch.sigmoid(nn_model(val_tensor_x)).cpu().numpy()
        test_prob_nn = torch.sigmoid(nn_model(test_tensor_x)).cpu().numpy()

    oof_preds_nn[val_mask] = val_prob_nn
    test_preds_nn += test_prob_nn / n_folds
    torch.save(best_weights, f"./working/models/nn_fold_{fold}.pt")

    fold_lgb_acc = accuracy_score(y_va, (val_prob_lgb >= 0.5).astype(int))
    fold_xgb_acc = accuracy_score(y_va, (val_prob_xgb >= 0.5).astype(int))
    fold_nn_acc = accuracy_score(y_va, (val_prob_nn >= 0.5).astype(int))
    print(
        f"Fold {fold+1}/5 - LGB Acc: {fold_lgb_acc:.4f} | XGB Acc: {fold_xgb_acc:.4f} | NN Acc: {fold_nn_acc:.4f}"
    )

# =========================================================================
# 4. Out-Of-Fold Evaluation and Ensemble Selection
# =========================================================================
# Evaluate on original unaugmented slice (indices 0..len(train_df)-1)
orig_n = len(train_df)
y_orig = y_train_all[:orig_n]

oof_lgb_orig = oof_preds_lgb[:orig_n]
oof_xgb_orig = oof_preds_xgb[:orig_n]
oof_nn_orig = oof_preds_nn[:orig_n]

acc_lgb = accuracy_score(y_orig, (oof_lgb_orig >= 0.5).astype(int))
acc_xgb = accuracy_score(y_orig, (oof_xgb_orig >= 0.5).astype(int))
acc_nn = accuracy_score(y_orig, (oof_nn_orig >= 0.5).astype(int))

blend_trees_oof = 0.5 * oof_lgb_orig + 0.5 * oof_xgb_orig
blend_trees_test = 0.5 * test_preds_lgb + 0.5 * test_preds_xgb
acc_blend_trees = accuracy_score(y_orig, (blend_trees_oof >= 0.5).astype(int))

blend_all_oof = 0.45 * oof_lgb_orig + 0.45 * oof_xgb_orig + 0.10 * oof_nn_orig
blend_all_test = 0.45 * test_preds_lgb + 0.45 * test_preds_xgb + 0.10 * test_preds_nn
acc_blend_all = accuracy_score(y_orig, (blend_all_oof >= 0.5).astype(int))

candidates = {
    "lgb": (acc_lgb, test_preds_lgb),
    "xgb": (acc_xgb, test_preds_xgb),
    "nn": (acc_nn, test_preds_nn),
    "blend_trees": (acc_blend_trees, blend_trees_test),
    "blend_all": (acc_blend_all, blend_all_test),
}

best_name = max(candidates, key=lambda k: candidates[k][0])
best_score, best_test_prob = candidates[best_name]

# =========================================================================
# 5. Test Inference and Submission Generation
# =========================================================================
test_predictions = (best_test_prob >= 0.5).astype(int)

submission_df = pd.DataFrame(
    {"id": np.arange(len(test_predictions), dtype=int), "Choice": test_predictions}
)

submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Final Validation Score: {best_score}")
