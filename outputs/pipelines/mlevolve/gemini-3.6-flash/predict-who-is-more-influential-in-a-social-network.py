import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Ensure required directories exist
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Device configuration and random seeds
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)


def extract_pairwise_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts scale-invariant and domain-specific pairwise comparison features for social influence.
    """
    features = pd.DataFrame(index=df.index)

    # Base feature names per entity
    base_cols = [
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

    # 1. Log transformations for heavy-tailed scale invariance
    for col in base_cols:
        features[f"log_A_{col}"] = np.log1p(np.maximum(0, df[f"A_{col}"]))
        features[f"log_B_{col}"] = np.log1p(np.maximum(0, df[f"B_{col}"]))

        # Log difference & relative difference
        features[f"diff_log_{col}"] = (
            features[f"log_A_{col}"] - features[f"log_B_{col}"]
        )
        features[f"diff_raw_{col}"] = df[f"A_{col}"] - df[f"B_{col}"]
        features[f"rel_diff_{col}"] = (df[f"A_{col}"] - df[f"B_{col}"]) / (
            df[f"A_{col}"] + df[f"B_{col}"] + 1e-5
        )

    # 2. Domain-Specific Efficiency & Engagement Ratios
    for p in ["A", "B"]:
        f_count = np.maximum(0, df[f"{p}_follower_count"])
        f_ing = np.maximum(0, df[f"{p}_following_count"])
        posts = np.maximum(0, df[f"{p}_posts"])
        retweets_rec = np.maximum(0, df[f"{p}_retweets_received"])
        mentions_rec = np.maximum(0, df[f"{p}_mentions_received"])
        listed = np.maximum(0, df[f"{p}_listed_count"])

        # Key Ratios
        features[f"{p}_follower_following_ratio"] = f_count / (f_ing + 1.0)
        features[f"{p}_listed_per_follower"] = listed / (f_count + 1.0)
        features[f"{p}_retweets_per_follower"] = retweets_rec / (f_count + 1.0)
        features[f"{p}_mentions_per_follower"] = mentions_rec / (f_count + 1.0)
        features[f"{p}_mentions_per_post"] = mentions_rec / (posts + 0.1)
        features[f"{p}_retweets_per_post"] = retweets_rec / (posts + 0.1)
        features[f"{p}_reciprocity_mentions"] = df[f"{p}_mentions_sent"] / (
            mentions_rec + 0.1
        )
        features[f"{p}_reciprocity_retweets"] = df[f"{p}_retweets_sent"] / (
            retweets_rec + 0.1
        )
        features[f"{p}_net1_per_follower"] = df[f"{p}_network_feature_1"] / (
            f_count + 1.0
        )

    # 3. Differences of Domain Ratios
    ratio_names = [
        "follower_following_ratio",
        "listed_per_follower",
        "retweets_per_follower",
        "mentions_per_follower",
        "mentions_per_post",
        "retweets_per_post",
        "reciprocity_mentions",
        "reciprocity_retweets",
        "net1_per_follower",
    ]

    for r in ratio_names:
        log_r_a = np.log1p(np.maximum(0, features[f"A_{r}"]))
        log_r_b = np.log1p(np.maximum(0, features[f"B_{r}"]))
        features[f"diff_log_ratio_{r}"] = log_r_a - log_r_b
        features[f"diff_raw_ratio_{r}"] = features[f"A_{r}"] - features[f"B_{r}"]

    # 4. Clean numerical non-finites
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return features


class SiameseInfluenceNet(nn.Module):
    """
    Strictly anti-symmetric Bradley-Terry neural network for pairwise social influence prediction.
    logit(A, B) = [g(x_A) - g(x_B)] + h(x_A - x_B)
    """

    def __init__(
        self,
        single_entity_dim: int = 20,
        diff_feature_dim: int = 51,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.latent_net = nn.Sequential(
            nn.Linear(single_entity_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.diff_layer = nn.Linear(diff_feature_dim, 1, bias=False)

    def forward(
        self, x_a: torch.Tensor, x_b: torch.Tensor, x_diff: torch.Tensor
    ) -> torch.Tensor:
        score_a = self.latent_net(x_a)
        score_b = self.latent_net(x_b)
        logit = (score_a - score_b) + self.diff_layer(x_diff)
        return logit.squeeze(-1)


def main():
    # 1. Load Raw Data and Engineer Features
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    X_train_df = extract_pairwise_features(train_df)
    y_train_full = train_df["Choice"].astype(int).values
    X_test_df = extract_pairwise_features(test_df)

    # Column mappings
    base_cols = [
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
    ratio_names = [
        "follower_following_ratio",
        "listed_per_follower",
        "retweets_per_follower",
        "mentions_per_follower",
        "mentions_per_post",
        "retweets_per_post",
        "reciprocity_mentions",
        "reciprocity_retweets",
        "net1_per_follower",
    ]

    cols_A = [f"log_A_{c}" for c in base_cols] + [f"A_{r}" for r in ratio_names]
    cols_B = [f"log_B_{c}" for c in base_cols] + [f"B_{r}" for r in ratio_names]
    cols_diff = [c for c in X_train_df.columns if c not in cols_A and c not in cols_B]

    single_dim = len(cols_A)
    diff_dim = len(cols_diff)

    n_samples = len(X_train_df)
    oof_preds = np.zeros(n_samples, dtype=np.float32)
    test_preds_accum = np.zeros(len(X_test_df), dtype=np.float32)

    epochs = 35
    batch_size = 128
    criterion = nn.BCEWithLogitsLoss()

    # 2. 5-Fold Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(X_train_df, y_train_full)
    ):
        X_tr, y_tr = X_train_df.iloc[train_idx], y_train_full[train_idx]
        X_va, y_va = X_train_df.iloc[val_idx], y_train_full[val_idx]

        # Leakage-Free Standard Scaling
        entity_scaler = StandardScaler()
        train_entity_combined = np.vstack([X_tr[cols_A].values, X_tr[cols_B].values])
        entity_scaler.fit(train_entity_combined)

        diff_scaler = StandardScaler()
        diff_scaler.fit(X_tr[cols_diff].values)

        xA_tr = entity_scaler.transform(X_tr[cols_A].values)
        xB_tr = entity_scaler.transform(X_tr[cols_B].values)
        xDiff_tr = diff_scaler.transform(X_tr[cols_diff].values)

        xA_va = entity_scaler.transform(X_va[cols_A].values)
        xB_va = entity_scaler.transform(X_va[cols_B].values)
        xDiff_va = diff_scaler.transform(X_va[cols_diff].values)

        xA_te = entity_scaler.transform(X_test_df[cols_A].values)
        xB_te = entity_scaler.transform(X_test_df[cols_B].values)
        xDiff_te = diff_scaler.transform(X_test_df[cols_diff].values)

        # Dynamic Anti-symmetric Mirror Augmentation on Training Fold
        xA_tr_aug = np.vstack([xA_tr, xB_tr])
        xB_tr_aug = np.vstack([xB_tr, xA_tr])
        xDiff_tr_aug = np.vstack([xDiff_tr, -xDiff_tr])
        y_tr_aug = np.concatenate([y_tr, 1 - y_tr])

        train_dataset = TensorDataset(
            torch.tensor(xA_tr_aug, dtype=torch.float32),
            torch.tensor(xB_tr_aug, dtype=torch.float32),
            torch.tensor(xDiff_tr_aug, dtype=torch.float32),
            torch.tensor(y_tr_aug, dtype=torch.float32),
        )
        val_dataset = TensorDataset(
            torch.tensor(xA_va, dtype=torch.float32),
            torch.tensor(xB_va, dtype=torch.float32),
            torch.tensor(xDiff_va, dtype=torch.float32),
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=2
        )

        model = SiameseInfluenceNet(
            single_entity_dim=single_dim, diff_feature_dim=diff_dim
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5
        )

        best_val_acc = 0.0
        best_model_path = f"./working/best_model_fold_{fold_idx}.pt"

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for b_xa, b_xb, b_xd, b_y in train_loader:
                b_xa, b_xb, b_xd, b_y = (
                    b_xa.to(device),
                    b_xb.to(device),
                    b_xd.to(device),
                    b_y.to(device),
                )

                optimizer.zero_grad()
                logits = model(b_xa, b_xb, b_xd)
                loss = criterion(logits, b_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(b_y)

            scheduler.step()
            train_loss /= len(train_dataset)

            # Evaluate on validation set
            model.eval()
            val_preds_list = []
            with torch.no_grad():
                for b_xa, b_xb, b_xd in val_loader:
                    b_xa, b_xb, b_xd = (
                        b_xa.to(device),
                        b_xb.to(device),
                        b_xd.to(device),
                    )
                    logits = model(b_xa, b_xb, b_xd)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    val_preds_list.extend(probs)

            val_preds_arr = np.array(val_preds_list)
            val_acc = accuracy_score(y_va, (val_preds_arr >= 0.5).astype(int))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)

            print(
                f"Fold {fold_idx + 1} | Epoch {epoch}/{epochs} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.5f}"
            )

        # Load best fold model for inference
        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        val_preds_list = []
        with torch.no_grad():
            for b_xa, b_xb, b_xd in val_loader:
                b_xa, b_xb, b_xd = (
                    b_xa.to(device),
                    b_xb.to(device),
                    b_xd.to(device),
                )
                logits = model(b_xa, b_xb, b_xd)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds_list.extend(probs)

        oof_preds[val_idx] = np.array(val_preds_list)

        # Test Set Inference
        test_dataset = TensorDataset(
            torch.tensor(xA_te, dtype=torch.float32),
            torch.tensor(xB_te, dtype=torch.float32),
            torch.tensor(xDiff_te, dtype=torch.float32),
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=2
        )

        fold_test_preds = []
        with torch.no_grad():
            for b_xa, b_xb, b_xd in test_loader:
                b_xa, b_xb, b_xd = (
                    b_xa.to(device),
                    b_xb.to(device),
                    b_xd.to(device),
                )
                logits = model(b_xa, b_xb, b_xd)
                probs = torch.sigmoid(logits).cpu().numpy()
                fold_test_preds.extend(probs)

        test_preds_accum += np.array(fold_test_preds) / 5.0

    # 3. Compute Final OOF Validation Metric
    oof_binary = (oof_preds >= 0.5).astype(int)
    final_score = accuracy_score(y_train_full, oof_binary)

    # 4. Save Predictions to Submission File
    test_binary = (test_preds_accum >= 0.5).astype(int)
    submission_df = pd.DataFrame(
        {"id": np.arange(len(test_binary)), "Choice": test_binary}
    )
    submission_df.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {final_score}")


if __name__ == "__main__":
    main()
