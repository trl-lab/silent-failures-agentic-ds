import os
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


seed_everything(42)

# 1. Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "class"
feature_cols = [col for col in train_df.columns if col != target_col]

y_train = train_df[target_col].values.astype(int)


def extract_features(df, cols):
    """Extract domain features for Connect-4 board positions."""
    raw = df[cols].values.astype(np.int8)
    N = len(raw)

    # Reconstruct 6 rows x 7 cols grid (column-major order in UCI Connect-4)
    boards = raw.reshape(N, 7, 6).transpose(0, 2, 1)  # (N, 6, 7)

    # 1. Base cell values encoded as: 0 for empty, 1 for P1, -1 for P2
    board_signed = np.zeros_like(boards, dtype=np.int8)
    board_signed[boards == 1] = 1
    board_signed[boards == 2] = -1

    p1_mask = (boards == 1).astype(np.int8)
    p2_mask = (boards == 2).astype(np.int8)
    empty_mask = (boards == 0).astype(np.int8)

    feats = []

    # Flattened original & signed representation
    feats.append(raw)
    feats.append(board_signed.reshape(N, 42))

    # Column piece counts (column fill heights)
    col_heights = (boards != 0).sum(axis=1)  # (N, 7)
    feats.append(col_heights)

    # Row, Column, Center piece counts per player
    feats.append(p1_mask.sum(axis=1))  # (N, 7) p1 count per column
    feats.append(p2_mask.sum(axis=1))  # (N, 7) p2 count per column
    feats.append(p1_mask.sum(axis=2))  # (N, 6) p1 count per row
    feats.append(p2_mask.sum(axis=2))  # (N, 6) p2 count per row

    # Center column dominance (column index 3 is crucial in Connect-4)
    center_p1 = p1_mask[:, :, 3].sum(axis=1, keepdims=True)
    center_p2 = p2_mask[:, :, 3].sum(axis=1, keepdims=True)
    feats.append(center_p1)
    feats.append(center_p2)
    feats.append(center_p1 - center_p2)

    # All possible 4-in-a-row lines (horizontal, vertical, diagonal)
    line_p1_counts = []
    line_p2_counts = []

    # Horizontal 4-windows (6 rows x 4 start cols = 24 windows)
    for r in range(6):
        for c in range(4):
            line_p1_counts.append(p1_mask[:, r, c : c + 4].sum(axis=1, keepdims=True))
            line_p2_counts.append(p2_mask[:, r, c : c + 4].sum(axis=1, keepdims=True))

    # Vertical 4-windows (3 start rows x 7 cols = 21 windows)
    for r in range(3):
        for c in range(7):
            line_p1_counts.append(p1_mask[:, r : r + 4, c].sum(axis=1, keepdims=True))
            line_p2_counts.append(p2_mask[:, r : r + 4, c].sum(axis=1, keepdims=True))

    # Positive diagonals (/) (3 start rows x 4 start cols = 12 windows)
    for r in range(3):
        for c in range(4):
            p1_diag = np.stack(
                [p1_mask[:, r + k, c + k] for k in range(4)], axis=1
            ).sum(axis=1, keepdims=True)
            p2_diag = np.stack(
                [p2_mask[:, r + k, c + k] for k in range(4)], axis=1
            ).sum(axis=1, keepdims=True)
            line_p1_counts.append(p1_diag)
            line_p2_counts.append(p2_diag)

    # Negative diagonals (\) (3 start rows x 4 start cols = 12 windows)
    for r in range(3, 6):
        for c in range(4):
            p1_diag = np.stack(
                [p1_mask[:, r - k, c + k] for k in range(4)], axis=1
            ).sum(axis=1, keepdims=True)
            p2_diag = np.stack(
                [p2_mask[:, r - k, c + k] for k in range(4)], axis=1
            ).sum(axis=1, keepdims=True)
            line_p1_counts.append(p1_diag)
            line_p2_counts.append(p2_diag)

    p1_lines = np.hstack(line_p1_counts)  # (N, 69)
    p2_lines = np.hstack(line_p2_counts)  # (N, 69)

    # Summary threat counts: 3-in-a-row without opponent block, 2-in-a-row
    p1_threats_3 = ((p1_lines == 3) & (p2_lines == 0)).sum(axis=1, keepdims=True)
    p2_threats_3 = ((p2_lines == 3) & (p1_lines == 0)).sum(axis=1, keepdims=True)
    p1_threats_2 = ((p1_lines == 2) & (p2_lines == 0)).sum(axis=1, keepdims=True)
    p2_threats_2 = ((p2_lines == 2) & (p1_lines == 0)).sum(axis=1, keepdims=True)

    feats.extend(
        [
            p1_lines,
            p2_lines,
            p1_threats_3,
            p2_threats_3,
            p1_threats_2,
            p2_threats_2,
        ]
    )

    return np.hstack(feats).astype(np.float32)


def get_horizontal_flip_df(df, cols):
    """Create horizontally mirrored dataframe for Connect-4 boards."""
    raw = df[cols].values.astype(np.int8)
    N = len(raw)
    boards = raw.reshape(N, 7, 6)
    # Flip columns: col 0 <-> col 6, col 1 <-> col 5, etc.
    boards_flipped = boards[:, ::-1, :]
    raw_flipped = boards_flipped.reshape(N, 42)
    return pd.DataFrame(raw_flipped, columns=cols)


# Generate features
X_train = extract_features(train_df, feature_cols)
X_test = extract_features(test_df, feature_cols)

test_flipped_df = get_horizontal_flip_df(test_df, feature_cols)
X_test_flipped = extract_features(test_flipped_df, feature_cols)

# 2. 5-Fold Stratified Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(train_df), 3))
test_preds = np.zeros((len(test_df), 3))

lgb_params = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": 8,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "verbosity": -1,
    "random_state": 42,
    "n_jobs": -1,
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_va, y_va = X_train[val_idx], y_train[val_idx]

    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va, reference=train_data)

    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1200,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    val_probs = model.predict(X_va, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_probs

    test_prob_orig = model.predict(X_test, num_iteration=model.best_iteration)
    test_prob_flip = model.predict(X_test_flipped, num_iteration=model.best_iteration)
    test_preds += (test_prob_orig + test_prob_flip) / (2.0 * skf.n_splits)

cv_accuracy = accuracy_score(y_train, np.argmax(oof_preds, axis=1))
print(f"5-Fold Cross-Validation Accuracy: {cv_accuracy:.5f}")

# 3. Save test predictions
test_classes = np.argmax(test_preds, axis=1)
submission = pd.DataFrame({"id": np.arange(len(test_classes)), "class": test_classes})
os.makedirs("./working", exist_ok=True)
submission.to_csv("./working/submission.csv", index=False)
