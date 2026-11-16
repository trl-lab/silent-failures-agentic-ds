import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import lightgbm as lgb

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

target_col = "class"
feature_cols = [c for c in train_df.columns if c != target_col]

# 2. Encode Target and Map Categorical Board Values safely
le = LabelEncoder()
y_train = le.fit_transform(train_df[target_col].astype(str))

val_map = {"b": 0, "x": 1, "o": 2, "0": 0, "1": 1, "2": 2, 0: 0, 1: 1, 2: 2}


def preprocess_board(df, feature_cols):
    X_mat = np.zeros((len(df), 42), dtype=np.int8)
    for idx, col in enumerate(feature_cols):
        X_mat[:, idx] = (
            df[col]
            .map(lambda v: val_map.get(v, int(v) if str(v).isdigit() else 0))
            .values
        )
    return X_mat


X_train_raw = preprocess_board(train_df, feature_cols)
X_test_raw = preprocess_board(test_df, feature_cols)


# 3. Exhaustive 4-in-a-row Win Window & Board Gravity Feature Engineering
def extract_board_features(X_raw):
    num_samples = len(X_raw)
    # Reshape into 6 rows x 7 cols grid: grid shape (N, 6, 7)
    grid = X_raw.reshape(num_samples, 7, 6).swapaxes(1, 2)

    features = [X_raw]

    # Piece totals for Player 1 and Player 2
    p1_total = (grid == 1).sum(axis=(1, 2))[:, None]
    p2_total = (grid == 2).sum(axis=(1, 2))[:, None]
    features.extend([p1_total, p2_total])

    # Column heights (gravity indicator)
    col_heights = (grid != 0).sum(axis=1)
    features.append(col_heights)

    # Column piece counts
    p1_cols = (grid == 1).sum(axis=1)
    p2_cols = (grid == 2).sum(axis=1)
    features.extend([p1_cols, p2_cols])

    # Row piece counts
    p1_rows = (grid == 1).sum(axis=2)
    p2_rows = (grid == 2).sum(axis=2)
    features.extend([p1_rows, p2_rows])

    # Build all 69 unique 4-in-a-row win window cell coordinate sets
    win_windows = []

    # 24 Horizontal 4-windows (6 rows x 4 start cols)
    for r in range(6):
        for c in range(4):
            win_windows.append([(r, c + i) for i in range(4)])

    # 21 Vertical 4-windows (3 start rows x 7 cols)
    for r in range(3):
        for c in range(7):
            win_windows.append([(r + i, c) for i in range(4)])

    # 12 Positive Diagonal 4-windows (3 start rows x 4 start cols)
    for r in range(3):
        for c in range(4):
            win_windows.append([(r + i, c + i) for i in range(4)])

    # 12 Negative Diagonal 4-windows (3 start rows x 4 start cols)
    for r in range(3):
        for c in range(3, 7):
            win_windows.append([(r + i, c - i) for i in range(4)])

    num_windows = len(win_windows)
    win_p1 = np.zeros((num_samples, num_windows), dtype=np.int8)
    win_p2 = np.zeros((num_samples, num_windows), dtype=np.int8)

    for idx, w in enumerate(win_windows):
        cells = np.stack([grid[:, r, c] for r, c in w], axis=1)  # shape (N, 4)
        win_p1[:, idx] = (cells == 1).sum(axis=1)
        win_p2[:, idx] = (cells == 2).sum(axis=1)

    features.extend([win_p1, win_p2])

    # Higher-level threat aggregates derived from the 69 win windows
    p1_threat3 = ((win_p1 == 3) & (win_p2 == 0)).sum(axis=1, keepdims=True)
    p1_threat2 = ((win_p1 == 2) & (win_p2 == 0)).sum(axis=1, keepdims=True)
    p1_threat1 = ((win_p1 == 1) & (win_p2 == 0)).sum(axis=1, keepdims=True)

    p2_threat3 = ((win_p2 == 3) & (win_p1 == 0)).sum(axis=1, keepdims=True)
    p2_threat2 = ((win_p2 == 2) & (win_p1 == 0)).sum(axis=1, keepdims=True)
    p2_threat1 = ((win_p2 == 1) & (win_p1 == 0)).sum(axis=1, keepdims=True)

    blocked_windows = ((win_p1 > 0) & (win_p2 > 0)).sum(axis=1, keepdims=True)
    empty_windows = ((win_p1 == 0) & (win_p2 == 0)).sum(axis=1, keepdims=True)

    features.extend(
        [
            p1_threat3,
            p1_threat2,
            p1_threat1,
            p2_threat3,
            p2_threat2,
            p2_threat1,
            blocked_windows,
            empty_windows,
        ]
    )

    return np.hstack(features)


X_train = extract_board_features(X_train_raw)
X_test = extract_board_features(X_test_raw)

# 4. Stratified 5-Fold Cross Validation Modeling
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds_prob = np.zeros((len(train_df), len(le.classes_)))
test_preds_prob = np.zeros((len(test_df), len(le.classes_)))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_va, y_va = X_train[val_idx], y_train[val_idx]

    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(le.classes_),
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=127,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
    )

    clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    oof_preds_prob[val_idx] = clf.predict_proba(X_va)
    test_preds_prob += clf.predict_proba(X_test) / skf.n_splits

# 5. Evaluate Accuracy Metric
oof_preds = np.argmax(oof_preds_prob, axis=1)
accuracy = accuracy_score(y_train, oof_preds)
print(f"Validation Accuracy: {accuracy:.5f}")

# 6. Save Submission File
os.makedirs("./working", exist_ok=True)
test_preds_idx = np.argmax(test_preds_prob, axis=1)
test_preds_labels = le.inverse_transform(test_preds_idx)

# Format correctly matching original class types (int/str)
if pd.api.types.is_integer_dtype(train_df[target_col]):
    test_preds_labels = test_preds_labels.astype(int)

submission = pd.DataFrame(
    {"id": np.arange(len(test_preds_labels)), "class": test_preds_labels}
)
submission.to_csv("./working/submission.csv", index=False)
