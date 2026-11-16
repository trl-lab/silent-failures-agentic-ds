import pandas as pd
import numpy as np
import ast
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/mlebench/stanford-covid-vaccine/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/mlebench/stanford-covid-vaccine/data/test.parquet"
TARGET = "reactivity"
RANDOM_STATE = 42


def parse_vector(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=float)
        return arr if arr.ndim == 1 else None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if s.startswith("[") and s.endswith("]"):
            try:
                arr = np.asarray(ast.literal_eval(s), dtype=float)
                return arr if arr.ndim == 1 else None
            except Exception:
                pass
            try:
                arr = np.asarray(json.loads(s), dtype=float)
                return arr if arr.ndim == 1 else None
            except Exception:
                pass
        try:
            return np.asarray([float(x) for x in s.split()], dtype=float)
        except Exception:
            return None
    try:
        return np.asarray([float(value)], dtype=float)
    except Exception:
        return None


def safe_str_series(s):
    return s.astype("string").fillna("")


def common_suffix_length(strings, max_len):
    strings = [s for s in strings if isinstance(s, str) and len(s) > 0]
    if not strings:
        return 0
    first = strings[0]
    limit = min(max_len, len(first))
    for k in range(1, limit + 1):
        suffix = first[-k:]
        if all(isinstance(s, str) and s.endswith(suffix) for s in strings):
            continue
        return k - 1
    return limit


def infer_offset(train_df, max_len):
    if "seq_length" in train_df.columns and "seq_scored" in train_df.columns:
        sl = pd.to_numeric(train_df["seq_length"], errors="coerce").dropna()
        ss = pd.to_numeric(train_df["seq_scored"], errors="coerce").dropna()
        if len(sl) and len(ss):
            total = int(sl.mode().iloc[0])
            scored = int(ss.mode().iloc[0])
            if total > scored:
                sample = train_df["sequence"].dropna().head(2000).tolist()
                suffix_len = common_suffix_length(sample, total - scored)
                offset = total - scored - suffix_len
                if 0 <= offset <= total - scored:
                    return int(offset)
                return int((total - scored) // 2)
    return 20


def position_features(df, offset, pos, scalar_cols):
    seq = safe_str_series(df["sequence"])
    struct = safe_str_series(df["structure"])
    loop = safe_str_series(df["predicted_loop_type"])

    nuc = seq.str[offset + pos]
    st = struct.str[offset + pos]
    lp = loop.str[offset + pos]

    prev_nuc = seq.str[offset + pos - 1] if pos > 0 else seq.str[offset - 1]
    next_nuc = seq.str[offset + pos + 1] if pos < 67 else seq.str[offset + pos + 1]

    feats = []
    for ch in ["A", "C", "G", "U"]:
        feats.append((nuc == ch).astype(np.float32).to_numpy())
        feats.append((prev_nuc == ch).astype(np.float32).to_numpy())
        feats.append((next_nuc == ch).astype(np.float32).to_numpy())

    for ch in [".", "(", ")"]:
        feats.append((st == ch).astype(np.float32).to_numpy())

    for ch in ["E", "S", "H", "M", "X", "I", "B"]:
        feats.append((lp == ch).astype(np.float32).to_numpy())

    feats.append(np.full(len(df), float(pos), dtype=np.float32))
    feats.append(np.full(len(df), float(offset + pos), dtype=np.float32))

    for col in scalar_cols:
        if col in df.columns:
            feats.append(pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(np.float32).to_numpy())
        else:
            feats.append(np.zeros(len(df), dtype=np.float32))

    return np.column_stack(feats)


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    if TARGET not in train_df.columns:
        raise ValueError("Target column 'reactivity' not found in train data.")

    y_raw = [parse_vector(v) for v in train_df[TARGET]]
    valid_lengths = [len(y) for y in y_raw if y is not None and len(y) > 0]
    if valid_lengths:
        max_len = int(max(valid_lengths))
    elif "seq_scored" in train_df.columns:
        max_len = int(pd.to_numeric(train_df["seq_scored"], errors="coerce").fillna(0).max())
    else:
        max_len = 68

    Y = np.full((len(train_df), max_len), np.nan, dtype=np.float32)
    for i, y in enumerate(y_raw):
        if y is not None and len(y) > 0:
            L = min(len(y), max_len)
            Y[i, :L] = y[:L]

    offset = infer_offset(train_df, max_len)

    scalar_cols = ["signal_to_noise", "SN_filter", "seq_length", "seq_scored"]
    scalar_cols = [c for c in scalar_cols if c in train_df.columns]

    train_idx, val_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=0.2,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    val_errors = []
    for pos in range(max_len):
        y_all = Y[:, pos]
        mask = ~np.isnan(y_all)
        if mask.sum() < 10:
            continue

        X_all = position_features(train_df, offset, pos, scalar_cols)
        X_tr = X_all[train_idx[mask[train_idx]]]
        y_tr = y_all[train_idx[mask[train_idx]]]
        X_va = X_all[val_idx[mask[val_idx]]]
        y_va = y_all[val_idx[mask[val_idx]]]

        if len(X_tr) == 0 or len(X_va) == 0:
            continue

        model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        val_errors.append(np.sqrt(np.mean((pred - y_va) ** 2)))

    if val_errors:
        print(f"Validation mCRMSE: {np.mean(val_errors):.6f}")

    if "seq_scored" in test_df.columns:
        lengths = pd.to_numeric(test_df["seq_scored"], errors="coerce").fillna(max_len).astype(int).clip(upper=max_len)
    else:
        lengths = pd.Series([max_len] * len(test_df))

    pred_matrix = np.zeros((len(test_df), max_len), dtype=np.float32)

    for pos in range(max_len):
        y_all = Y[:, pos]
        mask = ~np.isnan(y_all)
        if mask.sum() < 10:
            continue

        X_train = position_features(train_df, offset, pos, scalar_cols)
        X_test = position_features(test_df, offset, pos, scalar_cols)

        model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
        model.fit(X_train[mask], y_all[mask])
        pred_matrix[:, pos] = model.predict(X_test).astype(np.float32)

    reactivity_strings = []
    for i in range(len(test_df)):
        L = int(lengths.iloc[i])
        vals = pred_matrix[i, :L]
        reactivity_strings.append(json.dumps([float(v) for v in vals]))

    out = pd.DataFrame({
        "id": np.arange(len(test_df), dtype=int),
        "reactivity": reactivity_strings
    })
    out.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()