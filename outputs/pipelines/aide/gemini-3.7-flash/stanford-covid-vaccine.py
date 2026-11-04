import os
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

warnings.filterwarnings("ignore")

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

TARGET_COL = "reactivity"
if TARGET_COL not in train_df.columns:
    for col in train_df.columns:
        if "reactivity" in col.lower() or "target" in col.lower():
            TARGET_COL = col
            break


def parse_structure(structure_str):
    """Parse dot-bracket RNA secondary structure into paired indices."""
    stack = []
    pairs = [-1] * len(structure_str)
    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                pairs[i] = j
                pairs[j] = i
    return pairs


def extract_features(df, is_train=True):
    if (
        "sequence" in df.columns
        and isinstance(df["sequence"].iloc[0], str)
        and len(df["sequence"].iloc[0]) > 1
    ):
        features_list = []
        for idx, row in df.iterrows():
            seq = row["sequence"]
            struct = row.get("structure", "." * len(seq))
            loop = row.get("predicted_loop_type", "E" * len(seq))
            seq_len = len(seq)
            pairs = parse_structure(struct)

            # Precompute stem / loop segments and run lengths
            stem_run = [0] * seq_len
            loop_run = [0] * seq_len
            current_stem = 0
            for i in range(seq_len):
                if struct[i] in ["(", ")"]:
                    current_stem += 1
                else:
                    current_stem = 0
                stem_run[i] = current_stem

            current_loop = 0
            for i in range(seq_len):
                if struct[i] == ".":
                    current_loop += 1
                else:
                    current_loop = 0
                loop_run[i] = current_loop

            # Distance to nearest paired base
            paired_indices = [i for i, p in enumerate(pairs) if p != -1]
            unpaired_indices = [i for i, p in enumerate(pairs) if p == -1]

            for pos in range(seq_len):
                partner_idx = pairs[pos]
                is_paired = 1 if partner_idx != -1 else 0
                partner_char = (
                    seq[partner_idx] if (is_paired and partner_idx < seq_len) else "N"
                )
                partner_loop = (
                    loop[partner_idx]
                    if (is_paired and partner_idx < len(loop))
                    else "N"
                )

                # Pair bond type classification
                if is_paired:
                    b1, b2 = seq[pos], partner_char
                    pair_bond = "".join(sorted([b1, b2]))
                else:
                    pair_bond = "unpaired"

                # Distance to nearest paired and unpaired
                dist_nearest_paired = (
                    min([abs(pos - p) for p in paired_indices])
                    if paired_indices
                    else 999
                )
                dist_nearest_unpaired = (
                    min([abs(pos - u) for u in unpaired_indices])
                    if unpaired_indices
                    else 999
                )

                # Base feature dict
                feat = {
                    "orig_idx": idx,
                    "pos": pos,
                    "pos_rel": pos / seq_len,
                    "dist_from_end": min(pos, seq_len - 1 - pos),
                    "seq_char": seq[pos],
                    "struct_char": struct[pos] if pos < len(struct) else ".",
                    "loop_char": loop[pos] if pos < len(loop) else "E",
                    "is_paired": is_paired,
                    "pair_dist": abs(partner_idx - pos) if is_paired else -1,
                    "partner_char": partner_char,
                    "partner_loop": partner_loop,
                    "pair_bond": pair_bond,
                    "stem_run": stem_run[pos],
                    "loop_run": loop_run[pos],
                    "dist_nearest_paired": dist_nearest_paired,
                    "dist_nearest_unpaired": dist_nearest_unpaired,
                }

                # Contextual offsets (-3 to +3)
                for offset in [-3, -2, -1, 1, 2, 3]:
                    neighbor_pos = pos + offset
                    if 0 <= neighbor_pos < seq_len:
                        feat[f"seq_{offset}"] = seq[neighbor_pos]
                        feat[f"struct_{offset}"] = (
                            struct[neighbor_pos] if neighbor_pos < len(struct) else "."
                        )
                        feat[f"loop_{offset}"] = (
                            loop[neighbor_pos] if neighbor_pos < len(loop) else "E"
                        )
                    else:
                        feat[f"seq_{offset}"] = "N"
                        feat[f"struct_{offset}"] = "N"
                        feat[f"loop_{offset}"] = "N"

                # GC content windows
                window_5 = [
                    seq[p]
                    for p in range(pos - 2, pos + 3)
                    if 0 <= p < seq_len and seq[p] in ["G", "C"]
                ]
                window_7 = [
                    seq[p]
                    for p in range(pos - 3, pos + 4)
                    if 0 <= p < seq_len and seq[p] in ["G", "C"]
                ]
                feat["gc_5"] = len(window_5) / 5.0
                feat["gc_7"] = len(window_7) / 7.0

                # Dinucleotide and trimer context
                prev_c = seq[pos - 1] if pos > 0 else "N"
                next_c = seq[pos + 1] if pos < seq_len - 1 else "N"
                feat["dimer_left"] = f"{prev_c}{seq[pos]}"
                feat["dimer_right"] = f"{seq[pos]}{next_c}"
                feat["trimer"] = f"{prev_c}{seq[pos]}{next_c}"
                feat["struct_trimer"] = (
                    f"{struct[pos-1] if pos > 0 else 'N'}{struct[pos]}{struct[pos+1] if pos < len(struct)-1 else 'N'}"
                )
                feat["loop_trimer"] = (
                    f"{loop[pos-1] if pos > 0 else 'N'}{loop[pos]}{loop[pos+1] if pos < len(loop)-1 else 'N'}"
                )

                if is_train and TARGET_COL in row:
                    val = row[TARGET_COL]
                    if hasattr(val, "__len__") and not isinstance(val, str):
                        feat[TARGET_COL] = val[pos] if pos < len(val) else np.nan
                    else:
                        feat[TARGET_COL] = val

                features_list.append(feat)
        return pd.DataFrame(features_list)
    else:
        return df.copy()


if (
    "sequence" in train_df.columns
    and isinstance(train_df["sequence"].iloc[0], str)
    and len(train_df["sequence"].iloc[0]) > 1
):
    X_train_raw = extract_features(train_df, is_train=True)
    X_test_raw = extract_features(test_df, is_train=False)
    if TARGET_COL in X_train_raw.columns:
        X_train_raw = X_train_raw.dropna(subset=[TARGET_COL]).reset_index(drop=True)
else:
    X_train_raw = train_df.copy()
    X_test_raw = test_df.copy()

ignore_cols = [TARGET_COL, "id", "id_seqpos", "orig_idx"]
feature_cols = [
    c for c in X_train_raw.columns if c not in ignore_cols and c in X_test_raw.columns
]

# Convert categorical columns
for c in feature_cols:
    if (
        X_train_raw[c].dtype == "object"
        or X_train_raw[c].dtype.name == "category"
        or isinstance(X_train_raw[c].iloc[0], str)
    ):
        X_train_raw[c] = X_train_raw[c].astype("category")
        X_test_raw[c] = X_test_raw[c].astype("category")

X = X_train_raw[feature_cols]
y = X_train_raw[TARGET_COL].values
groups = (
    X_train_raw["orig_idx"].values
    if "orig_idx" in X_train_raw.columns
    else np.arange(len(X))
)
X_test = X_test_raw[feature_cols]

# GroupKFold Cross Validation by molecule
gkf = GroupKFold(n_splits=5)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": 8,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "min_child_samples": 25,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 2000,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_va = X.iloc[val_idx], y[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    val_pred = np.clip(model.predict(X_va), 0, None)
    oof_preds[val_idx] = val_pred
    test_preds += np.clip(model.predict(X_test), 0, None) / gkf.n_splits

    fold_rmse = root_mean_squared_error(y_va, val_pred)
    print(f"Fold {fold + 1} RMSE (MCRMSE): {fold_rmse:.5f}")

overall_rmse = root_mean_squared_error(y, oof_preds)
print(f"Overall CV MCRMSE: {overall_rmse:.5f}")

# Generate Submission
os.makedirs("./working", exist_ok=True)

if len(test_preds) != len(test_df):
    if "orig_idx" in X_test_raw.columns:
        X_test_raw["pred"] = test_preds
        agg_preds = X_test_raw.groupby("orig_idx")["pred"].mean().values
        final_preds = agg_preds
    else:
        final_preds = test_preds[: len(test_df)]
else:
    final_preds = test_preds

submission = pd.DataFrame(
    {"id": np.arange(len(final_preds)), "reactivity": final_preds}
)

submission.to_csv("./working/submission.csv", index=False)
print(f"Submission saved to ./working/submission.csv with shape {submission.shape}")
