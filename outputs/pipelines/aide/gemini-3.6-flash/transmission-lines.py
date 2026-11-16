import os
import re
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# 1. Load data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train = pd.read_parquet(train_path)
test = pd.read_parquet(test_path)

os.makedirs("./working", exist_ok=True)

# Target processing
target_col = "VOLT_CLASS"
y_raw = train[target_col].values
le_target = LabelEncoder()
y = le_target.fit_transform(y_raw)

test_ids = np.arange(len(test))

# Combine datasets for feature engineering
n_train = len(train)
df_all = pd.concat(
    [train.drop(columns=[target_col], errors="ignore"), test], ignore_index=True
)


# Enhanced WKT geometry feature extractor with curvature & quantile stats
def extract_wkt_features(wkt_str):
    num_feats = 39
    if not isinstance(wkt_str, str) or len(wkt_str) == 0:
        return [np.nan] * num_feats
    coords = re.findall(r"[-+]?\d*\.\d+|\d+", wkt_str)
    if len(coords) < 2:
        return [np.nan] * num_feats
    nums = [float(x) for x in coords]
    xs = np.array(nums[0::2])
    ys = np.array(nums[1::2])
    n_pts = min(len(xs), len(ys))
    if n_pts == 0:
        return [np.nan] * num_feats
    xs = xs[:n_pts]
    ys = ys[:n_pts]

    min_x, max_x = np.min(xs), np.max(xs)
    min_y, max_y = np.min(ys), np.max(ys)
    width = max_x - min_x
    height = max_y - min_y
    area = width * height
    aspect_ratio = height / (width + 1e-6)
    diag = np.sqrt(width**2 + height**2)
    centroid_x = np.mean(xs)
    centroid_y = np.mean(ys)
    std_x = np.std(xs)
    std_y = np.std(ys)

    start_x, start_y = xs[0], ys[0]
    end_x, end_y = xs[-1], ys[-1]
    disp = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
    orientation = np.arctan2(end_y - start_y, end_x - start_x)

    if n_pts > 1:
        dx = np.diff(xs)
        dy = np.diff(ys)
        seg_lens = np.sqrt(dx**2 + dy**2)
        path_len = np.sum(seg_lens)
        mean_seg = np.mean(seg_lens)
        min_seg = np.min(seg_lens)
        max_seg = np.max(seg_lens)
        std_seg = np.std(seg_lens)
        q25_seg = np.percentile(seg_lens, 25)
        median_seg = np.median(seg_lens)
        q75_seg = np.percentile(seg_lens, 75)
        cv_seg = std_seg / (mean_seg + 1e-6)
    else:
        path_len = mean_seg = min_seg = max_seg = std_seg = q25_seg = median_seg = (
            q75_seg
        ) = cv_seg = 0.0

    tortuosity = path_len / (disp + 1e-6)

    if n_pts > 2:
        seg_orientations = np.arctan2(dy, dx)
        angle_diffs = np.diff(seg_orientations)
        angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
        abs_turn_angles = np.abs(angle_diffs)
        mean_turn = np.mean(abs_turn_angles)
        max_turn = np.max(abs_turn_angles)
        std_turn = np.std(abs_turn_angles)
        sum_turn = np.sum(abs_turn_angles)
        curvature_density = sum_turn / (path_len + 1e-6)
    else:
        mean_turn = max_turn = std_turn = sum_turn = curvature_density = 0.0

    grid_x_01 = np.round(centroid_x / 0.1)
    grid_y_01 = np.round(centroid_y / 0.1)
    grid_x_05 = np.round(centroid_x / 0.5)
    grid_y_05 = np.round(centroid_y / 0.5)

    return [
        n_pts,
        min_x,
        max_x,
        min_y,
        max_y,
        width,
        height,
        area,
        aspect_ratio,
        diag,
        centroid_x,
        centroid_y,
        std_x,
        std_y,
        start_x,
        start_y,
        end_x,
        end_y,
        disp,
        orientation,
        path_len,
        mean_seg,
        min_seg,
        max_seg,
        std_seg,
        q25_seg,
        median_seg,
        q75_seg,
        cv_seg,
        tortuosity,
        mean_turn,
        max_turn,
        std_turn,
        sum_turn,
        curvature_density,
        grid_x_01,
        grid_y_01,
        grid_x_05,
        grid_y_05,
    ]


# Identify geometry columns
geom_cols = [
    c
    for c in df_all.columns
    if "wkt" in c.lower() or "geom" in c.lower() or "shape" in c.lower()
]
spatial_df = pd.DataFrame()

for gc in geom_cols:
    if df_all[gc].dtype == object or str(df_all[gc].dtype) == "string":
        feat_names = [
            f"{gc}_n_pts",
            f"{gc}_min_x",
            f"{gc}_max_x",
            f"{gc}_min_y",
            f"{gc}_max_y",
            f"{gc}_width",
            f"{gc}_height",
            f"{gc}_area",
            f"{gc}_aspect_ratio",
            f"{gc}_diag",
            f"{gc}_centroid_x",
            f"{gc}_centroid_y",
            f"{gc}_std_x",
            f"{gc}_std_y",
            f"{gc}_start_x",
            f"{gc}_start_y",
            f"{gc}_end_x",
            f"{gc}_end_y",
            f"{gc}_disp",
            f"{gc}_orientation",
            f"{gc}_path_len",
            f"{gc}_mean_seg",
            f"{gc}_min_seg",
            f"{gc}_max_seg",
            f"{gc}_std_seg",
            f"{gc}_q25_seg",
            f"{gc}_median_seg",
            f"{gc}_q75_seg",
            f"{gc}_cv_seg",
            f"{gc}_tortuosity",
            f"{gc}_mean_turn",
            f"{gc}_max_turn",
            f"{gc}_std_turn",
            f"{gc}_sum_turn",
            f"{gc}_curvature_density",
            f"{gc}_grid_x_01",
            f"{gc}_grid_y_01",
            f"{gc}_grid_x_05",
            f"{gc}_grid_y_05",
        ]
        feats = df_all[gc].apply(extract_wkt_features).tolist()
        sdf = pd.DataFrame(feats, columns=feat_names)
        spatial_df = pd.concat([spatial_df, sdf], axis=1)

# Process date columns
date_df = pd.DataFrame()
for col in df_all.columns:
    if col in geom_cols:
        continue
    if "date" in col.lower() or "year" in col.lower() or "time" in col.lower():
        try:
            dt_series = pd.to_datetime(df_all[col], errors="coerce")
            if dt_series.notnull().sum() > 0.1 * len(df_all):
                date_df[f"{col}_year"] = dt_series.dt.year
                date_df[f"{col}_month"] = dt_series.dt.month
                date_df[f"{col}_day"] = dt_series.dt.day
                date_df[f"{col}_dayofweek"] = dt_series.dt.dayofweek
                date_df[f"{col}_sec"] = (
                    dt_series - pd.Timestamp("1970-01-01")
                ) // pd.Timedelta(seconds=1)
        except Exception:
            pass

# Process categorical and numeric columns
cat_cols = []
num_cols = []

for col in df_all.columns:
    if col in geom_cols:
        continue
    if (
        df_all[col].dtype == "object"
        or df_all[col].dtype == "category"
        or str(df_all[col].dtype) == "string"
    ):
        cat_cols.append(col)
    else:
        num_cols.append(col)

freq_df = pd.DataFrame()
for c in cat_cols:
    freq_map = df_all[c].value_counts(dropna=False).to_dict()
    freq_df[f"{c}_freq"] = df_all[c].map(freq_map)
    freq_df[f"{c}_len"] = df_all[c].astype(str).str.len()

df_proc = pd.concat([df_all[num_cols], spatial_df, date_df, freq_df], axis=1)

for c in cat_cols:
    df_proc[c] = df_all[c].astype(str).fillna("missing")

X_train = df_proc.iloc[:n_train].copy().reset_index(drop=True)
X_test = df_proc.iloc[n_train:].copy().reset_index(drop=True)

# 5-Fold Stratified Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
n_classes = len(le_target.classes_)
oof_preds = np.zeros((n_train, n_classes))
test_preds = np.zeros((len(test), n_classes))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]

    cb_model = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.04,
        depth=7,
        l2_leaf_reg=3,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        cat_features=cat_cols if len(cat_cols) > 0 else None,
        random_seed=42 + fold,
        verbose=0,
        early_stopping_rounds=50,
    )

    cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)

    oof_preds[val_idx] = cb_model.predict_proba(X_va)
    test_preds += cb_model.predict_proba(X_test) / skf.n_splits

oof_pred_labels = np.argmax(oof_preds, axis=1)
acc = accuracy_score(y, oof_pred_labels)
print(f"Validation Accuracy: {acc:.5f}")

# Save test predictions
final_pred_indices = np.argmax(test_preds, axis=1)
final_pred_labels = le_target.inverse_transform(final_pred_indices)

submission = pd.DataFrame({"id": test_ids, "VOLT_CLASS": final_pred_labels})
submission.to_csv("./working/submission.csv", index=False)
