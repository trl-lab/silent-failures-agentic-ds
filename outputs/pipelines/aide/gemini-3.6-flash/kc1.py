import os
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import QuantileTransformer, RobustScaler

warnings.filterwarnings("ignore")

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Extract target variable
target_col = "defects"
if target_col in train_df.columns:
    y = (
        train_df[target_col]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "t"])
        .astype(int)
        .values
    )
    X_raw = train_df.drop(columns=[target_col])
else:
    raise ValueError(f"Target column '{target_col}' not found in training data.")

X_test_raw = test_df.drop(columns=[target_col], errors="ignore")

# Drop 'id' from features if present
if "id" in X_raw.columns:
    X_raw = X_raw.drop(columns=["id"])
if "id" in X_test_raw.columns:
    X_test_raw = X_test_raw.drop(columns=["id"])


# 2. Domain-Specific Feature Engineering Function
def create_features(df):
    data = df.copy()

    # Numeric columns ensuring float conversion
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

    # McCabe & Halstead Ratio & Interaction Features
    loc = data["loc"] if "loc" in data.columns else data.iloc[:, 0]
    vg = data["v(g)"] if "v(g)" in data.columns else 0
    evg = data["ev(g)"] if "ev(g)" in data.columns else 0
    ivg = data["iv(g)"] if "iv(g)" in data.columns else 0
    lOComment = data["lOComment"] if "lOComment" in data.columns else 0
    lOBlank = data["lOBlank"] if "lOBlank" in data.columns else 0
    lOCode = data["lOCode"] if "lOCode" in data.columns else 0
    uniq_Op = data["uniq_Op"] if "uniq_Op" in data.columns else 0
    uniq_Opnd = data["uniq_Opnd"] if "uniq_Opnd" in data.columns else 0
    total_Op = data["total_Op"] if "total_Op" in data.columns else 0
    total_Opnd = data["total_Opnd"] if "total_Opnd" in data.columns else 0
    e = data["e"] if "e" in data.columns else 0
    v = data["v"] if "v" in data.columns else 0
    b = data["b"] if "b" in data.columns else 0
    branchCount = data["branchCount"] if "branchCount" in data.columns else 0

    data["loc_total"] = loc + lOComment + lOBlank
    data["comment_density"] = lOComment / (loc + 1.0)
    data["blank_density"] = lOBlank / (loc + 1.0)
    data["code_comment_ratio"] = lOCode / (lOComment + 1.0)
    data["cyclomatic_density"] = vg / (loc + 1.0)
    data["essential_density"] = evg / (vg + 1.0)
    data["design_density"] = ivg / (vg + 1.0)
    data["op_vocab_total"] = uniq_Op + uniq_Opnd
    data["op_count_total"] = total_Op + total_Opnd
    data["uniq_op_ratio"] = uniq_Op / (uniq_Opnd + 1.0)
    data["total_op_ratio"] = total_Op / (total_Opnd + 1.0)
    data["volume_per_line"] = v / (loc + 1.0)
    data["effort_per_line"] = e / (loc + 1.0)
    data["branch_density"] = branchCount / (loc + 1.0)
    data["b_per_v"] = b / (v + 1.0)

    # Log1p transforms of numeric columns to handle high skewness
    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    for col in num_cols:
        data[f"{col}_log1p"] = np.log1p(np.maximum(0, df[col]))

    return data


X_engineered = create_features(X_raw)
X_test_engineered = create_features(X_test_raw)

# Align columns
common_cols = X_engineered.columns.intersection(X_test_engineered.columns)
X_engineered = X_engineered[common_cols]
X_test_engineered = X_test_engineered[common_cols]

# 3. Model Training with 5-Fold Stratified K-Fold Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(y))
test_preds = np.zeros(len(X_test_engineered))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_engineered, y)):
    X_tr, y_tr = X_engineered.iloc[train_idx], y[train_idx]
    X_va, y_va = X_engineered.iloc[val_idx], y[val_idx]

    # Preprocessing pipelines for distance/NN sensitive models
    scaler = QuantileTransformer(output_distribution="normal", random_state=42)
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)
    X_test_scaled = scaler.transform(X_test_engineered)

    # Model 1: HistGradientBoosting
    hgb = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.03,
        max_leaf_nodes=15,
        min_samples_leaf=10,
        random_state=42 + fold,
    )
    hgb.fit(X_tr, y_tr)
    val_p_hgb = hgb.predict_proba(X_va)[:, 1]
    test_p_hgb = hgb.predict_proba(X_test_engineered)[:, 1]

    # Model 2: ExtraTrees Classifier
    et = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        random_state=42 + fold,
        n_jobs=-1,
    )
    et.fit(X_tr, y_tr)
    val_p_et = et.predict_proba(X_va)[:, 1]
    test_p_et = et.predict_proba(X_test_engineered)[:, 1]

    # Model 3: MLP Classifier
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=200,
        alpha=0.01,
        random_state=42 + fold,
        early_stopping=True,
    )
    mlp.fit(X_tr_scaled, y_tr)
    val_p_mlp = mlp.predict_proba(X_va_scaled)[:, 1]
    test_p_mlp = mlp.predict_proba(X_test_scaled)[:, 1]

    # Blended fold predictions
    fold_val_preds = 0.50 * val_p_hgb + 0.35 * val_p_et + 0.15 * val_p_mlp
    fold_test_preds = 0.50 * test_p_hgb + 0.35 * test_p_et + 0.15 * test_p_mlp

    oof_preds[val_idx] = fold_val_preds
    test_preds += fold_test_preds / skf.n_splits

# Compute Out-of-Fold Evaluation Metric
cv_auc = roc_auc_score(y, oof_preds)
print(f"5-Fold Cross-Validation ROC AUC Score: {cv_auc:.5f}")

# 4. Save Test Predictions to ./working/submission.csv
os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame({"id": np.arange(len(test_preds)), "defects": test_preds})
submission.to_csv("./working/submission.csv", index=False)
