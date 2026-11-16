import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb
from xgboost import XGBClassifier

try:
    from catboost import CatBoostClassifier

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# Ensure working directory exists
os.makedirs("./working", exist_ok=True)

# Load data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")


# Domain-informed Feature Engineering
def engineer_features(df):
    df = df.copy()
    eps = 1e-6

    # Total edge strength, ratios, and differences
    if "vedge-mean" in df.columns and "hedge-mean" in df.columns:
        df["total_edge_mean"] = df["vedge-mean"] + df["hedge-mean"]
        df["edge_ratio"] = (df["vedge-mean"] + eps) / (df["hedge-mean"] + eps)
        df["edge_diff"] = np.abs(df["vedge-mean"] - df["hedge-mean"])

    if "vegde-sd" in df.columns and "hedge-sd" in df.columns:
        df["total_edge_sd"] = df["vegde-sd"] + df["hedge-sd"]

    # Edge Signal-to-Noise Ratios (mean / standard deviation)
    if "vedge-mean" in df.columns and "vegde-sd" in df.columns:
        df["vedge_snr"] = df["vedge-mean"] / (df["vegde-sd"] + eps)
    if "hedge-mean" in df.columns and "hedge-sd" in df.columns:
        df["hedge_snr"] = df["hedge-mean"] / (df["hedge-sd"] + eps)

    # Color channel ratios relative to intensity
    if "intensity-mean" in df.columns:
        if "rawred-mean" in df.columns:
            df["red_ratio"] = df["rawred-mean"] / (df["intensity-mean"] + eps)
        if "rawblue-mean" in df.columns:
            df["blue_ratio"] = df["rawblue-mean"] / (df["intensity-mean"] + eps)
        if "rawgreen-mean" in df.columns:
            df["green_ratio"] = df["rawgreen-mean"] / (df["intensity-mean"] + eps)

    # Chromaticity coordinates (r, g, b) normalized by RGB sum
    if all(c in df.columns for c in ["rawred-mean", "rawgreen-mean", "rawblue-mean"]):
        rgb_sum = df["rawred-mean"] + df["rawgreen-mean"] + df["rawblue-mean"] + eps
        df["chroma_r"] = df["rawred-mean"] / rgb_sum
        df["chroma_g"] = df["rawgreen-mean"] / rgb_sum
        df["chroma_b"] = df["rawblue-mean"] / rgb_sum
        df["rg_diff"] = df["rawred-mean"] - df["rawgreen-mean"]
        df["rb_diff"] = df["rawred-mean"] - df["rawblue-mean"]
        df["gb_diff"] = df["rawgreen-mean"] - df["rawblue-mean"]

    # Line density combination
    if "short-line-density-5" in df.columns and "short-line-density-2" in df.columns:
        df["total_line_density"] = (
            df["short-line-density-5"] + df["short-line-density-2"]
        )
        df["line_density_ratio"] = (df["short-line-density-5"] + eps) / (
            df["short-line-density-2"] + eps
        )

    # Trigonometric transformation of hue (circular encoding)
    if "hue-mean" in df.columns:
        df["sin_hue"] = np.sin(df["hue-mean"])
        df["cos_hue"] = np.cos(df["hue-mean"])

    return df


target_col = "class"
feature_cols = [c for c in train_df.columns if c != target_col]

train_engineered = engineer_features(train_df[feature_cols])
test_engineered = engineer_features(test_df[feature_cols])

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(train_df[target_col])
num_classes = len(le.classes_)

X_train = train_engineered.values
X_test = test_engineered.values

# Cross-validation setup
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

models_list = ["xgb", "lgb", "et", "rf", "mlp"]
if HAS_CATBOOST:
    models_list.append("cb")

oof_dict = {m: np.zeros((len(train_df), num_classes)) for m in models_list}
test_dict = {m: np.zeros((len(test_df), num_classes)) for m in models_list}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    # 1. XGBoost
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=900,
        learning_rate=0.035,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42 + fold,
        eval_metric="mlogloss",
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_dict["xgb"][val_idx] = xgb.predict_proba(X_val)
    test_dict["xgb"] += xgb.predict_proba(X_test) / n_splits

    # 2. LightGBM
    lgb_clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        n_estimators=900,
        learning_rate=0.035,
        max_depth=6,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    oof_dict["lgb"][val_idx] = lgb_clf.predict_proba(X_val)
    test_dict["lgb"] += lgb_clf.predict_proba(X_test) / n_splits

    # 3. ExtraTrees
    et = ExtraTreesClassifier(
        n_estimators=450,
        max_depth=16,
        min_samples_split=3,
        random_state=42 + fold,
        n_jobs=-1,
    )
    et.fit(X_tr, y_tr)
    oof_dict["et"][val_idx] = et.predict_proba(X_val)
    test_dict["et"] += et.predict_proba(X_test) / n_splits

    # 4. Random Forest
    rf = RandomForestClassifier(
        n_estimators=450,
        max_depth=16,
        min_samples_split=3,
        random_state=42 + fold,
        n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)
    oof_dict["rf"][val_idx] = rf.predict_proba(X_val)
    test_dict["rf"] += rf.predict_proba(X_test) / n_splits

    # 5. Multi-Layer Perceptron (Neural Network with Standard Scaler)
    mlp = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=0.001,
            max_iter=400,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=42 + fold,
        ),
    )
    mlp.fit(X_tr, y_tr)
    oof_dict["mlp"][val_idx] = mlp.predict_proba(X_val)
    test_dict["mlp"] += mlp.predict_proba(X_test) / n_splits

    # 6. CatBoost (if available)
    if HAS_CATBOOST:
        cb = CatBoostClassifier(
            iterations=900,
            learning_rate=0.035,
            depth=5,
            loss_function="MultiClass",
            random_seed=42 + fold,
            early_stopping_rounds=50,
            verbose=False,
            thread_count=-1,
        )
        cb.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        oof_dict["cb"][val_idx] = cb.predict_proba(X_val)
        test_dict["cb"] += cb.predict_proba(X_test) / n_splits


# Optimize ensemble weights to minimize multi-class log loss smoothly
def loss_func(weights):
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    blend_oof = np.zeros_like(oof_dict[models_list[0]])
    for i, m in enumerate(models_list):
        blend_oof += weights[i] * oof_dict[m]
    eps = 1e-15
    blend_oof = np.clip(blend_oof, eps, 1 - eps)
    return log_loss(y_train, blend_oof)


init_weights = [1.0 / len(models_list)] * len(models_list)
bounds = [(0.0, 1.0)] * len(models_list)
constraints = {"type": "eq", "fun": lambda w: 1.0 - sum(w)}

res = minimize(
    loss_func,
    init_weights,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"maxiter": 500},
)
opt_weights = res.x / np.sum(res.x)

# Evaluate ensemble performance
ensemble_oof = np.zeros_like(oof_dict[models_list[0]])
ensemble_test = np.zeros_like(test_dict[models_list[0]])

for i, m in enumerate(models_list):
    m_acc = accuracy_score(y_train, np.argmax(oof_dict[m], axis=1))
    print(f"Model {m} OOF Accuracy: {m_acc:.5f}, Blend Weight: {opt_weights[i]:.4f}")
    ensemble_oof += opt_weights[i] * oof_dict[m]
    ensemble_test += opt_weights[i] * test_dict[m]

final_cv_acc = accuracy_score(y_train, np.argmax(ensemble_oof, axis=1))
print(f"Ensemble 5-Fold Cross-Validation Accuracy: {final_cv_acc:.5f}")

# Save test predictions
test_preds_encoded = np.argmax(ensemble_test, axis=1)
test_preds_labels = le.inverse_transform(test_preds_encoded)

submission = pd.DataFrame({"id": np.arange(len(test_df)), "class": test_preds_labels})
submission.to_csv("./working/submission.csv", index=False)
print("Submission saved successfully with shape:", submission.shape)
