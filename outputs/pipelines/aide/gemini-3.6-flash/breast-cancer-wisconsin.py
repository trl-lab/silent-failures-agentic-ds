import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

# Load train and test data
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

# Target encoding (Malignant vs Benign)
target_col = "Diagnosis"
if train[target_col].dtype == object or isinstance(train[target_col].iloc[0], str):
    le = LabelEncoder()
    y = le.fit_transform(train[target_col])
else:
    y = train[target_col].values


# Feature engineering function for morphological ratios and shape factors
def engineer_features(df):
    df_feat = df.copy()
    eps = 1e-8

    # Ratios of worst (3) to mean (1) measurements
    df_feat["radius_ratio"] = df_feat["radius3"] / (df_feat["radius1"] + eps)
    df_feat["perimeter_ratio"] = df_feat["perimeter3"] / (df_feat["perimeter1"] + eps)
    df_feat["area_ratio"] = df_feat["area3"] / (df_feat["area1"] + eps)
    df_feat["compactness_ratio"] = df_feat["compactness3"] / (
        df_feat["compactness1"] + eps
    )
    df_feat["concavity_ratio"] = df_feat["concavity3"] / (df_feat["concavity1"] + eps)
    df_feat["concave_points_ratio"] = df_feat["concave_points3"] / (
        df_feat["concave_points1"] + eps
    )

    # Shape compactness factors
    df_feat["shape_factor1"] = df_feat["area1"] / (df_feat["perimeter1"] ** 2 + eps)
    df_feat["shape_factor3"] = df_feat["area3"] / (df_feat["perimeter3"] ** 2 + eps)

    ignore_cols = ["ID", "id", target_col]
    feature_cols = [col for col in df_feat.columns if col not in ignore_cols]
    return df_feat[feature_cols]


X_df = engineer_features(train)
X_test_df = engineer_features(test)

X = X_df.values
X_test = X_test_df.values

# Stratified K-Fold setup
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
num_models = 4
oof_model_ranks = np.zeros((len(train), num_models))
test_model_ranks = np.zeros((len(test), num_models))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Feature scaling fitted strictly on train fold to eliminate leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 1. LightGBM Model
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        random_state=42 + fold,
        verbosity=-1,
    )
    lgb_model.fit(
        X_train_scaled,
        y_train,
        eval_set=[(X_val_scaled, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    lgb_val_prob = lgb_model.predict_proba(X_val_scaled)[:, 1]
    lgb_test_prob = lgb_model.predict_proba(X_test_scaled)[:, 1]

    # 2. Logistic Regression Model
    lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42 + fold)
    lr_model.fit(X_train_scaled, y_train)
    lr_val_prob = lr_model.predict_proba(X_val_scaled)[:, 1]
    lr_test_prob = lr_model.predict_proba(X_test_scaled)[:, 1]

    # 3. Support Vector Classifier (RBF Kernel)
    svc_model = SVC(C=1.0, kernel="rbf", probability=True, random_state=42 + fold)
    svc_model.fit(X_train_scaled, y_train)
    svc_val_prob = svc_model.predict_proba(X_val_scaled)[:, 1]
    svc_test_prob = svc_model.predict_proba(X_test_scaled)[:, 1]

    # 4. ExtraTrees Classifier
    et_model = ExtraTreesClassifier(
        n_estimators=300, max_depth=8, random_state=42 + fold
    )
    et_model.fit(X_train_scaled, y_train)
    et_val_prob = et_model.predict_proba(X_val_scaled)[:, 1]
    et_test_prob = et_model.predict_proba(X_test_scaled)[:, 1]

    val_probs = [lgb_val_prob, lr_val_prob, svc_val_prob, et_val_prob]
    test_probs = [lgb_test_prob, lr_test_prob, svc_test_prob, et_test_prob]

    for m in range(num_models):
        oof_model_ranks[val_idx, m] = rankdata(val_probs[m]) / len(val_probs[m])
        test_model_ranks[:, m] += (
            rankdata(test_probs[m]) / len(test_probs[m])
        ) / kf.n_splits

# Grid search to find initial optimal model weights maximizing out-of-fold ROC AUC
weights_list = []
step = 0.05
steps = int(1.0 / step) + 1
grid_vals = np.linspace(0, 1, steps)

for w1 in grid_vals:
    for w2 in np.linspace(0, 1 - w1, int((1 - w1) / step) + 1):
        for w3 in np.linspace(0, 1 - w1 - w2, int((1 - w1 - w2) / step) + 1):
            w4 = 1.0 - w1 - w2 - w3
            if w4 >= -1e-5:
                weights_list.append([w1, w2, w3, max(0.0, w4)])

weights_grid = np.array(weights_list)

best_auc = 0
best_weights = np.array([0.25, 0.25, 0.25, 0.25])

for w in weights_grid:
    blend_pred = oof_model_ranks @ w
    score = roc_auc_score(y, blend_pred)
    if score > best_auc:
        best_auc = score
        best_weights = w


# Refine ensemble weights using Nelder-Mead optimization
def loss_func(weights):
    w = np.maximum(weights, 0)
    if np.sum(w) > 0:
        w = w / np.sum(w)
    pred = oof_model_ranks @ w
    return -roc_auc_score(y, pred)


res = minimize(loss_func, best_weights, method="Nelder-Mead")
refined_weights = np.maximum(res.x, 0)
if np.sum(refined_weights) > 0:
    refined_weights /= np.sum(refined_weights)

refined_auc = roc_auc_score(y, oof_model_ranks @ refined_weights)
if refined_auc > best_auc:
    best_weights = refined_weights
    best_auc = refined_auc

oof_final_preds = oof_model_ranks @ best_weights
test_final_preds = test_model_ranks @ best_weights

cv_auc = roc_auc_score(y, oof_final_preds)
print(f"5-Fold Cross-Validation ROC AUC Score (Optimized Blend): {cv_auc:.5f}")

# Save test predictions in the required directory and format
os.makedirs("./working", exist_ok=True)
sub = pd.DataFrame({"id": np.arange(len(test)), "Diagnosis": test_final_preds})
sub.to_csv("./working/submission.csv", index=False)
