import os
import warnings
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# -------------------------------------------------------------------
# Feature Engineering & Data Preprocessing
# -------------------------------------------------------------------
def load_and_normalize_data():
    train_df = pd.read_parquet("./input/train.parquet")
    test_df = pd.read_parquet("./input/test.parquet")

    def sanitize_cols(df):
        df.columns = [
            c.strip()
            .replace(" ", "_")
            .replace("'", "")
            .replace("?", "")
            .replace("-", "_")
            for c in df.columns
        ]
        return df

    train_df = sanitize_cols(train_df)
    test_df = sanitize_cols(test_df)
    return train_df, test_df


def find_col_by_keywords(df, keywords):
    for col in df.columns:
        if all(kw.lower() in col.lower() for kw in keywords):
            return col
    for col in df.columns:
        if any(kw.lower() in col.lower() for kw in keywords):
            return col
    return None


def engineer_features(df):
    data = df.copy()

    age_col = find_col_by_keywords(data, ["age"])
    wife_edu_col = find_col_by_keywords(
        data, ["wife", "education"]
    ) or find_col_by_keywords(data, ["edu"])
    husb_edu_col = find_col_by_keywords(data, ["husband", "education"])
    children_col = find_col_by_keywords(data, ["children"]) or find_col_by_keywords(
        data, ["born"]
    )
    religion_col = find_col_by_keywords(data, ["religion"])
    working_col = find_col_by_keywords(data, ["working"]) or find_col_by_keywords(
        data, ["work"]
    )
    husb_occ_col = find_col_by_keywords(data, ["occupation"])
    sol_col = find_col_by_keywords(
        data, ["standard", "living"]
    ) or find_col_by_keywords(data, ["sol"])
    media_col = find_col_by_keywords(data, ["media"])

    # Reproductive & Age Dynamics
    if age_col and children_col:
        age_val = data[age_col].astype(float)
        child_val = data[children_col].astype(float)
        fertility_years = np.maximum(1.0, age_val - 15.0)
        data["fertility_rate"] = child_val / fertility_years
        data["has_children"] = (child_val > 0).astype(int)
        data["high_parity"] = (child_val >= 4).astype(int)
        data["age_children_ratio"] = age_val / (child_val + 1.0)
        data["age_group"] = pd.cut(
            age_val, bins=[0, 25, 34, 44, 100], labels=[0, 1, 2, 3]
        ).astype(int)

    # Educational Dynamics & Parity
    if wife_edu_col and husb_edu_col:
        w_edu = data[wife_edu_col].astype(float)
        h_edu = data[husb_edu_col].astype(float)
        data["edu_diff"] = w_edu - h_edu
        data["edu_ratio"] = (w_edu + 1.0) / (h_edu + 1.0)
        data["total_household_edu"] = w_edu + h_edu
        data["wife_higher_edu"] = (w_edu > h_edu).astype(int)
        data["both_high_edu"] = ((w_edu == 4) & (h_edu == 4)).astype(int)

    # Socio-Economic & Living Standard Features
    if sol_col and children_col:
        sol_val = data[sol_col].astype(float)
        child_val = data[children_col].astype(float)
        data["sol_per_child"] = sol_val / (child_val + 1.0)

    if wife_edu_col and sol_col:
        w_edu = data[wife_edu_col].astype(float)
        sol_val = data[sol_col].astype(float)
        data["edu_x_sol"] = w_edu * sol_val

    # Empowerment Index
    if wife_edu_col and working_col and media_col:
        w_edu = data[wife_edu_col].astype(float)
        working_binary = (data[working_col].astype(float) == 0).astype(float)
        media_good = (data[media_col].astype(float) == 0).astype(float)
        data["empowerment_score"] = w_edu * (1.0 + working_binary) * (1.0 + media_good)

    if religion_col and working_col:
        data["religion_x_working"] = (
            data[religion_col].astype(str) + "_" + data[working_col].astype(str)
        )
        data["religion_x_working"] = (
            data["religion_x_working"].astype("category").cat.codes
        )

    if sol_col and husb_occ_col:
        data["sol_x_occ"] = (
            data[sol_col].astype(str) + "_" + data[husb_occ_col].astype(str)
        )
        data["sol_x_occ"] = data["sol_x_occ"].astype("category").cat.codes

    return data


def preprocess_data():
    train_df, test_df = load_and_normalize_data()

    target_col = find_col_by_keywords(
        train_df, ["contraceptive", "target", "class", "method"]
    )
    if target_col is None:
        target_col = train_df.columns[-1]

    y_raw = train_df[target_col].astype(int).values
    if np.min(y_raw) == 1:
        y = y_raw - 1
    else:
        y = y_raw

    X_train_raw = train_df.drop(columns=[target_col])
    X_test_raw = test_df.copy()

    X_train_fe = engineer_features(X_train_raw)
    X_test_fe = engineer_features(X_test_raw)

    feature_cols = [c for c in X_train_fe.columns if c not in ["id", "ID"]]
    X_train_fe = X_train_fe[feature_cols]
    X_test_fe = X_test_fe[feature_cols]

    return X_train_fe, y, X_test_fe


# -------------------------------------------------------------------
# PyTorch Tabular Neural Network Architecture
# -------------------------------------------------------------------
class TabularNN(nn.Module):

    def __init__(self, input_dim, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.35),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_pytorch_nn(
    X_train_np, y_train_np, X_val_np, num_classes=3, epochs=60, batch_size=32
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tr_t = torch.tensor(X_train_np, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train_np, dtype=torch.long)
    X_va_t = torch.tensor(X_val_np, dtype=torch.float32)

    dataset = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = TabularNN(input_dim=X_train_np.shape[1], num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(X_va_t.to(device))
        val_probs = torch.softmax(val_logits, dim=1).cpu().numpy()

    return model, val_probs


def predict_pytorch_nn(model, X_test_np):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    X_te_t = torch.tensor(X_test_np, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_te_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs


# -------------------------------------------------------------------
# Threshold Optimization Helper
# -------------------------------------------------------------------
def optimize_thresholds(oof_probs, y_true):
    def obj(weights):
        w = np.array(weights)
        scaled_probs = oof_probs * w
        preds = np.argmax(scaled_probs, axis=1)
        return -accuracy_score(y_true, preds)

    best_score = float("inf")
    best_w = np.array([1.0, 1.0, 1.0])

    seeds = [[1.0, 1.0, 1.0]]
    for w1 in [0.7, 0.9, 1.0, 1.1, 1.3]:
        for w2 in [0.7, 0.9, 1.0, 1.1, 1.3]:
            for w3 in [0.7, 0.9, 1.0, 1.1, 1.3]:
                seeds.append([w1, w2, w3])

    for seed in seeds:
        res = minimize(obj, seed, method="Nelder-Mead", options={"maxiter": 300})
        if res.fun < best_score:
            best_score = res.fun
            best_w = res.x

    return best_w


# -------------------------------------------------------------------
# Main Pipeline Execution
# -------------------------------------------------------------------
def main():
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    X, y, X_test = preprocess_data()
    n_samples = len(X)
    n_classes = len(np.unique(y))

    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_base_features = np.zeros((n_samples, 5 * n_classes))
    test_base_features = np.zeros((len(X_test), 5 * n_classes))

    X_np = X.values
    X_test_np = X_test.values

    feature_cols = list(X.columns)
    cat_cols = [c for c in feature_cols if X[c].dtype == "object" or X[c].nunique() < 10]

    te_keywords = [
        ["wife", "education"],
        ["husband", "education"],
        ["occupation"],
        ["standard", "living"],
        ["age_group"],
    ]
    te_cols = []
    for kw in te_keywords:
        c = find_col_by_keywords(X, kw)
        if c is not None and c in X.columns and c not in te_cols:
            te_cols.append(c)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr = X.iloc[train_idx].copy()
        y_tr = y[train_idx]
        X_va = X.iloc[val_idx].copy()
        y_va = y[val_idx]
        X_te = X_test.copy()

        # Out-Of-Fold Target Encoding for key categorical/ordinal features
        for col in te_cols:
            if col in X_tr.columns:
                for k in range(n_classes):
                    prior_k = np.mean(y_tr == k)
                    counts = X_tr[col].value_counts()
                    class_counts = X_tr[y_tr == k][col].value_counts()
                    m = 10.0
                    te_map = ((class_counts + m * prior_k) / (counts + m)).to_dict()

                    X_tr[f"{col}_te_{k}"] = X_tr[col].astype(object).map(te_map).fillna(prior_k).astype(float)
                    X_va[f"{col}_te_{k}"] = X_va[col].astype(object).map(te_map).fillna(prior_k).astype(float)
                    X_te[f"{col}_te_{k}"] = X_te[col].astype(object).map(te_map).fillna(prior_k).astype(float)

        for col in cat_cols:
            freq = X_tr[col].value_counts(normalize=True).to_dict()
            X_tr[f"{col}_freq"] = X_tr[col].astype(object).map(freq).fillna(0)
            X_va[f"{col}_freq"] = X_va[col].astype(object).map(freq).fillna(0)
            X_te[f"{col}_freq"] = X_te[col].astype(object).map(freq).fillna(0)

        scaler = StandardScaler()
        cols_to_scale = list(X_tr.columns)
        X_tr = pd.DataFrame(scaler.fit_transform(X_tr), columns=cols_to_scale)
        X_va = pd.DataFrame(scaler.transform(X_va), columns=cols_to_scale)
        X_te = pd.DataFrame(scaler.transform(X_te), columns=cols_to_scale)

        # 1. LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.025,
            max_depth=4,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=42 + fold,
            verbose=-1,
        )
        lgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        prob_lgb_val = lgb_model.predict_proba(X_va)
        prob_lgb_test = lgb_model.predict_proba(X_te)

        # 2. XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.025,
            max_depth=4,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=42 + fold,
            eval_metric="mlogloss",
            early_stopping_rounds=50,
            verbosity=0,
        )
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        prob_xgb_val = xgb_model.predict_proba(X_va)
        prob_xgb_test = xgb_model.predict_proba(X_te)

        # 3. CatBoost
        cat_model = cb.CatBoostClassifier(
            iterations=500,
            learning_rate=0.025,
            depth=4,
            l2_leaf_reg=4.0,
            random_seed=42 + fold,
            early_stopping_rounds=50,
            verbose=False,
        )
        cat_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
        prob_cat_val = cat_model.predict_proba(X_va)
        prob_cat_test = cat_model.predict_proba(X_te)

        # 4. ExtraTrees
        et_model = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=7,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42 + fold,
            n_jobs=-1,
        )
        et_model.fit(X_tr, y_tr)
        prob_et_val = et_model.predict_proba(X_va)
        prob_et_test = et_model.predict_proba(X_te)

        # 5. PyTorch Neural Network
        nn_model, prob_nn_val = train_pytorch_nn(
            X_tr.values,
            y_tr,
            X_va.values,
            num_classes=n_classes,
            epochs=60,
            batch_size=32,
        )
        prob_nn_test = predict_pytorch_nn(nn_model, X_te.values)

        val_concat = np.hstack([prob_lgb_val, prob_xgb_val, prob_cat_val, prob_et_val, prob_nn_val])
        test_concat = np.hstack([prob_lgb_test, prob_xgb_test, prob_cat_test, prob_et_test, prob_nn_test])

        oof_base_features[val_idx] = val_concat
        test_base_features += test_concat / n_splits

    # Logistic Regression Meta-Learner Stacking
    meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta_model.fit(oof_base_features, y)
    oof_stacked_probs = meta_model.predict_proba(oof_base_features)
    test_stacked_probs = meta_model.predict_proba(test_base_features)

    # Threshold Multiplier Search via Nelder-Mead Optimization
    best_weights = optimize_thresholds(oof_stacked_probs, y)

    # Compute Out-of-Fold accuracy score
    oof_preds = np.argmax(oof_stacked_probs * best_weights, axis=1)
    val_acc = accuracy_score(y, oof_preds)

    # Convert predictions back to original 1, 2, 3 label scheme
    final_test_preds = np.argmax(test_stacked_probs * best_weights, axis=1) + 1

    submission = pd.DataFrame(
        {
            "id": np.arange(len(X_test)),
            "Contraceptive_method_used": final_test_preds,
        }
    )
    submission.to_csv("./submission/submission.csv", index=False)

    print(f"Final Validation Score: {val_acc:.5f}")


if __name__ == "__main__":
    main()