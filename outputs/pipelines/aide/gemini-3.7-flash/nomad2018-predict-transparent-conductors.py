import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

# 1. Load Data
train_path = "./input/train.parquet"
test_path = "./input/test.parquet"
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

target_col = "formation_energy_ev_natom"


# 2. Feature Engineering Function
def extract_features(df):
    data = df.copy()

    # Identify lattice and composition columns if present
    cols = data.columns

    # Unit cell volume calculation from lattice parameters if available
    if all(
        col in cols
        for col in [
            "lattice_vector_1_ang",
            "lattice_vector_2_ang",
            "lattice_vector_3_ang",
            "lattice_angle_alpha_degree",
            "lattice_angle_beta_degree",
            "lattice_angle_gamma_degree",
        ]
    ):
        a = data["lattice_vector_1_ang"]
        b = data["lattice_vector_2_ang"]
        c = data["lattice_vector_3_ang"]
        alpha = np.radians(data["lattice_angle_alpha_degree"])
        beta = np.radians(data["lattice_angle_beta_degree"])
        gamma = np.radians(data["lattice_angle_gamma_degree"])

        # Parallelepiped volume formula
        val = (
            1
            - np.cos(alpha) ** 2
            - np.cos(beta) ** 2
            - np.cos(gamma) ** 2
            + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
        )
        val = np.clip(val, 0, None)
        vol = a * b * c * np.sqrt(val)
        data["cell_volume"] = vol
        data["density"] = data["number_of_total_atoms"] / (vol + 1e-6)
        data["a_b_ratio"] = a / (b + 1e-6)
        data["b_c_ratio"] = b / (c + 1e-6)
        data["a_c_ratio"] = a / (c + 1e-6)

    # Atomic composition descriptors (Al, Ga, In)
    # Approx properties: Atomic radii (pm): Al=143, Ga=153, In=167; Mass: Al=26.98, Ga=69.72, In=114.82
    if all(
        col in cols for col in ["percent_atom_al", "percent_atom_ga", "percent_atom_in"]
    ):
        al = data["percent_atom_al"]
        ga = data["percent_atom_ga"]
        in_ = data["percent_atom_in"]

        data["weighted_atomic_radius"] = al * 143.0 + ga * 153.0 + in_ * 167.0
        data["weighted_atomic_mass"] = al * 26.98 + ga * 69.72 + in_ * 114.82
        data["weighted_electronegativity"] = al * 1.61 + ga * 1.81 + in_ * 1.78
        data["cation_entropy"] = -(
            al * np.log(al + 1e-9) + ga * np.log(ga + 1e-9) + in_ * np.log(in_ + 1e-9)
        )

    return data


train_feat = extract_features(train_df)
test_feat = extract_features(test_df)

# Drop non-feature columns
drop_cols = [
    col for col in ["id", target_col, "bandgap_energy_ev"] if col in train_feat.columns
]
features = [col for col in train_feat.columns if col not in drop_cols]

# Target transformation for RMSLE
y_raw = train_feat[target_col].values
y_log = np.log1p(np.clip(y_raw, 0, None))

X = train_feat[features].copy()
X_test = test_feat[features].copy()

# Categorical features handling
cat_cols = [col for col in ["spacegroup"] if col in X.columns]
for c in cat_cols:
    X[c] = X[c].astype("category")
    X_test[c] = X_test[c].astype("category")

# 3. 5-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train_feat))
test_preds = np.zeros(len(test_feat))

lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_log)):
    X_tr, y_tr = X.iloc[train_idx], y_log[train_idx]
    X_va, y_va = X.iloc[val_idx], y_log[val_idx]

    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)

    model = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=1500,
        valid_sets=[trn_data, val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

    oof_preds[val_idx] = model.predict(X_va, num_iteration=model.best_iteration)
    test_preds += (
        model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits
    )

# 4. Evaluation (RMSLE on original scale)
oof_preds_original = np.expm1(oof_preds)
oof_preds_original = np.clip(oof_preds_original, 0, None)

cv_rmsle = root_mean_squared_error(np.log1p(y_raw), np.log1p(oof_preds_original))
print(f"5-Fold CV RMSLE: {cv_rmsle:.5f}")

# 5. Generate Submission
final_test_preds = np.expm1(test_preds)
final_test_preds = np.clip(final_test_preds, 0, None)

os.makedirs("./working", exist_ok=True)
submission = pd.DataFrame(
    {
        "id": np.arange(len(final_test_preds)),
        "formation_energy_ev_natom": final_test_preds,
    }
)

submission_path = "./working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path} with shape {submission.shape}")
