import pandas as pd
import numpy as np
import os
import json
import warnings
import re
from collections import deque
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

# Create directories
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ============ LOAD DATA ============
train = pd.read_parquet("./input/train.parquet")
test = pd.read_parquet("./input/test.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Train columns: {train.columns.tolist()}")
print(f"Train head:\n{train.head()}")
print(f"\nTarget stats:\n{train['scalar_coupling_constant'].describe()}")

# Identify SMILES column (only if it actually contains SMILES)
smiles_col = None
for col in train.columns:
    if "smiles" in col.lower():
        smiles_col = col
        break
# Verify it's actually SMILES (not just molecule names)
if smiles_col is not None:
    sample = train[smiles_col].dropna().iloc[0]
    # SMILES should contain typical organic elements and bond notation
    if not any(c in sample for c in ['C', 'N', 'O', 'c', 'n', 'o', '[', '(', '=']):
        smiles_col = None
    elif '_' in sample and not any(c in sample for c in ['=', '(', '[']):
        smiles_col = None
print(f"SMILES column: {smiles_col}")

# Identify atom index columns (use exact column names)
atom1_col = None
atom2_col = None
for col in train.columns:
    col_lower = col.lower()
    if "atom_index_0" in col_lower or col_lower == "atom_index_0":
        atom1_col = col
    elif "atom_index_1" in col_lower or col_lower == "atom_index_1":
        atom2_col = col

# Fallback if exact names not found
if atom1_col is None or atom2_col is None:
    for col in train.columns:
        col_lower = col.lower()
        if "atom" in col_lower or "idx" in col_lower:
            if "0" in col or "first" in col_lower:
                if atom1_col is None:
                    atom1_col = col
            elif "1" in col or "second" in col_lower:
                if atom2_col is None:
                    atom2_col = col

print(f"Atom columns: atom1={atom1_col}, atom2={atom2_col}")

# Identify coupling class column
coupling_col = None
for col in train.columns:
    if col == "scalar_coupling_constant":
        continue
    if "type" in col.lower() or "j_class" in col.lower() or "J_class" in col:
        coupling_col = col
        break
if coupling_col is None:
    # Fallback: look for a column with string values containing J patterns
    for col in train.columns:
        if col == "scalar_coupling_constant":
            continue
        if train[col].dtype == "object":
            sample = train[col].dropna().iloc[0]
            if isinstance(sample, str) and "J" in sample:
                coupling_col = col
                break
print(f"Coupling class column: {coupling_col}")

# ============ MOLECULE-LEVEL SPLIT ============
mol_col = "molecule_name"
unique_mols = train[mol_col].unique()
np.random.seed(42)
np.random.shuffle(unique_mols)

n_val = max(1, int(len(unique_mols) * 0.1))
val_mols = set(unique_mols[:n_val])
train_mols = set(unique_mols[n_val:])

train_mask = train[mol_col].isin(train_mols)
val_mask = train[mol_col].isin(val_mols)

train_split = train[train_mask].reset_index(drop=True)
val_split = train[val_mask].reset_index(drop=True)

print(
    f"Train split: {len(train_split)} rows, {train_split[mol_col].nunique()} molecules"
)
print(f"Val split: {len(val_split)} rows, {val_split[mol_col].nunique()} molecules")
print(f"Test: {len(test)} rows, {test[mol_col].nunique()} molecules")

# ============ FEATURE ENGINEERING ============
# The 'type' column encodes coupling type like "2JHC", "3JHH", "1JCH"
# Format: {bond_order}J{atom1_type}{atom2_type}
# This gives us the key physical features for J-coupling prediction.

ELEMENT_ATOMIC_NUMS = {
    'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16,
    'Cl': 17, 'Br': 35, 'I': 53, 'B': 5, 'Si': 14,
}

# Map element symbol to a numeric code for feature encoding
ELEMENT_CODES = {
    'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5, 'P': 6, 'S': 7,
    'Cl': 8, 'Br': 9, 'I': 10, 'B': 11, 'Si': 12,
}


def parse_type_string(type_str):
    """Parse a coupling type string like '2JHC', '3JHH', '1JCH'.
    Returns: (bond_order, atom1_elem, atom2_elem)
    """
    if not isinstance(type_str, str):
        return (1, 'C', 'C')
    # Match pattern: digit(s) + J + element(s)
    m = re.match(r'^(\d)J(.+)$', type_str)
    if m:
        bond_order = int(m.group(1))
        elems_str = m.group(2)
        # Parse two element symbols from the string
        # Common: single chars (H, C, N, O, F, P, S) or two-char (Cl, Br)
        if len(elems_str) == 2:
            elem1, elem2 = elems_str[0], elems_str[1]
        elif len(elems_str) == 3:
            # Could be Cl + single, or single + Cl
            if elems_str[:2] in ELEMENT_CODES:
                elem1, elem2 = elems_str[:2], elems_str[2]
            else:
                elem1, elem2 = elems_str[0], elems_str[1:3]
        elif len(elems_str) == 4:
            # Two two-char elements
            if elems_str[:2] in ELEMENT_CODES and elems_str[2:] in ELEMENT_CODES:
                elem1, elem2 = elems_str[:2], elems_str[2:]
            else:
                elem1, elem2 = elems_str[0], elems_str[1]
        else:
            elem1, elem2 = 'C', 'C'
        return (bond_order, elem1, elem2)
    return (1, 'C', 'C')


def engineer_features(df, coupling_col):
    """Engineer features from the coupling type string."""
    print(f"Engineering features for {len(df)} rows...")

    types = df[coupling_col].values
    parsed = [parse_type_string(t) for t in types]

    bond_orders = np.array([p[0] for p in parsed], dtype=np.int32)
    elem1_list = [p[1] for p in parsed]
    elem2_list = [p[2] for p in parsed]

    # Encode elements as numeric codes
    elem1_codes = np.array([ELEMENT_CODES.get(e, 0) for e in elem1_list], dtype=np.int32)
    elem2_codes = np.array([ELEMENT_CODES.get(e, 0) for e in elem2_list], dtype=np.int32)

    # Create sorted pair for symmetric features
    pair_min = np.minimum(elem1_codes, elem2_codes)
    pair_max = np.maximum(elem1_codes, elem2_codes)

    features = pd.DataFrame({
        'bond_order': bond_orders,
        'elem1_code': elem1_codes,
        'elem2_code': elem2_codes,
        'elem1_atomic_num': np.array([ELEMENT_ATOMIC_NUMS.get(e, 6) for e in elem1_list], dtype=np.float32),
        'elem2_atomic_num': np.array([ELEMENT_ATOMIC_NUMS.get(e, 6) for e in elem2_list], dtype=np.float32),
        'pair_min_code': pair_min,
        'pair_max_code': pair_max,
        'pair_type_code': pair_min * 20 + pair_max,  # unique pair identifier
        'elem_diff': np.abs(elem1_codes - elem2_codes),
        'elem_sum': elem1_codes + elem2_codes,
        'elem_product': elem1_codes * elem2_codes,
        'is_HH': ((elem1_codes == 1) & (elem2_codes == 1)).astype(np.int32),
        'is_HC': ((elem1_codes == 1) & (elem2_codes == 2)).astype(np.int32) | ((elem1_codes == 2) & (elem2_codes == 1)).astype(np.int32),
        'is_HN': ((elem1_codes == 1) & (elem2_codes == 3)).astype(np.int32) | ((elem1_codes == 3) & (elem2_codes == 1)).astype(np.int32),
        'is_HO': ((elem1_codes == 1) & (elem2_codes == 4)).astype(np.int32) | ((elem1_codes == 4) & (elem2_codes == 1)).astype(np.int32),
        'is_HF': ((elem1_codes == 1) & (elem2_codes == 5)).astype(np.int32) | ((elem1_codes == 5) & (elem2_codes == 1)).astype(np.int32),
        'is_CC': ((elem1_codes == 2) & (elem2_codes == 2)).astype(np.int32),
        'is_CN': ((elem1_codes == 2) & (elem2_codes == 3)).astype(np.int32) | ((elem1_codes == 3) & (elem2_codes == 2)).astype(np.int32),
        'is_CO': ((elem1_codes == 2) & (elem2_codes == 4)).astype(np.int32) | ((elem1_codes == 4) & (elem2_codes == 2)).astype(np.int32),
        'is_CF': ((elem1_codes == 2) & (elem2_codes == 5)).astype(np.int32) | ((elem1_codes == 5) & (elem2_codes == 2)).astype(np.int32),
        'is_NN': ((elem1_codes == 3) & (elem2_codes == 3)).astype(np.int32),
        'is_NO': ((elem1_codes == 3) & (elem2_codes == 4)).astype(np.int32) | ((elem1_codes == 4) & (elem2_codes == 3)).astype(np.int32),
        'is_NF': ((elem1_codes == 3) & (elem2_codes == 5)).astype(np.int32) | ((elem1_codes == 5) & (elem2_codes == 3)).astype(np.int32),
        'is_OO': ((elem1_codes == 4) & (elem2_codes == 4)).astype(np.int32),
        'is_OF': ((elem1_codes == 4) & (elem2_codes == 5)).astype(np.int32) | ((elem1_codes == 5) & (elem2_codes == 4)).astype(np.int32),
        'is_FF': ((elem1_codes == 5) & (elem2_codes == 5)).astype(np.int32),
        'has_H': ((elem1_codes == 1) | (elem2_codes == 1)).astype(np.int32),
        'has_C': ((elem1_codes == 2) | (elem2_codes == 2)).astype(np.int32),
        'has_N': ((elem1_codes == 3) | (elem2_codes == 3)).astype(np.int32),
        'has_O': ((elem1_codes == 4) | (elem2_codes == 4)).astype(np.int32),
        'has_F': ((elem1_codes == 5) | (elem2_codes == 5)).astype(np.int32),
        'is_1J': (bond_orders == 1).astype(np.int32),
        'is_2J': (bond_orders == 2).astype(np.int32),
        'is_3J': (bond_orders == 3).astype(np.int32),
        'is_4J': (bond_orders == 4).astype(np.int32),
        'bond_order_sq': bond_orders ** 2,
        'bond_order_inv': 1.0 / np.maximum(bond_orders, 1),
    })

    # Add atom index features if available
    if atom1_col is not None and atom1_col in df.columns:
        features['atom_index_0'] = df[atom1_col].values.astype(np.float32)
    if atom2_col is not None and atom2_col in df.columns:
        features['atom_index_1'] = df[atom2_col].values.astype(np.float32)

    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(0, inplace=True)
    print(f"  Feature engineering complete: {features.shape[1]} features")
    return features


# Engineer features
print("\n--- Engineering train features ---")
train_features = engineer_features(train_split, coupling_col)

print("\n--- Engineering val features ---")
val_features = engineer_features(val_split, coupling_col)

print("\n--- Engineering test features ---")
test_features = engineer_features(test, coupling_col)

# ============ PREPARE DATA FOR MODELING ============
# Determine coupling class from bond order in type string
def parse_coupling_type(val):
    """Parse coupling type string like '2JHC', '3JHH' to numeric class (1-4)."""
    if isinstance(val, str):
        m = re.match(r'^(\d)J', val)
        if m:
            return int(m.group(1))
        for ch in val:
            if ch.isdigit():
                return int(ch)
    elif isinstance(val, (int, float, np.integer, np.floating)):
        return int(val)
    return 1

if coupling_col is not None:
    train_classes = train_split[coupling_col].apply(parse_coupling_type).values
    val_classes = val_split[coupling_col].apply(parse_coupling_type).values
    test_classes = test[coupling_col].apply(parse_coupling_type).values
else:
    print("No explicit coupling class column, using bond_order from features")
    train_classes = train_features["bond_order"].values.astype(int)
    val_classes = val_features["bond_order"].values.astype(int)
    test_classes = test_features["bond_order"].values.astype(int)

y_train = train_split["scalar_coupling_constant"].values
y_val = val_split["scalar_coupling_constant"].values

feature_cols = train_features.columns.tolist()
X_train = train_features[feature_cols].values.astype(np.float32)
X_val = val_features[feature_cols].values.astype(np.float32)
X_test = test_features[feature_cols].values.astype(np.float32)

print(f"\nNumber of features: {len(feature_cols)}")
print(f"\nCoupling class distribution (train):")
for c in sorted(set(train_classes)):
    print(f"  Class {c}: {(train_classes == c).sum()} samples")

# ============ MULTI-SEED CLASS-CONDITIONAL LIGHTGBM ENSEMBLE ============

# Class-specific parameters
lgb_params_base = {
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "max_depth": -1,
    "learning_rate": 0.03,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 20,
    "min_sum_hessian_in_leaf": 1e-3,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "num_threads": -1,
}

# Class 1 (1J) specific: quantile regression (alpha=0.5 = MAE), more leaves, bynode fraction
lgb_params_class1 = lgb_params_base.copy()
lgb_params_class1["objective"] = "quantile"
lgb_params_class1["alpha"] = 0.5
lgb_params_class1["num_leaves"] = 255
lgb_params_class1["feature_fraction_bynode"] = 0.8

# Class 2 and 3: keep Huber loss
lgb_params_class23 = lgb_params_base.copy()
lgb_params_class23["objective"] = "huber"
lgb_params_class23["alpha"] = 1.0

# Seeds: 5 for Class 1 (high variance), 3 for others
SEEDS_CLASS1 = [42, 123, 456, 789, 1011]
SEEDS_OTHER = [42, 123, 456]

unique_classes = sorted(set(train_classes))

print(f"\nUnique coupling classes: {unique_classes}")
print(f"Seeds for Class 1: {len(SEEDS_CLASS1)}, Seeds for others: {len(SEEDS_OTHER)}")
print(f"Total models to train: {len(SEEDS_CLASS1) + len(unique_classes) * len(SEEDS_OTHER) - len(SEEDS_OTHER)}")

all_models = {cls: [] for cls in unique_classes}

for cls in unique_classes:
    train_mask_cls = train_classes == cls
    val_mask_cls = val_classes == cls

    n_train_cls = train_mask_cls.sum()
    n_val_cls = val_mask_cls.sum()

    if n_train_cls < 10:
        print(f"  Class {cls}: skipped (only {n_train_cls} samples)")
        continue

    X_train_cls = X_train[train_mask_cls]
    y_train_cls = y_train[train_mask_cls]
    X_val_cls = X_val[val_mask_cls]
    y_val_cls = y_val[val_mask_cls]

    # Select params and seeds based on class
    if cls == 1:
        params_template = lgb_params_class1
        seeds = SEEDS_CLASS1
    else:
        params_template = lgb_params_class23
        seeds = SEEDS_OTHER

    print(f"\nClass {cls}: train={n_train_cls}, val={n_val_cls}, objective={params_template['objective']}, seeds={len(seeds)}")

    for seed in seeds:
        params = params_template.copy()
        params["seed"] = seed
        params["bagging_seed"] = seed
        params["feature_fraction_seed"] = seed

        dtrain = lgb.Dataset(X_train_cls, label=y_train_cls, feature_name=feature_cols)
        dval = lgb.Dataset(
            X_val_cls, label=y_val_cls, feature_name=feature_cols, reference=dtrain
        )

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=3000,
            valid_sets=[dval],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )

        all_models[cls].append(model)
        print(f"  Seed {seed}: best_iter={model.best_iteration}")

# ============ VALIDATION PREDICTIONS ============
val_preds = np.zeros(len(y_val))

for cls in unique_classes:
    mask = val_classes == cls
    if mask.sum() == 0:
        continue
    if cls in all_models and len(all_models[cls]) > 0:
        X_cls = X_val[mask]
        cls_preds = np.zeros(len(X_cls))
        for model in all_models[cls]:
            cls_preds += model.predict(X_cls, num_iteration=model.best_iteration)
        cls_preds /= len(all_models[cls])
        val_preds[mask] = cls_preds
    else:
        # Fallback to nearest class
        nearest = min(
            unique_classes,
            key=lambda c: abs(c - cls) if c in all_models and len(all_models[c]) > 0 else 999,
        )
        if nearest in all_models and len(all_models[nearest]) > 0:
            X_cls = X_val[mask]
            cls_preds = np.zeros(len(X_cls))
            for model in all_models[nearest]:
                cls_preds += model.predict(X_cls, num_iteration=model.best_iteration)
            cls_preds /= len(all_models[nearest])
            val_preds[mask] = cls_preds
        else:
            val_preds[mask] = np.median(y_train)

val_preds = np.clip(val_preds, -30, 30)
val_mae = mean_absolute_error(y_val, val_preds)
print(f"\n{'='*50}")
print(f"OVERALL VALIDATION MAE: {val_mae:.4f}")
print(f"{'='*50}")

for cls in unique_classes:
    mask = val_classes == cls
    if mask.sum() > 0:
        cls_mae = mean_absolute_error(y_val[mask], val_preds[mask])
        print(f"  Class {cls} MAE: {cls_mae:.4f} (n={mask.sum()})")

# ============ TEST PREDICTIONS ============
print(f"\nGenerating test predictions...")
test_preds = np.zeros(len(X_test))

for cls in unique_classes:
    mask = test_classes == cls
    if mask.sum() == 0:
        continue
    if cls in all_models and len(all_models[cls]) > 0:
        X_cls = X_test[mask]
        cls_preds = np.zeros(len(X_cls))
        for model in all_models[cls]:
            cls_preds += model.predict(X_cls, num_iteration=model.best_iteration)
        cls_preds /= len(all_models[cls])
        test_preds[mask] = cls_preds
    else:
        nearest = min(
            unique_classes,
            key=lambda c: abs(c - cls) if c in all_models and len(all_models[c]) > 0 else 999,
        )
        if nearest in all_models and len(all_models[nearest]) > 0:
            X_cls = X_test[mask]
            cls_preds = np.zeros(len(X_cls))
            for model in all_models[nearest]:
                cls_preds += model.predict(X_cls, num_iteration=model.best_iteration)
            cls_preds /= len(all_models[nearest])
            test_preds[mask] = cls_preds
        else:
            test_preds[mask] = np.median(y_train)

test_preds = np.clip(test_preds, -30, 30)

# ============ SAVE SUBMISSION ============
submission = pd.DataFrame(
    {"id": np.arange(len(test_preds)), "scalar_coupling_constant": test_preds}
)
submission.to_csv("./submission/submission.csv", index=False)
print(f"\nSubmission saved: {submission.shape}")
print(
    f"Prediction stats: mean={test_preds.mean():.4f}, std={test_preds.std():.4f}, min={test_preds.min():.4f}, max={test_preds.max():.4f}"
)

# Save model metadata
model_info = {
    "architecture": "multi_seed_class_conditional_lightgbm_quantile_huber",
    "n_seeds_class1": len(SEEDS_CLASS1),
    "n_seeds_other": len(SEEDS_OTHER),
    "classes_trained": {str(k): len(v) for k, v in all_models.items()},
    "val_mae": val_mae,
    "n_features": len(feature_cols),
    "class1_objective": "quantile(alpha=0.5)",
    "class23_objective": "huber(alpha=1.0)",
}
with open("./working/model_info.json", "w") as f:
    json.dump(model_info, f, indent=2)

print(f"\nModel: Multi-seed class-conditional LightGBM (Class1: Quantile, Class2/3: Huber)")
print(f"Models: {sum(len(v) for v in all_models.values())} total")
print(f"Final Validation Score: {val_mae:.4f}")
