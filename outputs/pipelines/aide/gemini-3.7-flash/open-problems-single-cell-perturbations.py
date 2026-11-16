import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

# Try importing RDKit for SMILES Morgan fingerprints
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


def compute_mrrmse(y_true, y_pred):
    """Compute Mean Rowwise Root Mean Squared Error."""
    return np.mean(np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1)))


def smiles_to_fingerprint(smiles_list, n_bits=1024):
    """Generate Morgan fingerprints from a list of SMILES strings."""
    fps = []
    for s in smiles_list:
        if HAS_RDKIT and isinstance(s, str):
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
                fps.append(np.array(fp, dtype=np.float32))
                continue
        fps.append(np.zeros(n_bits, dtype=np.float32))
    return np.vstack(fps)


def main():
    # 1. Load data
    train_path = "./input/train.parquet"
    test_path = "./input/test.parquet"
    sample_sub_path = "./input/sample_submission.csv"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    meta_cols = ["cell_type", "sm_name", "sm_lincs_id", "SMILES", "control"]
    gene_cols = [c for c in train_df.columns if c not in meta_cols]
    n_genes = len(gene_cols)
    print(f"Loaded train shape: {train_df.shape}, test shape: {test_df.shape}")
    print(f"Number of gene targets: {n_genes}")

    # 2. Extract targets
    Y = train_df[gene_cols].values.astype(np.float32)

    # 3. Feature engineering
    all_cell_types = sorted(
        list(set(train_df["cell_type"]).union(set(test_df["cell_type"])))
    )
    all_sm_names = sorted(list(set(train_df["sm_name"]).union(set(test_df["sm_name"]))))

    def build_features(df):
        # One-hot encoded cell type
        cell_dummies = pd.get_dummies(
            pd.Categorical(df["cell_type"], categories=all_cell_types), dtype=float
        ).values
        # One-hot encoded small molecule name
        sm_dummies = pd.get_dummies(
            pd.Categorical(df["sm_name"], categories=all_sm_names), dtype=float
        ).values
        # Molecular fingerprints
        smiles_fps = smiles_to_fingerprint(df["SMILES"].values, n_bits=512)
        return np.hstack([cell_dummies, sm_dummies, smiles_fps]).astype(np.float32)

    X = build_features(train_df)
    X_test = build_features(test_df)
    print(f"Feature dimension: {X.shape[1]}")

    # 4. Cross-Validation setup
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros_like(Y)
    test_preds = np.zeros((len(test_df), n_genes), dtype=np.float32)

    # 5. Model Training & Prediction using Multi-Output Ridge Regression
    alpha = 10.0
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, y_tr = X[train_idx], Y[train_idx]
        X_va, y_val = X[val_idx], Y[val_idx]

        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X_tr, y_tr)

        val_pred = model.predict(X_va)
        oof_preds[val_idx] = val_pred

        fold_mrrmse = compute_mrrmse(y_val, val_pred)
        print(f"Fold {fold + 1} MRRMSE: {fold_mrrmse:.4f}")

        test_preds += model.predict(X_test) / n_splits

    cv_mrrmse = compute_mrrmse(Y, oof_preds)
    print(f"\nOverall 5-Fold CV MRRMSE: {cv_mrrmse:.4f}")

    # 6. Save Submission
    os.makedirs("./working", exist_ok=True)
    if os.path.exists(sample_sub_path):
        sub = pd.read_csv(sample_sub_path)
        sub.loc[:, gene_cols] = test_preds
    else:
        sub = pd.DataFrame(test_preds, columns=gene_cols)
        sub.insert(0, "id", np.arange(len(sub)))

    sub.to_csv("./working/submission.csv", index=False)
    print("Saved ./working/submission.csv successfully.")


if __name__ == "__main__":
    main()
