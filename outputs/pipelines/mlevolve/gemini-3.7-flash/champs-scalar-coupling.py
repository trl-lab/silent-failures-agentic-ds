import gc
import glob
import json
import os
from typing import Any, Dict

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

# =============================================================================
# 1. Constants & Physical-Chemical Properties
# =============================================================================
ATOM_PROPERTIES = {
    "H": {
        "atomic_num": 1,
        "mass": 1.008,
        "electronegativity": 2.20,
        "covalent_radius": 0.31,
    },
    "C": {
        "atomic_num": 6,
        "mass": 12.011,
        "electronegativity": 2.55,
        "covalent_radius": 0.76,
    },
    "N": {
        "atomic_num": 7,
        "mass": 14.007,
        "electronegativity": 3.04,
        "covalent_radius": 0.71,
    },
    "O": {
        "atomic_num": 8,
        "mass": 15.999,
        "electronegativity": 3.44,
        "covalent_radius": 0.66,
    },
    "F": {
        "atomic_num": 9,
        "mass": 18.998,
        "electronegativity": 3.98,
        "covalent_radius": 0.57,
    },
}

DEFAULT_ATOM_PROP = {
    "atomic_num": 0,
    "mass": 0.0,
    "electronegativity": 0.0,
    "covalent_radius": 0.0,
}


# =============================================================================
# 2. Feature Engineering & Structure Loading Functions
# =============================================================================
def get_structures(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Find and load structures file if available using exhaustive filesystem scanning."""
    candidates = []
    for root, _, files in os.walk("./input"):
        for f in files:
            if f.endswith((".csv", ".parquet")):
                candidates.append(os.path.join(root, f))

    struct_candidates = [p for p in candidates if "structure" in os.path.basename(p).lower()]
    all_candidates = struct_candidates + [p for p in candidates if p not in struct_candidates]

    for path in all_candidates:
        try:
            if path.endswith(".parquet"):
                df_candidate = pd.read_parquet(path)
            else:
                df_candidate = pd.read_csv(path)

            cols = set(df_candidate.columns)
            if {"molecule_name", "atom_index", "atom", "x", "y", "z"}.issubset(cols):
                print(f"Loaded structures from {path} with shape {df_candidate.shape}...")
                return df_candidate
            elif {"molecule_name", "x", "y", "z"}.issubset(cols):
                print(f"Loaded structures from {path} with shape {df_candidate.shape}...")
                if "atom_index" not in df_candidate.columns:
                    df_candidate["atom_index"] = df_candidate.groupby("molecule_name").cumcount()
                if "atom" not in df_candidate.columns:
                    df_candidate["atom"] = "C"
                return df_candidate
        except Exception as e:
            print(f"Skipping {path}: {e}")

    # Reconstruct from train_df and test_df if coordinate columns exist
    struct_parts = []
    for df in [train_df, test_df]:
        if df is None:
            continue
        cols = df.columns
        if {"molecule_name", "atom_index_0", "atom_0", "x_0", "y_0", "z_0"}.issubset(cols):
            s0 = df[["molecule_name", "atom_index_0", "atom_0", "x_0", "y_0", "z_0"]].copy()
            s0.columns = ["molecule_name", "atom_index", "atom", "x", "y", "z"]
            struct_parts.append(s0)
        if {"molecule_name", "atom_index_1", "atom_1", "x_1", "y_1", "z_1"}.issubset(cols):
            s1 = df[["molecule_name", "atom_index_1", "atom_1", "x_1", "y_1", "z_1"]].copy()
            s1.columns = ["molecule_name", "atom_index", "atom", "x", "y", "z"]
            struct_parts.append(s1)

    if struct_parts:
        all_struct = pd.concat(struct_parts, ignore_index=True)
        structures = all_struct.drop_duplicates(subset=["molecule_name", "atom_index"]).reset_index(drop=True)
        print(f"Reconstructed structures with {len(structures)} atoms across {structures['molecule_name'].nunique()} molecules.")
        return structures

    print("No external structures file found. Continuing with graph-based molecular features.")
    return pd.DataFrame(columns=["molecule_name", "atom_index", "atom", "x", "y", "z"])


def compute_graph_topology_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract comprehensive molecular graph topology, connectivity, and hybridization proxies."""
    # Decompose coupling type metadata
    if "type" in df.columns:
        df["n_bonds"] = df["type"].str[0].astype(np.int32)
        df["inferred_atom_0"] = df["type"].str[2]
        df["inferred_atom_1"] = df["type"].str[3]
        if "atom_0" not in df.columns or df["atom_0"].isna().all():
            df["atom_0"] = df["inferred_atom_0"]
        if "atom_1" not in df.columns or df["atom_1"].isna().all():
            df["atom_1"] = df["inferred_atom_1"]

    # Index relative distance and arithmetic features
    if "atom_index_0" in df.columns and "atom_index_1" in df.columns:
        df["atom_index_diff"] = (df["atom_index_1"] - df["atom_index_0"]).abs().astype(np.float32)
        df["atom_index_min"] = np.minimum(df["atom_index_0"], df["atom_index_1"]).astype(np.float32)
        df["atom_index_max"] = np.maximum(df["atom_index_0"], df["atom_index_1"]).astype(np.float32)
        df["atom_index_sum"] = (df["atom_index_0"] + df["atom_index_1"]).astype(np.float32)
        df["atom_index_ratio"] = ((df["atom_index_min"] + 1.0) / (df["atom_index_max"] + 1.0)).astype(np.float32)

    if "molecule_name" in df.columns:
        # Molecule-level span and total coupling counts
        mol_grp = df.groupby("molecule_name")
        df["mol_coupling_count"] = mol_grp["id"].transform("count").astype(np.float32)

        mol_max_0 = mol_grp["atom_index_0"].transform("max")
        mol_max_1 = mol_grp["atom_index_1"].transform("max")
        df["mol_max_atom_idx"] = np.maximum(mol_max_0, mol_max_1).astype(np.float32)

        mol_min_0 = mol_grp["atom_index_0"].transform("min")
        mol_min_1 = mol_grp["atom_index_1"].transform("min")
        df["mol_min_atom_idx"] = np.minimum(mol_min_0, mol_min_1).astype(np.float32)
        df["mol_atom_span"] = (df["mol_max_atom_idx"] - df["mol_min_atom_idx"] + 1.0).astype(np.float32)

        # Per-atom connectivity degrees
        df["atom_0_deg"] = df.groupby(["molecule_name", "atom_index_0"])["id"].transform("count").astype(np.float32)
        df["atom_1_deg"] = df.groupby(["molecule_name", "atom_index_1"])["id"].transform("count").astype(np.float32)

        df["atom_deg_diff"] = np.abs(df["atom_0_deg"] - df["atom_1_deg"]).astype(np.float32)
        df["atom_deg_sum"] = (df["atom_0_deg"] + df["atom_1_deg"]).astype(np.float32)
        df["atom_deg_prod"] = (df["atom_0_deg"] * df["atom_1_deg"]).astype(np.float32)
        df["atom_deg_ratio"] = (df["atom_0_deg"] / (df["atom_1_deg"] + 1e-5)).astype(np.float32)

        # Coupling path length distribution (1J, 2J, 3J)
        for bond_n in ["1J", "2J", "3J"]:
            is_bond = df["type"].str.startswith(bond_n).astype(np.float32)
            df[f"mol_{bond_n}_count"] = df.assign(_bond=is_bond).groupby("molecule_name")["_bond"].transform("sum").astype(np.float32)
            df[f"mol_{bond_n}_ratio"] = (df[f"mol_{bond_n}_count"] / (df["mol_coupling_count"] + 1e-5)).astype(np.float32)

            df[f"atom_0_deg_{bond_n}"] = df.assign(_bond=is_bond).groupby(["molecule_name", "atom_index_0"])["_bond"].transform("sum").astype(np.float32)
            df[f"atom_1_deg_{bond_n}"] = df.assign(_bond=is_bond).groupby(["molecule_name", "atom_index_1"])["_bond"].transform("sum").astype(np.float32)
            df[f"atom_{bond_n}_deg_sum"] = (df[f"atom_0_deg_{bond_n}"] + df[f"atom_1_deg_{bond_n}"]).astype(np.float32)

        # Per-subtype atom_1 degrees to resolve local carbon/nitrogen hybridization environments
        for ctype in ["1JHC", "2JHC", "3JHC", "1JHN", "2JHN", "3JHN", "2JHH", "3JHH"]:
            is_type = (df["type"] == ctype).astype(np.float32)
            df[f"atom_1_deg_{ctype}"] = df.assign(_ctype=is_type).groupby(["molecule_name", "atom_index_1"])["_ctype"].transform("sum").astype(np.float32)

        # Shared coupling partner network (two-hop common neighbors)
        edges = pd.concat([
            df[["molecule_name", "atom_index_0", "atom_index_1"]].rename(columns={"atom_index_0": "u", "atom_index_1": "v"}),
            df[["molecule_name", "atom_index_1", "atom_index_0"]].rename(columns={"atom_index_1": "u", "atom_index_0": "v"}),
        ]).drop_duplicates()

        two_hop = edges.merge(edges, left_on=["molecule_name", "v"], right_on=["molecule_name", "u"], suffixes=("_1", "_2"))
        two_hop = two_hop[two_hop["u_1"] != two_hop["v_2"]]
        common_neighbors = (
            two_hop.groupby(["molecule_name", "u_1", "v_2"])
            .size()
            .reset_index(name="shared_coupling_neighbors")
            .rename(columns={"u_1": "atom_index_0", "v_2": "atom_index_1"})
        )
        common_neighbors["shared_coupling_neighbors"] = common_neighbors["shared_coupling_neighbors"].astype(np.float32)
        df = df.merge(common_neighbors, on=["molecule_name", "atom_index_0", "atom_index_1"], how="left")
        df["shared_coupling_neighbors"] = df["shared_coupling_neighbors"].fillna(0.0).astype(np.float32)
        df["jaccard_coupling_neighbors"] = (
            df["shared_coupling_neighbors"] / (df["atom_0_deg"] + df["atom_1_deg"] - df["shared_coupling_neighbors"] + 1e-5)
        ).astype(np.float32)

    return df


def merge_structures(df: pd.DataFrame, structures: pd.DataFrame) -> pd.DataFrame:
    """Merge 3D coordinates and atom types for atom_0 and atom_1 if not already present."""
    if structures.empty:
        return df

    struct_cols = ["molecule_name", "atom_index", "atom", "x", "y", "z"]
    if not set(struct_cols).issubset(structures.columns):
        return df

    s = structures[struct_cols]

    if "x_0" not in df.columns or df["x_0"].isna().all():
        cols_to_drop = [c for c in ["atom_0", "x_0", "y_0", "z_0"] if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        df = df.merge(
            s.rename(
                columns={
                    "atom_index": "atom_index_0",
                    "atom": "atom_0",
                    "x": "x_0",
                    "y": "y_0",
                    "z": "z_0",
                }
            ),
            on=["molecule_name", "atom_index_0"],
            how="left",
        )

    if "x_1" not in df.columns or df["x_1"].isna().all():
        cols_to_drop = [c for c in ["atom_1", "x_1", "y_1", "z_1"] if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        df = df.merge(
            s.rename(
                columns={
                    "atom_index": "atom_index_1",
                    "atom": "atom_1",
                    "x": "x_1",
                    "y": "y_1",
                    "z": "z_1",
                }
            ),
            on=["molecule_name", "atom_index_1"],
            how="left",
        )
    return df


def compute_structure_features(structures: pd.DataFrame) -> pd.DataFrame:
    """Compute molecular-level geometric summary features from 3D structures."""
    if (
        structures.empty
        or "molecule_name" not in structures.columns
        or "x" not in structures.columns
    ):
        return pd.DataFrame(columns=["molecule_name"])

    counts = (
        pd.crosstab(structures["molecule_name"], structures["atom"])
        .add_prefix("n_atom_")
        .reset_index()
    )
    for elem in ["H", "C", "N", "O", "F"]:
        col = f"n_atom_{elem}"
        if col not in counts.columns:
            counts[col] = 0
        counts[col] = counts[col].astype(np.float32)

    agg_df = (
        structures.groupby("molecule_name")
        .agg(
            n_atoms=("atom_index", "count"),
            mol_center_x=("x", "mean"),
            mol_center_y=("y", "mean"),
            mol_center_z=("z", "mean"),
            mol_min_x=("x", "min"),
            mol_max_x=("x", "max"),
            mol_min_y=("y", "min"),
            mol_max_y=("y", "max"),
            mol_min_z=("z", "min"),
            mol_max_z=("z", "max"),
        )
        .reset_index()
    )

    agg_df["mol_size_x"] = (agg_df["mol_max_x"] - agg_df["mol_min_x"]).astype(np.float32)
    agg_df["mol_size_y"] = (agg_df["mol_max_y"] - agg_df["mol_min_y"]).astype(np.float32)
    agg_df["mol_size_z"] = (agg_df["mol_max_z"] - agg_df["mol_min_z"]).astype(np.float32)
    agg_df["mol_bounding_radius"] = np.sqrt(
        agg_df["mol_size_x"] ** 2 + agg_df["mol_size_y"] ** 2 + agg_df["mol_size_z"] ** 2
    ).astype(np.float32)

    summary = agg_df.merge(counts, on="molecule_name", how="left")
    return summary


def compute_pairwise_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 3D Euclidean distances, differentials, inverse decays, and angular coordinates."""
    if {"x_0", "y_0", "z_0", "x_1", "y_1", "z_1"}.issubset(df.columns):
        dx = df["x_0"] - df["x_1"]
        dy = df["y_0"] - df["y_1"]
        dz = df["z_0"] - df["z_1"]

        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        dist_xy = np.sqrt(dx**2 + dy**2)

        df["dist"] = dist.astype(np.float32)
        df["dist_xy"] = dist_xy.astype(np.float32)
        df["dx"] = dx.astype(np.float32)
        df["dy"] = dy.astype(np.float32)
        df["dz"] = dz.astype(np.float32)
        df["abs_dx"] = np.abs(dx).astype(np.float32)
        df["abs_dy"] = np.abs(dy).astype(np.float32)
        df["abs_dz"] = np.abs(dz).astype(np.float32)

        # Non-linear quantum interaction decay terms
        eps = 1e-6
        df["inv_dist"] = (1.0 / (dist + eps)).astype(np.float32)
        df["inv_dist2"] = (1.0 / (dist**2 + eps)).astype(np.float32)
        df["inv_dist3"] = (1.0 / (dist**3 + eps)).astype(np.float32)

        # Angular coordinate projections
        df["polar_theta"] = np.arccos(np.clip(dz / (dist + eps), -1.0, 1.0)).astype(
            np.float32
        )
        df["polar_phi"] = np.arctan2(dy, dx).astype(np.float32)
    return df


def map_atom_properties(df: pd.DataFrame) -> pd.DataFrame:
    """Map atomic masses, electronegativities, and covalent radii."""
    for atom_idx in ["0", "1"]:
        atom_col = f"atom_{atom_idx}"
        if atom_col in df.columns:
            for prop in ["atomic_num", "mass", "electronegativity", "covalent_radius"]:
                prop_map = {k: v[prop] for k, v in ATOM_PROPERTIES.items()}
                df[f"{atom_col}_{prop}"] = (
                    df[atom_col]
                    .map(prop_map)
                    .fillna(DEFAULT_ATOM_PROP[prop])
                    .astype(np.float32)
                )

    if (
        "atom_0_electronegativity" in df.columns
        and "atom_1_electronegativity" in df.columns
    ):
        df["en_diff"] = np.abs(
            df["atom_0_electronegativity"] - df["atom_1_electronegativity"]
        ).astype(np.float32)
        df["en_product"] = (
            df["atom_0_electronegativity"] * df["atom_1_electronegativity"]
        ).astype(np.float32)

    if (
        "atom_0_covalent_radius" in df.columns
        and "atom_1_covalent_radius" in df.columns
    ):
        df["cov_radius_sum"] = (
            df["atom_0_covalent_radius"] + df["atom_1_covalent_radius"]
        ).astype(np.float32)
        if "dist" in df.columns:
            df["dist_cov_ratio"] = (df["dist"] / (df["cov_radius_sum"] + 1e-6)).astype(
                np.float32
            )

    if "atom_0_mass" in df.columns and "atom_1_mass" in df.columns:
        df["mass_sum"] = (df["atom_0_mass"] + df["atom_1_mass"]).astype(np.float32)
        df["mass_diff"] = np.abs(df["atom_0_mass"] - df["atom_1_mass"]).astype(
            np.float32
        )

    return df


def compute_molecule_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute molecular geometry statistics and relative center offsets."""
    if "molecule_name" in df.columns and {"x_0", "y_0", "z_0"}.issubset(df.columns) and df["x_0"].notna().any():
        mol_counts = (
            df.groupby("molecule_name")["molecule_name"]
            .transform("count")
            .astype(np.float32)
        )
        df["mol_coupling_count"] = mol_counts

        mol_mean_x = (
            df.groupby("molecule_name")["x_0"].transform("mean").astype(np.float32)
        )
        mol_mean_y = (
            df.groupby("molecule_name")["y_0"].transform("mean").astype(np.float32)
        )
        mol_mean_z = (
            df.groupby("molecule_name")["z_0"].transform("mean").astype(np.float32)
        )

        if "mol_center_x" in df.columns:
            df["dist_to_center_0"] = np.sqrt(
                (df["x_0"] - df["mol_center_x"]) ** 2
                + (df["y_0"] - df["mol_center_y"]) ** 2
                + (df["z_0"] - df["mol_center_z"]) ** 2
            ).astype(np.float32)
            if "x_1" in df.columns:
                df["dist_to_center_1"] = np.sqrt(
                    (df["x_1"] - df["mol_center_x"]) ** 2
                    + (df["y_1"] - df["mol_center_y"]) ** 2
                    + (df["z_1"] - df["mol_center_z"]) ** 2
                ).astype(np.float32)
        else:
            df["dist_to_center_0"] = np.sqrt(
                (df["x_0"] - mol_mean_x) ** 2
                + (df["y_0"] - mol_mean_y) ** 2
                + (df["z_0"] - mol_mean_z) ** 2
            ).astype(np.float32)

            if "x_1" in df.columns:
                df["dist_to_center_1"] = np.sqrt(
                    (df["x_1"] - mol_mean_x) ** 2
                    + (df["y_1"] - mol_mean_y) ** 2
                    + (df["z_1"] - mol_mean_z) ** 2
                ).astype(np.float32)

        if "dist" in df.columns:
            df["mol_dist_mean"] = (
                df.groupby("molecule_name")["dist"].transform("mean").astype(np.float32)
            )
            df["mol_dist_min"] = (
                df.groupby("molecule_name")["dist"].transform("min").astype(np.float32)
            )
            df["mol_dist_max"] = (
                df.groupby("molecule_name")["dist"].transform("max").astype(np.float32)
            )
            df["mol_dist_std"] = (
                df.groupby("molecule_name")["dist"]
                .transform("std")
                .fillna(0.0)
                .astype(np.float32)
            )
            df["dist_to_mol_mean_ratio"] = (
                df["dist"] / (df["mol_dist_mean"] + 1e-6)
            ).astype(np.float32)
            df["dist_to_mol_min_diff"] = (df["dist"] - df["mol_dist_min"]).astype(np.float32)
            df["dist_to_mol_max_diff"] = (df["mol_dist_max"] - df["dist"]).astype(np.float32)
            if "dist_to_center_0" in df.columns and "dist_to_center_1" in df.columns:
                df["dist_center_diff"] = np.abs(
                    df["dist_to_center_0"] - df["dist_to_center_1"]
                ).astype(np.float32)
                df["dist_center_sum"] = (
                    df["dist_to_center_0"] + df["dist_to_center_1"]
                ).astype(np.float32)
            if "atom_index_0" in df.columns:
                df["atom_0_dist_mean"] = (
                    df.groupby(["molecule_name", "atom_index_0"])["dist"]
                    .transform("mean")
                    .astype(np.float32)
                )
                df["atom_0_dist_min"] = (
                    df.groupby(["molecule_name", "atom_index_0"])["dist"]
                    .transform("min")
                    .astype(np.float32)
                )
                df["atom_0_dist_max"] = (
                    df.groupby(["molecule_name", "atom_index_0"])["dist"]
                    .transform("max")
                    .astype(np.float32)
                )
            if "atom_index_1" in df.columns:
                df["atom_1_dist_mean"] = (
                    df.groupby(["molecule_name", "atom_index_1"])["dist"]
                    .transform("mean")
                    .astype(np.float32)
                )
                df["atom_1_dist_min"] = (
                    df.groupby(["molecule_name", "atom_index_1"])["dist"]
                    .transform("min")
                    .astype(np.float32)
                )
                df["atom_1_dist_max"] = (
                    df.groupby(["molecule_name", "atom_index_1"])["dist"]
                    .transform("max")
                    .astype(np.float32)
                )

    return df


# =============================================================================
# 3. Model Configuration Factory
# =============================================================================
def get_lgb_model_params(
    num_leaves: int = 255,
    learning_rate: float = 0.05,
    random_state: int = 42,
    min_child_samples: int = 30,
) -> Dict[str, Any]:
    """LightGBM hyperparameter configuration strictly aligned with the MAE evaluation metric."""
    return {
        "objective": "regression_l1",
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "max_depth": -1,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_samples": min_child_samples,
        "n_jobs": -1,
        "random_state": random_state,
        "importance_type": "gain",
        "verbose": -1,
    }


# =============================================================================
# 4. Main Pipeline Execution
# =============================================================================
def main():
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    print("Loading raw parquet datasets and 3D molecular structures...")
    train_raw = pd.read_parquet("./input/train.parquet")
    test_raw = pd.read_parquet("./input/test.parquet")
    structures = get_structures(train_raw, test_raw)

    print(f"Loaded train shape: {train_raw.shape}, test shape: {test_raw.shape}, structures shape: {structures.shape}")

    # Compute molecule-level structure aggregates from structures table
    struct_summary = compute_structure_features(structures)

    # Molecule-grouped validation split to prevent cross-atom molecular data leakage
    gkf = GroupKFold(n_splits=5)
    groups = train_raw["molecule_name"]
    train_idx, val_idx = next(gkf.split(train_raw, groups=groups))

    train_df = train_raw.iloc[train_idx].copy().reset_index(drop=True)
    val_df = train_raw.iloc[val_idx].copy().reset_index(drop=True)
    test_df = test_raw.copy().reset_index(drop=True)

    del train_raw, test_raw
    gc.collect()

    print(
        f"Train split molecules: {train_df['molecule_name'].nunique()} ({len(train_df)} rows)"
    )
    print(
        f"Val split molecules:   {val_df['molecule_name'].nunique()} ({len(val_df)} rows)"
    )

    # Merge coordinates and compute graph topological & chemical features
    print("Computing molecular graph topological features, chemical properties, and geometry...")
    splits = {"train": train_df, "val": val_df, "test": test_df}
    for name in ["train", "val", "test"]:
        df = splits[name]
        df = merge_structures(df, structures)
        df = compute_graph_topology_features(df)
        df = df.merge(struct_summary, on="molecule_name", how="left")
        df = compute_pairwise_geometry(df)
        df = map_atom_properties(df)
        df = compute_molecule_aggregates(df)
        splits[name] = df

    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    # Fit categorical encoders strictly on the training partition
    categorical_cols = [
        col for col in ["atom_0", "atom_1"] if col in train_df.columns
    ]

    for col in categorical_cols:
        encoder = LabelEncoder()
        encoder.fit(train_df[col].astype(str))

        def safe_transform(series: pd.Series, le: LabelEncoder) -> np.ndarray:
            classes = set(le.classes_)
            mapped = series.astype(str).map(
                lambda x: x if x in classes else le.classes_[0]
            )
            return le.transform(mapped)

        train_df[f"{col}_encoded"] = encoder.transform(train_df[col].astype(str))
        val_df[f"{col}_encoded"] = safe_transform(val_df[col], encoder)
        test_df[f"{col}_encoded"] = safe_transform(test_df[col], encoder)

    target_col = "scalar_coupling_constant"
    exclude_cols = {
        "id",
        "molecule_name",
        target_col,
        "type",
        "atom_0",
        "atom_1",
    }
    feature_cols = [
        c
        for c in train_df.columns
        if c not in exclude_cols and np.issubdtype(train_df[c].dtype, np.number)
    ]

    print(f"Selected {len(feature_cols)} numeric features for modeling across {train_df['type'].nunique()} coupling types.")

    # Train Type-Specific LightGBM Models
    coupling_types = sorted(train_df["type"].unique())
    val_preds = np.zeros(len(val_df), dtype=np.float32)
    test_preds = np.zeros(len(test_df), dtype=np.float32)

    for ctype in coupling_types:
        print(f"\n--- Training Type-Specific Model for Coupling Type: {ctype} ---")
        train_mask = train_df["type"] == ctype
        val_mask = val_df["type"] == ctype
        test_mask = test_df["type"] == ctype

        train_sub = train_df[train_mask]
        val_sub = val_df[val_mask]
        test_sub = test_df[test_mask]

        print(f"[{ctype}] Train samples: {len(train_sub)}, Val samples: {len(val_sub)}, Test samples: {len(test_sub)}")

        train_data = lgb.Dataset(
            train_sub[feature_cols],
            label=train_sub[target_col].values,
            free_raw_data=False,
        )
        val_data = lgb.Dataset(
            val_sub[feature_cols],
            label=val_sub[target_col].values,
            reference=train_data,
            free_raw_data=False,
        )

        num_leaves = 255 if len(train_sub) > 50000 else 127
        lgb_params = get_lgb_model_params(
            num_leaves=num_leaves,
            learning_rate=0.05,
            min_child_samples=30,
        )
        callbacks = [
            lgb.early_stopping(stopping_rounds=80, verbose=False),
            lgb.log_evaluation(period=200),
        ]

        model = lgb.train(
            params=lgb_params,
            train_set=train_data,
            num_boost_round=3500,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            callbacks=callbacks,
        )

        model_save_path = f"./working/lgb_model_{ctype}.txt"
        model.save_model(model_save_path)

        val_sub_preds = model.predict(val_sub[feature_cols], num_iteration=model.best_iteration)
        val_preds[val_mask] = val_sub_preds

        type_val_mae = mean_absolute_error(val_sub[target_col].values, val_sub_preds)
        print(f"[{ctype}] Best Iteration: {model.best_iteration}, Validation MAE: {type_val_mae:.5f}")

        if len(test_sub) > 0:
            test_sub_preds = model.predict(test_sub[feature_cols], num_iteration=model.best_iteration)
            test_preds[test_mask] = test_sub_preds

        del train_data, val_data, model
        gc.collect()

    # Overall validation inference & score calculation
    print("\n=================================================================")
    val_mae = float(mean_absolute_error(val_df[target_col].values, val_preds))
    print(f"Overall Multi-Type Validation MAE: {val_mae:.5f}")
    print("=================================================================")

    # Test inference & submission creation
    submission_path = "./submission/submission.csv"
    submission_df = pd.DataFrame(
        {"id": test_df["id"].values, "scalar_coupling_constant": test_preds}
    )

    assert len(submission_df) == len(
        test_df
    ), f"Row count mismatch: expected {len(test_df)}, got {len(submission_df)}"
    assert (
        not submission_df["scalar_coupling_constant"].isna().any()
    ), "NaN values detected in predictions"
    assert list(submission_df.columns) == [
        "id",
        "scalar_coupling_constant",
    ], "Incorrect submission columns"

    submission_df.to_csv(submission_path, index=False)
    print(f"Saved submission to {submission_path} with shape {submission_df.shape}")

    del train_df, val_df, test_df, structures, struct_summary
    gc.collect()

    print(f"Final Validation Score: {val_mae}")


if __name__ == "__main__":
    main()
