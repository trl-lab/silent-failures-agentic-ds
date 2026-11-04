import os
import random
import re
import warnings

warnings.filterwarnings("ignore")

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================================
# 1. PyTorch Tabular Residual Neural Network Architecture
# =========================================================================
class TabularResidualBlock(nn.Module):
    """Residual block for tabular data with LayerNorm, SiLU, and Dropout."""

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TabularResNet(nn.Module):
    """Deep Tabular ResNet architecture designed for tabular binary classification."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_blocks: int = 2,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()
        )
        self.res_blocks = nn.ModuleList(
            [TabularResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.input_projection(x)
        for block in self.res_blocks:
            feat = block(feat)
        logits = self.head(feat)
        return logits.squeeze(-1)


def build_pytorch_components(
    input_dim: int,
    hidden_dim: int = 64,
    num_blocks: int = 2,
    dropout_rate: float = 0.25,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
):
    """Instantiates TabularResNet model, binary cross entropy loss, and AdamW optimizer."""
    model = TabularResNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout_rate=dropout_rate,
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=1e-5
    )
    return model, criterion, optimizer, scheduler


# =========================================================================
# 2. Gradient Boosted & Tree-Based Model Model Configurations
# =========================================================================
def get_tree_and_linear_models(random_state: int = 42) -> dict:
    """Instantiates configured tree ensembles and regularized linear classifiers."""
    models = {
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.2,
            reg_lambda=1.0,
            min_child_samples=10,
            random_state=random_state,
            verbose=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.2,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=random_state,
            tree_method="hist",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_split=6,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_split=6,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        ),
        "logistic_regression": LogisticRegression(
            C=0.1,
            penalty="l2",
            solver="liblinear",
            max_iter=1000,
            random_state=random_state,
        ),
    }
    return models


# =========================================================================
# 3. Main Data Processing, Training, and Evaluation Pipeline
# =========================================================================
def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dir = "./input"
    output_dir = "./working"
    submission_dir = "./submission"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    train_path = os.path.join(input_dir, "train.parquet")
    test_path = os.path.join(input_dir, "test.parquet")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # Honorific Titles Extraction
    def extract_title(name_series):
        titles = name_series.str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
        title_mapping = {
            "Mlle": "Miss",
            "Ms": "Miss",
            "Mme": "Mrs",
            "Lady": "Noble",
            "Countess": "Noble",
            "Dona": "Noble",
            "Sir": "Noble",
            "Jonkheer": "Noble",
            "Don": "Noble",
            "Capt": "Officer",
            "Col": "Officer",
            "Major": "Officer",
            "Dr": "Officer",
            "Rev": "Officer",
        }
        titles = titles.map(lambda t: title_mapping.get(t, t))
        common_titles = {"Mr", "Miss", "Mrs", "Master", "Noble", "Officer"}
        titles = titles.map(lambda t: t if t in common_titles else "Rare")
        return titles

    train_df["Title"] = extract_title(train_df["Name"])
    test_df["Title"] = extract_title(test_df["Name"])

    # Group Size and Family Features
    combined_tickets = pd.concat([train_df["Ticket"], test_df["Ticket"]], axis=0)
    ticket_counts = combined_tickets.value_counts().to_dict()

    for df in [train_df, test_df]:
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
        df["TicketCount"] = df["Ticket"].map(ticket_counts).fillna(1).astype(int)
        df["GroupSize"] = df[["FamilySize", "TicketCount"]].max(axis=1)
        df["IsSmallFamily"] = ((df["GroupSize"] >= 2) & (df["GroupSize"] <= 4)).astype(
            int
        )
        df["IsLargeFamily"] = (df["GroupSize"] >= 5).astype(int)

    # Imputation derived strictly from train set
    embarked_mode = train_df["Embarked"].dropna().mode()[0]
    train_df["Embarked"] = train_df["Embarked"].fillna(embarked_mode)
    test_df["Embarked"] = test_df["Embarked"].fillna(embarked_mode)

    fare_medians = train_df.groupby("Pclass")["Fare"].median().to_dict()
    overall_fare_median = train_df["Fare"].median()

    for df in [train_df, test_df]:
        df["Fare"] = df["Fare"].fillna(
            df["Pclass"].map(fare_medians).fillna(overall_fare_median)
        )
        df["AdjustedFare"] = df["Fare"] / df["GroupSize"]
        df["LogFare"] = np.log1p(df["Fare"])
        df["LogAdjustedFare"] = np.log1p(df["AdjustedFare"])

    age_medians = train_df.groupby(["Title", "Pclass"])["Age"].median().to_dict()
    title_age_medians = train_df.groupby("Title")["Age"].median().to_dict()
    overall_age_median = train_df["Age"].median()

    for df in [train_df, test_df]:
        imputed_age = df.apply(
            lambda r: age_medians.get(
                (r["Title"], r["Pclass"]),
                title_age_medians.get(r["Title"], overall_age_median),
            ),
            axis=1,
        )
        df["Age_WasMissing"] = df["Age"].isna().astype(int)
        df["Age"] = df["Age"].fillna(imputed_age)
        df["IsChild"] = (df["Age"] < 12).astype(int)
        df["IsElderly"] = (df["Age"] >= 60).astype(int)
        df["Age_Pclass"] = df["Age"] * df["Pclass"]

    # Cabin Parsing and Deck Extraction
    deck_hierarchy = {
        "U": 0,
        "G": 1,
        "F": 2,
        "E": 3,
        "D": 4,
        "C": 5,
        "B": 6,
        "A": 7,
        "T": 4,
    }
    for df in [train_df, test_df]:
        df["HasCabin"] = df["Cabin"].notna().astype(int)
        df["CabinDeck"] = df["Cabin"].fillna("U").astype(str).str[0].str.upper()
        df["CabinDeck"] = df["CabinDeck"].map(
            lambda d: d if d in deck_hierarchy else "U"
        )
        df["DeckScore"] = df["CabinDeck"].map(deck_hierarchy).fillna(0).astype(int)
        df["CabinRoomCount"] = (
            df["Cabin"].fillna("").apply(lambda x: len(x.split()) if x else 0)
        )

    # Categorical Encodings & Interactions
    for df in [train_df, test_df]:
        df["Sex_Pclass"] = df["Sex"].astype(str) + "_" + df["Pclass"].astype(str)
        df["Title_Pclass"] = df["Title"].astype(str) + "_" + df["Pclass"].astype(str)
        df["Embarked_Pclass"] = (
            df["Embarked"].astype(str) + "_" + df["Pclass"].astype(str)
        )

    cat_cols = [
        "Sex",
        "Embarked",
        "Title",
        "CabinDeck",
        "Sex_Pclass",
        "Title_Pclass",
        "Embarked_Pclass",
    ]
    for col in cat_cols:
        le = LabelEncoder()
        train_df[col + "_encoded"] = le.fit_transform(train_df[col].astype(str))
        train_classes = set(le.classes_)
        test_df_col_safe = (
            test_df[col]
            .astype(str)
            .map(lambda c: c if c in train_classes else le.classes_[0])
        )
        test_df[col + "_encoded"] = le.transform(test_df_col_safe)

    # One-hot encoding for linear and neural network pipelines
    train_onehot = pd.get_dummies(train_df[cat_cols], drop_first=True, dtype=int)
    test_onehot = pd.get_dummies(test_df[cat_cols], drop_first=True, dtype=int)
    train_onehot, test_onehot = train_onehot.align(
        test_onehot, join="left", axis=1, fill_value=0
    )

    # Feature definitions
    cont_features = [
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "FamilySize",
        "IsAlone",
        "TicketCount",
        "GroupSize",
        "IsSmallFamily",
        "IsLargeFamily",
        "AdjustedFare",
        "LogFare",
        "LogAdjustedFare",
        "Age_WasMissing",
        "IsChild",
        "IsElderly",
        "Age_Pclass",
        "HasCabin",
        "DeckScore",
        "CabinRoomCount",
    ]
    cat_encoded_features = [col + "_encoded" for col in cat_cols]
    tree_features = cont_features + cat_encoded_features

    # Setup Stratified 5-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_df["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(skf.split(train_df, train_df["Survived"])):
        train_df.loc[val_idx, "fold"] = fold_idx

    y_train = train_df["Survived"].values
    n_train = len(train_df)
    n_test = len(test_df)
    n_folds = 5

    model_names = [
        "lightgbm",
        "xgboost",
        "random_forest",
        "extra_trees",
        "logistic_regression",
        "tabular_resnet",
    ]
    oof_preds = {name: np.zeros(n_train, dtype=float) for name in model_names}
    test_preds = {name: np.zeros(n_test, dtype=float) for name in model_names}

    for fold in range(n_folds):
        train_idx = train_df["fold"] != fold
        val_idx = train_df["fold"] == fold

        # Tree-based feature matrices
        X_tr_tree = train_df.loc[train_idx, tree_features].values
        y_tr = y_train[train_idx]
        X_val_tree = train_df.loc[val_idx, tree_features].values
        y_val = y_train[val_idx]
        X_test_tree = test_df[tree_features].values

        # Continuous feature scaling for dense representations
        scaler = StandardScaler()
        X_tr_cont_scaled = scaler.fit_transform(
            train_df.loc[train_idx, cont_features].values
        )
        X_val_cont_scaled = scaler.transform(
            train_df.loc[val_idx, cont_features].values
        )
        X_test_cont_scaled = scaler.transform(test_df[cont_features].values)

        X_tr_dense = np.hstack([X_tr_cont_scaled, train_onehot.loc[train_idx].values])
        X_val_dense = np.hstack([X_val_cont_scaled, train_onehot.loc[val_idx].values])
        X_test_dense = np.hstack([X_test_cont_scaled, test_onehot.values])

        models = get_tree_and_linear_models(random_state=42 + fold)

        # 1. Train LightGBM
        lgb_model = models["lightgbm"]
        lgb_model.fit(
            X_tr_tree,
            y_tr,
            eval_set=[(X_val_tree, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        oof_preds["lightgbm"][val_idx] = lgb_model.predict_proba(X_val_tree)[:, 1]
        test_preds["lightgbm"] += lgb_model.predict_proba(X_test_tree)[:, 1] / n_folds

        # 2. Train XGBoost
        xgb_model = models["xgboost"]
        xgb_model.fit(
            X_tr_tree,
            y_tr,
            eval_set=[(X_val_tree, y_val)],
            verbose=False,
        )
        oof_preds["xgboost"][val_idx] = xgb_model.predict_proba(X_val_tree)[:, 1]
        test_preds["xgboost"] += xgb_model.predict_proba(X_test_tree)[:, 1] / n_folds

        # 3. Train Random Forest
        rf_model = models["random_forest"]
        rf_model.fit(X_tr_tree, y_tr)
        oof_preds["random_forest"][val_idx] = rf_model.predict_proba(X_val_tree)[:, 1]
        test_preds["random_forest"] += (
            rf_model.predict_proba(X_test_tree)[:, 1] / n_folds
        )

        # 4. Train Extra Trees
        et_model = models["extra_trees"]
        et_model.fit(X_tr_tree, y_tr)
        oof_preds["extra_trees"][val_idx] = et_model.predict_proba(X_val_tree)[:, 1]
        test_preds["extra_trees"] += et_model.predict_proba(X_test_tree)[:, 1] / n_folds

        # 5. Train Logistic Regression
        lr_model = models["logistic_regression"]
        lr_model.fit(X_tr_dense, y_tr)
        oof_preds["logistic_regression"][val_idx] = lr_model.predict_proba(X_val_dense)[
            :, 1
        ]
        test_preds["logistic_regression"] += (
            lr_model.predict_proba(X_test_dense)[:, 1] / n_folds
        )

        # 6. Train TabularResNet
        input_dim = X_tr_dense.shape[1]
        nn_model, criterion, optimizer, scheduler = build_pytorch_components(
            input_dim=input_dim,
            hidden_dim=64,
            num_blocks=2,
            dropout_rate=0.25,
            lr=1e-3,
            weight_decay=1e-4,
        )
        nn_model = nn_model.to(device)

        tr_dataset = TensorDataset(
            torch.tensor(X_tr_dense, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
        )
        val_tensor = torch.tensor(X_val_dense, dtype=torch.float32).to(device)
        test_tensor = torch.tensor(X_test_dense, dtype=torch.float32).to(device)
        tr_loader = DataLoader(tr_dataset, batch_size=32, shuffle=True, drop_last=False)

        best_val_loss = float("inf")
        best_nn_state = None
        epochs = 45

        for _ in range(epochs):
            nn_model.train()
            for bx, by in tr_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                logits = nn_model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
            scheduler.step()

            nn_model.eval()
            with torch.no_grad():
                val_logits = nn_model(val_tensor)
                val_loss = criterion(
                    val_logits, torch.tensor(y_val, dtype=torch.float32).to(device)
                ).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_nn_state = {
                        k: v.cpu().clone() for k, v in nn_model.state_dict().items()
                    }

        nn_model.load_state_dict(best_nn_state)
        nn_model.eval()
        with torch.no_grad():
            oof_nn = torch.sigmoid(nn_model(val_tensor)).cpu().numpy().flatten()
            test_nn = torch.sigmoid(nn_model(test_tensor)).cpu().numpy().flatten()

        oof_preds["tabular_resnet"][val_idx] = oof_nn
        test_preds["tabular_resnet"] += test_nn / n_folds

        print(
            f"Fold {fold + 1}/{n_folds} completed | LGBM Acc: {accuracy_score(y_val, (oof_preds['lightgbm'][val_idx] >= 0.5).astype(int)):.4f} | ResNet Acc: {accuracy_score(y_val, (oof_nn >= 0.5).astype(int)):.4f}"
        )

    # Blending optimization on Out-of-Fold predictions
    oof_matrix = np.column_stack([oof_preds[name] for name in model_names])
    test_matrix = np.column_stack([test_preds[name] for name in model_names])

    def loss_func(weights):
        w = weights / np.sum(weights)
        blended_oof = np.dot(oof_matrix, w)
        eps = 1e-15
        blended_oof = np.clip(blended_oof, eps, 1.0 - eps)
        return -np.mean(
            y_train * np.log(blended_oof) + (1.0 - y_train) * np.log(1.0 - blended_oof)
        )

    init_weights = np.ones(len(model_names)) / len(model_names)
    bounds = [(0, 1) for _ in range(len(model_names))]
    res = minimize(
        loss_func,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints={"type": "eq", "fun": lambda w: 1.0 - np.sum(w)},
    )
    opt_weights = res.x / np.sum(res.x)

    oof_ensemble = np.dot(oof_matrix, opt_weights)
    test_ensemble = np.dot(test_matrix, opt_weights)

    # Tune decision threshold to maximize classification accuracy
    best_threshold = 0.5
    best_acc = 0.0
    for th in np.linspace(0.30, 0.70, 401):
        acc = accuracy_score(y_train, (oof_ensemble >= th).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_threshold = th

    print(
        f"Optimal Decision Threshold: {best_threshold:.4f} | OOF Ensemble Accuracy: {best_acc:.5f}"
    )

    # Generate final test predictions
    final_test_binary = (test_ensemble >= best_threshold).astype(int)
    submission_df = pd.DataFrame(
        {"id": np.arange(len(test_df)), "Survived": final_test_binary}
    )
    submission_path = os.path.join(submission_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Saved submission to {submission_path} with {len(submission_df)} rows.")

    # Output required final validation score
    print(f"Final Validation Score: {best_acc}")


if __name__ == "__main__":
    main()