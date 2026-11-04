import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/mercari/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/strable/mercari/data/test.parquet"
TARGET = "price"
RANDOM_STATE = 42


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def create_models():
    return {
        "lgbm": LGBMRegressor(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "xgb": XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        ),
        "cat": CatBoostRegressor(
            iterations=600,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            random_seed=RANDOM_STATE,
            verbose=0,
        ),
    }


class FeatureBuilder:
    def __init__(self):
        self.tfidf = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
            min_df=3,
            max_df=0.9,
            max_features=5000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.svd = None
        self.freq_maps = {}

    def _text(self, df):
        name = df["name"].fillna("").astype(str)
        desc = df["item_description"].fillna("").astype(str)
        return (name + " " + desc).astype(str)

    def _category_parts(self, df):
        cat = df["category_name"].fillna("missing").astype(str)
        top = cat.str.split("/").str[0].fillna("missing")
        mid = cat.str.split("/").str[1].fillna("missing")
        leaf = cat.str.split("/").str[-1].fillna("missing")
        return cat, top, mid, leaf

    def fit(self, df):
        text = self._text(df)
        try:
            self.tfidf.fit(text)
            X_tfidf = self.tfidf.transform(text)
            n_components = min(100, max(1, min(df.shape[0] - 1, X_tfidf.shape[1])))
            if X_tfidf.shape[1] > 0 and n_components > 0:
                self.svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
                self.svd.fit(X_tfidf)
            else:
                self.svd = None
        except Exception:
            self.tfidf = None
            self.svd = None

        cat, top, mid, leaf = self._category_parts(df)
        brand = df["brand_name"].fillna("missing").astype(str)

        self.freq_maps = {
            "category_name": cat.value_counts().to_dict(),
            "category_top": top.value_counts().to_dict(),
            "category_mid": mid.value_counts().to_dict(),
            "category_leaf": leaf.value_counts().to_dict(),
            "brand_name": brand.value_counts().to_dict(),
        }
        return self

    def transform(self, df):
        feats = pd.DataFrame(index=df.index)

        feats["item_condition_id"] = df["item_condition_id"].fillna(-1).astype(float)
        feats["shipping"] = df["shipping"].fillna(-1).astype(float)

        name = df["name"].fillna("").astype(str)
        desc = df["item_description"].fillna("").astype(str)

        name_words = name.str.split().str.len().fillna(0)
        desc_words = desc.str.split().str.len().fillna(0)

        feats["name_len"] = name.str.len().astype(float)
        feats["desc_len"] = desc.str.len().astype(float)
        feats["name_words"] = name_words.astype(float)
        feats["desc_words"] = desc_words.astype(float)
        feats["desc_avg_word_len"] = np.where(desc_words > 0, feats["desc_len"] / desc_words, 0.0)
        feats["desc_digits"] = desc.str.count(r"\d").astype(float)
        feats["desc_upper"] = desc.str.count(r"[A-Z]").astype(float)
        feats["desc_punct"] = desc.str.count(r"[^\w\s]").astype(float)
        feats["name_upper"] = name.str.count(r"[A-Z]").astype(float)

        keywords = [
            "free",
            "new",
            "used",
            "sale",
            "ship",
            "discount",
            "bundle",
            "lot",
            "size",
            "color",
            "original",
            "authentic",
            "rare",
            "vintage",
            "like new",
            "excellent",
            "good",
            "fair",
            "poor",
            "damaged",
            "defect",
            "warranty",
            "return",
            "exchange",
            "gift",
            "set",
            "pair",
        ]
        for kw in keywords:
            kw_clean = kw.replace(" ", "_")
            feats[f"kw_{kw_clean}"] = (
                name.str.contains(kw, regex=False, na=False)
                | desc.str.contains(kw, regex=False, na=False)
            ).astype(float)

        cat, top, mid, leaf = self._category_parts(df)
        brand = df["brand_name"].fillna("missing").astype(str)

        feats["category_name_freq"] = cat.map(self.freq_maps.get("category_name", {})).fillna(0.0)
        feats["category_top_freq"] = top.map(self.freq_maps.get("category_top", {})).fillna(0.0)
        feats["category_mid_freq"] = mid.map(self.freq_maps.get("category_mid", {})).fillna(0.0)
        feats["category_leaf_freq"] = leaf.map(self.freq_maps.get("category_leaf", {})).fillna(0.0)
        feats["brand_name_freq"] = brand.map(self.freq_maps.get("brand_name", {})).fillna(0.0)

        if self.tfidf is not None and self.svd is not None:
            try:
                X_tfidf = self.tfidf.transform(self._text(df))
                X_svd = self.svd.transform(X_tfidf)
                svd_cols = [f"tfidf_svd_{i}" for i in range(X_svd.shape[1])]
                feats = pd.concat(
                    [feats, pd.DataFrame(X_svd, index=df.index, columns=svd_cols)],
                    axis=1,
                )
            except Exception:
                pass

        return feats.fillna(0.0)


def main():
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    train_df = train_df.dropna(subset=[TARGET])
    test_df = test_df.drop(columns=[TARGET], errors="ignore")

    X_full = train_df.drop(columns=[TARGET], errors="ignore")
    y_full = train_df[TARGET].astype(float)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_full,
        y_full,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    val_builder = FeatureBuilder().fit(X_tr)
    X_tr_feat = val_builder.transform(X_tr)
    X_val_feat = val_builder.transform(X_val)

    val_models = create_models()
    val_rmses = {}
    for name, model in val_models.items():
        model.fit(X_tr_feat, y_tr)
        val_pred = model.predict(X_val_feat)
        val_rmses[name] = rmse(y_val, val_pred)
        print(f"Validation RMSE {name}: {val_rmses[name]:.6f}")

    weights = {name: 1.0 / max(rmse, 1e-8) for name, rmse in val_rmses.items()}
    weight_sum = sum(weights.values())
    weights = {name: w / weight_sum for name, w in weights.items()}

    final_builder = FeatureBuilder().fit(X_full)
    X_full_feat = final_builder.transform(X_full)
    X_test_feat = final_builder.transform(test_df)

    final_models = create_models()
    final_preds = np.zeros(len(test_df), dtype=float)
    for name, model in final_models.items():
        model.fit(X_full_feat, y_full)
        pred = model.predict(X_test_feat)
        final_preds += weights[name] * pred

    final_preds = np.nan_to_num(final_preds, nan=float(y_full.mean()), posinf=10.0, neginf=0.0)
    final_preds = np.clip(final_preds, 0.0, 10.0)

    pd.DataFrame({"id": np.arange(len(test_df), dtype=int), "price": final_preds}).to_csv(
        "predictions.csv",
        index=False,
    )


if __name__ == "__main__":
    main()