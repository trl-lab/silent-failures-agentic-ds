import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier
import warnings

warnings.filterwarnings("ignore")

# 1. Load Data
train_df = pd.read_parquet("./input/train.parquet")
test_df = pd.read_parquet("./input/test.parquet")

# Separate target and features
drop_cols = [c for c in ["Id", "id", "Cover_Type"] if c in train_df.columns]
features = [c for c in train_df.columns if c not in drop_cols]

y_train = train_df["Cover_Type"].values - 1  # map 1-7 to 0-6 for zero-based indexing


def add_features(df):
    df_feat = df.copy()

    # Distance features
    h_hydro = df_feat["Horizontal_Distance_To_Hydrology"]
    v_hydro = df_feat["Vertical_Distance_To_Hydrology"]
    h_road = df_feat["Horizontal_Distance_To_Roadways"]
    h_fire = df_feat["Horizontal_Distance_To_Fire_Points"]

    df_feat["Euclidean_Distance_To_Hydrology"] = np.sqrt(h_hydro**2 + v_hydro**2)
    df_feat["Elevation_Minus_V_Hydro"] = df_feat["Elevation"] - v_hydro
    df_feat["Elevation_Plus_V_Hydro"] = df_feat["Elevation"] + v_hydro

    df_feat["Hydro_Plus_Roadways"] = h_hydro + h_road
    df_feat["Hydro_Minus_Roadways"] = h_hydro - h_road
    df_feat["Hydro_Plus_Fire"] = h_hydro + h_fire
    df_feat["Hydro_Minus_Fire"] = h_hydro - h_fire
    df_feat["Roadways_Plus_Fire"] = h_road + h_fire
    df_feat["Roadways_Minus_Fire"] = h_road - h_fire

    # Aspect trigonometric features
    aspect_rad = np.radians(df_feat["Aspect"])
    df_feat["Aspect_Sin"] = np.sin(aspect_rad)
    df_feat["Aspect_Cos"] = np.cos(aspect_rad)

    # Hillshade aggregations
    hill_cols = ["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]
    df_feat["Hillshade_Mean"] = df_feat[hill_cols].mean(axis=1)
    df_feat["Hillshade_9am_Minus_3pm"] = (
        df_feat["Hillshade_9am"] - df_feat["Hillshade_3pm"]
    )
    df_feat["Hillshade_Noon_Minus_3pm"] = (
        df_feat["Hillshade_Noon"] - df_feat["Hillshade_3pm"]
    )

    # Categorical index features
    soil_cols = [c for c in df_feat.columns if c.startswith("Soil_Type")]
    wild_cols = [c for c in df_feat.columns if c.startswith("Wilderness_Area")]

    if soil_cols:
        soil_mat = df_feat[soil_cols].values
        df_feat["Soil_Type_Code"] = np.argmax(soil_mat, axis=1)
    if wild_cols:
        wild_mat = df_feat[wild_cols].values
        df_feat["Wilderness_Area_Code"] = np.argmax(wild_mat, axis=1)

    return df_feat


X_train = add_features(train_df[features])
X_test = add_features(test_df[[c for c in test_df.columns if c in features]])

# 2. Cross-Validation and Modeling
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros((len(X_train), 7))
test_preds = np.zeros((len(X_test), 7))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

    # LightGBM Classifier
    lgb_model = LGBMClassifier(
        objective="multiclass",
        num_class=7,
        n_estimators=300,
        learning_rate=0.05,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(X_tr, y_tr)

    # Extra Trees Classifier
    et_model = ExtraTreesClassifier(n_estimators=300, random_state=42 + fold, n_jobs=-1)
    et_model.fit(X_tr, y_tr)

    # Blend probabilities
    val_pred_lgb = lgb_model.predict_proba(X_va)
    val_pred_et = et_model.predict_proba(X_va)
    oof_preds[val_idx] = 0.5 * val_pred_lgb + 0.5 * val_pred_et

    test_pred_lgb = lgb_model.predict_proba(X_test)
    test_pred_et = et_model.predict_proba(X_test)
    test_preds += (0.5 * test_pred_lgb + 0.5 * test_pred_et) / skf.n_splits

# 3. Evaluation
oof_classes = np.argmax(oof_preds, axis=1)
cv_accuracy = accuracy_score(y_train, oof_classes)
print(f"5-Fold CV Accuracy: {cv_accuracy:.5f}")

# 4. Submission File
final_test_classes = np.argmax(test_preds, axis=1) + 1  # map back to 1-7

submission = pd.DataFrame(
    {"id": np.arange(len(test_df)), "Cover_Type": final_test_classes}
)
submission.to_csv("./working/submission.csv", index=False)
