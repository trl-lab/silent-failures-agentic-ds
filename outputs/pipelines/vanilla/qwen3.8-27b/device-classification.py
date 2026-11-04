import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/device-classification/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/device-classification/data/test.parquet')

# Separate features and target
y_train = train_df['device_class'].astype(str)
X_train = train_df.drop('device_class', axis=1)
X_test = test_df.copy()

# Fill missing values
for col in X_train.columns:
    X_train[col] = X_train[col].fillna('MISSING').astype(str)
    X_test[col] = X_test[col].fillna('MISSING').astype(str)

# Convert all columns to category dtype for native categorical handling
cat_cols = X_train.columns.tolist()
for col in cat_cols:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# Encode target
classes = sorted(y_train.unique())
class_to_idx = {c: i for i, c in enumerate(classes)}
y_train_encoded = y_train.map(class_to_idx).values

# Split for validation
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
)

# Train LightGBM
lgbm_model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbose=-1
)
lgbm_model.fit(X_tr, y_tr, categorical_feature=cat_cols)
val_pred_lgbm = lgbm_model.predict(X_val)
score_lgbm = accuracy_score(y_val, val_pred_lgbm)
print(f"LightGBM: Validation Accuracy = {score_lgbm:.4f}")

# Train CatBoost
cat_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    random_state=42,
    verbose=0,
    cat_features=cat_cols
)
cat_model.fit(X_tr, y_tr, cat_features=cat_cols)
val_pred_cat = cat_model.predict(X_val)
score_cat = accuracy_score(y_val, val_pred_cat)
print(f"CatBoost: Validation Accuracy = {score_cat:.4f}")

# Retrain on full training data
lgbm_model.fit(X_train, y_train_encoded, categorical_feature=cat_cols)
cat_model.fit(X_train, y_train_encoded, cat_features=cat_cols)

# Ensemble: average probabilities
proba_lgbm = lgbm_model.predict_proba(X_test)
proba_cat = cat_model.predict_proba(X_test)

# Align columns (both should have same number of classes)
avg_proba = (proba_lgbm + proba_cat) / 2.0
final_pred_indices = np.argmax(avg_proba, axis=1)
final_pred = np.array(classes)[final_pred_indices]

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(final_pred)),
    'device_class': final_pred
})
output_df.to_csv('predictions.csv', index=False)