import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/drug-enforcement/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/drug-enforcement/data/test.parquet')

# Separate target
y_train = train_df['classification']
X_train = train_df.drop('classification', axis=1)
X_test = test_df.copy()

# Identify columns to use - drop columns that are almost entirely missing
# and keep columns with meaningful information
cols_to_drop = []
for col in X_train.columns:
    if X_train[col].isna().mean() > 0.95:
        cols_to_drop.append(col)

X_train = X_train.drop(columns=cols_to_drop)
X_test = X_test.drop(columns=cols_to_drop)

# Convert all columns to string for consistent handling
for col in X_train.columns:
    X_train[col] = X_train[col].astype(str).replace('None', np.nan).replace('nan', np.nan)
    X_test[col] = X_test[col].astype(str).replace('None', np.nan).replace('nan', np.nan)

# Fill missing values with a placeholder
for col in X_train.columns:
    X_train[col] = X_train[col].fillna('MISSING')
    X_test[col] = X_test[col].fillna('MISSING')

# Encode target
le_target = LabelEncoder()
y_train_encoded = le_target.fit_transform(y_train)

# Encode categorical features
categorical_cols = X_train.columns.tolist()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Fit on combined train+test to handle unseen categories
    combined = pd.concat([X_train[col], X_test[col]], axis=0)
    le.fit(combined)
    X_train[col] = le.transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    label_encoders[col] = le

# Create validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
)

# Define models
models = {
    'LightGBM': LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        eval_metric='mlogloss',
        use_label_encoder=False
    ),
    'CatBoost': CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        random_state=42,
        verbose=0
    )
}

# Evaluate models on validation set
val_scores = {}
for name, model in models.items():
    model.fit(X_tr, y_tr)
    val_pred = model.predict(X_val)
    from sklearn.metrics import accuracy_score
    score = accuracy_score(y_val, val_pred)
    val_scores[name] = score
    print(f"{name}: Validation Accuracy = {score:.4f}")

# Retrain all models on full training data and ensemble
final_predictions = []
for name, model in models.items():
    model.fit(X_train, y_train_encoded)
    pred_proba = model.predict_proba(X_test)
    final_predictions.append(pred_proba)

# Average probabilities
avg_proba = np.mean(final_predictions, axis=0)
final_pred_encoded = np.argmax(avg_proba, axis=1)
final_pred = le_target.inverse_transform(final_pred_encoded)

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(final_pred)),
    'classification': final_pred
})
output_df.to_csv('predictions.csv', index=False)