import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/device-pma/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/device-pma/data/test.parquet')

# Encode target
le_target = LabelEncoder()
y_train = le_target.fit_transform(train_df['decision_code'])

# Drop target and non-feature columns
drop_cols = ['decision_code', 'openfda', 'pma_number', 'docket_number']
drop_cols = [c for c in drop_cols if c in train_df.columns]

X_train_full = train_df.drop(columns=drop_cols).copy()
X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns]).copy()

# Feature engineering
def engineer_features(df):
    df = df.copy()
    
    # Date features - extract numeric parts then drop original
    for col in ['date_received', 'decision_date', 'fed_reg_notice_date']:
        if col in df.columns:
            dt = pd.to_datetime(df[col], errors='coerce')
            df[f'{col}_year'] = dt.dt.year
            df[f'{col}_month'] = dt.dt.month
            df[f'{col}_day'] = dt.dt.day
            df.drop(columns=[col], inplace=True)
    
    # Review duration
    if 'date_received_year' in df.columns and 'decision_date_year' in df.columns:
        dr = pd.to_datetime(df.get('date_received', pd.Series(dtype='datetime64[ns]')), errors='coerce')
        dd = pd.to_datetime(df.get('decision_date', pd.Series(dtype='datetime64[ns]')), errors='coerce')
        # We already dropped these, so compute from year/month
        df['review_duration_approx'] = (
            (df['decision_date_year'] - df['date_received_year']) * 365 +
            (df['decision_date_month'] - df['date_received_month']) * 30
        )
    
    # Text length features
    for col in ['generic_name', 'trade_name', 'applicant', 'street_1', 'street_2']:
        if col in df.columns:
            df[f'{col}_len'] = df[col].astype(str).str.len()
    
    # Word count in generic_name
    if 'generic_name' in df.columns:
        df['generic_name_word_count'] = df['generic_name'].astype(str).str.split().str.len()
    
    return df

X_train_full = engineer_features(X_train_full)
X_test = engineer_features(X_test)

# Identify columns to keep
# Drop high-cardinality object columns (identifiers, addresses, etc.)
high_card_drop = ['applicant', 'street_1', 'street_2', 'city', 'trade_name', 'generic_name']
high_card_drop = [c for c in high_card_drop if c in X_train_full.columns]

# Also drop any remaining object columns with very high cardinality
for col in X_train_full.select_dtypes(include=['object', 'category']).columns:
    if X_train_full[col].nunique() > 200:
        high_card_drop.append(col)

high_card_drop = list(set(high_card_drop))
X_train_full.drop(columns=high_card_drop, inplace=True, errors='ignore')
X_test.drop(columns=high_card_drop, inplace=True, errors='ignore')

# Now encode remaining categorical columns
categorical_cols = X_train_full.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = X_train_full.select_dtypes(include=[np.number]).columns.tolist()

for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([X_train_full[col].astype(str), X_test[col].astype(str)], axis=0)
    le.fit(combined)
    X_train_full[col] = le.transform(X_train_full[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# Fill missing numeric values
for col in X_train_full.columns:
    if X_train_full[col].isna().any():
        median_val = X_train_full[col].median()
        if np.isnan(median_val):
            median_val = 0
        X_train_full[col] = X_train_full[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

# Ensure all columns are numeric
X_train_full = X_train_full.astype(float)
X_test = X_test.astype(float)

# Create validation split
X_tr, X_val, y_tr, y_val = train_test_split(X_train_full, y_train, test_size=0.2, random_state=42, stratify=y_train)

# Train models and evaluate
models = {
    'lgbm': LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        verbose=-1
    ),
    'xgb': XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    ),
    'cat': CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        random_state=42,
        verbose=0
    )
}

val_scores = {}
for name, model in models.items():
    model.fit(X_tr, y_tr)
    val_pred = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, val_pred)
    val_scores[name] = score
    print(f"{name}: ROC AUC = {score:.4f}")

# Retrain on full training data and predict test
test_preds = []
for name, model in models.items():
    model.fit(X_train_full, y_train)
    test_pred = model.predict_proba(X_test)[:, 1]
    test_preds.append(test_pred)

# Ensemble
final_pred = np.mean(test_preds, axis=0)

# Save predictions
submission = pd.DataFrame({
    'id': range(len(final_pred)),
    'decision_code': final_pred
})
submission.to_csv('predictions.csv', index=False)
print(f"Predictions saved. Shape: {submission.shape}")