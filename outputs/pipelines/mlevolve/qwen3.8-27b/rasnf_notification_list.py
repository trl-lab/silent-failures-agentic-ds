import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import os

# Read data
train = pd.read_parquet('./input/train.parquet')
test = pd.read_parquet('./input/test.parquet')

# Identify target
target = 'num_countries_affected'

# Keep only rows with a valid target and drop row identifiers from features
y_raw = train[target]
valid_mask = y_raw.notna()
train = train.loc[valid_mask].copy()
y_raw = y_raw[valid_mask]

drop_feature_cols = ['id']
X = train.drop(columns=[target] + [c for c in drop_feature_cols if c in train.columns])
test_data = test.drop(columns=[c for c in drop_feature_cols if c in test.columns], errors='ignore').copy()

# Align test columns with the training feature set (use NaN, not 0, for missing features)
for col in X.columns:
    if col not in test_data.columns:
        test_data[col] = np.nan
test_data = test_data[X.columns].copy()

# Convert datetime columns to numeric-safe types before feature encoding
datetime_cols = X.select_dtypes(include=['datetime64[ns]']).columns.tolist()
for col in datetime_cols:
    X[col] = X[col].astype('int64')
    test_data[col] = pd.to_datetime(test_data[col], errors='coerce').astype('int64')

# Label-encode target so the classifier always sees integer labels
le = LabelEncoder()
y = le.fit_transform(y_raw)

def prepare_features(X_transform, state=None, X_fit=None, numeric_cols=None, cat_cols=None):
    if state is None:
        if numeric_cols:
            numeric_medians = X_fit[numeric_cols].median().fillna(0.0)
        else:
            numeric_medians = pd.Series(dtype=float)
        categories = {}
        encodings = {}
        for col in cat_cols:
            vals = X_fit[col].astype('string').fillna('__MISSING__')
            cats = sorted(vals.unique())
            categories[col] = cats
            encodings[col] = 'onehot' if len(cats) <= 50 else 'ordinal'
        state = {
            'numeric_medians': numeric_medians,
            'categories': categories,
            'encodings': encodings,
        }

    if numeric_cols:
        num = X_transform[numeric_cols].copy()
        num = num.replace([np.inf, -np.inf], np.nan)
        num = num.fillna(state['numeric_medians'])
        num = num.astype(float)
    else:
        num = pd.DataFrame(index=X_transform.index)

    frames = []
    for col in cat_cols:
        s = X_transform[col].astype('string').fillna('__MISSING__')
        cats = state['categories'][col]
        s = s.where(s.isin(cats), '__UNKNOWN__')
        if state['encodings'][col] == 'onehot':
            frames.append(pd.get_dummies(s, prefix=col, prefix_sep='__', drop_first=False).astype(float))
        else:
            mapping = {c: i for i, c in enumerate(cats)}
            frames.append(pd.DataFrame({col: s.map(mapping).fillna(-1).astype(float)}))

    if frames:
        cat = pd.concat(frames, axis=1)
    else:
        cat = pd.DataFrame(index=X_transform.index)

    return pd.concat([num, cat], axis=1), state

# Create validation split FIRST (before any preprocessing to avoid leakage)
if len(np.unique(y)) > 1 and pd.Series(y).value_counts().min() >= 2:
    stratify = y
else:
    stratify = None
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify
)

# Fit feature statistics/categories on the training split only, then transform all sets
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X_train.columns if c not in numeric_cols]

X_train_enc, state = prepare_features(X_train, X_fit=X_train, numeric_cols=numeric_cols, cat_cols=cat_cols)
X_val_enc, _ = prepare_features(X_val, state=state, numeric_cols=numeric_cols, cat_cols=cat_cols)
test_enc, _ = prepare_features(test_data, state=state, numeric_cols=numeric_cols, cat_cols=cat_cols)

# Align transformed columns exactly (train columns are the reference)
X_train_enc = X_train_enc.reindex(columns=X_train_enc.columns, fill_value=0.0)
X_val_enc = X_val_enc.reindex(columns=X_train_enc.columns, fill_value=0.0)
test_enc = test_enc.reindex(columns=X_train_enc.columns, fill_value=0.0)

def probability_to_score(proba, model_classes, original_classes):
    if len(model_classes) == 2:
        return proba[:, 1]
    try:
        orig_float = np.asarray(original_classes, dtype=float)
        aligned = np.array([orig_float[int(c)] for c in model_classes], dtype=float)
        return proba @ aligned
    except (TypeError, ValueError, IndexError):
        return proba[:, -1]

def ordinal_roc_auc(y_true, score):
    y_true = np.asarray(y_true)
    score = np.asarray(score)
    unique = np.unique(y_true)
    if len(unique) <= 1:
        return 0.5
    if len(unique) == 2:
        return float(roc_auc_score(y_true, score))
    aucs = []
    for t in unique[1:]:
        y_bin = (y_true >= t).astype(int)
        if len(np.unique(y_bin)) > 1:
            aucs.append(roc_auc_score(y_bin, score))
    return float(np.mean(aucs)) if aucs else 0.5

# Train validation model on the training split only
model = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
model.fit(X_train_enc, y_train)

# Validation predictions using actual model output
val_proba = model.predict_proba(X_val_enc)
val_score = probability_to_score(val_proba, model.classes_, le.classes_)
val_auc = ordinal_roc_auc(y_val, val_score)

# Retrain on the complete training set using the same fitted preprocessing state
X_full_enc, _ = prepare_features(X, state=state, numeric_cols=numeric_cols, cat_cols=cat_cols)
X_full_enc = X_full_enc.reindex(columns=X_train_enc.columns, fill_value=0.0)

final_model = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
final_model.fit(X_full_enc, y)

# Test predictions using actual model output
test_proba = final_model.predict_proba(test_enc)
test_pred = probability_to_score(test_proba, final_model.classes_, le.classes_)

# Create submission with required columns
submission = pd.DataFrame({
    'id': range(len(test_pred)),
    'num_countries_affected': test_pred
})

os.makedirs('./submission', exist_ok=True)
submission.to_csv('./submission/submission.csv', index=False)
print(f"Final Validation Score: {val_auc:.6f}")