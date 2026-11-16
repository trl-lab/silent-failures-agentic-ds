import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Set random seed for reproducibility
np.random.seed(42)

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/lending-club-loan/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/lending-club-loan/data/test.parquet')

# Identify columns
target_col = 'loan_status'
feature_cols = [c for c in train_df.columns if c != target_col]

# Initialize preprocessing
preprocessor = ColumnTransformer(transformers=[('num', 'passthrough', [])], remainder=None) # Placeholder for handling later

def preprocess(df):
    # Ensure all data is loaded correctly
    df = df.copy()
    
    # Remove target from features
    cols_to_drop = [target_col, 'id'] if 'id' in df.columns else [target_col]
    df = df.drop(columns=cols_to_drop)
    
    # Identify numeric and categorical columns
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    # Fill missing values
    for col in numerical_features:
        df[col].fillna(df[col].mean(), inplace=True)
    
    for col in categorical_features:
        df[col].fillna(df[col].mode()[0], inplace=True)
        
    return df

# Load and preprocess
raw_train = train_df.copy()
train_processed = preprocess(raw_train)

raw_test = test_df.copy()
test_processed = preprocess(raw_test)

# Prepare features
X_train = train_processed[feature_cols]
X_test = test_processed[feature_cols]

y_train = train_df[target_col].values

# Handle target column (Binarization for B-Classification)
unique_vals = np.unique(y_train)
if len(unique_vals) == 2:
    # Already binary or close enough
    is_binary = True
    y_train_ = y_train
else:
    # Binarily encode: Assume "Paid" or high frequency as positive for success
    # Heuristic: Search for 'paid' or specific success status in string values
    positive_label = None
    for val in unique_vals:
        str_val = str(val) if not isinstance(val, (int, float)) else str(val)
        if 'paid' in str_val.lower() or 'closed' in str_val.lower():
            positive_label = val
            break
    
    # Fallback: First value or most frequent
    if positive_label is None:
        positive_label = unique_vals[0]
        # If unique count is 3+, assume 0 is negative? No, standard is usually 'default'. 
        # Weights: Success=1.
        
    # If already integer:
    # e.g. 1 vs 0, 0 is bad, 1 is good.
    
    # Create binary mapping
    if positive_label is not None and not isinstance(positive_label, (int, float)):
        # Map strings to binary
        encoder = LabelEncoder()
        # Custom mapping for robustness
        label_map = {val: 1 for val in unique_vals}
        label_map[positive_label] = 1
        for v in unique_vals:
            if v not in label_map:
                label_map[v] = 0
        y_train_ = np.array([label_map[v] for v in y_train])
    else:
        # Try to extract binary values
        if len(np.unique(y_train)) == 2:
             pass
        else:
            # Assuming numeric 1/0 or 0/1
             if min(y_train) == 0 and max(y_train) == 1:
                 pass
            else:
                 # Fallback to simple binning based on majority
                 is_binary = True
                 y_train_ = y_train
                 print(f"Warning: Target could be non-standard, assuming existing mapping.")

# Feature Engineering: Expand categorical features
for col in categorical_features:
    # One Hot Encoding
    le = LabelEncoder()
    # Fit transform on train
    X_train.loc[:, f'{col}_encoded'] = le.fit_transform(X_train[col].astype(str))
    
    # Transform test
    temp_test = X_test[col].apply(lambda x: le.transform(x.astype(str)))
    X_test.insert(0, f'{col}_encoded', temp_test)

# Select new feature list including encoded columns
final_feature_list = []
temp_train = X_train
temp_test = X_test

new_feat_idx = 0
for idx, col in enumerate(feature_cols):
    # Original numeric stays as is
    temp_train[f'{col}_orig'] = X_train[col] if col in X_train.columns else None # Reconstruct properly
    
    # Add encoded categorical
    if col in categorical_features:
        encoded_col = f'{col}_onehot'
        if encoded_col not in X_train.columns:
             # This path logic is slightly complex in loop, simplified:
             pass
        # Actually, let's rebuild properly.
        break

# Restart preprocessing cleanly for feature creation
X_train_temp = train_processed.copy()
X_test_temp = test_processed.copy()

cat_cols = [c for c in X_train_temp.columns if X_train_temp[c].dtype == 'object']
num_cols = [c for c in X_train_temp.columns if X_train_temp[c].dtype == 'number']

for col in cat_cols:
    # Encode categorical
    le = LabelEncoder()
    train_enc = le.fit_transform(X_train_temp[col].astype(str))
    test_enc = le.transform(X_test_temp[col].astype(str))
    X_train_temp.insert(len(num_cols), f'enc_{col}', train_enc)
    X_test_temp.insert(len(num_cols), f'enc_{col}', test_enc)
    
    # Drop old object column to prevent leakage
    X_train_temp.drop(col, axis=1, inplace=True)
    X_test_temp.drop(col, axis=1, inplace=True)

# Combine
all_features = list(X_train_temp.columns)
X_final_train = X_train_temp[all_features]
X_final_test = X_test_temp[all_features]

# Normalize (optional for tree models but good practice)
# Tree models (LGBM/XGB/CatBoost) scale-invariant, skip.

# Train Multiple Models
from sklearn.model_selection import GridSearchCV
# Simple Grid Search not possible due to time/complexity, use default params optimized

models = []

# XGBoost
model_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='auc')
model_xgb.fit(X_final_train, y_train_)
models.append(('xgb', model_xgb))

# LightGBM
model_lgb = lgb.LGBMClassifier(use_label_encoder=False, eval_metric='auc')
model_lgb.fit(X_final_train, y_train_)
models.append(('lgb', model_lgb))

# CatBoost
model_cat = cb.CatBoostClassifier(iterations=100, loss_function='Logloss', verbose=0, device='CPU')
model_cat.fit(X_final_train, y_train_)
models.append(('cat', model_cat))

# Ensemble Average Predictions
pred_proba = np.zeros(len(test_df))

for name, model in models:
    probs = model.predict_proba(X_final_test)
    pred_proba += probs[:, 1] # Sum positive class probabilities

pred_proba /= len(models)

# Convert probabilities to class labels (argmax)
predicted_labels = np.argmax(pred_proba, axis=1)

# Construct DataFrame for output
output_data = {
    'id': range(len(test_df)),
    'loan_status': predicted_labels.astype(int) if isinstance(predicted_labels[0], int) else predicted_labels.astype(object)
}
output_df = pd.DataFrame(output_data)
output_df.to_csv('predictions.csv', index=False)