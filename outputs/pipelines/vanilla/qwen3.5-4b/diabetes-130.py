import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import gc
warnings.filterwarnings('ignore')

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/uci/diabetes-130/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/uci/diabetes-130/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Column definitions
target_col = 'readmitted'
id_col = 'encounter_id'
exclude_cols = [id_col, target_col]

# All feature columns
all_feature_cols = [col for col in train_df.columns if col not in exclude_cols]

# Define categorical columns based on dataset description to handle high cardinality
categorical_feature_names = [
    'race', 'gender', 'age', 'weight', 'admission_type_id',
    'discharge_disposition_id', 'admission_source_id', 'payer_code',
    'medical_specialty', 'max_glu_serum', 'A1Cresult',
    'change', 'diabetesMed',
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
    'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton',
    'insulin', 'glyburide-metformin', 'glipizide-metformin',
    'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone',
    'diag_1', 'diag_2', 'diag_3'
]

# Filter categorical columns present in dataframe
actual_categorical_cols = [col for col in categorical_feature_names if col in all_feature_cols]
numerical_cols = [col for col in all_feature_cols if col not in actual_categorical_cols]

# Prepare Training Data
X_train = train_df[all_feature_cols].copy()
y_train = train_df[target_col].copy()

# Prepare Test Data
X_test = test_df[all_feature_cols].copy()

# Handle Missing Values
def impute_dataframe(df, categorical_list):
    df_filled = df.copy()
    for col in df.columns:
        if col in categorical_list:
            # Fill categorical with mode
            mask_nan = df_filled[col].isna()
            if mask_nan.any():
                mode_val = df_filled[col].mode()
                fill_val = mode_val.iloc[0] if len(mode_val) > 0 else 'missing'
                df_filled.loc[mask_nan, col] = fill_val
        else:
            # Fill numerical with median
            median_val = df_filled[col].quantile(0.5)
            fill_val = median_val
            if pd.isna(fill_val):
                fill_val = 0
            df_filled.loc[df_filled[col].isna(), col] = fill_val
    return df_filled

X_train_clean = impute_dataframe(X_train, categorical_feature_names)
X_test_clean = impute_dataframe(X_test, categorical_feature_names)

# Convert categorical columns to object dtype to ensure LightGBM treats them as categorical
for col in actual_categorical_cols:
    X_train_clean[col] = X_train_clean[col].astype(str)
    X_test_clean[col] = X_test_clean[col].astype(str)

# Feature indices for LightGBM categorical handling
categorical_indices = []
for idx, col in enumerate(all_feature_cols):
    if col in actual_categorical_cols:
        categorical_indices.append(idx)

# Initialize LightGBM Model
params = {
    'objective': 'multiclass',
    'num_classes': len(y_train.unique()),
    'metric': 'accuracy',
    'boosting_type': 'gbdt',
    'verbose': -1,
    'random_state': 42,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

train_data = lgb.Dataset(X_train_clean, label=y_train, categorical_feature=categorical_indices)
eval_data = train_data # Same for eval

model = lgb.train(params, train_data, num_boost_round=1000)

# Predictions
pred_proba = model.predict(X_test_clean, num_iteration=1000)
pred_probs = pred_proba.reshape(-1, len(y_train.unique()))
predictions = np.argmax(pred_probs, axis=1)

# Format Output
output_df = pd.DataFrame({
    'id': np.arange(len(predictions)),
    'readmitted': [y_train_unique[int(p)] for p in predictions]
}, dtype=str)
# Convert back to original category mapping if needed?
# Since predictions are indices (0, 1, ...), we map them to actual labels.
y_train_cats = sorted(list(y_train.drop_duplicates()))
label_map = {val: i for i, val in enumerate(sorted(list(set(y_train))))}
inv_label_map = {i: val for val, i in label_map.items()}

# Re-map prediction to actual class label string
final_predictions = [y_train_cats[pred] for pred in predictions]

output_final = pd.DataFrame({
    'id': np.arange(len(final_predictions)),
    'readmitted': final_predictions
})
output_final.to_csv('predictions.csv', index=False)