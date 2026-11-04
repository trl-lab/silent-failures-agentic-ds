import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
import lightgbm as lgb
from collections import defaultdict

def process_data(X, y):
    # Ensure numerical target
    if isinstance(y[0], (np.ndarray, str)):
        y = y.map(lambda x: float(x))
    else:
        y = y.astype('float')

    col_to_drop = ['id'] if 'id' in X.columns else []
    
    # Prepare dataframes
    X = X.drop(columns=col_to_drop)
    
    # Check for dtypes
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    return X, cat_cols, num_cols, y

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/meta-critic_whisky/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/strable/meta-critic_whisky/data/test.parquet')

# Initialize target
y_train = train_df['Meta Critic']
X_train = train_df.drop(columns=['Meta Critic'])
y_test_pred_target = None  # We predict for test

# Preprocessor for training
numeric_features, categorical_features = [], []
for col in X_train.columns:
    if X_train[col].dtype in ['int64', 'float64']:
        numeric_features.append(col)
    elif X_train[col].dtype == 'object':
        categorical_features.append(col)
    else:
        numeric_features.append(col)

numeric_preproc = ColumnTransformer([
    ('num_impute', SimpleImputer(strategy='median'), numeric_features)
], remainder='passthrough')

categorical_preproc = ColumnTransformer([
    ('cat_onehot', OneHotEncoder(drop='if_binary', sparse_output=False), categorical_features)
])

full_pipeline = Pipeline(steps=[
    ('cat_num', ColumnTransformer(
        [
            ('num_impute', SimpleImputer(strategy='median'), numeric_features),
            ('cat_onehot', OneHotEncoder(drop='if_binary', sparse_output=False), categorical_features)
        ]
    ))
])

model = lgb.LGBMRegressor(
    objective='regression',
    boosting_type='gbdt',
    num_leaves=31,
    learning_rate=0.1,
    n_estimators=500,
    subsample_for_bin=2**16,
    verbosity=-1,
    force_col_wise=False
)

model.fit(full_pipeline.fit_transform(X_train), y_train.values)

preds = full_pipeline.transform(test_df).apply(lambda x: x.to_numpy())
actual_preds = model.predict(preds)
final_preds = model.predict(pretrained_model=full_pipeline.transform(test_df))

# Fix prediction mapping
y_train = np.array([i for i in range(len(y_train))], dtype=int)
X_train = X_train.iloc[:len(y_train)]
X_test = test_df

feature_names = X_train.columns.tolist()

# Re-run fit properly since above was fragmented logic
final_y_train = y_train.values
final_model = lgb.LGBMRegressor(objective='regression', verbose=-1)

# Correct Logic
X_tr = train_df.copy()
y_tr = train_df['Meta Critic'].values

X_te = test_df.copy()

# Extract Target
y_pred_target = None

# Construct Features Matrix
cat_features = []
num_features = []
for col in X_tr.columns:
    if col != 'Meta Critic':
        if str(X_tr[col].dtype) == 'object':
            cat_features.append(col)
        elif str(X_tr[col].dtype) in ['int64', 'float64']:
            num_features.append(col)

num_imp = SimpleImputer(strategy='median')
cat_enc = OneHotEncoder(sparse_output=False, drop='if_binary')
trans_fom = ColumnTransformer([('num', num_imp, num_features), ('cat', cat_enc, cat_features)])

# Fit and Train
model_reg = lgb.LGBMRegressor(objective='regression', max_depth=-1, learning_rate=0.05, n_estimators=1000, random_state=42)
final_X = trans_fom.fit(X_tr[['id']].append(X_tr.drop(columns='Meta Critic'))).transform(train_df.drop(columns='Meta Critic'))
y_vals = final_X.reshape(-1, 1).flatten() if hasattr(final_X, 'reshape') else final_X.flatten()

# Re-train cleanly
from sklearn.metrics import mean_squared_error

X_tr_final = X_tr[['id']].copy()
for feat in X_tr_final.index:
    if feat not in X_tr_final.columns:
        X_tr_final.loc[:, feat] = 0

X_train_clean = X_tr_final.drop(columns=['id'])

cat_feats_list = []
num_feats_list = []
cols = X_train_clean.columns
for c in cols:
    if X_train_clean[c].dtype.name in ['int64', 'float64']:
        num_feats_list.append(c)
    else:
        cat_feats_list.append(c)

preproc_full = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), num_feats_list),
    ('cat', OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown='ignore'), cat_feats_list)
])

pipeline_model = Pipeline([
    ('feats', preproc_full),
    ('model', lgb.LGBMRegressor(objective='regression', boosting_type='gbdt', learning_rate=0.1, n_estimators=500, verbose=-1, force_col_wise=False))
])

pipeline_model.fit(X_tr_clean, y_vals.values)

preds_val = pipeline_model.predict(test_df.drop(columns=['id']))

# Final Output Generation
preds_array = np.array(preds_val)

# Prepare CSV
output_df = pd.DataFrame({
    'id': range(len(test_df)),
    'Meta Critic': preds_array
})

output_df.to_csv('predictions.csv', index=False)

# Cleanup
del pipeline_model
del trans_fom