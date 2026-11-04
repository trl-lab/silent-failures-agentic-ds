import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Load data
train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/openml/MiceProtein/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/openml/MiceProtein/data/test.parquet')

# Separate features and target
X_train = train_df.drop('class', axis=1).values
y_train_raw = train_df['class'].values
X_test = test_df.values

# Encode target labels
le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)

# Handle missing values with median imputation
imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

# Define models
lgbm = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)

xgb = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='mlogloss'
)

catboost = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    random_state=42,
    verbose=0
)

# Cross-validation to estimate performance
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("LightGBM CV accuracy:", cross_val_score(lgbm, X_train_imp, y_train, cv=skf, scoring='accuracy').mean())
print("XGBoost CV accuracy:", cross_val_score(xgb, X_train_imp, y_train, cv=skf, scoring='accuracy').mean())
print("CatBoost CV accuracy:", cross_val_score(catboost, X_train_imp, y_train, cv=skf, scoring='accuracy').mean())

# Train all models on full training data
lgbm.fit(X_train_imp, y_train)
xgb.fit(X_train_imp, y_train)
catboost.fit(X_train_imp, y_train)

# Ensemble via probability averaging
prob_lgbm = lgbm.predict_proba(X_test_imp)
prob_xgb = xgb.predict_proba(X_test_imp)
prob_cat = catboost.predict_proba(X_test_imp)

# Average probabilities
avg_prob = (prob_lgbm + prob_xgb + prob_cat) / 3.0

# Predict class indices
predicted_indices = np.argmax(avg_prob, axis=1)

# Decode back to original class labels
predicted_classes = le.inverse_transform(predicted_indices)

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(predicted_classes)),
    'class': predicted_classes
})
output_df.to_csv('predictions.csv', index=False)