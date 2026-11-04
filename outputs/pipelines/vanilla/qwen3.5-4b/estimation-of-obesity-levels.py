import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import catboost as cb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/estimation-of-obesity-levels/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/uci/estimation-of-obesity-levels/data/test.parquet"

def main():
    try:
        # Load data
        train_df = pd.read_parquet(TRAIN_PATH)
        test_df = pd.read_parquet(TEST_PATH)
        
        # Identify columns
        target_col = "NObeyesdad"
        feature_cols = [col for col in train_df.columns if col != target_col]
        
        # Separate X and y
        X_train = train_df[feature_cols].copy()
        y_train = train_df[target_col].copy()
        X_test = test_df[feature_cols].copy()
        
        # Identify categorical and numeric columns for preprocessing
        cat_cols = []
        num_cols = []
        for col in feature_cols:
            if isinstance(X_train[col].dtype, pd.api.types.CategoricalDtype) or str(X_train[col].dtype) == 'object':
                cat_cols.append(col)
            else:
                num_cols.append(col)
        
        # Preprocessing Pipeline
        transformer = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())  # Optional scaling
                ]), num_cols),
                ('cat', Pipeline([
                    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]), cat_cols)
            ])
        
        # Combine for training
        X_train_processed = transformer.fit_transform(X_train)
        y_train_processed = y_train.values
        
        # Use LightGBM or Random Forest. Let's use LightGBM for efficiency with pre-installed lib.
        # LightGBM does not support OneHotEncoder pipelines directly out of box easily without extra configs, 
        # but it works fine with dense array from Pipeline.
        # However, to ensure robustness with 'cat_cols' identification, Random Forest is a safe sklearn fallback.
        # Given CatBoost is also listed, it is better for auto-categoricals. But to avoid config issues with mixed types 
        # without explicit 'cat_features' flags which might break if inferred types differ:
        # We'll stick to Random Forest (sklearn) as it accepts the dense transformed matrix robustly.
        
        clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        clf.fit(X_train_processed, y_train_processed)
        
        # Predict
        y_pred_proba = clf.predict(X_test_processed)
        
        # Generate Output DF
        results_df = pd.DataFrame({
            'id': range(len(X_test)),
            'NObeyesdad': y_pred_proba
        })
        
        results_df.to_csv("predictions.csv", index=False)
        print(f"Predictions saved to predictions.csv")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()