import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier

# Load data
train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/forest-cover-type-prediction/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/forest-cover-type-prediction/data/test.parquet'

train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# Separate features and target
drop_cols = ['Id', 'Cover_Type'] if 'Id' in train_df.columns else ['Cover_Type']
X = train_df.drop(columns=drop_cols, errors='ignore')
y = train_df['Cover_Type']

test_drop_cols = ['Id'] if 'Id' in test_df.columns else []
X_test = test_df.drop(columns=test_drop_cols, errors='ignore')

# Feature Engineering
def engineer_features(df):
    df = df.copy()
    df['Euclidean_Distance_To_Hydrology'] = np.sqrt(
        df['Horizontal_Distance_To_Hydrology']**2 + df['Vertical_Distance_To_Hydrology']**2
    )
    df['Elevation_Minus_VDH'] = df['Elevation'] - df['Vertical_Distance_To_Hydrology']
    df['Elevation_Plus_VDH'] = df['Elevation'] + df['Vertical_Distance_To_Hydrology']
    df['Hydro_Fire_Sum'] = df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Fire_Points']
    df['Hydro_Fire_Diff'] = np.abs(df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Fire_Points'])
    df['Hydro_Road_Sum'] = df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Roadways']
    df['Hydro_Road_Diff'] = np.abs(df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Roadways'])
    df['Road_Fire_Sum'] = df['Horizontal_Distance_To_Roadways'] + df['Horizontal_Distance_To_Fire_Points']
    df['Road_Fire_Diff'] = np.abs(df['Horizontal_Distance_To_Roadways'] - df['Horizontal_Distance_To_Fire_Points'])
    return df

X = engineer_features(X)
X_test = engineer_features(X_test)

# Label encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Models
et = ExtraTreesClassifier(n_estimators=300, max_features='sqrt', random_state=42, n_jobs=-1)
rf = RandomForestClassifier(n_estimators=300, max_features='sqrt', random_state=42, n_jobs=-1)
lgb = LGBMClassifier(n_estimators=300, learning_rate=0.08, num_leaves=63, random_state=42, n_jobs=-1, verbose=-1)

# Fit models
et.fit(X, y_encoded)
rf.fit(X, y_encoded)
lgb.fit(X, y_encoded)

# Predict probabilities and ensemble
prob_et = et.predict_proba(X_test)
prob_rf = rf.predict_proba(X_test)
prob_lgb = lgb.predict_proba(X_test)

final_probs = 0.45 * prob_et + 0.30 * prob_rf + 0.25 * prob_lgb
preds_encoded = np.argmax(final_probs, axis=1)
predictions = le.inverse_transform(preds_encoded)

# Save predictions
output_df = pd.DataFrame({
    'id': range(len(predictions)),
    'Cover_Type': predictions
})
output_df.to_csv('predictions.csv', index=False)