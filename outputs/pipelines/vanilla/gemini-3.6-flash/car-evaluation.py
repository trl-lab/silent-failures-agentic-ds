import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

train_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/uci/car-evaluation/data/train.parquet')
test_df = pd.read_parquet('/path/to/user/workspace/ds_failures/datasets/uci/car-evaluation/data/test.parquet')

target_col = 'car'
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]

X_test = test_df.drop(columns=[target_col]) if target_col in test_df.columns else test_df.copy()

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train_enc = encoder.fit_transform(X_train.astype(str))
X_test_enc = encoder.transform(X_test.astype(str))

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)

models = [
    CatBoostClassifier(verbose=0, random_state=42, iterations=500),
    LGBMClassifier(random_state=42, verbose=-1, n_estimators=300),
    XGBClassifier(random_state=42, n_estimators=300, eval_metric='mlogloss'),
    RandomForestClassifier(n_estimators=300, random_state=42),
    ExtraTreesClassifier(n_estimators=300, random_state=42),
    HistGradientBoostingClassifier(random_state=42, max_iter=300)
]

probs = np.zeros((len(X_test_enc), len(le.classes_)))

for model in models:
    model.fit(X_train_enc, y_train_enc)
    probs += model.predict_proba(X_test_enc) / len(models)

preds_enc = np.argmax(probs, axis=1)
preds = le.inverse_transform(preds_enc)

output_df = pd.DataFrame({
    'id': np.arange(len(preds)),
    'car': preds
})
output_df.to_csv('predictions.csv', index=False)