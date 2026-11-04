= np.argmax(test_probs, axis=1)
preds = le.inverse_transform(pred_labels)

# Submission DataFrame
sub = pd.DataFrame({'id': np.arange(len(test)), 'Cover_Type': preds})

sub.to_csv('predictions.csv', index=False)
```

Let's double-check all requirements:
1. "Loads the train and test data from the parquet files above using pandas" -> Checked.
2. "Trains a model to predict 'Cover_Type'" -> Checked.
3. "Saves predictions to predictions.csv in the current working directory, with exactly two columns: 'id' (integer row index starting from 0) and 'Cover_Type'" -> Checked.
4. "Output only the Python code, no explanation or markdown fences" -> Checked.

Everything looks solid, bug-free, robust, and concise.import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

train_path = '/path/to/user/workspace/ds_failures/datasets/kaggle/forest-cover-type-prediction/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets