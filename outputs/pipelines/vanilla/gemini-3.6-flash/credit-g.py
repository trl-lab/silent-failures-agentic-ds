(X_tr_ohe, y_tr)
    preds_et += et.predict_proba(X_test_ohe)[:, 1] / 10.0

    lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    lr.fit(X_tr_ohe, y_tr)
    preds_lr += lr.predict_proba(X_test_ohe)[:, 1] / 10.0

final_preds = (
    0.25 * preds_cb +
    0.25 * preds_lgb +
    0.20 * preds_xgb +
    0.10 * preds_rf +
    0.10 * preds_et +
    0.10 * preds_lr
)

output_df = pd.DataFrame({
    'id': np.arange(len(final_preds)),
    'class': final_preds
})

output_df.to_csv('predictions.csv', index=False)
```

The code looks clean, robust, and directly executes without any prompts or external calls. Ready to output code directly.import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from