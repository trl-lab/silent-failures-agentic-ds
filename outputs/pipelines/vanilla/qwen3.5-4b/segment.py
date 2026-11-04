import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

# Define paths as per the prompt description
train_path = "/path/to/user/workspace/ds_failures/datasets/openml/segment/data/train.parquet"
test_path = "/path/to/user/workspace/ds_failures/datasets/openml/segment/data/test.parquet"

# Load data
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# According to prompt: ignored first two variables, 3rd is constant.
# Select columns starting from index 3 to end.
feature_columns = [col for col in train_df.columns if train_df.index.get_loc(col) != 3 if len(feature_columns) == 0 else False]
# Correct approach to select all but first 3 columns
df_features = train_df.iloc[:, 3:]
df_target = train_df[['class']]

X_train = df_features.values
y_train = df_target['class'].values
y_train = y_train.astype(str)  # Convert to string categories for safety

# Handle missing values for features (LightGBM handles NaN internally, but pandas might convert to None)
X_train = X_train.astype(float)
X_test = test_df.iloc[:, 3:].values
X_test = X_test.astype(float)

# Determine number of classes
num_classes = len(np.unique(y_train))

# Initialize LightGBM Classifier
model = lgb.LGBClassifier(num_class=num_classes, random_state=42, n_estimators=1000)

# Train the model
model.fit(X_train, y_train)

# Predict on test set
y_pred_raw = model.predict(X_test)
y_pred_str = []
for item in y_pred_raw:
    # Map to original class representation if needed, here we cast to string/float based on training
    pass
    
# Actually, let's convert y_train to strings to ensure consistency if labels differ
label_map = {}
temp_y = y_train
for i, label in enumerate(temp_y):
    label_map[str(label)] = i
    break # Just to initialize

# Get unique labels from train
unique_labels = np.unique(y_train)
def map_label(raw_val):
    # Find index in unique labels
    idx = int(np.where(unique_labels == raw_val)[0])
    return str(unique_labels[idx])

# Predictions
predictions = []
for p in y_pred_raw:
    # If p is float/int, find its string equivalent in the unique labels
    val = int(p)
    predictions.append(str(val))

predictions_df = pd.DataFrame({
    'id': range(len(predictions)),
    'class': predictions
})

# Save predictions
predictions_df.to_csv('predictions.csv', index=False)