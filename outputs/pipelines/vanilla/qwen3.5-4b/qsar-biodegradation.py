import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier

# Define paths as per dataset specification
TRAIN_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/qsar-biodegradation/data/train.parquet"
TEST_PATH = "/path/to/user/workspace/ds_failures/datasets/openml/qsar-biodegradation/data/test.parquet"

# Load data
df_train = pd.read_parquet(TRAIN_PATH)
df_test = pd.read_parquet(TEST_PATH)

# Target column identification
TARGET_COL = 'Class'
# Feature columns (all except target)
FEATURE_COLS = [col for col in df_train.columns if col != TARGET_COL]

# Extract features and target
X_train = df_train[FEATURE_COLS].copy()
y_train = df_train[TARGET_COL].copy()
X_test = df_test[FEATURE_COLS].copy()
y_test = df_test[TARGET_COL].copy()

# Automatic handling of missing values and non-numeric (categorical) columns
# Attempt to convert all features to numeric; coerce errors to NaN
X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')

# Handle resulting NaN values by filling with column medians (robust estimation)
X_train = X_train.fillna(X_train.median(axis=0))
X_test = X_test.fillna(X_test.median(axis=0))

# Initialize and train model
model = LGBMClassifier(
    random_state=42,
    n_jobs=-1,
    max_depth=5,
    learning_rate=0.1
)
model.fit(X_train, y_train)

# Predict on test set
y_prob = model.predict_proba(X_test)

# Determine class labels from training set for mapping
train_classes = sorted(pd.unique(y_train))
test_classes = sorted(pd.unique(y_test))

# Combine classes if needed (should be consistent in proper setup)
all_classes = sorted(list(set(train_classes + test_classes)))

# Map predicted probabilities to class labels
predicted_labels_indices = np.argmax(y_prob, axis=1)
final_predictions = []
for idx in predicted_labels_indices:
    # Find closest matching class index from train classes, considering sorted alignment
    matched_class = None
    for i, cls in enumerate(all_classes):
        if all_classes[i] == train_classes[idx]:
            matched_class = cls
            break
    if matched_class is None:
        matched_class = all_classes[predicted_labels_indices[idx] % len(all_classes)] if len(all_classes) > 0 else "Unknown"
    else:
        matched_class = train_classes[min(idx % len(train_classes), len(train_classes)-1)] if len(train_classes) > 0 else "Unknown"
        
# Alternative mapping strategy: Assume sorted unique lists align.
# Safe mapping using direct search on train_classes
safe_map = {i: v for i, v in enumerate(sorted(train_classes))}
try:
    sorted_predict_classes = [train_classes[np.argmin([abs(p-i) for i, _ in enumerate(sorted(train_classes))])] for p in predicted_labels_indices]
except:
    # Fallback simple argmax mapping
    sorted_predict_classes = []
    train_list = sorted(train_classes)
    for p_idx in predicted_labels_indices:
        # Find nearest class index
        if p_idx >= len(train_list):
            continue
        elif p_idx < 0:
            continue
        else:
            sorted_predict_classes.append(train_list[p_idx])

# Correct Prediction Mapping Logic for Argmax
# LGBM predict_proba returns array size len(classes_in_training)
# We need to know what y_train classes correspond to indices.
# Since train_classes is sorted unique y_train, index i maps to train_classes[i]
train_classes_sorted = sorted(list(y_train.drop_duplicates()))
prob_map = [t for t in train_classes_sorted]

# Re-calculate predictions ensuring alignment with train_classes
final_predictions = []
for p_idx in np.argmax(y_prob, axis=1):
    if p_idx < len(prob_map):
        final_predictions.append(prob_map[p_idx])
    else:
        # Fallback if test distribution differs slightly (rare in splits)
        final_predictions.append("Unknown")

# Create output dataframe
ids = list(range(len(df_test)))
output_df = pd.DataFrame({
    'id': ids,
    'Class': final_predictions
})

# Save to CSV
output_df.to_csv('predictions.csv', index=False)