# Memory

# Instructions


import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
train_data = pd.read_parquet("./input/train.parquet")
test_data = pd.read_parquet("./input/test.parquet")

# Preprocessing
# ...
# Feature Engineering
# ...
# CV
# ...
# Train
# ...
# Predict
# ...
