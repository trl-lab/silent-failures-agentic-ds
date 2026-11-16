import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

train_path = '/path/to/user/workspace/ds_failures/datasets/mlebench/nomad2018-predict-transparent-conductors/data/train.parquet'
test_path = '/path/to/user/workspace/ds_failures/datasets/mlebench/nomad2018-predict-transparent-conductors/data/test.parquet'

df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

target_col = 'formation_energy_ev_natom'

def create_features(df):
    df = df.copy()
    
    al = df['percent_atom_al'] if 'percent_atom_al' in df.columns else 0
    ga = df['percent_atom_ga'] if 'percent_atom_ga' in df.columns else 0
    in_ = df['percent_atom_in'] if 'percent_atom_in' in df.columns else 0
    
    metal_sum = al + ga + in_
    df['percent_atom_o'] = 1.0 - metal_sum
    
    df['rel_al'] =