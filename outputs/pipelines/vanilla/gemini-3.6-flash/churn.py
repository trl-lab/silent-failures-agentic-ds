import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMClassifier, early_stopping
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

def add_features(df):
    df = df.copy()
    cols_lower = {str(c).lower().replace(' ', '_').replace('-', '_'): c for c in df.columns}
    
    def get_col(name):
        return cols_lower.get(name, None)
    
    day_m = get_col('total_day_minutes')
    eve_m = get_col('total_eve_minutes')
    night_m = get_col('total_night_minutes')
    intl_m = get_col('total_intl_minutes')
    
    day_c = get_col('total_day_calls')
    eve_c = get_col('total_eve_calls')
    night_c = get_col('total_night_calls')
    intl_c = get_col('total_intl_calls')
    
    day_ch = get_col('total_day_charge')
    eve_ch = get_col('total_eve_charge')
    night_ch = get_col('total_night_charge')
    intl_ch = get_col('total_intl_charge')
    
    cust_calls