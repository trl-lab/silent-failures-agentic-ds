col}_year"] = df_all[col].dt.year
    df_all[f"{col}_month"] = df_all[col].dt.month
    df_all[f"{col}_day"] = df_all[col].dt.day
    df_all[f"{col}_dayofweek"] = df_all[col].dt.dayofweek
    df_all = df_all.drop(columns=[col])

# Cleaning missing value place-holders in HIFLD datasets (-999, -9999, etc.)
for col in df_all.columns:
  if pd.api.types.is_numeric_dtype(df_all[col]):
    df_all[col] = df_all[col].replace(
        [-999, -999.0, -9999, -9999.0, -999.00], np.nan
    )
    df_all[col] = df_all[col].replace([np.inf, -np.inf], np.nan)

# Column classification
num_cols = []
cat_cols = []
text_cols = []

for col in df_all.columns:
  if pd.api.types.is_numeric_dtype(df_all[col]):
    num_cols.append(col)
  else:
    nunique = df_all[col].nunique(dropna