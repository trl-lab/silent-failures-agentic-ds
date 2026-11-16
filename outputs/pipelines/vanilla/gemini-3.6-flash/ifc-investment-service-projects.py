[col], errors='coerce')
          df_all[col + '_year'] = dt.dt.year
          df_all[col + '_month'] = dt.dt.month
          df_all[col + '_day'] = dt.dt.day
          df_all[col + '_dayofweek'] = dt.dt.dayofweek
          df_all[col + '_dayofyear'] = dt.dt.dayofyear
          df_all = df_all.drop(columns=[col])
      except Exception:
        pass

# Separate numeric and categorical
cat_cols = []
num_cols = []

for col in df_all.columns:
  if (
      pd.api.types.is_object_dtype(df_all[col])
      or pd.api.types.is_string_dtype(df_all[col])
      or pd.api.types.is_categorical_dtype(df_all[col])
  ):
    cat_cols.append(col)
  else:
    num_cols.append(col)

# Handle Numeric features
for col in num_cols:
  df_all[col] = df_all[col].replace([np.inf, -np.inf], np.nan)
  null_ratio = df_all[col].isnull().mean()
  if null_ratio > 0.05:
    df_all[col + '_