mean() > 0.5
      ):
        try:
          df_all[col] = pd.to_datetime(df_all[col])
          date_cols.append(col)
          continue
        except Exception:
          pass
      cat_cols.append(col)

  X_features = pd.DataFrame(index=df_all.index)

  # 1. Date features
  for col in date_cols:
    dt = pd.to_datetime(df_all[col])
    X_features[f"{col}_year"] = dt.dt.year
    X_features[f"{col}_month"] = dt.dt.month
    X_features[f"{col}_day"] = dt.dt.day
    X_features[f"{col}_dayofweek"] = dt.dt.dayofweek
    X_features[f"{col}_timestamp"] = np.where(
        dt.isna(), np.nan, dt.astype("int64") / 1e9
    )

  # 2. Numeric features
  for col in numeric_cols:
    X_features[col] = df_all[col].values
    if df_all[col].isna().sum() > 0:
      X_features[f"{col}_isna"] = df_all[col].isna().astype(int)

  # 3. Categorical features
  for col in cat_