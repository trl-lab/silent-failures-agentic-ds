.reshape(-1, 1)
        )
        X_te_blocks.append(
            te_ser.astype(str)
            .map(cat_map)
            .fillna(-1)
            .values.astype(float)
            .reshape(-1, 1)
        )

    # Handle numeric
    else:
      tr_num = tr_ser.values.astype(float).reshape(-1, 1)
      te_num = te_ser.values.astype(float).reshape(-1, 1)
      med = np.nanmedian(tr_num)
      if np.isnan(med):
        med = 0.0
      X_tr_blocks.append(np.nan_to_num(tr_num, nan=med))
      X_te_blocks.append(np.nan_to_num(te_num, nan=med))

  X_tr = np.hstack(X_tr_blocks) if len(X_tr_blocks) > 0 else np.zeros((len(df_train), 1))
  X_te = np.hstack(X_te_blocks) if len(X_te_blocks) > 0 else np.zeros((len(df_test), 1))
  return X_tr, X_te


X_train, X_test = extract_features(train_df, test_df, feature_