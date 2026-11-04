= np.zeros((len(X_tr), len(cat_cols)), dtype=np.float32)
X_te_freq = np.zeros((len(X_te), len(cat_cols)), dtype=np.float32)

for i, col in enumerate(cat_cols):
  # Ordinal
  oe = OrdinalEncoder(
      handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.float32
  )
  X_tr_cat_encoded[:, i] = oe.fit_transform(
      X_tr[[col]].fillna('MISSING').astype(str)
  ).ravel()
  X_te_cat_encoded[:, i] = oe.transform(
      X_te[[col]].fillna('MISSING').astype(str)
  ).ravel()

  # Frequency
  freq_map = X_tr[col].fillna('MISSING').value_counts(normalize=True).to_dict()
  X_tr_freq[:, i] = (
      X_tr[col].fillna('MISSING').map(freq_map).fillna(0).values.astype(np.float32)
  )
  X_te_freq[:, i] = (
      X_te[col].fillna('MISSING').map(freq_map).fillna(0).values.astype(np.float32)
  )
```

And Numeric features: