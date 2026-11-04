)
  ridge.fit(X_tr_ohe, y_tr)
  oof_preds_ridge[val_idx] = expit(ridge.decision_function(X_va_ohe))
  test_preds_ridge += expit(ridge.decision_function(X_test_ohe)) / 5.0

  # 4. CatBoost Classifier on raw features
  X_tr_cb = train.iloc[train_idx][feature_cols]
  X_va_cb = train.iloc[val_idx][feature_cols]

  cb = CatBoostClassifier(
      iterations=600,
      learning_rate=0.08,
      depth=6,
      eval_metric='AUC',
      cat_features=feature_cols,
      random_seed=42 + fold,
      verbose=0,
      thread_count=-1,
  )
  cb.fit(X_tr_cb, y_tr, eval_set=(X_va_cb, y_va), early_stopping_rounds=50)

  oof_preds_cb[val_idx] = cb.predict_proba(X_va_cb)[:, 1]
  test_preds_cb += cb.predict_proba(test[feature_cols])[:, 1] / 5.0
```

Wait! Let's check how CatBoost handles `cat_features` when passed as column names