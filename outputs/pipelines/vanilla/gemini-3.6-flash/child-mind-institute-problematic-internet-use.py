+= model_lgb_reg.predict(X_test) / n_splits

  # LGBM Classifier expected value
  model_lgb_cls = lgb.LGBMClassifier(
      n_estimators=500,
      learning_rate=0.03,
      num_leaves=31,
      subsample=0.8,
      colsample_bytree=0.8,
      random_state=42,
      verbose=-1,
  )
  model_lgb_cls.fit(X_tr, y_tr)
  cls_labels = model_lgb_cls.classes_
  oof_lgb_cls[val_idx] = np.sum(
      model_lgb_cls.predict_proba(X_va) * cls_labels, axis=1
  )
  test_lgb_cls += (
      np.sum(model_lgb_cls.predict_proba(X_test) * cls_labels, axis=1) / n_splits
  )

  # XGBoost Regressor
  model_xgb_reg = xgb.XGBRegressor(
      n_estimators=500,
      learning_rate=0.03,
      max_depth=5,
      subsample=0.8,
      colsample_bytree=0.8,
      random_state=42,
      tree_method="hist