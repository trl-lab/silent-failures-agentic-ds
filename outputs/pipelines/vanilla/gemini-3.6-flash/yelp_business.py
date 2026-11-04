(len(X_train), num_classes))
oof_probs_cat = np.zeros((len(X_train), num_classes))
oof_probs_xgb = np.zeros((len(X_train), num_classes))

oof_preds_lgb_reg = np.zeros(len(X_train))
oof_preds_cat_reg = np.zeros(len(X_train))
oof_preds_xgb_reg = np.zeros(len(X_train))

test_probs_lgb = np.zeros((len(X_test), num_classes))
test_probs_cat = np.zeros((len(X_test), num_classes))
test_probs_xgb = np.zeros((len(X_test), num_classes))

test_preds_lgb_reg = np.zeros(len(X_test))
test_preds_cat_reg = np.zeros(len(X_test))
test_preds_xgb_reg = np.zeros(len(X_test))
```

Let's check execution speed:
Is 5-fold CV fast enough?
If dataset size is ~10,000 to 100,000 rows, 5 folds with 300-500 trees in LightGBM and CatBoost takes less than 1-2 minutes!
To be extra safe on time and resource usage:
- `n_estimators` = 40