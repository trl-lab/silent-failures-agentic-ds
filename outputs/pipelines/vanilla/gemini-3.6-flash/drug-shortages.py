split(X_train, y)):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]
    
    # LightGBM
    if num_classes == 2:
        lgb_model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            random_state=42 + fold,
            n_jobs=-1,
            verbosity=-1
        )
    else:
        lgb_model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=num_classes,
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=31,
            random_state=42 + fold,
            n_jobs=-1,
            verbosity=-1
        )
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    p_lgb = lgb_model.predict_proba(X_test)