_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)]
        )
        p_lgb = lgb_model.predict_proba(X_test)
        if num_classes == 2 and p_lgb.shape[1] == 1:
            p_lgb = np.column_stack([1 - p_lgb[:, 0], p_lgb[:, 0]])
        lgb_test_preds += p_lgb / skf.n_splits

        # CatBoost
        if num_classes > 2:
            cb_model = CatBoostClassifier(
                iterations=600,
                learning_rate=0.05,
                depth=6,
                loss_function='MultiClass',
                random_seed=42 + fold,
                verbose=0
            )
        else:
            cb_model = CatBoostClassifier(
                iterations=600,
                learning_rate=0.05,
                depth=6,
                loss_function='Logloss',
                random_seed=42 + fold,
                verbose=0
            )
        cb_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=40)
        p_cb = cb_model.predict_proba(X