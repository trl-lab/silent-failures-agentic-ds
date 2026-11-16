.fit(
            X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50
        )
        fold_val_prob += model_cb.predict_proba(X_va)
        fold_test_prob += model_cb.predict_proba(X_test)
        model_count += 1
    except Exception as e:
        pass

    # Model 3: HistGradientBoosting
    try:
        model_hgb = HistGradientBoostingClassifier(
            max_iter=300, random_state=42 + fold
        )
        model_hgb.fit(X_tr, y_tr)
        fold_val_prob += model_hgb.predict_proba(X_va)
        fold_test_prob += model_hgb.predict_proba(X_test)
        model_count += 1
    except Exception as e:
        pass

    # Model 4: Logistic Regression on TF-IDF (if available)
    if tfidf_mat is not None:
        try:
            tf_tr = tfidf_train[trn_idx]
            tf_va = tfidf_train[val_idx]
            model_lr = LogisticRegression(
                C=2.0, max_iter=1000, solver="lbfgs", multi_class="auto"
            )
            model_lr.