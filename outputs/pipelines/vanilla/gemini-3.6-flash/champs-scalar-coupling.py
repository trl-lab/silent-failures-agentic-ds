te_sub = test[te_mask]
        
        drop_cols = ['id', 'scalar_coupling_constant', 'molecule_name']
        feature_cols = [c for c in tr_sub.columns if c not in drop_cols]
        
        # Identify categorical features
        cat_features = [c for c in feature_cols if tr_sub[c].dtype.name in ['category', 'object']]
        for c in cat_features:
            tr_sub[c] = tr_sub[c].astype('category')
            te_sub[c] = te_sub[c].astype('category')
            
        X = tr_sub[feature_cols]
        y = tr_sub['scalar_coupling_constant'].values
        X_test = te_sub[feature_cols]
        
        preds = np.zeros(len(te_sub), dtype=np.float32)
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        
        for tr_idx, va_idx in kf.split(X, y):
            X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
            X_va, y_va = X.iloc[va_idx], y[va_idx]
            
            model = lgb.LGBMRegressor(
                objective='regression_