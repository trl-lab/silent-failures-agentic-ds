0-9_]", "_", str(c))
    if c_clean in seen:
        seen[c_clean] += 1
        c_clean = f"{c_clean}_{seen[c_clean]}"
    else:
        seen[c_clean] = 0
    clean_cols.append(c_clean)
df_all.columns = clean_cols

# Separate back into train and test
X_train_proc = df_all.iloc[:n_train].copy()
X_test_proc = df_all.iloc[n_train:].copy()

# Cross-validation model training
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(n_test)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_proc, y)):
    X_tr, y_tr = X_train_proc.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train_proc.iloc[val_idx], y[val_idx]

    clf = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8