=n_comp, random_state=42)
                svd_feats = svd.fit_transform(tfidf_mat)
                for i in range(n_comp):
                    processed_features[f"svd_{i}"] = svd_feats[:, i]
        except Exception:
            pass

    for col in processed_features.columns:
        if processed_features[col].isnull().any():
            processed_features[col] = processed_features[col].fillna(processed_features[col].median())

    # Sanitize column names
    sanitized_cols = []
    seen = {}
    for c in processed_features.columns:
        clean_c = re.sub(r'\W+', '_', str(c))
        if clean_c in seen:
            seen[clean_c] += 1
            sanitized_cols.append(f"{clean_c}_{seen[clean_c]}")
        else:
            seen[clean_c] = 0
            sanitized_cols.append(clean_c)
    processed_features.columns = sanitized_cols

    X_train = processed_features.iloc[:n_train].copy()
    X_test = processed_features.iloc[n_train:].copy()

    min_class_count = pd.Series(y_encoded).value_counts().min()
    n_splits = 5
    if min_class_count >=