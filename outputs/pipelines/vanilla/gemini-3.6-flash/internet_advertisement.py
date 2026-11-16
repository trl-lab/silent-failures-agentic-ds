def extract_target(df):
    target = df['class']
    if np.issubdtype(target.dtype, np.number) or target.dtype == bool:
        return target.astype(int).values
    else:
        s = target.astype(str).str.strip().str.lower().str.rstrip('.')
        return s.isin(['ad', '1', 'true', 'yes']).astype(int).values