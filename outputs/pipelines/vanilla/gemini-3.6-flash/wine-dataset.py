.sum() > 0.05 * len(combined):
      combined[f'{col}_ext_abv'] = pd.to_numeric(abvs, errors='coerce')

    # Volume size
    sizes = s.str.extract(r'(\d+(?:\.\d+)?)\s*(?:cl|ml|l)\b', expand=False)
    if sizes.notna().sum() > 0.05 * len(combined):
      combined[f'{col}_ext_size'] = pd.to_numeric(sizes, errors='coerce')

# Classify columns
num_cols = []
cat_cols = []
text_cols = []

for col in combined.columns:
  if pd.api.types.is_numeric_dtype(combined[col]):
    num_cols.append(col)
  elif pd.api.types.is_datetime64_any_dtype(combined[col]):
    combined[f'{col}_year'] = combined[col].dt.year
    combined[f'{col}_month'] = combined[col].dt.month
    combined[f'{col}_day'] = combined[col].dt.day
    num_cols.extend([f'{col}_year', f'{col}_month', f'{col}_day'])
  else:
    s = combined[col].astype(str).fillna('')
    mean_len = s.str.len().mean()
    nunique =