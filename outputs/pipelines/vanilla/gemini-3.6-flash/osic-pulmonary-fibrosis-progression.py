/base_Percent if not set from FVC
  if (
      'Percent' in df.columns
      and 'Weeks' in df.columns
      and 'base_Percent' not in df.columns
  ):
    known_pct = df[df['Percent'].notnull()].copy()
    if not known_pct.empty:
      known_pct['abs_weeks'] = known_pct['Weeks'].abs()
      base_idx = known_pct.groupby('Patient')['abs_weeks'].idxmin()
      base_pct_df = known_pct.loc[
          base_idx, ['Patient', 'Percent', 'Weeks']
      ].rename(columns={'Percent': 'base_Percent', 'Weeks': 'base_Weeks'})
      df = df.merge(base_pct_df, on='Patient', how='left')

  if 'Weeks' in df.columns:
    patient_min_w = df.groupby('Patient')['Weeks'].transform('min')
    patient_max_w = df.groupby('Patient')['Weeks'].transform('max')
    df['min_week'] = patient_min_w
    df['max_week'] = patient_max_w
    df['week_range'] = patient_max_w - patient_min_w
    df['week_count'] = df.groupby('Patient')['Weeks'].transform('count')

if 'base_Weeks' in df.columns and 'Weeks' in