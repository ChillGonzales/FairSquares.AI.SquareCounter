def Normalize(df):
  result = df.copy()
  for feature_name in df.columns:
      max_value = df[feature_name].max()
      min_value = df[feature_name].min()
      if max_value - min_value == 0:
        continue
      result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
  return result