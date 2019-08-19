import numpy as np

def NormalizeDataframe(df):
  if (len(df) == 0):
    return df, {}
  result = df.copy()
  range_values = {}
  for feature_name in df.columns:
      max_value = df[feature_name].max()
      min_value = df[feature_name].min()
      if max_value - min_value == 0:
        continue
      result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
      range_values[feature_name] = {"max": max_value, "min": min_value}
  return result, range_values

def DenormalizeWithRange(arr, range_values):
  if (len(arr) == 0):
    return
  result = []
  for item in arr:
    print (item)
    result.append(item * (range_values["max"] - range_values["min"]) + range_values["min"])
  return result

def NormalizeArray(arr):
  if (len(arr) == 0):
    return arr
  return (arr - arr.mean()) / arr.std()

