import numpy as np

def Normalize(data):
  return (data - data.mean()) / data.std()