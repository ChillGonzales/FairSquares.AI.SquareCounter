import os
import csv
import numpy as np
import random
from PIL import Image, ImageOps
from utility.utility import NormalizeDataframe, NormalizeArray
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd

def get_data(val_split):
    # Get top roof images
  img_directory = "C:\\Images"
  top_file_name = "top.png"
  img_size = 299, 299
  keys = os.listdir(img_directory)
  imgs = []
  for name in keys:
    top = Image.open(img_directory + "\\" + name + "\\" + top_file_name).convert("RGB")
    top_fitted = ImageOps.fit(top, img_size, Image.ANTIALIAS)
    top_arr = np.array(top_fitted)
    imgs.append(top_arr)

  images = dict(sorted(zip(keys, imgs), key=lambda x: x[0]))

  # Get order data
  orders = pd.read_csv('C:\\Predictions\\predictions.csv')

  # Make sure our count of data is correct
  assert (len(images) == len(orders))
  print ("Training using " + str(len(images)) + " total orders.")

  # randomize orders dataframe
  orders = orders.sample(frac=1).reset_index(drop=True)
  splitIndex = int(len(orders) * val_split)

  features_train = orders.iloc[splitIndex:, :-2]
  features_test = orders.iloc[:splitIndex:, :-2]
  output_train = orders.iloc[splitIndex:, -2:]
  output_test = orders.iloc[:splitIndex, -2:]

  # Construct image inputs in same order as randomized feature inputs
  images_train = []
  images_test = []
  for order in features_train["order_id"]:
    images_train.append(images[str(order)])
  for order in features_test["order_id"]:
    images_test.append(images[str(order)])
  
  features_train.drop(labels="order_id", axis=1, inplace=True)
  features_test.drop(labels="order_id", axis=1, inplace=True)

  features_norm_train = NormalizeDataframe(features_train)
  features_norm_test = NormalizeDataframe(features_test)
  images_norm_train = NormalizeArray(np.array(images_train))
  images_norm_test = NormalizeArray(np.array(images_test))

  return ([images_train, features_train.values], [images_test, features_test.values],
  [images_norm_train, features_norm_train.values], [images_norm_test, features_norm_test.values], 
  [output_train["area_slope"].values, output_train["area_intercept"].values], [output_test["area_slope"].values, output_test["area_intercept"].values])