import os
import csv
import numpy as np
import random
from PIL import Image, ImageOps
from utility.utility import Normalize
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

  images = pd.DataFrame(sorted(zip(keys, imgs), key=lambda x: x[0]), columns=["order_id", "image"])
  print (images)

  # Get order data
  orders = pd.read_csv('C:\\Predictions\\predictions.csv')

  # Make sure our count of data is correct
  print ("Training using " + str(len(images)) + " total orders.")
  assert (len(images) == len(orders))

  # randomize orders data frame and join onto images
  orders = orders.sample(frac=1).reset_index(drop=True)
  combined = orders.join(images, on="order_id", how="left", lsuffix="_o", rsuffix="_i")
  combined.drop(labels=["order_id_o", "order_id_i"], axis=1, inplace=True)

  splitIndex = int(len(combined) * val_split)

  features_train = combined.iloc[splitIndex:, :-3]
  features_test = combined.iloc[:splitIndex:, :-3]
  output_train = combined.iloc[splitIndex:, -3:-1]
  output_test = combined.iloc[:splitIndex, -3:-1]
  images_train = combined.iloc[splitIndex:, [-1]]
  images_test = combined.iloc[:splitIndex, [-1]]

  features_norm_train = Normalize(features_train)
  features_norm_test = Normalize(features_test)
  images_norm_train = Normalize(images_train)
  images_norm_test = Normalize(images_test)

  result = []
  for r in images_train["image"]:
    print (r)
    result.append(r)
  print (np.shape(result))

  return ([images_train.values, features_train.values], [images_test.values, features_test.values],
  [images_norm_train.values, features_norm_train.values], [images_norm_test.values, features_norm_test.values], 
  [output_train["area_slope"].values, output_train["area_intercept"].values], [output_test["area_slope"].values, output_test["area_intercept"].values])