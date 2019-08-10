import os
import csv
import numpy as np
from PIL import Image, ImageOps
from utility.utility import Normalize
from sklearn.preprocessing import MinMaxScaler

def get_data():
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

  images = sorted(zip(keys, imgs), key= lambda x: x[0])

  # Get predictions
  y_actual=[]
  features=[]
  with open('C:\\Predictions\\predictions.csv') as file_reader:
    reader = csv.reader(file_reader)
    for row in reader:
      y_actual.append((row[0], [float(i) for i in row[-2:]]))
      features.append((row[0], row[1:-2]))
  y_actual = sorted(y_actual, key= lambda x: x[0])
  features = sorted(features, key= lambda x: x[0])

  # Make sure our count of data is correct
  print (len(images))
  print (len(y_actual))
  assert (len(images) == len(y_actual))
  assert (len(features) == len(images))

  # Remove junk (id's) from input data
  images_trimmed = np.array([list(x[1:]) for x in images])
  features_trimmed = np.array([list(x[1:]) for x in features])
  outputs = np.array([list(x[1:]) for x in y_actual])
  out_shape = outputs.shape
  outputs = np.reshape(outputs, (out_shape[0], out_shape[-1]))
  ft_shape = features_trimmed.shape
  im_shape = images_trimmed.shape
  print (im_shape)

  # Scale data to be between (0, 1)
  image_input = np.reshape(images_trimmed, (im_shape[0], im_shape[2], im_shape[3], im_shape[4]))
  feature_input = np.reshape(features_trimmed, (ft_shape[0], ft_shape[2])).astype("int")
  scaled_inputs = Normalize(image_input)
  scaled_features = Normalize(feature_input)

  return ([image_input, feature_input], [scaled_inputs, scaled_features], outputs)