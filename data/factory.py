import os
import csv
import numpy as np
from PIL import Image, ImageOps
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
  with open('predictions.csv') as file_reader:
    reader = csv.reader(file_reader)
    for row in reader:
      y_actual.append((row[0], float(row[-1])))
      features.append((row[0], row[1:-1]))
  y_actual = sorted(y_actual, key= lambda x: x[0])
  features = sorted(features, key= lambda x: x[0])

  # Make sure our count of data is correct
  assert (len(images) == len(y_actual))
  assert (len(features) == len(images))

  # Remove junk (id's) from input data
  images_trimmed = np.array([list(x[1:]) for x in images])
  features_trimmed = np.array([list(x[1:]) for x in features])
  outputs = np.array([list(x[1:]) for x in y_actual])
  ft_shape = features_trimmed.shape
  im_shape = images_trimmed.shape

  # Scale data to be between (0, 1)
  image_input = np.reshape(images_trimmed, (im_shape[0], im_shape[2], im_shape[3], im_shape[4]))
  feature_input = np.reshape(features_trimmed, (ft_shape[0], ft_shape[2])).astype("int")
  scaler_out = MinMaxScaler()
  scaled_inputs = np.interp(image_input, (image_input.min(), image_input.max()), (0, +1))
  scaled_features = np.interp(feature_input, (feature_input.min(), feature_input.max()), (0, +1))
  scaled_outputs = scaler_out.fit_transform(outputs)

  return ([image_input, feature_input], outputs, [scaled_inputs, scaled_features], scaled_outputs, scaler_out)