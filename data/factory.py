import os
import csv
import numpy as np
from PIL import Image, ImageOps
from sklearn.preprocessing import MinMaxScaler

def get_data():
    # Get top roof images
  img_directory = "C:\\Images"
  top_file_name = "top.png"
  front_file_name = "front.png"
  img_size = 299, 299
  keys = os.listdir(img_directory)
  imgs = []
  for name in keys:
    top = Image.open(img_directory + "\\" + name + "\\" + top_file_name).convert("RGB")
    front = Image.open(img_directory + "\\" + name + "\\" + front_file_name).convert("RGB")

    top_fitted = ImageOps.fit(top, img_size, Image.ANTIALIAS)
    front_fitted = ImageOps.fit(front, img_size, Image.ANTIALIAS)

    top_arr = np.array(top_fitted)
    front_arr = np.array(front_fitted)

    combined = np.concatenate((top_arr, front_arr))
    imgs.append(combined)

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
  inputs = np.pad(images_trimmed, ((0, 0), (0, 0), (1, 0), (0, 0), (0, 0)), mode='constant')
  features_trimmed = np.array([list(x[1:]) for x in features])
  outputs = np.array([list(x[1:]) for x in y_actual])
  ft_shape = features_trimmed.shape
  it_shape = inputs.shape
  # adding features to image array
  for i in range(ft_shape[0]):
    padded = np.pad(features_trimmed[i][0][:].reshape((ft_shape[2], 1, 1)), [(0, 0), (0, it_shape[3] - 1), (0, it_shape[4] - 1)], mode='constant')
    inputs[i][0][:ft_shape[2]][:][:] = padded

  # Scale data to be between (0, 1)
  inputs = np.reshape(inputs, (it_shape[0], it_shape[2], it_shape[3], it_shape[4]))
  scaler_out = MinMaxScaler()
  scaled_inputs = np.interp(inputs, (inputs.min(), inputs.max()), (0, +1))
  scaled_outputs = scaler_out.fit_transform(outputs)

  return (inputs, outputs, scaled_inputs, scaled_outputs, scaler_out)