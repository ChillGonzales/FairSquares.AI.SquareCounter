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
  with open('predictions.csv') as file_reader:
    reader = csv.reader(file_reader)
    for row in reader:
      y_actual.append((row[0], float(row[1])))
  y_actual = sorted(y_actual, key= lambda x: x[0])

  # Make sure our count of data is correct
  assert (len(images) == len(y_actual))

  # Remove junk (id's) from input data
  inputs = np.array([list(x[1:]) for x in images])
  outputs = np.array([list(x[1:]) for x in y_actual])

  # Scale data to be between (0, 1)
  old_shape = inputs.shape
  inputs = np.reshape(inputs, (old_shape[0], old_shape[2], old_shape[3], old_shape[4]))
  scaler_out = MinMaxScaler()
  scaled_inputs = np.interp(inputs, (inputs.min(), inputs.max()), (0, +1))
  scaled_outputs = scaler_out.fit_transform(outputs)

  return (inputs, outputs, scaled_inputs, scaled_outputs, scaler_out)