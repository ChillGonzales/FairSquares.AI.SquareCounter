import keras
import numpy as np
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model
from PIL import Image
import os
import csv

def main():
  # Get top roof images
  img_directory = "C:\\Images"
  img_file_name = "top.png"
  img_size = 299, 299
  keys = os.listdir(img_directory)
  imgs = []
  for name in keys:
    img = Image.open(img_directory + "\\" + name + "\\" + img_file_name).convert("RGB")
    img.thumbnail(img_size, Image.ANTIALIAS)
    arr = np.array(img)
    padded = np.pad(arr, ((120, 0), (0,0), (0,0)), 'constant')
    imgs.append(padded)

  images = sorted(zip(keys, imgs), key= lambda x: x[0])

  # Get predictions
  y_actual=[]
  with open('predictions.csv') as file_reader:
    reader = csv.reader(file_reader)
    for row in reader:
      y_actual.append((row[0], float(row[1])))
  y_actual = sorted(y_actual, key= lambda x: x[0])

  assert (len(images) == len(y_actual))
  # Keras model
  base_model = keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_tensor=None, input_shape=(299, 299, 3), pooling=None)
  for layer in base_model.layers:
    layer.trainable = False
  x = Flatten()(base_model.output)
  x = Dense(4096, activation='relu')(x)
  x = Dropout(0.5)(x)
  prediction = Dense(1, activation='linear')(x)

  inputs = np.array([list(x[1:]) for x in images])
  outputs = np.array([list(x[1:]) for x in y_actual])

  print(inputs)
  print(outputs)
  head_model = Model(input=base_model.input, output=prediction)
  head_model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])
  for i in range(len(inputs)):
    head_model.fit(inputs[i], outputs[i], batch_size=1, epochs=10, verbose=2)

if __name__ == "__main__":
  main()