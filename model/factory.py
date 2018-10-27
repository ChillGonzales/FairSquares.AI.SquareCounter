import keras
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam

def create_model():
  # Keras model
  base_model = keras.applications.inception_v3.InceptionV3(include_top=False,
                                                           weights='imagenet',
                                                           input_tensor=None,
                                                           input_shape=(299, 299, 3),
                                                           pooling=None)
  for layer in base_model.layers:
      layer.trainable = False
  x = Flatten()(base_model.output)
  x = Dense(4096, activation='relu')(x)
  x = Dropout(0.5)(x)
  prediction = Dense(1, activation='linear')(x)
  head_model = Model(input=base_model.input, output=prediction)
  opt = Adam(lr = 0.0005)
  head_model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
  return head_model