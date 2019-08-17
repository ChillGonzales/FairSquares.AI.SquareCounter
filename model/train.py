from model.factory import create_model
from data.factory import get_data
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

def train(epochs,
          save_weights=True):

  # Get training data
  _, _, scaled_train, scaled_test, output_train, output_test = get_data(val_split=0.1)

  # Get model
  head_model = create_model("strided", (299, 299, 3), (6, ))
  INIT_LR = 1e-5
  opt = Adam(lr=INIT_LR, decay=INIT_LR / epochs)
  losses = {
    "slope_output": "mean_squared_error",
    "intercept_output": "mean_squared_error"
  }
  lossWeights = { "slope_output": 1.0, "intercept_output": 1.0 }
  head_model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics=['accuracy'])

  # Train model and save weights
  checkpoint = ModelCheckpoint("saved-model-{epoch:02d}-{loss:.1f}-{val_loss:.1f}.hdf5", monitor='val_loss', verbose=0, save_best_only=True, 
    save_weights_only=True, mode='auto', period=30)
  stopping = EarlyStopping(monitor='loss', patience=50, verbose=1)
  history = head_model.fit(
    x=scaled_train, y={"slope_output": output_train[0], "intercept_output": output_train[1]}, 
    validation_data=(scaled_test, {"slope_output": output_test[0], "intercept_output": output_test[1]}),
    batch_size=32, epochs=epochs, verbose=2, shuffle=True, callbacks=[checkpoint, stopping])
  if (save_weights):
    head_model.save_weights("weights.hdf5")

  print("Training complete!")
  # Plot training & validation loss values
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')
  plt.show()