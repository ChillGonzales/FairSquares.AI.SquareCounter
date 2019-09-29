from model.factory import create_model
from data.factory import get_data, DataKeys
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

def train(epochs: int, save_model: bool, test_split: float):
  # Get training data
  data = get_data(val_split=0.15, test_split=test_split, randomize=True)
  normalized_features_train = data[DataKeys.NormalizedFeaturesTrain]
  normalized_features_test = data[DataKeys.NormalizedFeaturesTest]
  normalized_output_train = data[DataKeys.NormalizedOutputTrain]
  normalized_output_test = data[DataKeys.NormalizedOutputTest]

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
    save_weights_only=True, mode='auto', period=100)
  stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=1)
  history = head_model.fit(
    x = {"image_input": normalized_features_train[0], "feature_input": normalized_features_train[1]},
    y = {"slope_output": normalized_output_train[0], "intercept_output": normalized_output_train[1]}, 
    validation_data=({"image_input": normalized_features_test[0], "feature_input": normalized_features_test[1]}, 
      {"slope_output": normalized_output_test[0], "intercept_output": normalized_output_test[1]}),
    batch_size=32, 
    epochs=epochs, 
    verbose=2, 
    shuffle=True, 
    callbacks=[checkpoint, stopping]
  )
  if (save_model):
    head_model.save("model.hdf5")

  print("Training complete!")
  # Plot training & validation loss values
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')
  plt.show()