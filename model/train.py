from model.factory import create_model
from data.factory import get_data
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

def train(epochs,
          save_weights=True):

  # Get training data
  _, scaled_inputs, outputs = get_data()

  # Get model
  head_model = create_model("strided", (299, 299, 3), (38, ))
  opt = Adam(lr=1e-5, decay=1e-6 / epochs)
  head_model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])

  # Train model and save weights
  checkpoint = ModelCheckpoint("saved-model-{epoch:02d}-{loss:.1f}-{val_loss:.1f}.hdf5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=15)
  stopping = EarlyStopping(monitor='val_loss', patience=300, verbose=1)
  history = head_model.fit(scaled_inputs, outputs, batch_size=32, epochs=epochs, verbose=2, shuffle=True, validation_split=0.25, callbacks=[checkpoint, stopping])
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