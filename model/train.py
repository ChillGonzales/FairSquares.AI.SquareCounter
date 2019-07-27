from model.factory import create_model
from data.factory import get_data
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

def train(epochs=250,
          save_weights=True):

  # Get training data
  _, scaled_inputs, outputs = get_data()

  # Get model
  head_model = create_model("strided", (299, 299, 3), (38, ))
  opt = Adam(lr=1e-4, decay=1e-4 / epochs)
  head_model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])

  # Train model and save weights
  checkpoint = ModelCheckpoint("saved-model-{epoch:02d}-{val_acc:.2f}.hdf5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=2)
  stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3, verbose=0, mode='auto')
  head_model.fit(scaled_inputs, outputs, batch_size=20, epochs=epochs, verbose=2, shuffle=True, validation_split=0.2, callbacks=[checkpoint, stopping])
  if (save_weights):
    head_model.save_weights("weights.hdf5")
  print("Training complete!")