from model.factory import create_model
from data.factory import get_data

def train(epochs=250,
          save_weights=True):

  # Get training data
  _, _, scaled_inputs, scaled_outputs, _ = get_data()

  # Get model
  head_model = create_model("vgg19", (299, 299, 3), (38, ))

  # Train model and save weights
  head_model.fit(scaled_inputs, scaled_outputs, batch_size=20, epochs=epochs, verbose=2, shuffle=True, validation_split=0.2)
  if (save_weights):
    head_model.save_weights("weights.hdf5")
  print("Training complete!")