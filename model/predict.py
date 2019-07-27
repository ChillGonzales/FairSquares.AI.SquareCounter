from data.factory import get_data
from model.factory import create_model
import numpy as np

def predict():
  # Get data
  _, scaled_inputs, outputs = get_data()

  # Create model and load weights
  head_model = create_model("vgg19", (299, 299, 3), (38, ))
  head_model.load_weights("weights.hdf5")

  # Predict
  predicted = head_model.predict(scaled_inputs, batch_size=20)
  errors = []
  for i in range(len(predicted)):
    error = abs((predicted[i] - outputs[i])) / outputs[i]
    errors.append(error)
    # unscaled_accuracy = abs((predicted[i] - scaled_outputs[i])) / scaled_outputs[i]
    print("Predicted: " + str(predicted[i]) + ". Actual: " + str(outputs[i]) + ". Error: " + str(error))
  print ("Average error: " + str(np.asarray(errors).mean()))
  print("Prediction complete!")