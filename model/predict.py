from data.factory import get_data
from model.factory import create_model
import numpy as np

def predict():
  # Get data
  _, scaled_inputs, outputs = get_data()

  # Create model and load weights
  head_model = create_model("strided", (299, 299, 3), (38, ))
  head_model.load_weights("weights.hdf5")

  # Predict
  predicted = head_model.predict(scaled_inputs, batch_size=20)
  slopeErrors = []
  interceptErrors = []
  for i in range(len(predicted)):
    slopeError = abs(predicted[i][0] - outputs[i][0]) / outputs[i][0]
    interceptError = abs(predicted[i][1] - outputs[i][1]) / outputs[i][1]
    slopeErrors.append(slopeError)
    interceptErrors.append(interceptError)
    print("Predicted: " + str(predicted[i]) + ". Actual: " + str(outputs[i]) + ". Slope error: " + str(slopeError) + ". Intercept error: " + str(interceptError))

  print ("Average slope error: " + str(np.asarray(slopeErrors).mean()))
  print ("Average intercept error: " + str(np.asarray(interceptErrors).mean()))
  print("Prediction complete!")