from data.factory import get_data
from model.factory import create_model
import numpy as np

def predict():
  # Get data
  _, scaled_inputs, slope_output, intercept_output = get_data()

  # Create model and load weights
  head_model = create_model("strided", (299, 299, 3), (38, ))
  head_model.load_weights("weights.hdf5")

  # Predict
  predicted = head_model.predict(scaled_inputs, batch_size=20)
  slopeErrors = []
  interceptErrors = []
  for i in range(len(predicted[0])):
    slopeError = abs(predicted[0][i] - slope_output[i]) / slope_output[i]
    interceptError = abs(predicted[1][1] - intercept_output[i]) / intercept_output[i]
    slopeErrors.append(slopeError)
    interceptErrors.append(interceptError)
    print("Predicted: " + str(predicted[0][i]) + ". Actual: " + str(slope_output[i]) + ". Slope error: " + str(slopeError))
    print("Predicted: " + str(predicted[1][i]) + ". Actual: " + str(intercept_output[i]) + ". Intercept error: " + str(interceptError))

  print ("Average slope error: " + str(np.asarray(slopeErrors).mean()))
  print ("Average intercept error: " + str(np.asarray(interceptErrors).mean()))
  print ("Prediction complete!")