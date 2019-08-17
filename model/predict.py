from data.factory import get_data
from model.factory import create_model
import numpy as np

def predict():
  # Get data
  _, _, scaled_train, scaled_test, output_train, output_test = get_data(val_split=0.0)

  slope_output = output_train[0]
  intercept_output = output_train[1]
  # Create model and load weights
  head_model = create_model("strided", (299, 299, 3), (6, ))
  head_model.load_weights("weights.hdf5")

  # Predict
  predicted = head_model.predict(scaled_train, batch_size=20)
  slopeErrors = []
  interceptErrors = []
  for i in range(len(predicted[0])):
    slopeError = abs(predicted[0][i] - slope_output[i]) / slope_output[i]
    interceptError = abs(predicted[1][i] - intercept_output[i]) / intercept_output[i]
    slopeErrors.append(slopeError)
    interceptErrors.append(interceptError)
    print("Predicted: " + str(predicted[0][i]) + ". Actual: " + str(slope_output[i]) + ". Slope error: " + str(slopeError))
    print("Predicted: " + str(predicted[1][i]) + ". Actual: " + str(intercept_output[i]) + ". Intercept error: " + str(interceptError))

  print ("Average slope error: " + str(np.asarray(slopeErrors).mean()))
  print ("Average intercept error: " + str(np.asarray(interceptErrors).mean()))
  print ("Prediction complete!")