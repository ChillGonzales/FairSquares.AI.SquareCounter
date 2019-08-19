from data.factory import get_data
from model.factory import create_model
from utility.utility import DenormalizeWithRange
import pandas as pd
import numpy as np

def predict():
  # Get data
  _, _, scaled_train, scaled_test, output_train, _, _, _, output_ranges = get_data(val_split=0.0, randomize=False)

  slope_output = output_train[0]
  intercept_output = output_train[1]
  # Create model and load weights
  head_model = create_model("strided", (299, 299, 3), (6, ))
  head_model.load_weights("weights.hdf5")

  ranges = output_ranges[0]

  # Predict
  predicted = head_model.predict(scaled_train, batch_size=20)
  denormed_slope = DenormalizeWithRange(predicted[0][:], ranges["area_slope"])
  denormed_intercept = DenormalizeWithRange(predicted[1][:], ranges["area_intercept"])
  slopeErrors = []
  interceptErrors = []
  for i in range(len(predicted[0])):
    slopeError = abs(denormed_slope[i] - slope_output[i]) / slope_output[i]
    interceptError = abs(denormed_intercept[i] - intercept_output[i]) / intercept_output[i]
    slopeErrors.append(slopeError)
    interceptErrors.append(interceptError)
    print("Predicted: " + str(denormed_slope[i]) + ". Actual: " + str(slope_output[i]) + ". Slope error: " + str(slopeError))
    print("Predicted: " + str(denormed_intercept[i]) + ". Actual: " + str(intercept_output[i]) + ". Intercept error: " + str(interceptError))

  print ("Average slope error: " + str(np.asarray(slopeErrors).mean()))
  print ("Average intercept error: " + str(np.asarray(interceptErrors).mean()))
  print ("Prediction complete!")