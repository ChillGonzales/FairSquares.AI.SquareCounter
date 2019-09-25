from data.factory import get_data
from model.factory import create_model
from utility.utility import DenormalizeWithRange
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def predict(weights_file_name: str, test_split: float):
  # Get data
  _, _, scaled_train, scaled_test, output_train, _, _, _, output_ranges = get_data(val_split=0.0, test_split=0.0, randomize=False)
  slope_output = output_train[0]
  intercept_output = output_train[1]

  # Create model and load weights
  head_model = create_model("strided", (299, 299, 3), (6, ))
  head_model.load_weights(weights_file_name + ".hdf5")
  ranges = output_ranges[0]

  # Predict
  predicted = head_model.predict(scaled_train, batch_size=20)
  if (test_split > 0):
    predicted = predicted[0][:(test_split * len(predicted))]
  denormed_slope = DenormalizeWithRange(predicted[0][:], ranges["area_slope"])
  denormed_intercept = DenormalizeWithRange(predicted[1][:], ranges["area_intercept"])
  slopeDiffs = []
  interceptDiffs = []
  for i in range(len(predicted[0])):
    slopeDiff = denormed_slope[i] - slope_output[i]
    interceptDiff = denormed_intercept[i] - intercept_output[i] 
    slopeDiffs.append(slopeDiff)
    interceptDiffs.append(interceptDiff)
    print("[SLOPE] Predicted: " + str(denormed_slope[i]) + ". Actual: " + str(slope_output[i]) + ". Difference: " + str(slopeDiff))
    print("[INTERCEPT] Predicted: " + str(denormed_intercept[i]) + ". Actual: " + str(intercept_output[i]) + ". Difference: " + str(interceptDiff))

  print ("Average slope diff: " + str(np.asarray(slopeDiffs).mean()))
  print ("Slope diff std. dev.: " + str(np.asarray(slopeDiffs).std()))
  print ("Average intercept diff: " + str(np.asarray(interceptDiffs).mean()))
  print ("Intercept diff std. dev.: " + str(np.asarray(interceptDiffs).std()))
  print ("Prediction complete!")

  plt.subplot(2, 1, 1)
  plt.plot(slopeDiffs)
  plt.title('Predicted Differences for Slope')
  plt.ylabel('Difference (sq. ft.)')
  plt.xlabel('Order')
  plt.legend(['Slope'], loc='upper left')
  plt.subplot(2, 1, 2)
  plt.plot(interceptDiffs)
  plt.title('Predicted Differences for Intercept')
  plt.ylabel('Difference (sq. ft.)')
  plt.xlabel('Order')
  plt.legend(['Intercept'], loc='upper left')
  plt.show()