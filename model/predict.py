from data.factory import get_data, DataKeys
from model.factory import create_model
from utility.utility import DenormalizeWithRange
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def predict(model_file_name: str, test_split: float):
  # Get data
  # _, _, normalized_features_train, normalized_features_test, raw_output_train, _, _, _, output_ranges_train = get_data(val_split=0.0, test_split=0.0, randomize=False)
  data = get_data(val_split=0.15, test_split=test_split, randomize=True)
  normalized_features_train = data[DataKeys.NormalizedFeaturesTrain]
  output_ranges_train = data[DataKeys.OutputRangesTrain]
  raw_output_train = data[DataKeys.RawOutputTrain]
  slope_output = raw_output_train[0]
  intercept_output = raw_output_train[1]

  # Create model and load weights
  head_model = load_model(model_file_name + ".hdf5")
  ranges = output_ranges_train

  # Predict
  predicted = head_model.predict({"image_input": normalized_features_train[0], "feature_input": normalized_features_train[1]}, batch_size=20)
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