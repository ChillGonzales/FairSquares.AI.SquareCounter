from data.factory import get_data
from model.factory import create_model

def predict():
  # Get data
  _, outputs, scaled_inputs, scaled_outputs, output_scaler = get_data()

  # Create model and load weights
  head_model = create_model()
  head_model.load_weights("weights.hdf5")

  # Predict
  predicted = head_model.predict(scaled_inputs, batch_size=len(scaled_inputs))
  unscaled_predictions = output_scaler.inverse_transform(predicted)
  for i in range(len(unscaled_predictions)):
    print("Predicted: " + str(unscaled_predictions[i]) + ". Actual: " + str(outputs[i]))
  print("Prediction complete!")