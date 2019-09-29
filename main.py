from model.train import train
from model.predict import predict
import sys
import argparse

def main(mode: str, epochs: int, weights_name: str):
  if (mode == "train"):
    print("Train mode was chosen.")
    test_split = 0.05
    train(epochs=epochs, save_model=True, test_split=test_split)
    # Use the weights just trained to run a prediction on the test samples to see how the model performs. 
    predict(model_file_name="model", test_split=test_split)
  elif (mode == "predict"):
    print("Predict mode was chosen.")
    predict(model_file_name=weights_name, test_split=0.05)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("mode", help="The mode to run the program in.", choices=["predict", "train"])
  parser.add_argument("--e", help="Number of epochs (if in train mode).", default=20000, type=int)
  parser.add_argument("--m", help="Model file name (no extension) to use if (in predict mode).", default="model", type=str, required="predict" in sys.argv)
  args = parser.parse_args()
  main(args.mode, args.e, args.m)