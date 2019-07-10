from model.train import train
from model.predict import predict
import sys

def main(mode="predict"):
  if (mode == "train"):
    print("Train mode was chosen.")
    train(125, True)
  else:
    print("Predict mode was chosen.")
    predict()

if __name__ == "__main__":
  main(sys.argv[1])