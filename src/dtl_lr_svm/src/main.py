from eda import *
from cart import *
from logistic_regression import *
from preprocessing import *
from svm import *
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
train_path = BASE_DIR / "data" / "train.csv"
test_path = BASE_DIR / "data" / "test.csv"
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print("EDA: ")
EDA(train,test)
X_train, y_train, X_val, y_val, X_train_scaled, X_val_scaled = preprocessing(train, test)

print()
CARTfromScratch(X_train, y_train, X_val, y_val)
CARTwithsklearn(X_train, y_train, X_val, y_val)
logisticRegressionfromScratch(X_train_scaled, y_train, X_val_scaled, y_val)
logisticRegressionwithsklearn(X_train_scaled, y_train, X_val_scaled, y_val)
SVMfromScratch(X_train_scaled, y_train, X_val_scaled, y_val)
SVMwithsklearn(X_train_scaled, y_train, X_val_scaled, y_val)