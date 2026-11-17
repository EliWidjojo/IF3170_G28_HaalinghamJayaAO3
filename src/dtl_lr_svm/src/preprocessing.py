import pandas as pd
import numpy as np

def preprocessing (train, test):
    # Remove person_id
    train = train.drop(columns=["person_id"])
    test = test.drop(columns=["person_id"])

    # Separate them into X and y 
    # y = f(X)
    X = train.drop(columns=["loan_status"])
    y = train["loan_status"]

    # One-hot encoding
    X = pd.get_dummies(X, columns=["person_gender", "person_home_ownership", "previous_loan_defaults_on_file"], drop_first=True)

    # Split train-validation
    np.random.seed(67)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    split = int(0.8 * len(X))

    train_idx = indices[:split]
    val_idx = indices[split:]

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    # Turn into numpy
    X_train = X_train.to_numpy(dtype=float)
    y_train = y_train.to_numpy(dtype=float)
    X_val = X_val.to_numpy(dtype=float)
    y_val = y_val.to_numpy(dtype=float)

    # Scaling
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    X_train_scaled = (X_train - mean)/std
    X_val_scaled = (X_val - mean)/std

    return X_train, y_train, X_val, y_val, X_train_scaled, X_val_scaled
