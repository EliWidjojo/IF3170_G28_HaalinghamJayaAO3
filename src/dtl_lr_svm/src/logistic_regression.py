import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

def sigmoid(z):
    return 1/(1+np.exp(-z))

def predictProbability(X, weights, bias):
    z = np.dot(X, weights) + bias
    return sigmoid(z)

def train(X, y, learning_rate, epochs):
    weights = np.zeros(X.shape[1])
    bias = 0
    
    for i in range(epochs):
        prediction = predictProbability(X, weights, bias)

        dw = (1/len(y)) * np.dot(X.T, (prediction-y))
        db = (1/len(y)) * np.sum(prediction-y)

        weights -= learning_rate*dw
        bias -= learning_rate*db

    return weights, bias

def predict(X, weights, bias):
    probability = sigmoid(np.dot(X, weights)+bias)

    predictions = np.where(probability >= 0.5, 1, 0)
    return predictions

def logisticRegressionfromScratch(X_train_scaled, y_train, X_val_scaled, y_val):
    weights, bias = train(X_train_scaled, y_train, 0.01, 1000)
    y_pred_regression_scratch = predict(X_val_scaled, weights, bias)

    print("=== Logistic Regression from Scratch ===")
    print("F1 Score:", f1_score(y_val, y_pred_regression_scratch, average="macro"))
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_regression_scratch))

def logisticRegressionwithsklearn(X_train_scaled, y_train, X_val_scaled, y_val):
    model = LogisticRegression(random_state=67, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    y_pred_regression_sklearn = model.predict(X_val_scaled)

    print("=== Logistic Regression from sklearn ===")
    print("F1 Score:", f1_score(y_val, y_pred_regression_sklearn, average="macro"))
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_regression_sklearn))