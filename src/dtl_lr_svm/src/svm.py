import numpy as np

from sklearn.metrics import classification_report, f1_score
from sklearn.svm import LinearSVC

def predict(X, weights, bias):
    scores = np.dot(X, weights) + bias
    predictions = np.where(scores >= 0, 1, -1)
    
    return predictions

def trainSVM(X, y, learning_rate, epochs, lambda_param):
    weights = np.zeros(X.shape[1])
    bias = 0

    for epoch in range(epochs):
        for i in range(len(X)):
            condition = y[i] * (np.dot(X[i], weights)+bias)
            
            if condition >= 1:
                dw = 2 * lambda_param * weights
                db = 0
            else:
                dw = 2 * lambda_param * weights - y[i]*X[i]
                db = -y[i]

            weights -= learning_rate * dw
            bias -= learning_rate * db
    return weights, bias

def SVMfromScratch(X_train_scaled, y_train, X_val_scaled, y_val):    
    y_train_svm = np.where(y_train == 0, -1, 1)
    weights, bias = trainSVM(X_train_scaled, y_train_svm, 0.001, 100, 0.01)
    y_pred_SVM_scratch = predict(X_val_scaled, weights, bias)
    y_pred_SVM_scratch = np.where(y_pred_SVM_scratch == -1, 0, 1)

    print("=== SVM from Scratch ===")
    print("F1 Score:", f1_score(y_val, y_pred_SVM_scratch, average="macro"))
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_SVM_scratch))

def SVMwithsklearn(X_train_scaled, y_train, X_val_scaled, y_val): 
    model = LinearSVC(random_state=67, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    y_pred_SVM_sklearn = model.predict(X_val_scaled)

    print("=== SVM from sklearn ===")
    print("F1 Score:", f1_score(y_val, y_pred_SVM_sklearn, average="macro"))
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_SVM_sklearn)) 