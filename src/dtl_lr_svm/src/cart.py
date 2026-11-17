import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, f1_score

class Node:
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.prediction = None

def calculateGini(y):
    classes = np.unique(y)
    impurity = 1

    for c in classes:
        pi = np.sum(y==c)/len(y)
        impurity -= pi*pi

    return impurity

def findThresholds(values):
    values = np.unique(values)
    values = np.sort(values)
    numThresholds = 30
    if len(values) <= numThresholds:
        indices = range(len(values)-1)
    else:
        indices = np.linspace(0, len(values)-2, numThresholds, endpoint=True, dtype=int)
    
    thresholds = []

    for i in indices:
        threshold = (values[i]+values[i+1])/2
        thresholds.append(threshold)

    return thresholds

def splitY(X, y, feature, threshold):
    left_y = []
    right_y = []
    for i in range(len(X)):
        if X[i, feature] <= threshold:
            left_y.append(y[i])
        else:
            right_y.append(y[i])
    return left_y, right_y
    
def splitData(X, y, feature, threshold):
    left_y = []
    right_y = []
    left_X = []
    right_X = []
    
    for i in range(len(X)):
        if X[i, feature] <= threshold:
            left_y.append(y[i])
            left_X.append(X[i])
        else:
            right_y.append(y[i])
            right_X.append(X[i])
    return np.array(left_X), np.array(left_y), np.array(right_X), np.array(right_y)
                                                
def weightedGini(left, right):
    total = len(left)+len(right)
    return (len(left)/total)*calculateGini(left) + (len(right)/total)*calculateGini(right)

def findOptimalSplit(X, y):
    bestGini = float("inf")
    bestFeature = None
    bestThreshold = None
    
    for feature in range(X.shape[1]):
        thresholds = findThresholds(X[:, feature])

        for threshold in thresholds:
            left_y, right_y = splitY(X, y, feature, threshold)

            gini = weightedGini(left_y, right_y)
    
            if gini < bestGini:
                bestGini = gini
                bestFeature = feature
                bestThreshold = threshold

    return bestGini, bestFeature, bestThreshold

def majorityClass(y):
    classes, counts = np.unique(y, return_counts=True)
    return classes[np.argmax(counts)]
    
def buildTree(X, y, depth, maxDepth):
    node = Node()

    if depth >= maxDepth:
        node.prediction = majorityClass(y)
        return node
        
    if len(np.unique(y)) == 1:
        node.prediction = y[0]
        return node

    gini, feature, threshold = findOptimalSplit(X, y)

    if feature is None:
        node.prediction = majorityClass(y)
        return node

    node.feature = feature
    node.threshold = threshold

    left_X, left_y, right_X, right_y = splitData(X, y, feature, threshold)

    node.left = buildTree(left_X, left_y, depth+1, maxDepth)
    node.right = buildTree(right_X, right_y, depth+1, maxDepth)

    return node

def predictSample(node, X):
    if node.prediction is not None:
        return node.prediction
        
    if X[node.feature] <= node.threshold:
        return predictSample(node.left, X)
    else:
        return predictSample(node.right, X)
        
def predict(node, X):
    predictions = []

    for x in X:
        prediction = predictSample(node, x)
        predictions.append(prediction)

    return np.array(predictions)

def CARTfromScratch(X_train, y_train, X_val, y_val):    
    tree = buildTree(X_train, y_train, 0, 5)
    y_pred_CART_scratch = predict(tree, X_val)

    print("=== CART from Scratch ===")
    print("F1 Score:", f1_score(y_val, y_pred_CART_scratch, average="macro"))
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_CART_scratch))

def CARTwithsklearn(X_train, y_train, X_val, y_val):
    model = DecisionTreeClassifier(criterion="gini", max_depth = 5, random_state=67)
    model.fit(X_train, y_train)

    y_pred_CART_sklearn = model.predict(X_val)

    print("=== CART from sklearn ===")
    print("F1 Score:", f1_score(y_val, y_pred_CART_sklearn, average="macro"))
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_CART_sklearn))