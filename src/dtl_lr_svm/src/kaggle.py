from cart import *
from preprocessing import *
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
train_path = BASE_DIR / "data" / "train.csv"
test_path = BASE_DIR / "data" / "test.csv"
submission_path = BASE_DIR / "data" / "submission.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

test_ids = test["person_id"]
# Remove person_id
train = train.drop(columns=["person_id"])
test = test.drop(columns=["person_id"])

# Separate them into X and y 
# y = f(X)
X_train = train.drop(columns=["loan_status"])
y_train = train["loan_status"]
X_test = test.copy()

# One-hot encoding
X_train = pd.get_dummies(X_train, columns=["person_gender", "person_home_ownership", "previous_loan_defaults_on_file"], drop_first=True)
X_test = pd.get_dummies(X_test, columns=["person_gender", "person_home_ownership", "previous_loan_defaults_on_file"], drop_first=True)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Turn into numpy
X_train = X_train.to_numpy(dtype=float)
y_train = y_train.to_numpy(dtype=float)
X_test = X_test.to_numpy(dtype=float)


tree = buildTree(X_train, y_train, 0, 7)
y_pred = predict(tree, X_test)


submission = pd.DataFrame({
    "person_id": test_ids,
    "loan_status": y_pred.astype(int)
})

submission.to_csv(
    submission_path,
    index=False
)