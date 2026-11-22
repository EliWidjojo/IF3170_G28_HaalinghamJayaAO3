
import numpy as np
from src.decision_tree.cart import DecisionTreeCART

X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [6, 7],
    [7, 8],
    [8, 9]
])

y = np.array([0, 0, 0, 1, 1, 1])

model = DecisionTreeCART(max_depth=3)
model.fit(X, y)

predictions = model.predict(X)

print("Prediksi:", predictions)
print("Label asli:", y)
print("Akurasi training:", np.mean(predictions == y))
print("Struktur tree:", model.tree)
