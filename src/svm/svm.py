import os
import pickle

import numpy as np

class SVM:
    def __init__(
        self,
        kernel="linear",
        C=1.0,
        degree=3,
        gamma="scale",
        coef0=1.0,
        lr=0.001,
        epochs=100,
        max_kernel_samples=2000,
        random_state=42,
    ):
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.lr = lr
        self.epochs = epochs
        self.max_kernel_samples = max_kernel_samples
        self.random_state = random_state

        self.weights = None
        self.alpha = None
        self.bias = 0.0
        self.support_X = None
        self.support_y = None
        self.gamma_val = None
        self.classes_ = None
        self.slack_total = 0.0 

    def _resolve_gamma(self, X):
        if isinstance(self.gamma, (int, float)):
            return float(self.gamma)
        n_features = X.shape[1]
        if self.gamma == "auto":
            return 1.0 / max(1, n_features)
        # Default 'scale': 1 / (n_features * X.var())
        var = np.var(X)
        return 1.0 / max(1e-8, n_features * var)

    def _compute_kernel(self, X1, X2):
        if self.kernel == "linear":
            return np.dot(X1, X2.T)
        elif self.kernel == "poly":
            return (self.gamma_val * np.dot(X1, X2.T) + self.coef0) ** self.degree
        elif self.kernel == "rbf":
            X1_sq = np.sum(X1**2, axis=1, keepdims=True)
            X2_sq = np.sum(X2**2, axis=1, keepdims=True).T
            sq_dist = np.maximum(0.0, X1_sq + X2_sq - 2.0 * np.dot(X1, X2.T))
            return np.exp(-self.gamma_val * sq_dist)
        else:
            raise ValueError(
                f"Kernel '{self.kernel}' tidak didukung. Pilih 'linear', 'poly', atau 'rbf'."
            )

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Implementasi SVM ini khusus klasifikasi biner (0 dan 1).")

        # Konversi label ke {-1, +1}
        y_svm = np.where(y == self.classes_[0], -1.0, 1.0)
        n_samples, n_features = X.shape
        self.gamma_val = self._resolve_gamma(X)

        rng = np.random.default_rng(self.random_state)

        # linear kernel
        if self.kernel == "linear":
            self.weights = np.zeros(n_features, dtype=float)
            self.bias = 0.0
            lambda_param = 1.0 / (self.C * n_samples)

            for epoch in range(self.epochs):
                indices = rng.permutation(n_samples)
                lr_t = self.lr / (1.0 + 0.01 * epoch)

                for idx in indices:
                    x_i = X[idx]
                    y_i = y_svm[idx]
                    margin = y_i * (np.dot(x_i, self.weights) + self.bias)

                    if margin >= 1.0:
                        dw = lambda_param * self.weights
                        db = 0.0
                    else:
                        dw = lambda_param * self.weights - y_i * x_i
                        db = -y_i

                    self.weights -= lr_t * dw
                    self.bias -= lr_t * db

            scores = np.dot(X, self.weights) + self.bias
            slacks = np.maximum(0.0, 1.0 - y_svm * scores)
            self.slack_total = float(np.sum(slacks))

        # kernel sisanya
        else:
            if n_samples > self.max_kernel_samples:
                sub_idx = rng.choice(
                    n_samples, size=self.max_kernel_samples, replace=False
                )
                X_sub = X[sub_idx]
                y_sub = y_svm[sub_idx]
            else:
                X_sub = X
                y_sub = y_svm

            m = X_sub.shape[0]
            self.support_X = X_sub
            self.support_y = y_sub
            self.alpha = np.zeros(m, dtype=float)
            self.bias = 0.0

            K = self._compute_kernel(X_sub, X_sub)
            lambda_param = 1.0 / (self.C * m)

            for epoch in range(self.epochs):
                indices = rng.permutation(m)
                lr_t = self.lr / (1.0 + 0.01 * epoch)

                for i in indices:
                    # f(x_i) = sum_j(alpha_j * y_j * K(x_j, x_i)) + bias
                    score_i = np.dot(self.alpha * y_sub, K[:, i]) + self.bias
                    margin = y_sub[i] * score_i

                    self.alpha *= 1.0 - lr_t * lambda_param
                    if margin < 1.0:
                        self.alpha[i] += lr_t
                        self.bias += lr_t * y_sub[i]

            scores = np.dot(K, self.alpha * y_sub) + self.bias
            slacks = np.maximum(0.0, 1.0 - y_sub * scores)
            self.slack_total = float(np.sum(slacks))

        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        if self.kernel == "linear":
            return np.dot(X, self.weights) + self.bias
        else:
            K_test = self._compute_kernel(X, self.support_X)
            return np.dot(K_test, self.alpha * self.support_y) + self.bias

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores >= 0.0, self.classes_[1], self.classes_[0])

    def save_model(self, filepath):
        """
        Menyimpan model ke dalam file menggunakan pickle.
        """
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"Model berhasil disimpan di {filepath}")

    @classmethod
    def load_model(cls, filepath):
        """
        Memuat model dari file pickle.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File model tidak ditemukan: {filepath}")
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        print(f"Model berhasil dimuat dari {filepath}")
        return model