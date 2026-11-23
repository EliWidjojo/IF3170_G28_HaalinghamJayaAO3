import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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


    def plot_svm_kernels_2d(
        X,
        y,
        use_pca=True,
        feature_names=("Fitur 1", "Fitur 2"),
        C=1.0,
        degree=3,
        gamma="scale",
        lr=0.005,
        epochs=200,
        filepath="svm_kernels_boundary.png",
    ):
        """
        Melatih dan memvisualisasikan bidang pemisah, margin, dan support vector
        untuk kernel Linear, Polynomial, dan Gaussian (RBF) pada data 2D.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        # 1. Pastikan input dalam 2D (pilih 2 fitur atau reduksi dimensi dengan PCA)
        if X.shape[1] > 2 or use_pca:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            pca = PCA(n_components=2, random_state=42)
            X_2d = pca.fit_transform(X_scaled)
            var_ratio = pca.explained_variance_ratio_
            x_label = f"PC1 ({var_ratio[0]*100:.1f}% var)"
            y_label = f"PC2 ({var_ratio[1]*100:.1f}% var)"
        elif X.shape[1] == 2:
            X_2d = X
            x_label, y_label = feature_names[0], feature_names[1]
        else:
            raise ValueError("Data minimal harus memiliki 2 fitur.")

        # Siapkan meshgrid untuk menggambar area klasifikasi, decision boundary, dan margin
        pad_x = (X_2d[:, 0].max() - X_2d[:, 0].min()) * 0.12 + 0.5
        pad_y = (X_2d[:, 1].max() - X_2d[:, 1].min()) * 0.12 + 0.5
        x_min, x_max = X_2d[:, 0].min() - pad_x, X_2d[:, 0].max() + pad_x
        y_min, y_max = X_2d[:, 1].min() - pad_y, X_2d[:, 1].max() + pad_y

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 250),
            np.linspace(y_min, y_max, 250),
        )
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        kernels = [
            ("linear", "Linear Kernel"),
            ("poly", f"Polynomial Kernel (degree={degree})"),
            ("rbf", "Gaussian (RBF) Kernel"),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150)

        for ax, (kernel_name, title) in zip(axes, kernels):
            # Latih model SVM dari svm.py
            model = SVM(
                kernel=kernel_name,
                C=C,
                degree=degree,
                gamma=gamma,
                lr=lr,
                epochs=epochs,
                random_state=42,
            )
            model.fit(X_2d, y)

            # Hitung nilai decision_function pada seluruh area grid
            Z = model.decision_function(grid_points).reshape(xx.shape)

            # Warna latar wilayah kelas
            ax.contourf(xx, yy, Z, levels=[-np.inf, 0.0, np.inf], colors=["#cfe2ff", "#f8d7da"], alpha=0.45)

            # Garis Margin (-1 dan +1) serta Bidang Pemisah (0)
            contours = ax.contour(
                xx, yy, Z,
                levels=[-1.0, 0.0, 1.0],
                colors=["#333333", "#000000", "#333333"],
                linestyles=["--", "-", "--"],
                linewidths=[1.3, 2.2, 1.3],
            )
            ax.clabel(contours, inline=True, fontsize=8, fmt={-1.0: "-1", 0.0: "0", 1.0: "+1"})

            # Ekstrak Support Vectors sesuai arsitektur svm.py
            y_svm = np.where(y == model.classes_[0], -1.0, 1.0)
            if kernel_name == "linear":
                scores = model.decision_function(X_2d)
                sv_mask = (y_svm * scores) <= 1.0 + 1e-3
                support_vectors = X_2d[sv_mask]
            else:
                # Pada kernel non-linear, cek alpha aktif atau titik pada/dalam margin
                scores_sub = model.decision_function(model.support_X)
                sv_mask = (model.alpha > 1e-5) | ((model.support_y * scores_sub) <= 1.0 + 1e-3)
                support_vectors = model.support_X[sv_mask]

            # Plot titik data tiap kelas
            for cls_idx, cls_val in enumerate(model.classes_):
                mask = y == cls_val
                ax.scatter(
                    X_2d[mask, 0],
                    X_2d[mask, 1],
                    c="#0d6efd" if cls_idx == 0 else "#dc3545",
                    label=f"Kelas {cls_val}",
                    edgecolors="white",
                    s=35,
                    alpha=0.85,
                    zorder=3,
                )

            # Lingkari Support Vectors
            if len(support_vectors) > 0:
                ax.scatter(
                    support_vectors[:, 0],
                    support_vectors[:, 1],
                    s=110,
                    facecolors="none",
                    edgecolors="#111111",
                    linewidths=1.4,
                    label=f"Support Vectors ({len(support_vectors)})",
                    zorder=4,
                )

            acc = np.mean(model.predict(X_2d) == y)
            ax.set_title(f"{title}\n(Train Acc: {acc*100:.1f}%)", fontsize=11, fontweight="bold")
            ax.set_xlabel(x_label, fontsize=9.5)
            ax.set_ylabel(y_label, fontsize=9.5)
            ax.legend(loc="best", fontsize=8, framealpha=0.9)
            ax.grid(True, linestyle=":", alpha=0.4)

        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"Visualisasi SVM berhasil disimpan di: {filepath}")