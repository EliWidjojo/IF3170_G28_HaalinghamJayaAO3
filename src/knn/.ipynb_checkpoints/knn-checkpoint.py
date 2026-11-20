import numpy as np
import pickle
import os

class KNN:
    def __init__(self, k=3, distance_metric='euclidean', weights='uniform'):
        """
        Inisialisasi model K-Nearest Neighbors.
        
        Parameters:
        - k (int): Jumlah tetangga terdekat.
        - distance_metric (str): Metrik jarak ('euclidean', 'manhattan', 'minkowski').
        - weights (str): Bobot voting ('uniform' atau 'distance').
        """
        self.k = k
        self.distance_metric = distance_metric
        self.weights = weights
        
        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def _calculate_distance(self, X1, X2):
        """
        Menghitung jarak antara matriks X1 (data uji) dan X2 (data latih).
        Mengembalikan matriks jarak berukuran (len(X1), len(X2)).
        """
        if self.distance_metric == 'euclidean':
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            return np.sqrt(np.sum(diff ** 2, axis=2))
        elif self.distance_metric == 'manhattan':
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            return np.sum(np.abs(diff), axis=2)
        elif self.distance_metric == 'minkowski':
            p = 3
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            return np.sum(np.abs(diff) ** p, axis=2) ** (1 / p)
        else:
            raise ValueError("Metrik jarak tidak didukung.")

    def fit(self, X, y):
        """
        Melatih model (menyimpan data training).
        
        Parameters:
        - X (np.ndarray): Fitur data training.
        - y (np.ndarray): Label data training.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X harus berupa array 2 dimensi.")
        if y.ndim != 1:
            raise ValueError("y harus berupa array 1 dimensi.")
        if len(X) == 0:
            raise ValueError("Data training tidak boleh kosong.")
        if len(X) != len(y):
            raise ValueError("Jumlah sampel X dan y harus sama.")
        if self.k <= 0:
            raise ValueError("k harus berupa bilangan positif.")
        if self.k > len(X):
            raise ValueError("k tidak boleh lebih besar dari jumlah data training.")
        if self.distance_metric not in ("euclidean", "manhattan", "minkowski"):
            raise ValueError("Metrik jarak tidak didukung.")
        if self.weights not in ("uniform", "distance"):
            raise ValueError("Jenis weights harus 'uniform' atau 'distance'.")

        self.X_train = X
        self.y_train = y
        self.classes_ = np.unique(self.y_train)
        return self

    def _predict_single(self, x):
        """
        Memprediksi kelas untuk satu sampel data uji.
        """
        if self.X_train is None or self.y_train is None or self.classes_ is None:
            raise ValueError("Model belum dilatih. Panggil fit() terlebih dahulu.")

        X_train = self.X_train
        y_train = self.y_train
        classes = self.classes_

        x = np.asarray(x)
        if x.ndim != 1 or x.shape[0] != X_train.shape[1]:
            raise ValueError("Dimensi sampel uji tidak sesuai dengan data training.")

        distances = self._calculate_distance(x.reshape(1, -1), X_train)[0]
        neighbor_indices = np.argsort(distances, kind="stable")[:self.k]
        neighbor_labels = y_train[neighbor_indices]
        neighbor_distances = distances[neighbor_indices]

        if self.weights == "uniform":
            labels, counts = np.unique(neighbor_labels, return_counts=True)
            return labels[np.argmax(counts)]

        zero_mask = neighbor_distances == 0
        if np.any(zero_mask):
            labels, counts = np.unique(neighbor_labels[zero_mask], return_counts=True)
            return labels[np.argmax(counts)]

        sample_weights = 1.0 / neighbor_distances
        class_scores = np.array([
            np.sum(sample_weights[neighbor_labels == label])
            for label in classes
        ])
        return classes[np.argmax(class_scores)]

    def predict(self, X):
        """
        Memprediksi kelas untuk sekumpulan data uji.
        
        Parameters:
        - X (np.ndarray): Fitur data uji.
        
        Returns:
        - np.ndarray: Array berisi label prediksi.
        """
        X = np.array(X)
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)
    
    def predict_proba(self, X):
        """
        (Opsional/Bonus Cost-Sensitive) Mengembalikan probabilitas prediksi tiap kelas.
        """
        if self.X_train is None or self.y_train is None or self.classes_ is None:
            raise ValueError("Model belum dilatih. Panggil fit() terlebih dahulu.")

        X_train = self.X_train
        y_train = self.y_train
        classes = self.classes_

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X harus berupa array 2 dimensi.")
        if X.shape[1] != X_train.shape[1]:
            raise ValueError("Jumlah fitur data uji tidak sesuai dengan data training.")

        probabilities = []
        for x in X:
            distances = self._calculate_distance(x.reshape(1, -1), X_train)[0]
            neighbor_indices = np.argsort(distances, kind="stable")[:self.k]
            neighbor_labels = y_train[neighbor_indices]
            neighbor_distances = distances[neighbor_indices]

            if self.weights == "uniform":
                sample_weights = np.ones(self.k, dtype=float)
            else:
                zero_mask = neighbor_distances == 0
                if np.any(zero_mask):
                    sample_weights = zero_mask.astype(float)
                else:
                    sample_weights = 1.0 / neighbor_distances

            class_scores = np.array([
                np.sum(sample_weights[neighbor_labels == label])
                for label in classes
            ], dtype=float)
            probabilities.append(class_scores / np.sum(class_scores))

        return np.asarray(probabilities, dtype=float)

    def score(self, X, y):
        """
        Menghitung akurasi model pada data uji.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == np.array(y))

    def save_model(self, filepath):
        """
        Menyimpan model ke dalam file menggunakan pickle.
        """
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model berhasil disimpan di {filepath}")

    @classmethod
    def load_model(cls, filepath):
        """
        Memuat model dari file pickle.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File model tidak ditemukan: {filepath}")
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model berhasil dimuat dari {filepath}")
        return model


