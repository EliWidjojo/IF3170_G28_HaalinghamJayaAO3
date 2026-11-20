#log-reg dengan optimasi stochastic gradient ascent dan regularisasi L1, L2, serta elastic net
#referensi:
#1) slide ppt kuliah AI materi linear dan logistic regression, https://edunex.itb.ac.id/courses/97453/preview/486049
#2) https://medium.com/@koushikkushal95/logistic-regression-from-scratch-dfb8527a4226
#3) https://www.geeksforgeeks.org/machine-learning/implementation-of-logistic-regression-from-scratch-using-python/
#4) https://medium.com/@helloitsdaksh007/understanding-l1-l2-and-elastic-net-regularization-2ba1aadf3159

import os
import pickle
import numpy as np

class LogisticReg:
    VALID_PENALTIES = (None, "none", "l1", "l2", "elastic")

    def __init__(self, learning_rate=0.01, n_epochs=100, penalty=None, lambda1=0.0, lambda2=0.0, batch_size=1, shuffle=True, tol=None, random_state=None, record_history=False):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.penalty = penalty
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.batch_size = batch_size #1 untuk stochastic gradient ascent murni, >1 untuk mini-batch (rata-rata gradien, lbh cpt)
        self.shuffle = shuffle #acak urutan data/contoh tiap epoch
        self.tol = tol #berhenti lbh awal kalau absolut dari selisih loss antar epoch < tol
        self.random_state = random_state #seed untuk pengacakan biar reproducible
        self.record_history = record_history #simpen [b0, b1, ...] tiap epoch di param_history 

        self.bobot = None
        self.bias = 0.0
        self.classes = None
        self.logloss_history = []
        self.loss_history = []
        self.param_history = []
        self.n_epochs_run = 0

    @staticmethod
    def sigmoid(z): #fungsi logistik p = 1 / (1 + e^-z)
        z = np.clip(z, -500, 500) #di-clip biar exp ga overflow
        return 1.0 / (1.0 + np.exp(-z))

    def effective_lambdas(self): #lambda aktif sesuai jenis penalti yg dipilih
        if self.penalty == "l1": return float(self.lambda1), 0.0
        if self.penalty == "l2": return 0.0, float(self.lambda2)
        if self.penalty == "elastic": return float(self.lambda1), float(self.lambda2)
        return 0.0, 0.0

    def compute_loss(self, X, y): #ngembaliin log-loss rata-rata dan log-loss + penalty/N, dimana log-loss memakai bentuk stabil (-[y log p + (1-y) log(1-p)] = log(1+e^z) - y*z) sehingga tidak ada log(0)
        z = X @ self.bobot + self.bias
        logloss = np.mean(np.logaddexp(0.0, z) - y * z)
        l1, l2 = self.effective_lambdas()
        penalty = (l1 * np.sum(np.abs(self.bobot)) + l2 * np.sum(self.bobot ** 2)) / len(y)
        return float(logloss), float(logloss + penalty)

    def validate_param(self):
        if self.penalty not in self.VALID_PENALTIES: raise ValueError(f"penalty harus None, 'l1', 'l2', atau 'elasticnet'")
        if self.learning_rate <= 0: raise ValueError("learning_rate harus > 0")
        if self.n_epochs <= 0: raise ValueError("n_epochs harus > 0")
        if self.batch_size <= 0: raise ValueError("batch_size harus > 0")
        if self.lambda1 < 0 or self.lambda2 < 0: raise ValueError("lambda1 dan lambda2 harus >= 0")

    @staticmethod
    def validate_X(X, n_features=None):
        X = np.asarray(X)
        if X.ndim != 2: raise ValueError("X harus berupa array 2D")
        if not np.all(np.isfinite(X)): raise ValueError("X mengandung NaN atau inf, tolong lakukan imputasi pada tahap preprocessing")
        if n_features is not None and X.shape[1] != n_features: raise ValueError(f"Jumlah fitur tidak sesuai dengan data training")
        return X

    def fit(self, X, y): #pelatihan model dgn stochastic gradient ascent
        self.validate_param()
        X = self.validate_X(X)
        y = np.asarray(y)
        if y.ndim != 1: raise ValueError("y harus berupa array 1D")
        if len(X) == 0: raise ValueError("Data training tidak boleh kosong")
        if len(X) != len(y): raise ValueError("Jumlah sample pada X dan y harus sama")
        self.classes = np.unique(y)
        if len(self.classes) != 2: raise ValueError("Logistic regression hanya mendukung klasifikasi biner")
        y_binary = (y == self.classes[1]).astype(int) #ubah label ke 0/1
        n_samples, n_features = X.shape
        batch_size = min(self.batch_size, n_samples)
        l1, l2 = self.effective_lambdas()
        rng = np.random.default_rng(self.random_state)
        #init b = 0 dan reset history tiap fit()
        self.weight = np.zeros(n_features)
        self.bias = 0.0
        self.logloss_history = []
        self.loss_history = []
        self.param_history = []
        self.n_epochs_run = 0
        if self.record_history: self.param_history.append(np.concatenate(([self.bias], self.weight)))
        prev_loss = None
        for _ in range(int(self.n_epochs)):
            order = rng.permutation(n_samples) if self.shuffle else np.arange(n_samples)
            for start in range(0, n_samples, batch_size):
                idx = order[start:start + batch_size]
                X_b, y_b = X[idx], y_binary[idx]
                p = self.sigmoid(X_b @ self.weight + self.bias)
                error = y_b - p #(y_i - p_i)
                grad_w = (X_b.T @ error) / len(idx) #rata-rata (y_i - p_i) * x_ij
                grad_b = np.mean(error) #x_i0 = 1
                #gradien penalti (subgradien untuk l1)
                penalty_grad = (l1 * np.sign(self.weight) + 2.0 * l2 * self.weight) / n_samples
                #gradient ascent
                self.weight += self.learning_rate * (grad_w - penalty_grad)
                self.bias += self.learning_rate * grad_b
            logloss, loss = self.compute_loss(X, y_binary)
            self.logloss_history.append(logloss)
            self.loss_history.append(loss)
            self.n_epochs_run += 1
            if self.record_history: self.param_history.append(np.concatenate(([self.bias], self.weight)))
            if self.tol is not None and prev_loss is not None and abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss
        return self

    def check_fitted(self):
        if self.weight is None or self.classes is None: raise ValueError("Model belum dilatih, panggil fit() terlebih dahulu")

    def decision_func(self, X): #nilai log-odds z = b0 + b^T x
        self.check_fitted()
        X = self.validate_X(X, n_features=len(self.weight))
        return X @ self.weight + self.bias

    def predict_prob(self, X): #probabilitas tiap kelas, shape (n_samples, 2), urutan kolom = classes. Kolom ke-2 itu P(y = classes_[1] | x)
        p1 = self.sigmoid(self.decision_func(X))
        return np.column_stack((1.0 - p1, p1))

    def predict(self, X, threshold=0.5): #buat prediksi kelas. Kelas classes_[1] dipilih bila p >= threshold, dgn threshold bisa diubah
        p1 = self.sigmoid(self.decision_func(X))
        return self.classes[(p1 >= threshold).astype(int)]

    def score(self, X, y): #akurasi model pada data uji
        return np.mean(self.predict(X) == np.asarray(y))

    def save_model(self, filepath): #simpan model ke file
        dir = os.path.dirname(filepath)
        if dir: os.makedirs(dir, exist_ok=True)
        with open(filepath, "wb") as f: pickle.dump(self, f)
        print(f"Model berhasil disimpan di {filepath}")

    @classmethod
    def load_model(cls, filepath): #memuat model dari file
        if not os.path.exists(filepath): raise FileNotFoundError(f"File model tidak ditemukan: {filepath}")
        with open(filepath, "rb") as f: model = pickle.load(f)
        print(f"Model berhasil dimuat dari {filepath}")
        return model