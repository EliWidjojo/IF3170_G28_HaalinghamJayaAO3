import pandas as pd
import numpy as np
import math


class NaiveBayes:
    UNKNOWN = "__UNK__"

    def __init__(self, alpha=1.0):
        if alpha <= 0:
            raise ValueError("alpha harus > 0")
        self.alpha = alpha

    def fit(self, X, y):
        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        if len(X) != len(y):
            raise ValueError(f"Panjang X ({len(X)}) != panjang y ({len(y)})")
        if y.isna().any():
            raise ValueError("y mengandung NaN")
        if len(y) == 0:
            raise ValueError("Data training kosong")

        try:
            self.classes = sorted(y.unique().tolist())
        except TypeError:
            raise ValueError(
                "Label kelas bertipe campuran; konversi ke tipe seragam dulu"
            )

        self.features = X.columns.tolist()
        K = len(self.classes)

        self.log_prior = {}
        self.log_cond = {f: {} for f in self.features}
        self.vocabularies = {}

        # log P(vj)
        for c in self.classes:
            self.log_prior[c] = math.log(((y == c).sum() + self.alpha) / (len(y) + self.alpha * K))

        # log P(ai | vj) + smoothing
        for f in self.features:
            values = X[f].dropna().unique().tolist()

            # __UNK__ untuk kategori baru
            if self.UNKNOWN in values:
                raise ValueError(
                    f"Nilai {self.UNKNOWN} sudah digunakan sebagai kategori asli pada fitur '{f}'"
                )

            vocabulary = values + [self.UNKNOWN]
            self.vocabularies[f] = set(vocabulary)

            V = len(vocabulary)

            for c in self.classes:
                col = X.loc[y == c, f].dropna()
                counts = col.value_counts()
                n_c = len(col)

                self.log_cond[f][c] = {
                    v: math.log((counts.get(v, 0) + self.alpha) / (n_c + self.alpha * V))
                    for v in self.vocabulary
                }

        return self

    def predict_proba(self, X):
        if not hasattr(self, "log_cond"):
            raise RuntimeError("Model belum di-fit")
        
        missing_cols = [f for f in self.features if f not in X.columns]
        if missing_cols:
            raise ValueError(f"Kolom hilang saat predict: {missing_cols}")

        X = X.reset_index(drop=True)
        # matriks (n_sampel, n_kelas) berisi log_prior tiap kolom
        log_scores = np.tile([self.log_prior[c] for c in self.classes], (len(X), 1))

        # skor akhir = log P(kelas) + Σ_fitur log P(nilai_fitur | kelas)
        for f in self.features:
            col = X[f].astype(object)
            missing = col.isna().to_numpy()

            known = self.vocabularies[f]

            # Kategori baru dipetakan ke __UNK__
            mapped_col = col.where(col.isna() | col.isin(known), self.UNKNOWN)

            # akumulasi log P(fitur=f | kelas=c) untuk tiap kelas
            for j, c in enumerate(self.classes):
                s = col.map(self.log_cond[f][c]).to_numpy(dtype=float)
                log_scores[:, j] += np.where(missing, 0.0, s) # NaN -> skip

        # normalisasi, log-sum-exp
        log_scores -= log_scores.max(axis=1, keepdims=True)
        probs = np.exp(log_scores)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X):
        idx = np.argmax(self.predict_proba(X), axis=1)
        return np.array([self.classes[i] for i in idx])