import pandas as pd
import numpy as np
import math

class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        # reset index biar X dan y sejajar
        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        self.features = X.columns.tolist()
        self.classes = sorted(y.unique().tolist())
        K = len(self.classes)

        self.log_prior = {} # log P(vj)
        self.log_cond = {f: {} for f in self.features} # log P(ai | vj)
        self.log_unseen = {f: {} for f in self.features} # log prob utk nilai baru saat testing

        # log P(vj) utk setiap kelas
        for c in self.classes:
            self.log_prior[c] = math.log(((y == c).sum() + self.alpha) / (len(y) + self.alpha * K))

        # log P(ai | vj) + smoothing
        for f in self.features:
            values = X[f].unique()
            V = len(values)
            
            for c in self.classes:
                # ambil data kolom f yg kelasnya c
                col = X.loc[y == c, f]
                counts = col.value_counts()
                n_c = len(col)
                
                # penyebut utk laplace smoothing: n_c + (alpha * V)
                denom = n_c + self.alpha * V

                # log probabilitas utk SEMUA nilai yg ada di dataset
                self.log_cond[f][c] = {
                    v: math.log((counts.get(v, 0) + self.alpha) / denom) for v in values
                }
                
                # log probabilitas jika saat testing ada nilai yg BARU (V+1)
                self.log_unseen[f][c] = math.log(self.alpha / (n_c + self.alpha * (V + 1)))
                
        return self

    def predict_proba(self, X):
        X = X.reset_index(drop=True)
        
        log_scores = np.zeros((len(X), len(self.classes)))

        for j, c in enumerate(self.classes):
            score = np.full(len(X), self.log_prior[c])
            
            for f in self.features:
                table = self.log_cond[f][c]
                unseen = self.log_unseen[f][c]
                
                # cari log P(ai | vj) utk tiap nilai di kolom f. kalo ga ketemu, pakai unseen
                score += X[f].map(lambda v: table.get(v, unseen)).to_numpy()
            
            log_scores[:, j] = score

        # log-sum-exp, mencegah underflow
        log_scores -= log_scores.max(axis=1, keepdims=True)
        probs = np.exp(log_scores)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X):
        idx = np.argmax(self.predict_proba(X), axis=1)
        return np.array([self.classes[i] for i in idx])