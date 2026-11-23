from dataclasses import dataclass
from pathlib import Path
import pickle
import numpy as np


def _missing(value):
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        try:
            return bool(value != value)
        except (TypeError, ValueError):
            return False


@dataclass
class _Node:
    probabilities: np.ndarray
    prediction_index: int
    feature: int = -1
    threshold: object = None
    categorical: bool = False
    missing_left: bool = True
    left: object = None
    right: object = None

    @property
    def leaf(self):
        return self.feature < 0


class DecisionTreeCART:

    def __init__(self, max_depth=8, min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=1e-10, max_features=None, random_state=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.random_state = random_state

    @staticmethod
    def _gini_from_counts(counts):
        counts = np.asarray(counts, dtype=float)
        totals = counts.sum(axis=-1)
        return 1.0 - np.divide((counts ** 2).sum(axis=-1), totals ** 2,
                               out=np.zeros_like(totals, dtype=float), where=totals > 0)

    def _feature_subset(self):
        p = self.n_features_in_
        mf = self.max_features
        if mf is None:
            return np.arange(p)
        if isinstance(mf, str):
            n = max(1, int(np.sqrt(p))) if mf == 'sqrt' else max(1, int(np.log2(p)))
        elif isinstance(mf, (int, np.integer)):
            n = int(mf)
        else:
            n = max(1, int(float(mf) * p))
        if n == p:
            return np.arange(p)
        return np.sort(self._rng.choice(p, size=n, replace=False))


    def _best_numeric(self, column, yi, parent_counts):
        # Cari pemisahan numerik terbaik.
        # Data numerik tidak perlu dikonversi per elemen.
        if np.issubdtype(column.dtype, np.number): values = column.astype(float, copy=False)
        else:
            values = np.fromiter(
                (np.nan if _missing(v) else float(v) for v in column),
                dtype=float, count=len(column)
            )
        missing = np.isnan(values)
        x = values[~missing]
        if len(x) < 2:
            return None
        order = np.argsort(x, kind='stable')
        xs = x[order]
        ys = yi[~missing][order]
        # Ambang berada di antara nilai berbeda.
        positions = np.flatnonzero(xs[:-1] < xs[1:])
        if len(positions) == 0:
            return None

        k = len(self.classes_)
        counts_per_position = np.zeros((len(xs), k), dtype=np.int64)
        counts_per_position[np.arange(len(xs)), ys] = 1
        left_valid = np.cumsum(counts_per_position, axis=0)[positions]
        valid_counts = np.bincount(ys, minlength=k)
        right_valid = valid_counts - left_valid
        missing_counts = parent_counts - valid_counts

        best = None
        best_score = np.inf
        # Uji nilai kosong ke cabang kiri lebih dulu.
        for missing_left in (True, False) if missing.any() else (True,):
            lc = left_valid + (missing_counts if missing_left else 0)
            rc = right_valid + (0 if missing_left else missing_counts)
            nl = lc.sum(axis=1)
            nr = rc.sum(axis=1)
            allowed = (nl >= self.min_samples_leaf) & (nr >= self.min_samples_leaf)
            if not allowed.any():
                continue
            scores = (nl * self._gini_from_counts(lc) +
                      nr * self._gini_from_counts(rc)) / len(column)
            scores[~allowed] = np.inf
            ix = int(np.argmin(scores))
            score = float(scores[ix])
            # Hitung titik tengah sebagai ambang.
            threshold = float(xs[positions[ix]] + (xs[positions[ix] + 1] - xs[positions[ix]]) / 2)
            if score < best_score - 1e-12:
                best_score = score
                best = (score, threshold, missing_left)
        return best

    def _best_categorical(self, column, yi, parent_counts):
        # Pisahkan satu kategori dari kategori lainnya.
        missing = np.array([_missing(v) for v in column], dtype=bool)
        observed = []
        for value in column[~missing]:
            if not any(value == item for item in observed):
                observed.append(value)
        if not observed:
            return None
        if len(observed) == 1 and not missing.any():
            return None

        missing_counts = np.bincount(yi[missing], minlength=len(self.classes_))
        best = None
        best_score = np.inf
        for value in observed:
            equal = np.array([not missing[i] and v == value
                              for i, v in enumerate(column)], dtype=bool)
            value_counts = np.bincount(yi[equal], minlength=len(self.classes_))
            for missing_left in ((True, False) if missing.any() else (True,)):
                left_counts = value_counts + (missing_counts if missing_left else 0)
                right_counts = parent_counts - left_counts
                n_left = int(left_counts.sum())
                n_right = int(right_counts.sum())
                if min(n_left, n_right) < self.min_samples_leaf:
                    continue
                score = float((n_left * self._gini_from_counts(left_counts) +
                               n_right * self._gini_from_counts(right_counts)) / len(column))
                if score < best_score - 1e-12:
                    best_score = score
                    best = (score, value, missing_left)
        return best

    def _mask(self, X, feature, threshold, categorical, missing_left):
        col = X[:, feature]
        if not categorical:
            if np.issubdtype(col.dtype, np.number):
                numeric = col.astype(float, copy=False)
            else:
                numeric = np.fromiter(
                    (np.nan if _missing(v) else float(v) for v in col),
                    dtype=float, count=len(col)
                )
            mask = numeric <= threshold
            if missing_left:
                mask[np.isnan(numeric)] = True
            return mask
        return np.array([missing_left if _missing(v) else v == threshold for v in col], dtype=bool)

    def _build(self, X, yi, depth):
        n = len(yi)
        counts = np.bincount(yi, minlength=len(self.classes_))
        node = _Node(probabilities=counts / n, prediction_index=int(np.argmax(counts)))
        parent_gini = float(self._gini_from_counts(counts))
        if (parent_gini <= 1e-15 or n < self.min_samples_split or
                (self.max_depth is not None and depth >= self.max_depth) or
                n < 2 * self.min_samples_leaf):
            return node

        best = None
        best_score = parent_gini
        for j in self._feature_subset():
            categorical = self.feature_types_[j] == 'categorical'
            candidate = (self._best_categorical(X[:, j], yi, counts) if categorical
                         else self._best_numeric(X[:, j], yi, counts))
            if candidate is None:
                continue
            score, threshold, missing_left = candidate
            if score < best_score - 1e-12:
                best_score = score
                best = (j, threshold, categorical, missing_left)

        if best is None or parent_gini - best_score < self.min_impurity_decrease:
            return node
        j, threshold, categorical, missing_left = best
        mask = self._mask(X, j, threshold, categorical, missing_left)
        node.feature = int(j)
        node.threshold = threshold
        node.categorical = categorical
        node.missing_left = missing_left
        node.left = self._build(X[mask], yi[mask], depth + 1)
        node.right = self._build(X[~mask], yi[~mask], depth + 1)
        return node

    def _validate_parameters(self, n_features):
        if self.max_depth is not None and (not isinstance(self.max_depth, (int, np.integer)) or self.max_depth < 0):
            raise ValueError('max_depth harus bilangan bulat nonnegatif atau None')
        if (not isinstance(self.min_samples_split, (int, np.integer)) or self.min_samples_split < 2 or
            not isinstance(self.min_samples_leaf, (int, np.integer)) or self.min_samples_leaf < 1):
            raise ValueError('Parameter jumlah sampel tidak valid')
        if not np.isfinite(self.min_impurity_decrease) or self.min_impurity_decrease < 0:
            raise ValueError('min_impurity_decrease harus bernilai hingga dan >= 0')
        mf = self.max_features
        if mf is not None:
            if isinstance(mf, str):
                if mf not in ('sqrt', 'log2'):
                    raise ValueError('max_features tidak valid')
            elif isinstance(mf, (int, np.integer)):
                if not (1 <= mf <= n_features):
                    raise ValueError('max_features melebihi jumlah fitur')
            elif isinstance(mf, (float, np.floating)):
                if not (0 < mf <= 1):
                    raise ValueError('Rasio max_features harus berada di (0,1]')
            else:
                raise ValueError('max_features tidak valid')

    def fit(self, X, y, feature_types=None):
        if hasattr(X, 'columns'):
            self.feature_names_ = list(X.columns)
            if len(set(self.feature_names_)) != len(self.feature_names_):
                raise ValueError('Nama fitur harus unik')
            raw = X.to_numpy()
            if feature_types is None:
                from pandas.api.types import is_numeric_dtype
                feature_types = ['numeric' if is_numeric_dtype(X[c]) else 'categorical'
                                 for c in X.columns]
        else:
            self.feature_names_ = None
            raw = np.asarray(X)
        targets = np.asarray(y).ravel()
        if raw.ndim != 2 or len(raw) == 0 or raw.shape[1] == 0 or len(targets) != len(raw):
            raise ValueError('X harus matriks 2D dan panjangnya sesuai y')
        if any(_missing(v) for v in targets):
            raise ValueError('Label target tidak boleh kosong')
        self._validate_parameters(raw.shape[1])
        if feature_types is None:
            feature_types = []
            for j in range(raw.shape[1]):
                try:
                    for v in raw[:, j]:
                        if not _missing(v):
                            float(v)
                    feature_types.append('numeric')
                except (TypeError, ValueError):
                    feature_types.append('categorical')
        if len(feature_types) != raw.shape[1] or any(t not in ('numeric', 'categorical') for t in feature_types):
            raise ValueError('feature_types harus berisi numeric/categorical untuk setiap kolom')
        self.feature_types_ = list(feature_types)
        self.n_features_in_ = raw.shape[1]
        # Pertahankan array numerik agar proses split lebih cepat.
        if all(t == 'numeric' for t in self.feature_types_):
            raw = raw.astype(float, copy=False)
        # Tolak nilai tak hingga.
        for j, t in enumerate(self.feature_types_):
            if t == 'numeric':
                col = raw[:, j]
                if np.issubdtype(col.dtype, np.number):
                    invalid = np.isinf(col.astype(float, copy=False)).any()
                else:
                    invalid = any(not _missing(v) and not np.isfinite(float(v)) for v in col)
                if invalid:
                    raise ValueError(f'Fitur numerik {j} mengandung nilai tak hingga')
        self.classes_, yi = np.unique(targets, return_inverse=True)
        self._rng = np.random.default_rng(self.random_state)
        self.tree_ = self._build(raw, yi, 0)
        return self

    def _validate_X(self, X):
        if not hasattr(self, 'tree_'):
            raise RuntimeError('Jalankan fit sebelum predict')
        if hasattr(X, 'columns') and self.feature_names_ is not None:
            missing = set(self.feature_names_) - set(X.columns)
            if missing:
                raise ValueError(f'Kolom tidak ditemukan: {sorted(missing)}')
            X = X.loc[:, self.feature_names_]
            raw = X.to_numpy(dtype=object)
        else:
            raw = np.asarray(X, dtype=object)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        if raw.ndim != 2 or raw.shape[1] != self.n_features_in_:
            raise ValueError('Jumlah fitur tidak sesuai')
        return raw

    def predict_proba(self, X):
        raw = self._validate_X(X)
        result = np.empty((len(raw), len(self.classes_)), dtype=float)
        for i, row in enumerate(raw):
            node = self.tree_
            while not node.leaf:
                v = row[node.feature]
                if _missing(v):
                    left = node.missing_left
                elif node.categorical:
                    left = v == node.threshold
                else:
                    left = float(v) <= node.threshold
                node = node.left if left else node.right
            result[i] = node.probabilities
        return result

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def save(self, path):
        if not hasattr(self, 'tree_'):
            raise RuntimeError('Latih model terlebih dahulu')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with Path(path).open('rb') as f:
            result = pickle.load(f)  # Muat hanya file tepercaya.
        if not isinstance(result, DecisionTreeCART):
            raise TypeError('File bukan model DecisionTreeCART')
        return result
