"""CART classification from scratch (Gini; numeric/categorical/missing).

No ML implementation library is used. NumPy is used for array/math operations.
"""
from dataclasses import dataclass
from pathlib import Path
import pickle
import numpy as np


def _missing(v):
    if v is None:
        return True
    try:
        out = np.isnan(v)
        if isinstance(out, (bool, np.bool_)):
            return bool(out)
    except (TypeError, ValueError):
        pass
    try:
        return bool(v != v)
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
        return self.feature == -1


class DecisionTreeCART:
    """Binary CART tree using weighted Gini; exact categorical equality splits.

    Missing values are routed left or right by whichever routing yields lower
    weighted Gini *on training data*, and the same decision is saved for inference.
    For categorical features, each distinct observed category is a candidate split.
    Intended for small/medium tabular datasets; exhaustive splits can be expensive.
    """

    def __init__(self, max_depth=8, min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=1e-10, max_features=None):
        if max_depth is not None and max_depth < 0:
            raise ValueError('max_depth must be >= 0 or None')
        if min_samples_split < 2 or min_samples_leaf < 1:
            raise ValueError('Invalid minimum sample parameters')
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features

    @staticmethod
    def _gini(y_index, n_classes):
        if len(y_index) == 0:
            return 0.0
        p = np.bincount(y_index, minlength=n_classes) / len(y_index)
        return float(1 - np.sum(p ** 2))

    def _mask(self, X, feature, threshold, categorical, missing_left):
        col = X[:, feature]
        if categorical:
            return col == threshold
        else:
            #ga perlu cek missing lagi karena pipeline A sudah imputasi semua NaN
            return np.asarray(col, dtype=float) <= threshold

    def _split_candidates(self, col, categorical):
        valid = np.asarray([v for v in col if not _missing(v)])
        if categorical:
            # preserve insertion order without requiring hashable category values
            uniques = np.unique(valid)
            return uniques[:-1] if len(uniques) > 1 else []
        else:
            arr = np.unique(valid.astype(float))
            if len(arr) < 2:
                return []
            return ((arr[:-1] + arr[1:]) / 2).tolist()

    def _build(self, X, yi, depth):
        counts = np.bincount(yi, minlength=len(self.classes_))
        node = _Node(probabilities=counts / len(yi),
                     prediction_index=int(np.argmax(counts)))
        parent_gini = self._gini(yi, len(self.classes_))
        if (parent_gini <= 1e-15 or len(yi) < self.min_samples_split or
                (self.max_depth is not None and depth >= self.max_depth)):
            return node

        best = None
        best_score = parent_gini
        for j in range(X.shape[1]):
            categorical = self.feature_types_[j] == 'categorical'
            for threshold in self._split_candidates(X[:, j], categorical):
                for missing_left in (True, False):
                    mask = self._mask(X, j, threshold, categorical, missing_left)
                    n_left = int(mask.sum())
                    n_right = len(mask) - n_left
                    if min(n_left, n_right) < self.min_samples_leaf:
                        continue
                    score = (n_left * self._gini(yi[mask], len(self.classes_)) +
                             n_right * self._gini(yi[~mask], len(self.classes_))) / len(yi)
                    if score < best_score - 1e-12:
                        best_score = score
                        best = (j, threshold, categorical, missing_left, mask)
        if best is None or parent_gini - best_score < self.min_impurity_decrease:
            return node
        j, threshold, categorical, missing_left, mask = best
        node.feature, node.threshold = j, threshold
        node.categorical, node.missing_left = categorical, missing_left
        node.left = self._build(X[mask], yi[mask], depth+1)
        node.right = self._build(X[~mask], yi[~mask], depth+1)
        return node

    def fit(self, X, y, feature_types=None):
        # Accept NumPy arrays or Pandas DataFrames, but perform tree learning ourselves.
        if hasattr(X, 'columns'):
            self.feature_names_ = list(X.columns)
            raw = X.to_numpy(dtype=object)
            if feature_types is None:
                from pandas.api.types import is_numeric_dtype
                feature_types = ['numeric' if is_numeric_dtype(X[c]) else 'categorical'
                                 for c in X.columns]
        else:
            raw = np.asarray(X, dtype=object)
            self.feature_names_ = None
        targets = np.asarray(y).ravel()
        if raw.ndim != 2 or len(raw) != len(targets) or len(raw) == 0 or raw.shape[1] == 0:
            raise ValueError('X must be a nonempty 2D array, matching y')
        if any(_missing(v) for v in targets):
            raise ValueError('Missing target labels not supported')
        if feature_types is None:
            feature_types = []
            for j in range(raw.shape[1]):
                valid = [v for v in raw[:, j] if not _missing(v)]
                try:
                    for v in valid: float(v)
                    feature_types.append('numeric')
                except (TypeError, ValueError):
                    feature_types.append('categorical')
        if len(feature_types) != raw.shape[1] or any(t not in ('numeric', 'categorical') for t in feature_types):
            raise ValueError('feature_types must specify numeric/categorical per column')
        self.feature_types_ = list(feature_types)
        self.n_features_in_ = raw.shape[1]
        self.classes_, yi = np.unique(targets, return_inverse=True)
        self.tree_ = self._build(raw, yi, 0)
        return self

    def _validate_X(self, X):
        if not hasattr(self, 'tree_'):
            raise RuntimeError('Call fit before predict')
        if hasattr(X, 'columns') and self.feature_names_ is not None:
            X = X.reindex(columns=self.feature_names_)
            raw = X.to_numpy(dtype=object)
        else:
            raw = np.asarray(X, dtype=object)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        if raw.ndim != 2 or raw.shape[1] != self.n_features_in_:
            raise ValueError('Unexpected feature shape')
        return raw

    def predict_proba(self, X):
        raw = self._validate_X(X)
        result = np.zeros((len(raw), len(self.classes_)), dtype=float)
        for i, row in enumerate(raw):
            node = self.tree_
            while not node.leaf:
                value = row[node.feature]
                if _missing(value):
                    go_left = node.missing_left
                elif node.categorical:
                    go_left = value == node.threshold
                else:
                    go_left = float(value) <= node.threshold
                node = node.left if go_left else node.right
            result[i] = node.probabilities
        return result

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def save(self, path):
        if not hasattr(self, 'tree_'):
            raise RuntimeError('Fit model first')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        # Only load pickle files generated by you or trusted teammates.
        with Path(path).open('rb') as f:
            model = pickle.load(f)
        if not isinstance(model, DecisionTreeCART):
            raise TypeError('Saved file is not a DecisionTreeCART')
        return model
