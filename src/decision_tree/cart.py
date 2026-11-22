"""CART classifier from scratch, optimized Gini split search.

Numerical splits use sorted values and cumulative class counts instead of
recomputing a boolean mask/Gini for every threshold. Supports categorical
features and missing values; uses NumPy for numerical computations only.
"""
from dataclasses import dataclass
from pathlib import Path
import pickle
import numpy as np


def _missing(value):
    if value is None:
        return True
    try:
        result = np.isnan(value)
        if isinstance(result, (bool, np.bool_)):
            return bool(result)
    except (TypeError, ValueError):
        pass
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
        return self.feature == -1


class DecisionTreeCART:
    """Binary CART with weighted Gini; keeps original public API.

    Numeric candidates are evaluated in O(n log n + n*K) per feature/node,
    instead of approximately O(n^2) with repeated masks. K = classes.
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
        return float(1.0 - np.sum(p * p))

    @staticmethod
    def _gini_counts(counts):
        counts = np.asarray(counts, dtype=float)
        n = counts.sum(axis=-1)
        sq = np.sum(counts * counts, axis=-1)
        return np.divide(sq, n, out=np.zeros_like(n, dtype=float), where=n > 0)

    def _best_numeric(self, col, yi, n_classes, parent_gini):
        # One conversion per feature/node, rather than per candidate split.
        missing = np.fromiter((_missing(v) for v in col), dtype=bool, count=len(col))
        valid_idx = np.flatnonzero(~missing)
        if len(valid_idx) < 2:
            return None
        vals = np.asarray(col[valid_idx], dtype=float)
        # NaN is missing, infinity cannot be split safely into finite midpoints.
        if not np.isfinite(vals).all():
            return None
        order = np.argsort(vals, kind='stable')
        vals = vals[order]
        sorted_y = yi[valid_idx[order]]
        boundaries = np.flatnonzero(vals[:-1] < vals[1:])
        if not len(boundaries):
            return None
        n_valid = len(vals)
        n_total = len(col)
        missing_counts = np.bincount(yi[missing], minlength=n_classes)
        valid_counts = np.bincount(sorted_y, minlength=n_classes)
        # Cumulative class counts at each possible boundary.
        one_hot = np.eye(n_classes, dtype=np.int64)[sorted_y]
        left_counts = np.cumsum(one_hot, axis=0)[boundaries]
        right_counts = valid_counts - left_counts
        left_size = boundaries + 1
        right_size = n_valid - left_size
        best = None
        best_score = parent_gini
        for missing_left in (True, False):
            if missing_left:
                lc = left_counts + missing_counts
                rc = right_counts
                nl = left_size + int(missing.sum())
                nr = right_size
            else:
                lc = left_counts
                rc = right_counts + missing_counts
                nl = left_size
                nr = right_size + int(missing.sum())
            legal = (nl >= self.min_samples_leaf) & (nr >= self.min_samples_leaf)
            if not np.any(legal):
                continue
            # Weighted Gini = 1 - (sum squared counts / child size) / n_total.
            scores = 1.0 - (self._gini_counts(lc) + self._gini_counts(rc)) / n_total
            scores[~legal] = np.inf
            k = int(np.argmin(scores))
            score = float(scores[k])
            if score < best_score - 1e-12:
                b = int(boundaries[k])
                threshold = float(vals[b] / 2.0 + vals[b + 1] / 2.0)
                # For very close extreme floats, avoid midpoint rounding to upper.
                if threshold >= vals[b + 1]:
                    threshold = float(vals[b])
                best_score = score
                best = (threshold, missing_left, score)
        return best

    def _best_categorical(self, col, yi, n_classes, parent_gini):
        missing = np.fromiter((_missing(v) for v in col), dtype=bool, count=len(col))
        valid = [v for v, miss in zip(col, missing) if not miss]
        unique = []
        for v in valid:
            if not any(v == seen for seen in unique):
                unique.append(v)
        if len(unique) < 2:
            return None
        missing_counts = np.bincount(yi[missing], minlength=n_classes)
        all_counts = np.bincount(yi, minlength=n_classes)
        best = None
        best_score = parent_gini
        # The last category is intentionally omitted, as in the original implementation.
        for category in unique[:-1]:
            mask = np.fromiter((not _missing(v) and v == category for v in col), dtype=bool, count=len(col))
            cat_counts = np.bincount(yi[mask], minlength=n_classes)
            for missing_left in (True, False):
                left_counts = cat_counts + (missing_counts if missing_left else 0)
                right_counts = all_counts - left_counts
                nl = int(left_counts.sum())
                nr = len(col) - nl
                if min(nl, nr) < self.min_samples_leaf:
                    continue
                score = 1.0 - (self._gini_counts(left_counts) + self._gini_counts(right_counts)) / len(col)
                if score < best_score - 1e-12:
                    best_score = score
                    best = (category, missing_left, score)
        return best

    def _mask(self, X, feature, threshold, categorical, missing_left):
        col = X[:, feature]
        if not categorical:
            try:
                numbers = np.asarray(col, dtype=float)
                missing = np.isnan(numbers)
                return np.where(missing, missing_left, numbers <= threshold)
            except (TypeError, ValueError):
                pass
        result = np.zeros(len(col), dtype=bool)
        for i, value in enumerate(col):
            if _missing(value):
                result[i] = missing_left
            elif categorical:
                result[i] = value == threshold
            else:
                result[i] = float(value) <= threshold
        return result

    def _build(self, X, yi, depth):
        n_classes = len(self.classes_)
        counts = np.bincount(yi, minlength=n_classes)
        node = _Node(probabilities=counts / len(yi), prediction_index=int(np.argmax(counts)))
        parent_gini = self._gini(yi, n_classes)
        if (parent_gini <= 1e-15 or len(yi) < self.min_samples_split or
                (self.max_depth is not None and depth >= self.max_depth)):
            return node
        best = None
        best_score = parent_gini
        for j in range(X.shape[1]):
            categorical = self.feature_types_[j] == 'categorical'
            candidate = (self._best_categorical(X[:, j], yi, n_classes, parent_gini) if categorical
                         else self._best_numeric(X[:, j], yi, n_classes, parent_gini))
            if candidate is None:
                continue
            threshold, missing_left, score = candidate
            if score < best_score - 1e-12:
                best_score = score
                best = (j, threshold, categorical, missing_left)
        if best is None or parent_gini - best_score < self.min_impurity_decrease:
            return node
        j, threshold, categorical, missing_left = best
        mask = self._mask(X, j, threshold, categorical, missing_left)
        node.feature, node.threshold = j, threshold
        node.categorical, node.missing_left = categorical, missing_left
        node.left = self._build(X[mask], yi[mask], depth + 1)
        node.right = self._build(X[~mask], yi[~mask], depth + 1)
        return node

    def fit(self, X, y, feature_types=None):
        if hasattr(X, 'columns'):
            self.feature_names_ = list(X.columns)
            raw = X.to_numpy(dtype=object)
            if feature_types is None:
                from pandas.api.types import is_numeric_dtype
                feature_types = ['numeric' if is_numeric_dtype(X[c]) else 'categorical' for c in X.columns]
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
                    for v in valid:
                        float(v)
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
        # Load only files you or your trusted teammates generated.
        with Path(path).open('rb') as f:
            model = pickle.load(f)
        if not isinstance(model, DecisionTreeCART):
            raise TypeError('Saved file is not a DecisionTreeCART')
        return model
