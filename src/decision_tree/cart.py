"""CART classification from scratch: efficient exact Gini splitting.

References (conceptual, NOT copied implementation):
[6]-[14] N. U. Maulidevi, IF3270 Decision Tree Learning / Issues in DTL,
         ITB lecture videos, 2021 (bibliography supplied in assignment).
[23] GeeksforGeeks, 'CART (Classification And Regression Tree)',
     https://www.geeksforgeeks.org/machine-learning/cart-classification-and-regression-tree-in-machine-learning/
[24] GeeksforGeeks, 'Decision Tree Algorithms',
     https://www.geeksforgeeks.org/machine-learning/decision-tree-algorithms/
[27] L. Breiman, J. H. Friedman, R. A. Olshen, C. J. Stone,
     Classification and Regression Trees, Wadsworth, 1984.
[28] scikit-learn User Guide, Decision Trees / Mathematical formulation,
     https://scikit-learn.org/stable/modules/tree.html

All learning, split evaluation, and inference are implemented here using NumPy;
no sklearn estimator or sklearn tree routines are used.

Split search: sort numerical values once per feature/node, then compute all
left/right class frequencies with cumulative sums, avoiding Python loops over
candidate thresholds. Categorical splits are one category vs the rest.
"""
from dataclasses import dataclass
from pathlib import Path
import pickle
import numpy as np
from collections import deque
import matplotlib.pyplot as plt



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
    """Binary, greedy CART classifier with exact Gini candidate search.

    max_features: None (all), positive integer, proportion (0,1], 'sqrt', 'log2'.
    Notes: no post-pruning, sample weighting, or surrogate splits; probabilities
    are unsmoothed empirical class frequencies at a leaf.
    """

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
        # Returns (score, threshold, missing_left) or None.
        values = np.asarray([np.nan if _missing(v) else float(v) for v in column], dtype=float)
        missing = np.isnan(values)
        x = values[~missing]
        if len(x) < 2:
            return None
        order = np.argsort(x, kind='stable')
        xs = x[order]
        ys = yi[~missing][order]
        # Candidates only between distinct consecutive values.
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
        # Evaluate missing->left first to preserve old implementation tie preference.
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
            # Midpoint stable against overflow for finite positive values.
            threshold = float(xs[positions[ix]] + (xs[positions[ix] + 1] - xs[positions[ix]]) / 2)
            if score < best_score - 1e-12:
                best_score = score
                best = (score, threshold, missing_left)
        return best

    def _best_categorical(self, column, yi, parent_counts):
        values = []
        for v in column:
            if not _missing(v) and not any(v == u for u in values):
                values.append(v)
        if len(values) < 2:
            return None
        miss = np.array([_missing(v) for v in column], dtype=bool)
        miss_counts = np.bincount(yi[miss], minlength=len(self.classes_))
        best = None
        best_score = np.inf
        for v in values[:-1]:
            equals = np.array([not miss[i] and a == v for i, a in enumerate(column)], dtype=bool)
            for missing_left in (True, False) if miss.any() else (True,):
                left_counts = np.bincount(yi[equals], minlength=len(self.classes_))
                if missing_left:
                    left_counts = left_counts + miss_counts
                right_counts = parent_counts - left_counts
                nl, nr = int(left_counts.sum()), int(right_counts.sum())
                if min(nl, nr) < self.min_samples_leaf:
                    continue
                score = float((nl * self._gini_from_counts(left_counts) +
                               nr * self._gini_from_counts(right_counts)) / len(column))
                if score < best_score - 1e-12:
                    best_score = score
                    best = (score, v, missing_left)
        return best

    def _mask(self, X, feature, threshold, categorical, missing_left):
        col = X[:, feature]
        if not categorical:
            numeric = np.asarray([np.nan if _missing(v) else float(v) for v in col])
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
            raise ValueError('max_depth must be a nonnegative integer or None')
        if (not isinstance(self.min_samples_split, (int, np.integer)) or self.min_samples_split < 2 or
            not isinstance(self.min_samples_leaf, (int, np.integer)) or self.min_samples_leaf < 1):
            raise ValueError('Invalid minimum sample parameters')
        if not np.isfinite(self.min_impurity_decrease) or self.min_impurity_decrease < 0:
            raise ValueError('min_impurity_decrease must be finite and >= 0')
        mf = self.max_features
        if mf is not None:
            if isinstance(mf, str):
                if mf not in ('sqrt', 'log2'):
                    raise ValueError('max_features must be None, sqrt, log2, integer, or fraction')
            elif isinstance(mf, (int, np.integer)):
                if not (1 <= mf <= n_features):
                    raise ValueError('max_features integer must be 1..n_features')
            elif isinstance(mf, (float, np.floating)):
                if not (0 < mf <= 1):
                    raise ValueError('max_features fraction must be in (0,1]')
            else:
                raise ValueError('Invalid max_features')

    def fit(self, X, y, feature_types=None):
        if hasattr(X, 'columns'):
            self.feature_names_ = list(X.columns)
            if len(set(self.feature_names_)) != len(self.feature_names_):
                raise ValueError('Feature names must be unique')
            raw = X.to_numpy(dtype=object)
            if feature_types is None:
                from pandas.api.types import is_numeric_dtype
                feature_types = ['numeric' if is_numeric_dtype(X[c]) else 'categorical'
                                 for c in X.columns]
        else:
            self.feature_names_ = None
            raw = np.asarray(X, dtype=object)
        targets = np.asarray(y).ravel()
        if raw.ndim != 2 or len(raw) == 0 or raw.shape[1] == 0 or len(targets) != len(raw):
            raise ValueError('X must be a nonempty 2D matrix with matching y')
        if any(_missing(v) for v in targets):
            raise ValueError('Missing target labels are not supported')
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
            raise ValueError('feature_types must contain numeric/categorical per column')
        self.feature_types_ = list(feature_types)
        self.n_features_in_ = raw.shape[1]
        # Reject +/- infinity, because split logic assumes finite nonmissing numbers.
        for j, t in enumerate(self.feature_types_):
            if t == 'numeric':
                for v in raw[:, j]:
                    if not _missing(v) and not np.isfinite(float(v)):
                        raise ValueError(f'Numeric feature {j} contains infinity')
        self.classes_, yi = np.unique(targets, return_inverse=True)
        self._rng = np.random.default_rng(self.random_state)
        self.tree_ = self._build(raw, yi, 0)
        return self

    def _validate_X(self, X):
        if not hasattr(self, 'tree_'):
            raise RuntimeError('Call fit before predict')
        if hasattr(X, 'columns') and self.feature_names_ is not None:
            missing = set(self.feature_names_) - set(X.columns)
            if missing:
                raise ValueError(f'Missing columns: {sorted(missing)}')
            X = X.loc[:, self.feature_names_]
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
            raise RuntimeError('Fit model first')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with Path(path).open('rb') as f:
            result = pickle.load(f)  # Only load trusted files.
        if not isinstance(result, DecisionTreeCART):
            raise TypeError('Saved object is not DecisionTreeCART')
        return result
    
    def export_tree_image(self, filepath="decision_tree.png", top_n=5):
        """
        Menghasilkan gambar percabangan tree dan menyimpannya ke file.

        Parameters:
        - filepath (str | Path): Path/nama file output (misal 'tree.png', 'tree.pdf', 'tree.svg').
        - top_n (int | None): Jumlah maksimal percabangan (internal split) top-N yang ditampilkan
                              berdasarkan urutan level dari root (BFS). Jika None, tampilkan semua.
        """
        if not hasattr(self, "tree_"):
            raise RuntimeError("Call fit before exporting tree image")
        if top_n is not None and (not isinstance(top_n, (int, np.integer)) or top_n < 1):
            raise ValueError("top_n must be a positive integer or None")

        # 1. Tentukan node percabangan mana saja yang masuk dalam Top-N (BFS dari root)
        expanded_nodes = set()
        queue = deque([self.tree_])
        while queue:
            curr = queue.popleft()
            if curr is None or curr.leaf:
                continue
            if top_n is not None and len(expanded_nodes) >= top_n:
                break
            expanded_nodes.add(id(curr))
            queue.append(curr.left)
            queue.append(curr.right)

        # 2. Hitung lebar subtree secara rekursif agar kotak node tidak tumpang tindih
        def _subtree_width(node):
            if node is None or node.leaf or id(node) not in expanded_nodes:
                return 1
            return _subtree_width(node.left) + _subtree_width(node.right)

        # 3. Hitung kedalaman visual maksimum untuk mengatur tinggi kanvas
        def _visual_depth(node):
            if node is None or node.leaf or id(node) not in expanded_nodes:
                return 1
            return 1 + max(_visual_depth(node.left), _visual_depth(node.right))

        total_width = _subtree_width(self.tree_)
        max_vis_depth = _visual_depth(self.tree_)

        fig_w = max(10.0, total_width * 2.6)
        fig_h = max(6.0, max_vis_depth * 2.2)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")

        # 4. Format teks di dalam setiap kotak node
        def _format_node_text(node, is_truncated):
            pred_label = self.classes_[node.prediction_index]
            probs_str = np.array2string(node.probabilities, precision=2, separator=", ")

            if node.leaf:
                return f"LEAF\nPred: {pred_label}\nProbs: {probs_str}"

            feat_name = (
                self.feature_names_[node.feature]
                if getattr(self, "feature_names_", None) is not None
                else f"X[{node.feature}]"
            )
            if node.categorical:
                cond = f"{feat_name} == {node.threshold!r}"
            else:
                cond = f"{feat_name} <= {node.threshold:.4g}"

            miss_dir = "Left" if node.missing_left else "Right"
            header = f"{cond}\n(NaN -> {miss_dir})"
            if is_truncated:
                return f"{header}\n[...cabang disembunyikan]\nPred: {pred_label}"
            return f"{header}\nPred: {pred_label} | {probs_str}"

        # 5. Gambar node dan garis percabangan secara rekursif
        def _draw(node, x, y, width_span):
            if node is None:
                return

            is_truncated = (not node.leaf) and (id(node) not in expanded_nodes)
            text = _format_node_text(node, is_truncated)

            if node.leaf:
                facecolor = "#d4edda"   # Hijau muda untuk Leaf
                edgecolor = "#28a745"
            elif is_truncated:
                facecolor = "#fff3cd"   # Kuning muda untuk cabang yang dipotong oleh top_n
                edgecolor = "#d39e00"
            else:
                facecolor = "#d1ecf1"   # Biru muda untuk Split aktif
                edgecolor = "#17a2b8"

            ax.text(
                x, y, text,
                ha="center", va="center", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.5", facecolor=facecolor, edgecolor=edgecolor, lw=1.5),
                zorder=3,
            )

            if node.leaf or is_truncated:
                return

            left_w = _subtree_width(node.left)
            right_w = _subtree_width(node.right)
            tot_w = left_w + right_w

            left_span = width_span * (left_w / tot_w)
            right_span = width_span * (right_w / tot_w)

            x_left = x - (width_span / 2.0) + (left_span / 2.0)
            x_right = x + (width_span / 2.0) - (right_span / 2.0)
            y_next = y - 1.0

            # Cabang Kiri (True)
            if node.left is not None:
                ax.plot([x, x_left], [y, y_next], color="#555555", lw=1.3, zorder=1)
                ax.text(
                    (x + x_left) / 2.0, (y + y_next) / 2.0, "True",
                    ha="right", va="center", fontsize=8, color="#155724",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8),
                    zorder=2,
                )
                _draw(node.left, x_left, y_next, left_span)

            # Cabang Kanan (False)
            if node.right is not None:
                ax.plot([x, x_right], [y, y_next], color="#555555", lw=1.3, zorder=1)
                ax.text(
                    (x + x_right) / 2.0, (y + y_next) / 2.0, "False",
                    ha="left", va="center", fontsize=8, color="#721c24",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8),
                    zorder=2,
                )
                _draw(node.right, x_right, y_next, right_span)

        _draw(self.tree_, x=0.0, y=0.0, width_span=float(total_width))

        ax.set_xlim(-total_width / 2.0 - 0.5, total_width / 2.0 + 0.5)
        ax.set_ylim(-max_vis_depth + 0.3, 0.7)

        title_suffix = f"Top-{top_n} Percabangan" if top_n is not None else "Semua Percabangan"
        plt.title(f"Decision Tree CART ({title_suffix})", fontsize=12, fontweight="bold", pad=12)

        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out_path