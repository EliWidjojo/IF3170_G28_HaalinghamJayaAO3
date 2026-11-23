import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D



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
                    for v in vocabulary
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
                s = mapped_col.map(self.log_cond[f][c]).to_numpy(dtype=float)
                log_scores[:, j] += np.where(missing, 0.0, s) # NaN -> skip

        # normalisasi, log-sum-exp
        log_scores -= log_scores.max(axis=1, keepdims=True)
        probs = np.exp(log_scores)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X):
        idx = np.argmax(self.predict_proba(X), axis=1)
        return np.array([self.classes[i] for i in idx])


    def _mutual_information(s1, s2):
        """Menghitung Mutual Information I(X; Y) untuk dua variabel diskrit/kategorikal."""
        df_pair = pd.DataFrame({"x": s1, "y": s2}).dropna()
        if len(df_pair) == 0:
            return 0.0
        joint = pd.crosstab(df_pair["x"], df_pair["y"], normalize=True)
        px = joint.sum(axis=1).to_numpy()[:, None]
        py = joint.sum(axis=0).to_numpy()[None, :]
        pxy = joint.to_numpy()

        mask = pxy > 0
        mi = np.sum(pxy[mask] * np.log2(pxy[mask] / (px * py)[mask]))
        return float(max(0.0, mi))


    def plot_naive_bayes_graph(
        nb_model,
        X=None,
        y=None,
        class_name="Target (Class)",
        show_feature_links=True,
        feature_mi_threshold=0.05,
        top_k_features=None,
        filepath="naive_bayes_graph.png",
    ):
        """
        Menghasilkan gambar graf hubungan antarfitur dan variabel kelas untuk NaiveBayes.

        Parameters:
        - nb_model: Instance NaiveBayes yang sudah di-fit.
        - X (pd.DataFrame, opsional): Data fitur untuk menghitung bobot Mutual Information.
        - y (pd.Series/array, opsional): Label kelas untuk menghitung Mutual Information fitur-kelas.
        - class_name (str): Nama label node variabel kelas.
        - show_feature_links (bool): Jika True, menampilkan garis putus-putus hubungan empiris antarfitur.
        - feature_mi_threshold (float): Ambang batas MI minimal untuk menampilkan garis antarfitur.
        - top_k_features (int | None): Batasi jumlah fitur teratas (berdasarkan MI dengan kelas) agar graf rapi.
        - filepath (str): Nama file output gambar.
        """
        if not hasattr(nb_model, "log_cond"):
            raise RuntimeError("Model NaiveBayes belum di-fit.")

        features = list(nb_model.features)

        # Hitung Mutual Information antara tiap fitur dan kelas jika X & y diberikan
        fc_mi = {}
        if X is not None and y is not None:
            y_series = pd.Series(y).reset_index(drop=True)
            X_reset = X.reset_index(drop=True)
            for f in features:
                fc_mi[f] = _mutual_information(X_reset[f], y_series)

            # Urutkan fitur berdasarkan MI tertinggi dengan kelas
            features = sorted(features, key=lambda f: fc_mi[f], reverse=True)
            if top_k_features is not None:
                features = features[:top_k_features]
        elif top_k_features is not None:
            features = features[:top_k_features]

        G = nx.DiGraph()

        # Ringkasan prior probabilitas P(Y) dari model
        priors_text = ", ".join(
            f"{c}: {math.exp(nb_model.log_prior[c]):.2f}" for c in nb_model.classes
        )
        class_node_label = f"{class_name}\nP(Y) = [{priors_text}]"
        G.add_node(class_node_label, node_type="class")

        # Tambahkan node fitur dan edge berarah Class -> Fitur (struktur graf Naive Bayes)
        for f in features:
            n_vocab = len(nb_model.vocabularies[f]) - 1  # dikurangi token __UNK__
            f_label = f"{f}\n(|V|={n_vocab})"
            G.add_node(f_label, node_type="feature", raw_name=f)
            weight = fc_mi.get(f, 0.0)
            G.add_edge(class_node_label, f_label, relation="class_to_feature", weight=weight)

        # Tambahkan hubungan empiris antarfitur (garis putus-putus) jika diminta
        feature_edges = []
        if show_feature_links and X is not None:
            X_reset = X.reset_index(drop=True)
            feature_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "feature"]
            for i in range(len(feature_nodes)):
                for j in range(i + 1, len(feature_nodes)):
                    u, v = feature_nodes[i], feature_nodes[j]
                    raw_u = G.nodes[u]["raw_name"]
                    raw_v = G.nodes[v]["raw_name"]
                    mi_ff = _mutual_information(X_reset[raw_u], X_reset[raw_v])
                    if mi_ff >= feature_mi_threshold:
                        feature_edges.append((u, v, mi_ff))

        # Tata letak radial: Variabel Kelas di tengah, fitur-fitur mengelilinginya
        pos = {class_node_label: np.array([0.0, 0.0])}
        n_feat = len(features)
        radius = 2.6
        feature_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "feature"]

        for idx, f_node in enumerate(feature_nodes):
            angle = 2 * np.pi * idx / max(1, n_feat) + (np.pi / 2)
            pos[f_node] = np.array([radius * np.cos(angle), radius * np.sin(angle)])

        fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
        ax.axis("off")

        # 1. Gambar garis hubungan antarfitur (dashed, abu-abu)
        if feature_edges:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v) for u, v, _ in feature_edges],
                ax=ax,
                style="dashed",
                edge_color="#adb5bd",
                width=[1.0 + 3.0 * w for _, _, w in feature_edges],
                arrows=False,
                connectionstyle="arc3,rad=0.12",
            )
            ff_edge_labels = {(u, v): f"MI={w:.2f}" for u, v, w in feature_edges}
            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels=ff_edge_labels,
                font_size=7,
                font_color="#6c757d",
                label_pos=0.5,
                ax=ax,
            )

        # 2. Gambar panah berarah Class -> Fitur (solid, biru)
        cf_edges = [(u, v) for u, v, d in G.edges(data=True) if d["relation"] == "class_to_feature"]
        cf_weights = [G.edges[e]["weight"] for e in cf_edges]
        widths = [1.5 + (6.0 * w if fc_mi else 0.5) for w in cf_weights]

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=cf_edges,
            ax=ax,
            edge_color="#0d6efd",
            width=widths,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=18,
            node_size=3600,
        )

        if fc_mi:
            cf_edge_labels = {(u, v): f"MI={G.edges[(u, v)]['weight']:.3f}" for u, v in cf_edges}
            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels=cf_edge_labels,
                font_size=8,
                font_color="#084298",
                label_pos=0.55,
                ax=ax,
            )

        # 3. Gambar Node Kelas (Tengah) dan Node Fitur (Sekeliling)
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[class_node_label],
            node_color="#f8d7da",
            edgecolors="#dc3545",
            linewidths=2.2,
            node_size=4800,
            ax=ax,
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=feature_nodes,
            node_color="#d1ecf1",
            edgecolors="#0dcaf0",
            linewidths=1.8,
            node_size=3400,
            ax=ax,
        )

        nx.draw_networkx_labels(
            G,
            pos,
            labels={class_node_label: class_node_label},
            font_size=9,
            font_weight="bold",
            font_color="#721c24",
            ax=ax,
        )
        nx.draw_networkx_labels(
            G,
            pos,
            labels={n: n for n in feature_nodes},
            font_size=8.5,
            font_color="#055160",
            ax=ax,
        )

        # Legenda
        legend_elements = [
            Line2D([0], [0], color="#0d6efd", lw=2, marker=">", markersize=8,
                label="Dependensi Bersyarat Naive Bayes: P(Fitur | Kelas)"),
            Line2D([0], [0], color="#adb5bd", lw=1.5, linestyle="--",
                label=f"Asosiasi Empiris Antarfitur (MI ≥ {feature_mi_threshold})"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8.5, framealpha=0.9)

        plt.title(
            "Graf Hubungan Antarfitur dan Variabel Kelas (Naive Bayes)",
            fontsize=12,
            fontweight="bold",
            pad=14,
        )
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"Graf berhasil disimpan di: {filepath}")