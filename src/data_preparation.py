import pandas as pd
import numpy as np
from typing import Optional # Tambahkan ini
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

class DataPreprocessor:
    def __init__(self, numerical_cols: Optional[list] = None, categorical_cols: Optional[list] = None):
        """
        Inisialisasi kelas DataPreprocessor.
        """
        self.numerical_cols = numerical_cols if numerical_cols is not None else []
        self.categorical_cols = categorical_cols if categorical_cols is not None else []
        
        self.num_imputer: SimpleImputer = SimpleImputer(strategy='median')
        self.cat_imputer: SimpleImputer = SimpleImputer(strategy='most_frequent')
        self.scaler: StandardScaler = StandardScaler() 
        self.encoder: Optional[OneHotEncoder] = None 
        self.pca: Optional[PCA] = None
        self.selector: Optional[SelectKBest] = None

    def clean_data(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        1. Data Cleaning: Menangani missing values dan duplikat.
        """
        df = df.copy()
        
        if is_train:
            initial_len = len(df)
            df = df.drop_duplicates()
            print(f"[Cleaning] Menghapus {initial_len - len(df)} baris duplikat.")
            
        num_cols = [c for c in self.numerical_cols if c in df.columns]
        cat_cols = [c for c in self.categorical_cols if c in df.columns]
        
        if is_train:
            if num_cols:
                self.num_imputer.fit(df[num_cols])
            if cat_cols:
                self.cat_imputer.fit(df[cat_cols])
                
        if num_cols:
            df[num_cols] = self.num_imputer.transform(df[num_cols])
        if cat_cols:
            df[cat_cols] = self.cat_imputer.transform(df[cat_cols])
            
        return df

    def transform_features(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        2. Data Transformation: Encoding kategorikal dan Scaling numerik.
        """
        df = df.copy()
        
        num_cols = [c for c in self.numerical_cols if c in df.columns]
        cat_cols = [c for c in self.categorical_cols if c in df.columns]
        
        #  Encoding Categorical 
        if cat_cols:
            if is_train:
                self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                encoded_data = self.encoder.fit_transform(df[cat_cols])
            else:
                # Tambahkan assert/guard agar type checker yakin encoder sudah di-fit
                assert self.encoder is not None, "Encoder belum di-fit pada data training!"
                encoded_data = self.encoder.transform(df[cat_cols])
                
            encoded_cols = self.encoder.get_feature_names_out(cat_cols)
            df_encoded = pd.DataFrame(encoded_data, columns=encoded_cols, index=df.index)
            df = pd.concat([df.drop(columns=cat_cols), df_encoded], axis=1)

        #  Scaling Numerical 
        if num_cols:
            if is_train:
                self.scaler.fit(df[num_cols])
            
            df[num_cols] = self.scaler.transform(df[num_cols])
            
        return df

    def feature_selection(self, X: pd.DataFrame, y: Optional[pd.Series] = None, k_best: Optional[int] = None, is_train: bool = True) -> pd.DataFrame:
        """
        3. Feature Selection: Memilih fitur yang paling relevan.
        """
        if k_best is None or k_best >= X.shape[1]:
            return X 
            
        if is_train:
            assert y is not None, "Target y dibutuhkan untuk proses training feature selection."
            self.selector = SelectKBest(score_func=f_classif, k=k_best)
            self.selector.fit(X, y)
        else:
            assert self.selector is not None, "Selector belum di-fit!"
            
        X_selected = self.selector.transform(X)
        selected_cols = X.columns[self.selector.get_support()]
        
        return pd.DataFrame(X_selected, columns=selected_cols, index=X.index)

    def dimensionality_reduction(self, X: pd.DataFrame, n_components: Optional[int] = None, is_train: bool = True) -> pd.DataFrame:
        """
        4. Dimensionality Reduction: PCA.
        """
        if n_components is None:
            return X
            
        if is_train:
            self.pca = PCA(n_components=n_components, random_state=42)
            X_pca = self.pca.fit_transform(X)
        else:
            assert self.pca is not None, "PCA belum di-fit!"
            X_pca = self.pca.transform(X)
            
        pca_cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]
        return pd.DataFrame(X_pca, columns=pca_cols, index=X.index)

    def handle_imbalance(self, X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """
        5. Imbalanced Data Handling menggunakan Random Oversampling (Pandas/NumPy).
        """
        print("[Imbalance] Memeriksa distribusi kelas target...")
        print(y_train.value_counts())
        
        class_counts = y_train.value_counts()
        max_class = class_counts.idxmax()
        min_class = class_counts.idxmin()
        
        X_majority = X_train[y_train == max_class]
        X_minority = X_train[y_train == min_class]
        y_majority = y_train[y_train == max_class]
        y_minority = y_train[y_train == min_class]
        
        n_to_add = len(X_majority) - len(X_minority)
        if n_to_add > 0:
            X_minority_resampled = X_minority.sample(n=n_to_add, replace=True, random_state=42)
            y_minority_resampled = y_minority.sample(n=n_to_add, replace=True, random_state=42)
            
            X_resampled = pd.concat([X_majority, X_minority, X_minority_resampled])
            y_resampled = pd.concat([y_majority, y_minority, y_minority_resampled])
            
            idx = np.random.RandomState(42).permutation(len(X_resampled))
            X_resampled = X_resampled.iloc[idx].reset_index(drop=True)
            y_resampled = y_resampled.iloc[idx].reset_index(drop=True)
            
            print(f"[Imbalance] Oversampling selesai. Kelas minoritas ditambah {n_to_add} data.")
            return X_resampled, y_resampled
        else:
            return X_train, y_train