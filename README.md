# Tugas Besar 2 IF3170 Inteligensi Artifisial — Prediksi Pembatalan Reservasi Hotel

Repository ini berisi implementasi algoritma *Machine Learning* dari awal (*from scratch*) maupun menggunakan pustaka standar untuk memprediksi pembatalan reservasi hotel (`is_canceled`) berdasarkan data historis pemesanan tamu. Proyek ini dikerjakan oleh **Kelompok 28** sebagai pemenuhan Tugas Besar 2 mata kuliah **IF3170 Inteligensi Artifisial**.

---

## Deskripsi Singkat

Permasalahan yang diangkat adalah klasifikasi biner untuk memprediksi apakah suatu reservasi hotel akan dibatalkan (`1`) atau tidak (`0`) dengan dataset yang terdiri dari **27.373 baris data** dan berbagai fitur seperti lama menginap, jumlah tamu, tarif per malam (`nightly_rate_idr`), waktu tunggu pemesanan (`days_before_arrival`), serta riwayat pemesanan pelanggan.

Pipeline sistem mencakup:
1. **Exploratory Data Analysis (EDA) & Data Cleaning:** Penanganan *missing values*, duplikat, dan pengecekan distribusi fitur serta *class imbalance*.
2. **Feature Engineering & Preprocessing:** *Encoding* fitur kategorikal, *scaling* fitur numerik, penanganan ketidakseimbangan kelas (*Random Oversampling*), dan reduksi dimensi (PCA).
3. **Implementasi & Evaluasi Model:** Perbandingan performa 5 algoritma klasifikasi (*K-Nearest Neighbors*, *Decision Tree CART*, *Logistic Regression*, *Support Vector Machine*, dan *Naive Bayes*).

---

## Cara Setup dan Menjalankan Program

### 1. Prasyarat
Pastikan perangkat Anda sudah terpasang:
- **Python 3.9+**
- **Jupyter Notebook** atau **JupyterLab** (atau menggunakan **Google Colab**)

### 2. Instalasi Lokal
1. **Clone repository ini:**
   ```bash
   git clone https://github.com/username/Tubes2-IF3170-Group28.git
   cd Tubes2-IF3170-Group28
   ```

2. **Install semua dependensi yang diperlukan:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Atau install pustaka utama secara manual: `pip install numpy pandas matplotlib seaborn scikit-learn jupyter`)*

### 3. Menjalankan Program
1. Pastikan file dataset (`train.csv` dan `test.csv`) sudah berada di dalam folder `data/` (atau sesuaikan path direktori pada bagian konfigurasi `Settings` di dalam notebook).
2. Jalankan Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Buka file `notebooks/Tubes2_IF3170_Group28.ipynb`.
4. Sesuaikan konfigurasi eksperimen pada sel **`Settings`** (seperti opsi *oversampling*, *scaling*, atau pemilihan model).
5. Jalankan seluruh sel secara berurutan melalui menu **Kernel > Restart & Run All**.

---

## Pembagian Tugas Anggota Kelompok (Group 28)

| NIM | Nama Lengkap | Pembagian Tugas |
| :--- | :--- | :--- |
| 13524114 | Mirza Tsabita Wafa’ana | DTL |
| 13524116 | Eliana Natalie Widjojo | SVM |
| 13524128 | Safira Berlianti | Naive Bayes dan integrasi |
| 13524135 | Varistha Devi | Logistic Regression |
| 13524138 | Ahmad Rinofaros Muchtar | KNN dan Data Preparation |
