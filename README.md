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

Program dikembangkan dan dijalankan menggunakan Google Colab. Seluruh eksperimen utama tersedia pada notebook di folder `notebooks/`.

Pustaka yang digunakan meliputi NumPy, pandas, Matplotlib, Seaborn, scikit-learn, NetworkX, dan IPython.

Untuk menjalankan secara lokal, pengguna juga memerlukan Python 3 dan Jupyter Notebook atau JupyterLab.

### 2. Menjalankan di Google Colab (Direkomendasikan)

Notebook memiliki konfigurasi untuk melakukan cloning repository GitHub secara otomatis. Pada versi saat ini, proses tersebut menggunakan autentikasi melalui Google Colab Secrets.

Langkah menjalankan program:

1. Buka notebook yang tersedia di folder `notebooks/` menggunakan Google Colab.
2. Buka menu **Secrets** pada sidebar Google Colab.
3. Tambahkan secret dengan nama `GITHUB_TOKEN` yang berisi GitHub Personal Access Token.
4. Jika repository masih private, pastikan token memiliki izin membaca repository tersebut. Aktifkan akses notebook terhadap secret.
5. Jalankan seluruh cell notebook secara berurutan.

Autentikasi dilakukan menggunakan Colab Secrets sehingga token tidak perlu ditulis secara langsung dalam kode.

**Catatan:** Repository direncanakan menjadi public. Namun, konfigurasi notebook saat ini masih meminta `GITHUB_TOKEN` meskipun repository sudah public. Pengguna tetap perlu menyiapkan secret tersebut selama blok autentikasi belum diubah, atau menyesuaikan cell setup untuk menggunakan cloning tanpa autentikasi.

### 3. Menjalankan Secara Lokal

Clone repository menggunakan Git:

```bash
git clone https://github.com/EliWidjojo/IF3170_G28_HaalinghamJayaAO3.git
cd IF3170_G28_HaalinghamJayaAO3
```

Selama repository masih private, proses cloning memerlukan akun yang memiliki izin akses.

Instal pustaka yang diperlukan:

```bash
python -m pip install numpy pandas matplotlib seaborn scikit-learn networkx ipython jupyter
```

Jalankan Jupyter Notebook:

```bash
jupyter notebook
```

Buka notebook pada folder `notebooks/`.

**Penyesuaian untuk lingkungan lokal:**

Notebook memiliki blok setup khusus Google Colab pada cell **Import Libraries**. Pengguna yang menjalankan secara lokal perlu menonaktifkan bagian tersebut, mulai dari:

```python
from google.colab import userdata
```

hingga blok:

```python
print(f"Repository siap: {os.getcwd()}")
```

Selanjutnya, atur working directory ke root repository dan tambahkan lokasi tersebut ke Python path sebelum bagian import algoritma from scratch.

Bagian import pustaka umum dan implementasi algoritma tetap harus dijalankan.

Pastikan data tersedia pada folder `data/`, kemudian jalankan seluruh cell secara berurutan.

---

## Pembagian Tugas Anggota Kelompok (Group 28)

| NIM | Nama Lengkap | Pembagian Tugas |
| :--- | :--- | :--- |
| 13524114 | Mirza Tsabita Wafa’ana | DTL |
| 13524116 | Eliana Natalie Widjojo | SVM |
| 13524128 | Safira Berlianti | Naive Bayes dan integrasi |
| 13524135 | Varistha Devi | Logistic Regression |
| 13524138 | Ahmad Rinofaros Muchtar | KNN dan Data Preparation |
