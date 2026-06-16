# Klasifikasi Profit — Skincare & Beauty E-Store (Streamlit App)

Aplikasi web untuk men-deploy model klasifikasi `Profit_Category`
("Rugi" / "Untung Rendah" / "Untung Tinggi") dari notebook analisis
Global Skincare & Beauty E-Store.

## Struktur File

```
.
├── train_model.py        # Script training: hasilkan model & artifact
├── app.py                 # Aplikasi Streamlit (dashboard, EDA, prediksi)
├── requirements.txt
├── .streamlit/config.toml
└── artifacts/              # Dihasilkan otomatis oleh train_model.py
    ├── model.pkl              (Random Forest tuned)
    ├── scaler.pkl
    ├── label_encoder.pkl
    ├── feature_columns.pkl
    ├── model_comparison.csv
    ├── feature_importance.csv
    └── ui_metadata.json
```

## 1. Jalankan di Lokal

```bash
pip install -r requirements.txt

# Latih model & buat artifact (sekali saja, ambil 5-10 menit karena ada GridSearchCV)
python train_model.py

# Jalankan aplikasi
streamlit run app.py
```

Buka browser ke `http://localhost:8501`.

> Folder `artifacts/` HARUS ada sebelum `streamlit run app.py` dijalankan.
> Jika belum ada, jalankan `train_model.py` dulu.

## 2. Deploy ke Streamlit Community Cloud

1. Push seluruh folder ini (termasuk folder `artifacts/` yang sudah berisi
   hasil training) ke repository GitHub.
   - Jalankan `python train_model.py` di lokal dulu sebelum push, supaya
     `artifacts/` tidak kosong — Streamlit Cloud tidak menjalankan
     training otomatis.
2. Buka [share.streamlit.io](https://share.streamlit.io), hubungkan ke
   repository tersebut, pilih `app.py` sebagai entry point.
3. Deploy. Selesai.

## 3. Halaman pada Aplikasi

- **Dashboard** — ringkasan jumlah transaksi, total sales, rata-rata
  profit, dan akurasi model terbaik, plus distribusi kategori profit.
- **Eksplorasi Data** — korelasi variabel numerik terhadap profit,
  rata-rata profit per segment/market, distribusi kategori, scatter
  discount vs profit.
- **Prediksi Profit** — form input detail transaksi (Segment, Region,
  Market, Category, Subcategory, Quantity, Sales, Discount, Order Date)
  yang akan diprediksi kategori profit-nya secara real-time.
- **Perbandingan Model** — tabel & grafik perbandingan 4 algoritma
  (Logistic Regression, Decision Tree, Random Forest, SVM), termasuk
  cek overfitting (gap train vs test accuracy) dan feature importance.

## 4. Catatan Teknis

- Model utama yang dipakai untuk prediksi adalah **Random Forest** yang
  sudah di-tuning (`GridSearchCV`), dengan `max_depth` dibatasi (tidak
  `None`) agar ukuran file model tetap ringkas (~40-50 MB, bukan 100+ MB)
  sehingga aman untuk disimpan di GitHub dan cepat saat di-load Streamlit.
- Tema visual memakai font Times New Roman dengan palet warna cream,
  abu-abu, dan merah — diatur lewat custom CSS di `app.py`.
- Jika ingin melatih ulang model dengan data baru, cukup jalankan ulang
  `train_model.py` (bisa override sumber data lewat environment variable
  `DATA_URL`).
