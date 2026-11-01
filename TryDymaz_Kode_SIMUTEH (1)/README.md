# SIMUTEH — Evaluasi Praktikum Computer Vision P1–P4

**Mahasiswa:** Try Dymaz Prayoga  
**NIM:** 2330511020  
**Mata kuliah:** Computer Vision  
**Kelas:** 7A/7B *(konfirmasi kelas aktual sebelum dikumpulkan)*  
**SEED dataset:** 1020  
**Repository:** https://github.com/Brayen077/simuteh-2330511020

## Deskripsi
SIMUTEH merupakan studi kasus sistem bantu pemeriksaan mutu daun teh berbasis citra. Praktikum mencakup lingkungan kerja, pembentukan citra dan ruang warna, histogram dan perbaikan citra, konvolusi dan penapisan spasial, serta integrasi praproses. Dataset yang digunakan bersifat sintetis dan bukan acuan diagnosis penyakit tanaman nyata.

## Struktur proyek
- `notebook/`: notebook B1 sampai B5 yang telah dijalankan ulang menggunakan SEED 1020.
- `src/`: generator dataset, fungsi bantu, dan modul praproses.
- `hasil/`: tabel, grafik, citra hasil eksperimen, dan parameter praproses.
- `laporan/`: laporan PDF/DOCX dan slide presentasi.
- `requirements.txt`: dependensi Python.

## Menjalankan
1. Gunakan Python 3.10 atau lebih baru.
2. Pasang pustaka dengan `pip install -r requirements.txt`.
3. Pastikan `SEED = 1020` pada `src/buat_dataset_sintetis.py`.
4. Dari root repository, jalankan `python src/buat_dataset_sintetis.py`.
5. Jalankan notebook B1 sampai B5 dari folder `notebook/` dengan Restart & Run All.

## Ringkasan hasil
- Kanal R memiliki rerata Kontras Bercak (KB) tertinggi: 7,33 untuk kelas berat dan 6,21 untuk kelas ringan.
- Rerata KB 40 citra berbercak meningkat dari 6,77 menjadi 11,85 setelah praproses adaptif.
- Deteksi lima kondisi cahaya tepat pada 60 dari 60 citra sintetis.
- Waktu praproses adaptif sekitar 1,22 ms per citra 640 × 480.

## Reprodusibilitas dan keterbatasan
Dataset sintetis dibangkitkan dengan SEED 1020. Derau tambahan pada eksperimen B4 menggunakan seed 42. Data citra sintetis dapat dibuat ulang melalui skrip generator. Hasil hanya diuji pada dataset sintetis dan masih memerlukan validasi pada foto kebun teh nyata.

## Tag versi
Tag versi praktikum yang ditargetkan adalah `v0.1-praproses`. Buat tag setelah commit final dengan perintah:

```bash
git tag v0.1-praproses
git push --tags
```

## Publikasi ke GitHub
Dari folder proyek ini, jalankan:

```bash
git init -b main
git add .
git commit -m "B1-B5: evaluasi praktikum Computer Vision SIMUTEH"
git remote add origin https://github.com/Brayen077/simuteh-2330511020.git
git push -u origin main
git tag v0.1-praproses
git push --tags
```

Pastikan repository berada di akun pemilik tugas dan akses push telah tersedia.
