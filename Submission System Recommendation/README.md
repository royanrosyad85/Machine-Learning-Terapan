# **Laporan Proyek Machine Learning - Royan Rosyad**

## Project Overview

Proyek ini bertujuan untuk membangun **sistem rekomendasi film** yang cerdas dan relevan menggunakan dataset **MovieLens**. Dataset ini mencakup lebih dari 45.000 judul film yang dirilis hingga Juli 2017, dilengkapi dengan informasi metadata seperti nama pemeran, kru, kata kunci plot, anggaran produksi, serta pendapatan. Selain itu, terdapat lebih dari 26 juta data rating yang diberikan oleh 270.000 pengguna, masing-masing dalam skala 1 sampai 5.

Dalam era digital saat ini, layanan *streaming* seperti Netflix atau Amazon Prime menyediakan ribuan pilihan film dan serial. Namun, banyaknya konten justru bisa membuat pengguna kesulitan memilih. Sistem rekomendasi hadir sebagai solusi untuk membantu pengguna menemukan film sesuai preferensi mereka, sehingga pengalaman menonton menjadi lebih personal dan menyenangkan. Di sisi lain, sistem ini juga memberikan nilai tambah bagi penyedia layanan dengan meningkatkan retensi dan kepuasan pengguna.

Proyek ini menggabungkan dua pendekatan utama dalam sistem rekomendasi, yaitu *content-based filtering* dan *collaborative filtering*, untuk membangun model yang mampu memberikan rekomendasi yang lebih akurat dan efisien berdasarkan pola perilaku pengguna dan karakteristik film.

### Latar Belakang Industri

Pertumbuhan pesat industri hiburan digital mendorong layanan *streaming online* menjadi kebutuhan utama di kalangan masyarakat modern. Platform-platform ini bersaing tidak hanya dalam hal konten, tetapi juga dalam kemampuan memberikan rekomendasi yang tepat sasaran. Berdasarkan data dari Muvi.com, sekitar 35–40% pendapatan layanan *streaming* berasal dari sistem rekomendasi. Hal ini membuktikan pentingnya peran sistem rekomendasi dalam mendukung keberlangsungan bisnis dan meningkatkan keterlibatan pengguna.

### Kajian Penelitian Terkait

Beberapa studi terdahulu memberikan kontribusi besar terhadap pengembangan sistem rekomendasi, antara lain:

- **Matrix Factorization** sebagai teknik dalam *collaborative filtering* terbukti mampu meningkatkan akurasi rekomendasi dan mengurangi beban komputasi dalam skala besar (Koren et al., 2009).
- **Item-based Collaborative Filtering** menunjukkan hasil yang lebih stabil dan efektif dibanding pendekatan berbasis pengguna, terutama dalam kondisi data yang jarang (Sarwar et al., 2001).
- **Model untuk Implicit Feedback** yang dikembangkan oleh Hu et al. (2008) memberikan pendekatan baru untuk menangani data tanpa rating eksplisit, seperti histori penelusuran atau klik pengguna.

Pengetahuan dari studi-studi ini menjadi fondasi penting dalam membangun sistem rekomendasi yang lebih adaptif dan sesuai dengan kebutuhan pengguna.

---

## 🎯 Business Understanding

### Permasalahan

Layanan *streaming online* menghadapi tantangan besar dalam menyajikan konten yang sesuai dengan preferensi masing-masing pengguna. Dua masalah utama yang sering muncul adalah:

1. Banyaknya pilihan membuat pengguna kesulitan menemukan film yang sesuai dengan selera mereka.
2. Rekomendasi yang tidak relevan dapat menurunkan kepuasan pengguna, bahkan membuat mereka enggan untuk kembali menggunakan layanan.

Selain itu, dari sisi pengembangan, terdapat tantangan teknis seperti:

- Bagaimana memahami struktur dan informasi yang terkandung dalam dataset MovieLens?
- Bagaimana membangun sistem rekomendasi berbasis konten yang memanfaatkan metadata film?
- Bagaimana menerapkan pendekatan *collaborative filtering* untuk mengenali pola perilaku pengguna?
- Bagaimana mengevaluasi efektivitas dari model yang dikembangkan?

### Tujuan Proyek

Untuk menjawab tantangan tersebut, proyek ini menetapkan beberapa tujuan utama:

- Melakukan eksplorasi dan analisis awal terhadap dataset untuk memahami struktur dan pola data.
- Mengembangkan model rekomendasi berbasis konten (*content-based filtering*) dengan memanfaatkan atribut film seperti genre, kata kunci, dan pemeran.
- Membangun model rekomendasi berbasis kolaboratif (*collaborative filtering*) dengan memanfaatkan pola interaksi antar pengguna.
- Mengevaluasi performa model menggunakan metrik yang sesuai, guna memastikan akurasi dan relevansi rekomendasi yang dihasilkan.

Dengan strategi ini, sistem yang dikembangkan diharapkan mampu meningkatkan pengalaman pengguna dan mendukung keberlanjutan bisnis dari platform *streaming*.

---

## 📚 Referensi

- Koren et al. (2009). [Matrix Factorization Techniques for Recommender Systems](https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf)
- Sarwar, B. et al. (2001). [Item-Based Collaborative Filtering Recommendation Algorithms](https://www.researchgate.net/publication/2369002_Item-based_Collaborative_Filtering_Recommendation_Algorithms)
- Hu, Y. et al. (2008). [Collaborative Filtering for Implicit Feedback Datasets](https://ieeexplore.ieee.org/document/4781121)
- Sekartaji (2023). *Studi Perilaku Konsumen Digital*
- Muhd Shukri et al. (2024). *Tren Platform Streaming di Era Digital*
- Muvi.com. *Streaming Industry Insights*

