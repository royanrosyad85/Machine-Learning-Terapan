# LAPORAN PROYEK MACHINE LEARNING - Klasifikasi Kelayakan Kualitas Air Minum

## Latar Belakang (Domain Proyek)

Air merupakan salah satu kebutuhan dasar manusia yang tidak dapat dipisahkan dari kehidupan sehari-hari [[1]([https://ojs.serambimekkah.ac.id/jurnal-biologi/article/view/1592](https://doi.org/10.24002/konstelasi.v2i1.5630))]. Fungsi air bagi tubuh sangat vital karena ia menjadi elemen utama untuk menjaga kesehatan dan memastikan fungsi organ tubuh berjalan optimal. Air tidak hanya mendukung proses pencernaan dan penyerapan nutrisi, tetapi juga membantu mengeluarkan sisa metabolisme dari tubuh [[2](https://doi.org/10.1109/csnt57126.2023.10134661.)]. Selain itu, air memiliki peran penting dalam menjaga suhu tubuh stabil, mengatur tekanan darah, serta melumasi sendi-sendi. Kurangnya konsumsi air dapat menyebabkan dehidrasi, yang jika dibiarkan tanpa penanganan dapat berakibat fatal [[3](https://ijhn.ub.ac.id/index.php/ijhn/article/view/114](https://doi.org/10.58864/mrijet.2023.10.1.8))]. Oleh karena itu, setiap individu harus memastikan konsumsi air yang cukup setiap hari untuk menjaga kesehatan tubuh.

Di Indonesia, masalah kualitas air minum masih menjadi isu yang serius. Baik di daerah perkotaan maupun pedesaan, kondisi air minum yang tersedia semakin menurun [[4](https://doi.org/10.1016/j.jenvman.2020.111682.)]. Berdasarkan data riset dari Kementerian Kesehatan pada tahun 2020, sekitar 74,4% rumah tangga di Indonesia mengalami kontaminasi air minum oleh bakteri _E.coli_. Selain itu, air dengan tingkat keasaman atau kebasaan yang tidak sesuai dapat memengaruhi sistem pencernaan, lambung, ginjal, dan pembuluh darah. Hal ini menunjukkan bahwa upaya memastikan ketersediaan air minum yang aman dan layak konsumsi harus menjadi prioritas pemerintah dan lembaga terkait untuk menjaga kesejahteraan masyarakat Indonesia.

Untuk memastikan air minum aman untuk dikonsumsi, diperlukan analisis komprehensif berdasarkan parameter-parameter tertentu yang dapat mengukur kualitas air secara akurat. Dalam hal ini, teknologi **Machine Learning** dan **Deep Learning** dapat menjadi solusi efektif [[5]([https://www.researchgate.net/publication/360650780_Analisis_Komparatif_Algoritme_Machine_Learning_dan_Penanganan_Imbalanced_Data_pada_Klasifikasi_Kualitas_Air_Layak_Minum](https://doi.org/10.1016/j.envres.2021.111643.))]. Machine Learning mampu mendeteksi pola dari data historis yang dikumpulkan dari berbagai sumber, sehingga dapat memberikan prediksi yang tepat tentang kualitas air [[6](https://doi.org/10.1007/s13762-020-02887-1.)]. Beberapa algoritma klasifikasi yang umum digunakan dalam konteks ini antara lain **KNN**, **SVM**, dan **Random Forest** [[7]([https://ieeexplore.ieee.org/document/10134661](https://doi.org/10.33299/jpkop.22.2.1752))]. Dengan memanfaatkan data variabel yang relevan dan menggunakan pendekatan klasifikasi, model machine learning dapat dibuat untuk mengklasifikasikan kualitas air dengan lebih presisi [[8]([https://mrijet.mrpublishers.com/index.php/mrijet/article/view/10-1-8](https://doi.org/10.1007/s13762-020-02887-1.))].

# Business Understanding

### Problem Statements
Air minum yang layak konsumsi merupakan kebutuhan dasar manusia. Namun, tidak semua sumber air memiliki kualitas yang memenuhi standar potabilitas. Untuk memastikan air tersebut aman dikonsumsi, diperlukan metode klasifikasi kualitas air yang akurat dan efisien.  
Tantangan yang dihadapi dalam project ini adalah:
- Adanya **missing value** dalam data yang dapat mengganggu akurasi model.
- **Distribusi data** yang tidak seimbang (imbalance) antara air layak dan tidak layak minum.
- **Hubungan antar fitur** yang lemah dengan target variabel, sehingga sulit menemukan pola klasifikasi secara sederhana.
- **Adanya outlier** pada beberapa fitur yang berpotensi mempengaruhi performa model.
- Mencari **model terbaik** yang mampu melakukan klasifikasi dengan akurasi tinggi baik menggunakan pendekatan machine learning maupun deep learning.

### Goals
Project ini bertujuan untuk:
- **Membersihkan dan menyiapkan data** kualitas air dengan melakukan penanganan missing value, deteksi duplikasi, scaling fitur, serta penanganan imbalance data.
- **Melakukan exploratory data analysis (EDA)** untuk memahami karakteristik data, distribusi fitur, keberadaan outlier, serta hubungan antar fitur dan target.
- **Membangun model machine learning** (RandomForest, KNN, XGBoost) dengan hyperparameter tuning untuk klasifikasi potabilitas air.
- **Mengembangkan model deep learning** berbasis feedforward neural network untuk meningkatkan performa klasifikasi.
- **Mengevaluasi kinerja model** dengan menggunakan metrik akurasi, precision, recall, dan F1-score pada data uji.
- **Menyusun rekomendasi** terkait pengembangan lebih lanjut untuk meningkatkan generalisasi model di masa depan.

### Solution Statements
Untuk mencapai tujuan tersebut, pendekatan yang dilakukan dalam project ini meliputi:

- **Data Cleaning**  
  - Mengidentifikasi dan menangani missing value pada kolom `ph`, `Sulfate`, dan `Trihalomethanes` dengan metode imputasi median.
  - Mendeteksi dan menghapus duplikasi data agar dataset bersih dan representatif.
  - Memastikan tipe data setiap kolom sesuai untuk proses modelling.

- **Data Understanding & Exploratory Data Analysis (EDA)**  
  - Melakukan analisis univariat dan multivariat untuk memahami distribusi data dan hubungan antar fitur.
  - Menggunakan visualisasi (countplot, histogram, boxplot, heatmap) untuk mendeteksi imbalance, outlier, dan korelasi.

- **Data Preparation**  
  - Menyeimbangkan distribusi kelas dengan metode Random OverSampling.
  - Melakukan normalisasi fitur menggunakan MinMaxScaler untuk mempercepat dan menstabilkan proses training.

- **Modelling Machine Learning**  
  - Menerapkan algoritma RandomForestClassifier, KNeighborsClassifier, dan XGBClassifier.
  - Melakukan hyperparameter tuning dengan GridSearchCV untuk mendapatkan model dengan performa terbaik.

- **Modelling Deep Learning**  
  - Mendesain arsitektur deep learning sederhana namun efektif, terdiri dari beberapa dense layer dan dropout untuk menghindari overfitting.
  - Melatih model dengan teknik ModelCheckpoint dan evaluasi terhadap validation split.

- **Model Evaluation**  
  - Membandingkan performa model menggunakan metrik akurasi, precision, recall, dan F1-score.
  - Menyajikan hasil evaluasi dengan confusion matrix dan plot learning curves.

# Data Understanding
Dataset yang digunakan dalam proyek _machine learning_ ini adalah "Water Quality and Potability," yang tersedia di platform [Kaggle](https://www.kaggle.com/datasets/uom190346a/water-quality-and-potability). _Dataset_ ini merupakan kumpulan data kuantitatif yang mencakup berbagai kolom untuk menentukan apakah air layak diminum atau tidak. Secara keseluruhan, dataset ini terdiri dari 3276 baris dan 10 kolom.

Dataset_ ini sangat sesuai untuk membangun model _supervised learning_, khususnya untuk tugas _binary classification_. Dalam konteks ini, model akan digunakan untuk mengklasifikasikan sampel air menjadi dua kategori: layak diminum atau tidak layak diminum.
Berikut ini adalah informasi lainnya mengenai variabel-variabel yang terdapat di dataset tersebut:

### Variabel-variabel pada _Dataset "Water Quality and Potability"_ adalah sebagai berikut:
- ```pH```: Tingkat pH air. 
- ```Hardness```: Ukuran kandungan mineral. 
- ```Solids```: Total padatan terlarut dalam air. 
- ```Chloramines```: Konsentrasi kloramin dalam air. 
- ```Sulfate```: Konsentrasi sulfat dalam air. 
- ```Conductivity```: Konduktivitas listrik di air. 
- ```Organic_carbon```: Kandungan karbon organik dalam air. 
- ```Trihalomethanes```: Konsentrasi trihalometan dalam air. 
- ```Turbidity```: Tingkat kekeruhan, ukuran kejernihan air. 
- ```Potability```: Variabel target. menunjukkan potabilitas air dengan nilai 1 (layak minum) dan 0 (tidak layak minum).

Kemudian, untuk meningkatkan pemahaman atas data terkait, dilakukannya _exploratory data analysis_ dan Visualisasi Data.

**_Exploratory Data Analysis_**

Exploratory Data Analysis (EDA) adalah pendekatan analisis data yang bertujuan untuk memahami karakteristik utama dari kumpulan data. EDA melibatkan penggunaan teknik statistik dan visualisasi grafis untuk menemukan pola, hubungan, atau anomali untuk membentuk hipotesis. Proses ini sering kali tidak terstruktur dan dianggap sebagai langkah awal penting dalam analisis data yang membantu menentukan arah analisis lebih lanjut.

Berikut ini adalah EDA yang dilakukan:
- ```python
  dataset.shape
  ```
  Kode diatas memiliki output:
  ```python
  (3276, 10)
  ```

  Berdasarkan _output_ tersebut, didapatkan informasi bahwa dataset ini memiliki **3276 baris** dan **10 kolom** data sesuai dengan dengan keterangan yang tertera diatas. Pada bagian ini, belum dapat diketahui **nama** dari **kolom-kolom** yang ada.
- ```python
   dataset.keys()
  ```
  Kode diatas memiliki output:
  ```python
  Index(['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
       'Organic_carbon', 'Trihalomethanes', 'Turbidity', 'Potability'],
      dtype='object')
  ```

  Berdasarkan _output_ tersebut, didapatkan informasi lebih lanjut bahwa dataset ini memiliki 10 kolom sesuai dengan keterangan yang tertera diatas. Pada bagian ini, belum dapat diketahui **jumlah** dan **tipe data** dari tiap kolom.
- ```python
   dataset.info()
  ```
  Kode diatas memiliki output:
  ```python
  RangeIndex: 3276 entries, 0 to 3275
  Data columns (total 10 columns):
   #   Column           Non-Null Count  Dtype  
  ---  ------           --------------  -----  
   0   ph               2785 non-null   float64
   1   Hardness         3276 non-null   float64
   2   Solids           3276 non-null   float64
   3   Chloramines      3276 non-null   float64
   4   Sulfate          2495 non-null   float64
   5   Conductivity     3276 non-null   float64
   6   Organic_carbon   3276 non-null   float64
   7   Trihalomethanes  3114 non-null   float64
   8   Turbidity        3276 non-null   float64
   9   Potability       3276 non-null   int64  
  dtypes: float64(9), int64(1)
  ```

   Berdasarkan _output_ tersebut, didapatkan informasi mengenai jumlah data dan tipe data dari setiap kolom yang ada. Ada beberapa kolom yang tidak miliki jumlah data sesuai dengan total baris, yaitu 3276 baris. Hal ini mengindikasikan adanya _missing value_. Kemudian, hanya 1 kolom yang bertipe ```int64```, yaitu kolom ```'Potability'```. Kolom lainnya bertipe ```float64```.
- ```python
   df.describe().T
  ```
  Kode diatas memiliki output:



|                       | **Count** | **Mean**    | **Std**     | **Min**       | **25%**       | **50%**       | **75%**       | **Max**       |
|-----------------------|-----------|-------------|-------------|---------------|---------------|---------------|---------------|---------------|
| **ph**               | 3276.0    | 7.074194    | 1.470040    | 0.000000      | 6.277673      | 7.036752      | 7.870050      | 14.000000     |
| **Hardness**         | 3276.0    | 196.369496  | 32.879761   | 47.432000     | 176.850538    | 196.967627    | 216.667456    | 323.124000    |
| **Solids**           | 3276.0    | 22014.092526| 8768.570828 | 320.942611    | 15666.690297  | 20927.833607  | 27332.762127  | 61227.196008  |
| **Chloramines**      | 3276.0    | 7.122277    | 1.583085    | 0.352000      | 6.127421      | 7.130299      | 8.114887      | 13.127000     |
| **Sulfate**          | 3276.0    | 333.608364  | 36.143851   | 129.000000    | 317.094638    | 333.073546    | 350.385756    | 481.030642    |
| **Conductivity**     | 3276.0    | 426.205111  | 80.824064   | 181.483754    | 365.734414    | 421.884968    | 481.792304    | 753.342620    |
| **Organic_carbon**   | 3276.0    | 14.284970   | 3.308162    | 2.200000      | 12.065801     | 14.218338     | 16.557652     | 28.300000     |
| **Trihalomethanes**  | 3276.0    | 66.407478   | 15.769958   | 0.738000      | 56.647656     | 66.622485     | 76.666609     | 124.000000    |
| **Turbidity**        | 3276.0    | 3.966786    | 0.780382    | 1.450000      | 3.439711      | 3.955028      | 4.500320      | 6.739000      |
| **Potability**       | 3276.0    | 0.390110    | 0.487849    | 0.000000      | 0.000000      | 0.000000      | 1.000000      | 1.000000      |

Berdasarkan _output_ yang diberikan, dapat diperoleh informasi mengenai statistik deskriptif dari _dataset_ yang digunakan. Berikut adalah penjelasan untuk setiap komponen:

- ```count```: Menunjukkan total jumlah data pada suatu kolom.
- ```mean```: Menyatakan nilai rata-rata dari data di sebuah kolom.
- ```std```: Mengacu pada standar deviasi, yaitu ukuran seberapa jauh data tersebar dari nilai rata-ratanya.
- ```min```: Menampilkan nilai terkecil yang ada dalam sebuah kolom.
- ```25%```: Merupakan nilai kuartil pertama (Q1), yang membagi 25% data terbawah dari sisa data.
- ```50%```: Menunjukkan nilai median atau titik tengah data, yang juga dikenal sebagai kuartil kedua (Q2).
- ```75%```: Adalah nilai kuartil ketiga (Q3), yang memisahkan 75% data terbawah dari 25% data teratas.
- ```max```: Menyajikan nilai tertinggi yang terdapat pada sebuah kolom.

- ```python
  dataset.isnull().sum()
  ```
  
  ```python
  
  ph                 491
  Hardness             0
  Solids               0
  Chloramines          0
  Sulfate            781
  Conductivity         0
  Organic_carbon       0
  Trihalomethanes    162
  Turbidity            0
  Potability           0
  dtype: int64
  ```
Dari hasil analisis tersebut, teridentifikasi adanya sejumlah nilai yang hilang (missing value) pada beberapa variabel, meliputi ```pH```, ```Sulfat```, dan ```Trihalometana```. Penanganan terhadap nilai-nilai yang hilang ini penting untuk mencegah pengaruh negatif pada model yang akan dikembangkan.

**Visualisasi Data**
  - _Univariate Analysis_
    
    _Univariate Analysis_ merupakan metode analisis data yang berfokus pada pemeriksaan satu variabel atau kolom data secara individual. Tujuannya adalah untuk memberikan gambaran deskriptif mengenai data tersebut serta mengidentifikasi pola-pola yang terdapat dalam sebaran nilainya. Teknik yang umum digunakan meliputi statistik deskriptif, histogram, dan diagram kotak (box plot) untuk menganalisis distribusi dan memahami karakteristik variabel yang bersangkutan.
    
    ![Univariate Analysis](https://github.com/user-attachments/assets/818b98a4-5dbe-4457-896f-87cfeed56218)
    <div align="center">Gambar 1.1 - Univariate Analysis Categorical Column</div>

  Berdasarkan visualisasi pada ``` Gambar 1.1  ``` , dapat diamati bahwa variabel ```Potability``` memiliki dua nilai unik, yaitu '1' yang mengindikasikan air minum layak dikonsumsi dan '0' yang mengindikasikan sebaliknya. Akan tetapi, visualisasi tersebut juga memperlihatkan adanya ketidakseimbangan data (_imbalance data_). Jumlah baris data dengan nilai '0' mencapai hampir 2000, sementara nilai '1' hanya memiliki sekitar 1250 baris data. Mengingat kondisi ini, tindakan penyeimbangan data menjadi krusial untuk menghindari terjadinya bias pada model _machine learning_ yang akan dikembangkan.

  ![Distribusi Numerical Fitur](https://github.com/user-attachments/assets/3fc00fbc-fc2c-475a-ad54-50484de4d225)
  <div align="center">Gambar 1.2 - Distribution of Numerical Feature</div>
  
  
  Merujuk pada visualisasi dalam ```Gambar 1.2```, yang menyajikan distribusi untuk setiap kolom numerik dalam dataset (```pH```, ```Hardness```, ```Solids```, ```Chrolamines```, ```Sulfate```, ```Conductivity```, ```Organic_carbon```, ```Trihalomethanes```, ```Turbidity```), terlihat bahwa hanya kolom ```Solids``` dan ```Conductivity``` yang menunjukkan kemiringan (skewness) ke kiri. Informasi ringkas yang dapat disarikan dari visualisasi tersebut adalah sebagai berikut:

   - ```pH```: Tingkat keasaman air bervariasi dari 0 hingga 14, dengan mayoritas sampel memiliki nilai pH di sekitar 7, yang menandakan kondisi netral.
   - ```Hardness```: Tingkat kesadahan air beragam, dengan konsentrasi sampel yang signifikan menunjukkan tingkat kesadahan sekitar 200.
   - ```Solids```: Jumlah total padatan terlarut dalam sampel bervariasi, dengan frekuensi tertinggi berada di dekat nilai 20.000.
   - ```Chloramines```: Kadar kloramin dalam sampel mencapai puncak frekuensi pada rentang nilai 7 hingga 8.
   - ```Sulfate```: Konsentrasi sulfat dalam sampel paling sering ditemukan di sekitar nilai 300.
   - ```Conductivity```: Tingkat konduktivitas sampel memiliki frekuensi tertinggi di sekitar nilai 400.
   - ```Organic_carbon```: Kandungan karbon organik yang paling umum dalam sampel berkisar antara 14 hingga 15.
   - ```Trihalomethanes```: Kadar trihalometana dalam sampel paling sering berada di rentang 65 hingga 70.
   - ```Turbidity```: Tingkat kekeruhan sampel memiliki frekuensi tertinggi di sekitar nilai 3,5.

  
- _Multivariate Analysis_

  _Multivariate Analysis_ adalah prosedur statistik yang digunakan untuk memeriksa hubungan antara beberapa variabel secara bersamaan. Teknik ini mencakup berbagai metode seperti regresi berganda, analisis faktor, dan analisis kluster, yang membantu dalam memahami struktur dan pola yang kompleks dalam data dengan lebih dari satu variabel.

  ![Multivariate](https://github.com/user-attachments/assets/4bcf0802-4672-470f-9487-b9eb55666c14)
  <div align="center">Gambar 1.3 - Pairplot of Features by Potabillity</div>

  Berdasarkan hasil visualisasi di atas, tampak bahwa hampir semua variabel terdistribusi di sekitar nilai tengah dan tidak memperlihatkan pola atau karakteristik khusus terhadap variabel label, yaitu ```'Potability'```. Bahkan pada visualisasi tersebut meskipun data telah dipisahkan berdasarkan kategori ```0``` dan ```1``` (dengan warna biru dan oranye), tetap tidak ditemukan pola atau ciri khas tertentu pada masing-masing nilai label. Hal ini menunjukkan bahwa hubungan antar fitur, termasuk dengan variabel label, cenderung lemah atau berkorelasi rendah.

- _Correlation_

  Uji Korelasi adalah metode statistik yang digunakan untuk menentukan apakah ada hubungan antara dua variabel kuantitatif dan seberapa kuat hubungan tersebut. Uji ini menghasilkan nilai koefisien korelasi, seperti Pearson atau Spearman, yang berkisar antara -1 hingga +1. Nilai mendekati +1 menunjukkan korelasi positif yang kuat, sedangkan nilai mendekati -1 menunjukkan korelasi negatif yang kuat. Nilai mendekati 0 menunjukkan tidak adanya korelasi. Uji korelasi penting dalam menentukan arah dan kekuatan hubungan antar variabel, yang dapat membantu dalam pemodelan prediktif dan analisis penyebab.

  ![Correlation](https://github.com/user-attachments/assets/d3d64f6f-8215-4f53-a1ab-93bb3bcb9460)
  <div align="center">Gambar 1.4 - Pairplot of Features by Potabillity</div>

  **Kesimpulan:**
  - **Korelasi Lemah:** Semua fitur memiliki korelasi yang sangat lemah terhadap `Potability`, sehingga tidak ada fitur tunggal yang secara signifikan memengaruhi target.
  - **Pendekatan Model Kompleks:** Karena hubungan linier tidak signifikan, model machine learning yang lebih kompleks (seperti Random Forest atau XGBoost) atau deep learning diperlukan untuk menangkap pola non-linier dalam data.
  - **Dimensi Dataset:** Fitur dengan korelasi sangat rendah (misalnya, `Trihalomethanes`, `Turbidity`, `pH`) dapat dipertimbangkan untuk di-drop guna mengurangi dimensi dataset, tetapi perlu analisis lebih lanjut untuk memastikan dampaknya terhadap performa model.

- _Box Plots_

  Visualisasi boxplot bertujuan untuk memahami distribusi setiap fitur numerik berdasarkan label `Potability` (0: Tidak Layak Minum, 1: Layak Minum), mengidentifikasi perbedaan distribusi antar kelas, serta mendeteksi keberadaan outliers yang dapat memengaruhi performa model. Dengan melihat median, rentang interkuartil (IQR), dan pencilan, boxplot membantu menentukan apakah fitur tertentu memiliki pengaruh signifikan terhadap target dan memberikan wawasan awal untuk pemilihan fitur atau penanganan data sebelum pemodelan.

  ![Box Plot](https://github.com/user-attachments/assets/3b11465c-5cb0-4a22-b542-bb5546bfa739)
  <div align="center">Gambar 1.5 - Box Plots Correlation</div>

- _Heat Map Correlation_

  Heatmap korelasi ini menampilkan koefisien korelasi Pearson antar variabel numerik, dengan warna biru tua menandakan korelasi positif yang lebih kuat dan warna terang menunjukkan korelasi lemah atau negatif. Secara keseluruhan, hubungan linier antar fitur tampak lemah, dengan korelasi terkuat yang teramati hanya -0.150 antara Sulfate dan Solids. Variabel target Potability juga menunjukkan korelasi yang sangat rendah dengan semua fitur lainnya, mengimplikasikan bahwa hubungan non-linier atau interaksi antar fitur kemungkinan lebih berperan dalam menentukan kelayakan minum air, sehingga mendukung penggunaan model yang lebih kompleks.

  ![Heat Map](https://github.com/user-attachments/assets/7bbae0b1-5c3c-4a6a-abaa-fae2b60dba61)
  <div align="center">Gambar 1.6 - Correlation With Heatmap</div>

# Data Preparation
_Data Preparation_ adalah proses pembersihan, transformasi, dan pengorganisasian data mentah ke dalam format yang dapat dipahami oleh algoritma pembelajaran mesin. Berikut ini adalah **urutan** langkah-langkah Data Preparation yang dilakukan beserta penjelasan dan alasannya:

- _Data Cleaning_
  
  _Data cleaning_ memegang peranan krusial dalam tahapan Machine Learning karena mencakup identifikasi serta penghapusan data yang absen, ganda, maupun tidak relevan dari dataset. Proses ini melibatkan serangkaian langkah yang perlu ditempuh agar dataset layak digunakan dalam pengembangan model Machine Learning.
    
  **Alasan**: _Data Cleaning_ menjadi sebuah keharusan agar data yang dimanfaatkan memiliki tingkat akurasi dan konsistensi yang tinggi serta terbebas dari berbagai kesalahan. Hal ini dikarenakan keberadaan data yang keliru atau tidak konsisten berpotensi menurunkan kinerja model Machine Learning secara signifikan. Salah satu langkah penting dalam proses ini adalah _Detection and Removal Duplicates_.
      
  - Data duplikat merujuk pada baris-baris data yang identik di seluruh variabel yang ada. Penting untuk memverifikasi keberadaan data serupa atau duplikat dalam dataset yang digunakan. Jika ditemukan, langkah yang perlu diambil adalah menangani data tersebut dengan cara menghapusnya.

      **Alasan**: Pendeteksian dan penghapusan data duplikat merupakan langkah penting karena keberadaannya dalam dataset berpotensi menimbulkan bias pada model, yang pada akhirnya dapat menyebabkan overfitting. Kondisi overfitting ditandai dengan performa akurasi model yang tinggi pada data pelatihan, namun rendah ketika dihadapkan pada data baru. Dengan menghilangkan data duplikat, diharapkan model dapat mengidentifikasi pola-pola yang mendasari data dengan lebih efektif.

      Berikut ini adalah proses pendeteksian dan penghapusan data duplikatnya:
      ```python
      # Cek baris duplikat dalam dataset
      duplicates = dataset.duplicated()
      
      # Hitung jumlah baris duplikat
      duplicate_count = duplicates.sum()
      
      # Cetak jumlah baris duplikat
      print(f"Number of duplicate rows: {duplicate_count}")

      ```

      Berikut ini adalah hasilnya:

      ```python
        Number of duplicate rows: 0
      ```

      Berdasarkan hasil analisis, tidak ditemukan adanya duplikasi data dalam dataset. Oleh karena itu, proses penghapusan data duplikat tidak diperlukan.
  
  - _Handle Missing Value_
      
      _Missing Value_ terjadi ketika variabel atau barus tertentu kekurangan titik data, sehingga menghasilkan informasi yang tidak lengkap. Nilai yang hilang dapat ditangani dengan berbagai cara seperti imputasi (mengisi nilai yang hilang dengan mean, median, modus, dll), atau penghapusan (menghilangkan baris atau kolom yang nilai hilang)
 
      **Alasan**: _Missing Value_ perlu ditangani karena jika dibiarkan dapat berpengaruh ke rendahnya akurasi model yang akan dibuat. Maka dari itu, penting untuk mengatasi missing value secara efisien untuk mendapatkan model _Machine Learning_ yang baik juga.
 
      Berikut ini adalah kode untuk mencari tahu kolom mana saja dan berapa jumlah _missing value_-nya:
      ```python
       df.isnull().sum()
      ```
 
      Berikut ini adalah _output_-nya:
      ```python
      ph                491
      Hardness           0
      Solids             0
      Chloramines        0
      Sulfate          781
      Conductivity       0
      Organic_carbon     0
      Trihalomethanes  162
      Turbidity          0
      Potability         0
      dtype: int64
      ```

      Berikut ini kode untuk menghapus baris data yang memiliki _missing value:_
      ```python
       df.fillna(df.median(), inplace=True)
       df.isnull().sum()
      ```

      Penanganan _missing value_ sudah berhasil dilakukan.

  - _Imbalance Data_
      
      _Oversampling_ adalah teknik yang digunakan untuk menangani masalah ketidakseimbangan kelas dalam dataset. Ketidakseimbangan ini terjadi ketika jumlah data pada satu kelas jauh lebih sedikit dibandingkan kelas lainnya, yang dapat menyebabkan model menjadi bias terhadap kelas mayoritas. Dalam oversampling, data dari kelas minoritas ditambahkan secara sintetis atau diduplikasi hingga jumlahnya seimbang dengan kelas mayoritas. Salah satu metode populer adalah RandomOverSampler, yang menduplikasi data kelas minoritas secara acak, atau SMOTE, yang menciptakan sampel sintetis berdasarkan interpolasi data kelas minoritas.

      Berikut ini adalah untuk memeriksa ada berapa baris data untuk masing-masing kelas pada kolom ```'Potability'```:
      ```python
      # Memisahkan fitur dan target
      x = df.drop('Potability', axis=1).values
      y = df['Potability'].values
      
      # Melihat distribusi kelas awal
      count_0 = np.sum(y == 0)
      count_1 = np.sum(y == 1)
      print("Sebelum oversampling:")
      print(f"Jumlah baris data yang bernilai '0' ada sebanyak: {count_0}")
      print(f"Jumlah baris data yang bernilai '1' ada sebanyak: {count_1}")
      print(f"Persentase kelas 0: {count_0/(count_0+count_1)*100:.2f}%")
      print(f"Persentase kelas 1: {count_1/(count_0+count_1)*100:.2f}%")
      ```
 
      Berikut ini adalah hasilnya:
      ```markdown
      Sebelum oversampling:
      Jumlah baris data yang bernilai '0' ada sebanyak: 1998
      Jumlah baris data yang bernilai '1' ada sebanyak: 1278
      Persentase kelas 0: 60.99%
      Persentase kelas 1: 39.01%
      ```

      Dalam hal ini, perlunya dilakukannya _oversampling_ terhadap kelas ```'0'``` agar menyesuaikan jumlah baris datanya dengan kelas ```'1'```
 
      Berikut ini adalah bagian untuk melakukan proses _oversampling_:
      ```python
        # Over-sampling dengan keseimbangan penuh
        over_sampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
        x_resampled, y_resampled = over_sampler.fit_resample(x, y)
        
        # Melihat distribusi kelas setelah oversampling
        count_0 = np.sum(y_resampled == 0)
        count_1 = np.sum(y_resampled == 1)
        print("\nSetelah oversampling:")
        print(f"Jumlah baris data yang bernilai '0' ada sebanyak: {count_0}")
        print(f"Jumlah baris data yang bernilai '1' ada sebanyak: {count_1}")
        print(f"Persentase kelas 0: {count_0/(count_0+count_1)*100:.2f}%")
        print(f"Persentase kelas 1: {count_1/(count_0+count_1)*100:.2f}%")
      ```

      Berikut ini adalah hasil setelah dilakukannya _undesampling_:
      ```markdown
        Setelah oversampling:
        Jumlah baris data yang bernilai '0' ada sebanyak: 1998
        Jumlah baris data yang bernilai '1' ada sebanyak: 1998
        Persentase kelas 0: 50.00%
        Persentase kelas 1: 50.00%   
      ```

     Proses penyeimbangan dataset sudah berhasil dilakukan.
  
  - _Scaling Feature_

    Scaling fitur adalah langkah penting dalam machine learning, terutama untuk algoritma yang sensitif terhadap skala data (misalnya, KNN, SVM). Setelah oversampling, scaling dengan MinMaxScaler dalam rentang (-1, 1) memberikan manfaat berikut:

      - **Normalisasi**: Menyamakan skala semua fitur agar tidak ada fitur yang mendominasi.
      - **Konvergensi Lebih Cepat**: Mempercepat proses optimasi pada algoritma berbasis gradien.
      - **Konsistensi**: Memastikan data asli dan sintetis memiliki skala yang sama.

    Langkah ini penting untuk menjaga kualitas data setelah oversampling.

    Berikut langkah yang dilakukan:
    ```python
      # Scaling fitur setelah oversampling
      scaler = MinMaxScaler((-1, 1))
      x_resampled = scaler.fit_transform(x_resampled)
    ```

# Modelling

- _Machine Learning_

    Dengan pendekatan machine learning, langkah-langkah utama dalam pembuatan model meliputi pemilihan algoritma yang sesuai. Berbagai algoritma klasifikasi seperti Random Forest, K-Nearest Neighbors (KNN), dan Support Vector Machine (SVM) dipertimbangkan berdasarkan      karakteristik dataset dan tujuan proyek.

    ``` python

        parameters = {
        'RandomForestClassifier': {
            'n_estimators': [50, 100,150, 170,200,230,250,300]
        },
    
        'KNeighborsClassifier': {
            'n_neighbors': [3,5,7,10,15,20,30],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        },
    
        'XGBClassifier': {
            'learning_rate': [0.001,0.01, 0.1, 0.5,1],
            'n_estimators': [50, 100,150, 200,250,350,300,350]
        }
    }
    ```
    Parameter grid didefinisikan untuk beberapa algoritma seperti RandomForestClassifier, KNeighborsClassifier, dan XGBClassifier. Parameter ini digunakan untuk mencari kombinasi terbaik melalui proses Grid Search.

    Grid Search untuk Hyperparameter Tuning:
    ```python
        results =list()
    for model_name, model in models.items():
        print(f"Grid searching for {model_name}")
        param_grid = parameters[model_name]
        grid_search = GridSearchCV(model, param_grid, cv=4, scoring='accuracy')
        grid_search.fit(x_resampled,y_resampled)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_accuracy = grid_search.best_score_
    
        results.append({'Model': model_name, 'Best Params': best_params, 'Best Accuracy': best_accuracy})
    results_df = pd.DataFrame(results)
    results_df.head(3)
    ```

    Berikut hasil nya:
        Model                   | Best Params                                                     | Best Accuracy
    ------------------------|-----------------------------------------------------------------|--------------
    RandomForestClassifier  | {'n_estimators': 150}                                           | 0.729229
    KNeighborsClassifier    | {'metric': 'euclidean', 'n_neighbors': 30, 'weights': 'uniform'} | 0.696446
    XGBClassifier           | {'learning_rate': 0.5, 'n_estimators': 300}                     | 0.714214

    Belum cukup puas dengan hasilnya, saya bereksperimen menggunakan deep learning untuk meningkatkan accuracy training

- _Deep Learning_

    Dengan pendekatan deep learning, model dibangun menggunakan arsitektur yang terdiri dari input layer dengan 9 fitur, diikuti oleh beberapa hidden layers. Hidden layers mencakup 3 dense layers dengan 64 neuron dan fungsi aktivasi ReLU, 1 dropout layer dengan rate       0.2 untuk mengurangi overfitting, serta 1 dense layer tambahan dengan 16 neuron dan fungsi aktivasi ReLU. Output layer terdiri dari 1 neuron dengan fungsi aktivasi sigmoid untuk menghasilkan prediksi biner. Model dikompilasi menggunakan optimizer Adam, loss            function binary_crossentropy, serta metrics berupa accuracy, precision, dan recall untuk mengevaluasi performa model secara komprehensif.

    ``` python
        # Define the ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        filepath='model_checkpoint.weights.h5',
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )
    
    # Define the model architecture
    input_layer = Input(shape=(9,))
    layer_1 = Dense(64, activation='relu')(input_layer)
    layer_2 = Dense(64, activation='relu')(layer_1)
    layer_3 = Dense(64, activation='relu')(layer_2)
    layer_4 = Dropout(0.2)(layer_3)
    layer_5 = Dense(16, activation='relu')(layer_4)
    output_layer = Dense(1, activation='sigmoid')(layer_5)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model
    model.compile(
        optimizer="adam",
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    ```

    Kemudian Proses pelatihan model dilakukan dengan menggunakan 20% data sebagai validation split untuk mengevaluasi performa model selama pelatihan. Model dilatih selama 200 epoch dengan batch size sebesar 64 untuk memastikan proses pembelajaran yang efisien. Selain     itu, callback `ModelCheckpoint` digunakan untuk secara otomatis menyimpan model terbaik berdasarkan performa validasi tertinggi, sehingga memastikan hasil akhir yang optimal.

    ``` python
        history=model.fit(
        x_resampled, y_resampled,
        validation_split=0.2,
        epochs=200,
        batch_size=64,
        verbose=1,
        callbacks=[checkpoint_callback]
    )
    ```
    Berikut preview hasil training nya:

    ![epoch](https://github.com/user-attachments/assets/cd43d189-f548-4148-a1b2-ad8ef06dfb21)

# Evaluation

Ketika model sudah dibangun dan sudah melakukan uji dengan data test, perlu dilakukan evaluasi untuk melihat performa dari model tersebut.

  Berikut ini hasil dari Evaluasi pada data testing:
  ```python
    model.evaluate(x_test,y_test)
  ```
  Output:
  25/25 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.9706 - loss: 0.3183 - precision_1: 0.9911 - recall_1: 0.9515

  **Penjelasan**
  - Model memiliki performa yang sangat baik pada dataset uji, dengan akurasi tinggi (97%), precision tinggi (99%), dan recall tinggi (95%).
  - Nilai loss yang rendah (`0.1959`) menunjukkan bahwa model tidak mengalami overfitting atau underfitting yang signifikan.
  - Dengan precision dan recall yang tinggi, model ini sangat andal untuk tugas klasifikasi biner, khususnya dalam menentukan apakah air layak minum atau tidak.

Dari hasil diatas menujukkan bahwa Accuracy model menggunakan deep learning pada dataset test sebesar 97%. Hasilnya lebih besar dibandingkan dengan machine learning model dengan algoritma RandomForestClassifier, KNeighborsClassifier, dan XGBClassifier.

Berikut hasil visualisasi dari evaluasi training model deep learning:

![eval](https://github.com/user-attachments/assets/bdf40886-9681-4c6f-b310-7315d542c644)

### Kesimpulan

- **Performa Model**: Model menunjukkan performa yang baik pada data pelatihan, dengan metrik (accuracy, precision, recall) yang meningkat secara konsisten.

- **Generalization**: Meskipun performa pada data validasi cukup baik, fluktuasi pada grafik validasi (terutama pada precision dan recall) menunjukkan bahwa model mungkin mengalami sedikit overfitting pada data pelatihan.

- **Rekomendasi**: Untuk mengurangi fluktuasi dan meningkatkan generalisasi, beberapa langkah dapat dipertimbangkan:
  - Menambahkan regularisasi (seperti Dropout tambahan atau L2 regularization).
  - Menggunakan teknik early stopping untuk menghentikan pelatihan sebelum model mulai overfit.
  - Meningkatkan ukuran dataset atau menggunakan augmentasi data jika memungkinkan.

Kemudian berikut adalah hasil dari visualisasi confusion matrix:

![image](https://github.com/user-attachments/assets/fe34742c-66f8-4364-a840-c49914c42ac4)


_Confusion matrix_ di atas menunjukkan hasil evaluasi model klasifikasi biner. Berikut adalah rincian nilai-nilai dalam matriks:

- **True Positive (TP):** 394  
  Model memprediksi kelas positif (1) dengan benar.

- **True Negative (TN):** 385  
  Model memprediksi kelas negatif (0) dengan benar.

- **False Positive (FP):** 5  
  Model memprediksi kelas positif (1), tetapi sebenarnya kelas negatif (0).

- **False Negative (FN):** 16  
  Model memprediksi kelas negatif (0), tetapi sebenarnya kelas positif (1).

Di bawah ini adalah detail lengkap terkait evaluasi performa model deep learning dengan memanfaatkan serangkaian metrik yang relevan:

  ```python
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average='weighted')
    recall = metrics.recall_score(y_test, y_pred, average='weighted')
    f1 = metrics.f1_score(y_test, y_pred, average='weighted')
    
    classification_report = metrics.classification_report(y_test, y_pred)
    print(f"Classification Report :\n{classification_report}")
  ```

Hasil:
  ```python
    Classification Report:
                precision    recall  f1-score   support
  
             0       0.96      0.97      0.97       392
             1       0.97      0.96      0.97       408
  
      accuracy                           0.97       800
     macro avg       0.97      0.97      0.97       800
  weighted avg       0.97      0.97      0.97       800
  ```
## Referensi

[1] G. Lukhayu Pritalia, “Analisis Komparatif Algoritme Machine Learning dan Penanganan Imbalanced Data pada Klasifikasi Kualitas Air Layak Minum,” *KONSTELASI: Konvergensi Teknologi dan Sistem Informasi*, vol. 2, no. 1, Apr. 2022, doi: https://doi.org/10.24002/konstelasi.v2i1.5630.

[2] K. Abirami, P. Radhakrishna, and M. Venkatesan, “Water Quality Analysis and Prediction using Machine Learning,” *IEEE Conference on Signal Processing and Communication Technologies (CSNT)*, Apr. 2023, doi: https://doi.org/10.1109/csnt57126.2023.10134661.

[3] S. Iyer, S. Kaushik, and Poonam Nandal, “Water Quality Prediction Using Machine Learning,” *MR International Journal of Engineering and Technology*, vol. 10, no. 1, pp. 59–62, May 2023, doi: https://doi.org/10.58864/mrijet.2023.10.1.8.

[4] A. K. Mishra and B. K. Panigrahi, “A Comparative Study of Machine Learning Algorithms for Water Quality Prediction,” *Journal of Environmental Management*, vol. 280, Feb. 2021, doi: https://doi.org/10.1016/j.jenvman.2020.111682.

[5] R. Salim and T. Taslim, “EDUKASI MANFAAT AIR MINERAL PADA TUBUH BAGI ANAK SEKOLAH DASAR SECARA ONLINE,” *JPKM*, vol. 27, no. 2, Mar. 2021.

[6] Mega Fia Lestari et al., “Analysis of mineral water quality based on SNI 3553:2015 and its consequences from legal perspectives,” *IOP Conference Series: Earth and Environmental Science*, vol. 1190, no. 1, pp. 012041–012041, Jun. 2023, doi: https://doi.org/10.1088/1755-1315/1190/1/012041.

[7] E. B. Sasongko, E. Widyastuti, and R. E. Priyono, “KAJIAN KUALITAS AIR DAN PENGGUNAAN SUMUR GALI OLEH MASYARAKAT DI SEKITAR SUNGAI KALIYASA KABUPATEN CILACAP,” *Jurnal Ilmu Lingkungan*, vol. 12, no. 2, p. 72, Oct. 2014, doi: https://doi.org/10.14710/jil.12.2.72-82.

[8] I. Fitriyaningsih, Y. Basani, and L. M. Ginting, “MACHINE LEARNING: PROSPERITY OF RAINFALL, WATER DISCHARGE, AND FLOOD WITH WEB APPLICATION IN DELI SERDANG,” *JURNAL PENELITIAN KOMUNIKASI DAN OPINI PUBLIK*, vol. 22, no. 2, Dec. 2018, doi: https://doi.org/10.33299/jpkop.22.2.1752.

[9] Z. Zhang et al., “Deep Learning-Based Water Quality Prediction: A Comprehensive Review,” *Environmental Research*, vol. 200, Aug. 2021, doi: https://doi.org/10.1016/j.envres.2021.111643.

[10] A. Kumar and S. Jha, “Application of Deep Learning in Water Quality Monitoring and Prediction,” *International Journal of Environmental Science and Technology*, vol. 18, no. 5, pp. 1235–1248, May 2021, doi: https://doi.org/10.1007/s13762-020-02887-1.
