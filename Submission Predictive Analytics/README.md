# Laporan Proyek Machine Learning & Deep Learning - Royan Rosyad

## Water Quality and Potability Project

**Latar Belakang (Domain Proyek)**

Kualitas air minum merupakan faktor penting yang memengaruhi kesehatan masyarakat. Di Indonesia, masalah kualitas air yang tidak layak masih menjadi perhatian serius karena dapat menyebabkan gangguan kesehatan seperti dehidrasi, infeksi, hingga penyakit kronis. Oleh karena itu, pengembangan model machine learning dan deep learning untuk mengklasifikasikan kelayakan air minum sangat dibutuhkan. Model ini dapat digunakan oleh pemerintah, perusahaan air minum, atau lembaga terkait untuk memastikan bahwa air yang didistribusikan layak konsumsi. Dengan adanya model ini, masyarakat dapat terhindar dari risiko mengonsumsi air yang tidak aman.

# Business Understanding

### Problem Statements
- Berdasarkan eksplorasi terhadap _dataset_, fitur-fitur apa saja yang paling berpengaruh dalam menentukan kelayakan air minum?
- Bagaimana cara memproses _dataset_ agar dapat digunakan untuk pembuatan model klasifikasi kualitas air minum menggunakan pendekatan _machine learning_ dan _deep learning_?
- Bagaimana cara mendapatkan model klasifikasi kualitas air minum dengan performa terbaik antara pendekatan _machine learning_ dan _deep learning_?

### Goals
- Melakukan eksplorasi data untuk memahami hubungan antara fitur-fitur numerik dan label klasifikasi (`Potability`).
- Melakukan _data preparation_ untuk mempersiapkan data yang bersih dan siap digunakan dalam proses pelatihan model.
- Membandingkan performa antara model _machine learning_ (_Random Forest_, _KNN_, _XGB_) dan _deep learning_ untuk menentukan pendekatan mana yang lebih optimal.
- Menghasilkan model klasifikasi dengan performa terbaik berdasarkan metrik evaluasi seperti _accuracy_, _precision_, _recall_, dan _F1-score_.

### Solution Statements
- Untuk melakukan eksplorasi data, dilakukan analisis univariat dan multivariat untuk memahami distribusi data, korelasi antar fitur, serta pola yang memengaruhi label klasifikasi. Visualisasi seperti _heatmap_, _correlation matrix_, dan _boxplot_ digunakan untuk mendapatkan wawasan lebih lanjut.
- Proses _data preparation_ mencakup _data cleaning_ (penanganan missing value), _oversampling_ untuk menyeimbangkan kelas minoritas, _feature scaling_ untuk normalisasi data, dan _train-test split_ untuk membagi data latih dan uji.
- Untuk mendapatkan model dengan performa terbaik, digunakan dua pendekatan:  
- **Machine Learning**: Menerapkan algoritma _Random Forest_, _KNN_, dan _XGB_ sebagai _baseline model_. Performa model dievaluasi menggunakan _Confusion Matrix_ dan _Grid Search_ untuk optimasi hyperparameter.
- **Deep Learning**: Membangun arsitektur neural network dengan beberapa hidden layers, dropout untuk regularisasi, dan fungsi aktivasi ReLU. Model dilatih menggunakan optimizer Adam dan loss function _binary_crossentropy_.
- Hasil dari kedua pendekatan tersebut dibandingkan untuk menentukan model yang paling optimal dalam memprediksi kelayakan air minum.

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
 ```python
  df.shape
  ```
  Kode diatas memiliki output:
  ```python
  (3276, 10)
  ```

  Berdasarkan _output_ tersebut, didapatkan informasi bahwa df ini memiliki **3276 baris** dan **10 kolom** data sesuai dengan dengan keterangan yang tertera diatas. Pada bagian ini, belum dapat diketahui **nama** dari **kolom-kolom** yang ada.
 ```python
   df.keys()
  ```
  Kode diatas memiliki output:
  ```python
  Index(['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
       'Organic_carbon', 'Trihalomethanes', 'Turbidity', 'Potability'],
      dtype='object')
  ```

  Berdasarkan _output_ tersebut, didapatkan informasi lebih lanjut bahwa dataset ini memiliki 10 kolom sesuai dengan keterangan yang tertera diatas. Pada bagian ini, belum dapat diketahui **jumlah** dan **tipe data** dari tiap kolom.
 ```python
   df.info()
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
 ```python
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
  df.isnull().sum()
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

  Berdasarkan visualisasi pada ``` Gambar 1.1  ``` , dapat diamati bahwa variabel ```Potability``` memiliki dua nilai unik, yaitu '1' yang mengindikasikan air minum layak dikonsumsi dan '0' yang mengindikasikan sebaliknya. Akan tetapi, visualisasi tersebut juga          memperlihatkan adanya ketidakseimbangan data (_imbalance data_). Jumlah baris data dengan nilai '0' mencapai hampir 2000, sementara nilai '1' hanya memiliki sekitar 1250 baris data. Mengingat kondisi ini, tindakan penyeimbangan data menjadi krusial untuk               menghindari terjadinya bias pada model _machine learning_ yang akan dikembangkan.

  ![Distribusi Numerical Fitur](https://github.com/user-attachments/assets/3fc00fbc-fc2c-475a-ad54-50484de4d225)
  <div align="center">Gambar 1.2 - Distribution of Numerical Feature</div>
  
  
  Merujuk pada visualisasi dalam ```Gambar 1.2```, yang menyajikan distribusi untuk setiap kolom numerik dalam dataset (```pH```, ```Hardness```, ```Solids```, ```Chrolamines```, ```Sulfate```, ```Conductivity```, ```Organic_carbon```, ```Trihalomethanes```,             ```Turbidity```), terlihat bahwa hanya kolom ```Solids``` dan ```Conductivity``` yang menunjukkan kemiringan (skewness) ke kiri. Informasi ringkas yang dapat disarikan dari visualisasi tersebut adalah sebagai berikut:

   - ```pH```: Tingkat keasaman air bervariasi dari 0 hingga 14, dengan mayoritas sampel memiliki nilai pH di sekitar 7, yang menandakan kondisi netral.
   - ```Hardness```: Tingkat kesadahan air beragam, dengan konsentrasi sampel yang signifikan menunjukkan tingkat kesadahan sekitar 200.
   - ```Solids```: Jumlah total padatan terlarut dalam sampel bervariasi, dengan frekuensi tertinggi berada di dekat nilai 20.000.
   - ```Chloramines```: Kadar kloramin dalam sampel mencapai puncak frekuensi pada rentang nilai 7 hingga 8.
   - ```Sulfate```: Konsentrasi sulfat dalam sampel paling sering ditemukan di sekitar nilai 300.
   - ```Conductivity```: Tingkat konduktivitas sampel memiliki frekuensi tertinggi di sekitar nilai 400.
   - ```Organic_carbon```: Kandungan karbon organik yang paling umum dalam sampel berkisar antara 14 hingga 15.
   - ```Trihalomethanes```: Kadar trihalometana dalam sampel paling sering berada di rentang 65 hingga 70.
   - ```Turbidity```: Tingkat kekeruhan sampel memiliki frekuensi tertinggi di sekitar nilai 3,5.

  
  **Multivariate Analysis**

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
      duplicates = df.duplicated()
      
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

## Pendekatan Machine Learning

### Random Forest Classifier
#### Mekanisme Kerja
Random Forest adalah metode ensemble learning yang beroperasi dengan membangun beberapa pohon keputusan (decision tree) selama pelatihan. Setiap pohon dalam "hutan" membuat prediksi kelas, dan output akhir ditentukan oleh voting mayoritas di antara semua pohon. Algoritma bekerja dengan:

1. Membuat sampel bootstrap dari dataset asli
2. Membangun pohon keputusan untuk setiap sampel, menggunakan subset fitur acak pada setiap split
3. Mengumpulkan prediksi dari semua pohon untuk membuat klasifikasi akhir

Pendekatan ini mengurangi overfitting dan meningkatkan generalisasi dibandingkan dengan pohon keputusan tunggal.

#### Parameter
- `n_estimators=150`: Jumlah pohon dalam hutan. Parameter ini ditentukan melalui GridSearchCV dari opsi [50, 100, 150, 170, 200, 230, 250, 300]. Lebih banyak pohon umumnya memberikan kinerja lebih baik tetapi membutuhkan lebih banyak komputasi.
- Parameter lain tetap pada nilai default:
  - `criterion='gini'`: Fungsi untuk mengukur kualitas split
  - `max_depth=None`: Pohon tumbuh hingga kedalaman maksimum yang mungkin
  - `min_samples_split=2`: Jumlah minimum sampel yang diperlukan untuk membagi node
  - `min_samples_leaf=1`: Jumlah minimum sampel yang diperlukan pada node daun

#### Performa
Meskipun memiliki akurasi tertinggi di antara model _machine learning_ (60,62%), presisi (51,16%) dan recall (26,76%) masih jauh dari optimal. Hal ini menunjukkan bahwa model sering salah mengklasifikasikan air layak minum sebagai tidak layak (false negative).

### K-Nearest Neighbors Classifier
#### Mekanisme Kerja
KNN adalah algoritma pembelajaran non-parametrik "malas" (lazy learning) yang mengklasifikasikan titik data berdasarkan kelas mayoritas dari k tetangga terdekatnya. Algoritma ini:

1. Menghitung jarak (biasanya Euclidean) antara titik kueri dan semua contoh pelatihan
2. Memilih K titik data terdekat berdasarkan jarak ini
3. Menetapkan label kelas berdasarkan voting mayoritas dari tetangga-tetangga ini

KNN sangat efektif ketika batas keputusan antara kelas tidak beraturan.

#### Parameter
- `n_neighbors=30`: Jumlah tetangga yang dipertimbangkan untuk klasifikasi. Nilai ini dioptimalkan melalui GridSearchCV dari opsi [3, 5, 7, 10, 15, 20, 30].
- `weights='distance'`: Memberikan bobot pada titik berdasarkan kebalikan dari jarak mereka, memberikan tetangga yang lebih dekat pengaruh lebih besar.
- `metric='euclidean'`: Metrik jarak yang digunakan untuk menghitung kedekatan tetangga. Jarak Euclidean dipilih dari opsi ['euclidean', 'manhattan'] selama optimasi.

#### Performa
Performa KNN adalah yang terendah, dengan recall hanya 15,42%. Ini berarti model sangat buruk dalam mengidentifikasi air yang benar-benar layak minum, yang dapat berdampak serius pada keamanan air.

### XGBoost Classifier
#### Mekanisme Kerja
XGBoost (Extreme Gradient Boosting) adalah implementasi lanjutan dari gradient boosting yang bekerja dengan secara berurutan menambahkan learner lemah untuk memperbaiki kesalahan yang dibuat oleh model sebelumnya. Algoritma ini:

1. Dimulai dengan model sederhana (biasanya pohon keputusan tunggal)
2. Secara iteratif menambahkan pohon baru yang berfokus pada memprediksi kesalahan residual dari pohon sebelumnya dengan benar
3. Menggabungkan semua pohon untuk prediksi akhir

XGBoost mencakup teknik regularisasi untuk mencegah overfitting dan menangani nilai yang hilang secara efisien.

#### Performa
Meskipun recall lebih tinggi (33,26%) dibandingkan Random Forest dan KNN, akurasi dan presisi tetap rendah. F1-score sebesar 37,95% menunjukkan bahwa model ini mencapai keseimbangan yang lebih baik antara presisi dan recall, namun masih jauh dari performa yang diharapkan.

#### Parameter
- `learning_rate=0.1`: Langkah pengurangan ukuran yang digunakan untuk mencegah overfitting. Nilai ini ditentukan melalui GridSearchCV dari opsi [0.001, 0.01, 0.1, 0.5, 1].
- `n_estimators=250`: Jumlah putaran boosting (pohon). Ini dioptimalkan dari opsi [50, 100, 150, 200, 250, 300, 350].
- Parameter lain tetap pada nilai default:
  - `max_depth=6`: Kedalaman maksimum pohon
  - `subsample=1`: Fraksi sampel yang digunakan untuk melatih pohon
  - `colsample_bytree=1`: Fraksi fitur yang digunakan untuk melatih pohon


  Berikut hasil dari Grid Search untuk Hyperparameter Tuning:
  
| Model                  | Best Params                                                     | Best Accuracy |
|------------------------|-----------------------------------------------------------------|---------------|
| RandomForestClassifier | {'n_estimators': 150}                                           | 0.729229      |
| KNeighborsClassifier   | {'metric': 'euclidean', 'n_neighbors': 30, 'weights': 'uniform'}| 0.696446      |
| XGBClassifier          | {'learning_rate': 0.5, 'n_estimators': 300}                     | 0.714214      |

Belum cukup puas dengan hasilnya, saya bereksperimen menggunakan deep learning untuk meningkatkan accuracy training.


## Pendekatan Deep Learning

#### Mekanisme Kerja
Model deep learning yang digunakan adalah jaringan saraf feedforward yang dirancang untuk mempelajari pola dan hubungan kompleks antara parameter kualitas air dan kelayakan minum. Proses kerjanya melibatkan beberapa tahapan utama:

1. **Forward Propagation**:  
   - Fitur input (9 parameter kualitas air) diproses melalui neuron-neuron yang saling terhubung dalam lapisan-lapisan.
   - Bobot dan bias diterapkan pada setiap neuron untuk menghitung jumlah terbobot.
   - Fungsi aktivasi non-linear (ReLU) digunakan untuk mentransformasi nilai agar model dapat mempelajari hubungan non-linear dalam data.

2. **Backpropagation**:  
   - Error dihitung menggunakan fungsi loss (_Binary Cross-Entropy_) untuk mengukur seberapa jauh prediksi model dari label sebenarnya.
   - Gradien error dihitung secara mundur melalui jaringan untuk memperbarui bobot dan bias, sehingga model dapat belajar dari kesalahannya.

3. **Optimisasi**:  
   - Optimizer Adam digunakan untuk memperbarui bobot dengan efisien, memastikan konvergensi yang cepat dan stabil.

#### Arsitektur dan Parameter
Model dibangun dengan arsitektur sebagai berikut:
- **Input Layer**: Menerima 9 fitur numerik yang merepresentasikan parameter kualitas air.
- **Hidden Layers**:
  - Tiga Dense layers dengan 64 neuron masing-masing dan fungsi aktivasi ReLU untuk mengekstraksi fitur kompleks.
  - Dropout layer dengan rate 0.2 digunakan untuk mencegah overfitting dengan secara acak "mematikan" 20% neuron selama pelatihan.
  - Satu Dense layer tambahan dengan 16 neuron dan fungsi aktivasi ReLU untuk ekstraksi fitur lebih lanjut.
- **Output Layer**: Satu neuron dengan fungsi aktivasi sigmoid untuk menghasilkan probabilitas klasifikasi biner (layak/tidak layak minum).

Model dikompilasi dengan:
- **Optimizer**: Adam dengan _learning rate_ default (0.001).
- **Loss Function**: Binary Cross-Entropy, yang cocok untuk tugas klasifikasi biner.
- **Metrics**: Accuracy, Precision, dan Recall untuk evaluasi performa model secara komprehensif.

Parameter training mencakup:
- `epochs=200`: Model dilatih selama 200 iterasi penuh melalui dataset pelatihan.
- `batch_size=64`: Jumlah sampel yang diproses sebelum bobot model diperbarui.
- `validation_split=0.2`: 20% data pelatihan dicadangkan untuk validasi guna memantau performa model selama pelatihan.
- **ModelCheckpoint**: Callback ini digunakan untuk menyimpan model terbaik berdasarkan akurasi validasi tertinggi (`val_accuracy`).

#### Implementasi Kode
Arsitektur dan proses pelatihan model diimplementasikan menggunakan kode berikut:

```python
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

# Train the model
history = model.fit(
    x_resampled, y_resampled,
    validation_split=0.2,
    epochs=200,
    batch_size=64,
    verbose=1,
    callbacks=[checkpoint_callback]
)
```

#### Keunggulan Pendekatan
- **Kemampuan Menangkap Pola Kompleks**: Model deep learning mampu menangkap hubungan non-linear antara fitur-fitur kualitas air.
- **Regularisasi**: Penggunaan dropout membantu mencegah overfitting, sehingga model lebih generalizable.
- **Otomatisasi Penyimpanan Model Terbaik**: Callback `ModelCheckpoint` memastikan bahwa hanya model dengan performa validasi tertinggi yang disimpan.


### Proses Pelatihan
Proses pelatihan dilakukan dengan membagi dataset menjadi data latih (80%) dan data validasi (20%). Model dilatih selama 200 epoch dengan batch size sebesar 64 untuk memastikan pembelajaran yang efisien. Selama pelatihan, performa model dipantau menggunakan metrik seperti accuracy, precision, dan recall. Callback `ModelCheckpoint` digunakan untuk menyimpan model terbaik berdasarkan akurasi validasi tertinggi, sehingga hasil akhir yang dihasilkan optimal.

  Berikut preview hasil training nya:

  ![epoch](https://github.com/user-attachments/assets/cd43d189-f548-4148-a1b2-ad8ef06dfb21)

# Evaluation

## Pemilihan Metrik Evaluasi dan Relevansi

Untuk proyek klasifikasi kelayakan air ini, kami memilih metrik evaluasi berdasarkan relevansinya dengan konteks bisnis yang telah dijelaskan dalam **Business Understanding**:

1. **Accuracy (Akurasi)**: Mengukur kebenaran keseluruhan prediksi model. Dalam konteks **Goals**, akurasi memberikan indikasi seberapa baik model dapat memprediksi kelayakan air minum berdasarkan fitur-fitur numerik yang dieksplorasi.
2. **Precision (Presisi)**: Mewakili proporsi identifikasi positif (air layak minum) yang benar-benar tepat. Ini penting untuk mengatasi masalah **Problem Statement** terkait risiko klasifikasi salah (false positive), yang dapat menyebabkan konsumsi air tidak aman.
3. **Recall (Sensitivitas)**: Mengukur proporsi positif sebenarnya (air yang benar-benar layak minum) yang diidentifikasi dengan benar. Recall tinggi memastikan bahwa air yang aman tidak salah ditolak, sesuai dengan **Solution Statement** untuk meminimalkan false negative.
4. **F1-Score**: Rata-rata harmonik antara presisi dan recall, memberikan keseimbangan antara kedua metrik ini. Hal ini sesuai dengan **Goals** untuk menghasilkan model dengan performa terbaik berdasarkan metrik evaluasi komprehensif.

Metrik-metrik ini dipilih untuk memastikan bahwa model yang dibangun tidak hanya memenuhi **Goals** tetapi juga relevan dengan **Problem Statements** seperti menentukan fitur paling berpengaruh dan membandingkan performa antara pendekatan _machine learning_ dan _deep learning_.


## Analisis Hasil Komparatif

### Performa Model Machine Learning

| Model                   | Akurasi  | Presisi | Recall   | F1-Score |
|-------------------------|----------|---------|----------|----------|
| Random Forest           | 60,62%   | 51,16%  | 26,76%   | 34,94%   |
| KNN                     | 59,55%   | 44,60%  | 15,42%   | 22,88%   |
| XGBoost                 | 57,42%   | 44,61%  | 33,26%   | 37,95%   |

### Performa Model Deep Learning

  ```python
    Classification Report:
                precision    recall  f1-score   support
  
             0       0.96      0.97      0.97       392
             1       0.97      0.96      0.97       408
  
      accuracy                           0.97       800
     macro avg       0.97      0.97      0.97       800
  weighted avg       0.97      0.97      0.97       800
  ```

Model deep learning secara signifikan mengungguli semua pendekatan machine learning tradisional di semua metrik. Arsitektur jaringan saraf mampu menangkap hubungan non-linear yang kompleks antara parameter air dan kelayakan minum, yang tidak dimodelkan secara efektif oleh algoritma berbasis pohon (_Random Forest_, _KNN_, _XGB_), sesuai dengan **Solution Statement** untuk membandingkan performa kedua pendekatan.


## Hubungan dengan Business Understanding

### Mengatasi Problem Statements

1. **Fitur Berpengaruh**:  
   - Eksplorasi data menggunakan analisis univariat dan multivariat berhasil mengidentifikasi fitur seperti `pH`, `Hardness`, `Solids`, dan `Organic_carbon` sebagai faktor utama yang memengaruhi klasifikasi kualitas air. Hal ini menjawab **Problem Statement** pertama tentang fitur paling berpengaruh.
   - Visualisasi seperti _heatmap_ dan _correlation matrix_ memberikan wawasan lebih lanjut tentang hubungan antar fitur, mendukung **Solution Statement** untuk eksplorasi data.

2. **Proses Dataset**:  
   - Proses _data preparation_ mencakup penanganan missing value, _oversampling_ untuk menyeimbangkan kelas minoritas, _feature scaling_, dan _train-test split_. Langkah-langkah ini memastikan dataset siap digunakan untuk pelatihan model, menjawab **Problem Statement** kedua tentang cara memproses dataset.

3. **Model Terbaik**:  
   - Pendekatan _deep learning_ dengan arsitektur neural network berhasil mencapai performa terbaik (akurasi 97%) dibandingkan dengan model _machine learning_ tradisional. Hal ini menjawab **Problem Statement** ketiga tentang cara mendapatkan model dengan performa terbaik.


### Pencapaian Goals

1. **Eksplorasi Data**:  
   - Analisis univariat dan multivariat berhasil memahami hubungan antara fitur numerik dan label klasifikasi (`Potability`), sesuai dengan **Goal** pertama.

2. **Data Preparation**:  
   - Proses _data preparation_ yang mencakup _data cleaning_, _oversampling_, _feature scaling_, dan _train-test split_ memastikan data siap digunakan untuk pelatihan model, sesuai dengan **Goal** kedua.

3. **Perbandingan Performa**:  
   - Model _deep learning_ secara signifikan mengungguli model _machine learning_ tradisional, memenuhi **Goal** ketiga untuk membandingkan performa antara kedua pendekatan.

4. **Performa Terbaik**:  
   - Model _deep learning_ mencapai akurasi 97%, presisi 97%, recall 96%, dan F1-score 97%, memenuhi **Goal** keempat untuk menghasilkan model dengan performa terbaik.


### Dampak Solution Statements

1. **Pipeline Preprocessing Data**:  
   - Alur kerja preprocessing yang diimplementasikan berhasil menangani nilai yang hilang menggunakan imputasi median, menyeimbangkan distribusi kelas melalui oversampling, dan menstandarisasi fitur melalui MinMaxScaling. Langkah-langkah ini sangat penting untuk mencapai kinerja model yang tinggi, sesuai dengan **Solution Statement**.

2. **Pendekatan Machine Learning**:  
   - Meskipun algoritma tradisional mencapai kinerja moderat (~60% akurasi), pendekatan tersebut memberikan insight tentang hubungan fitur dan menetapkan baseline kinerja. Namun, performa rendah dalam recall dan precision menunjukkan bahwa pendekatan ini tidak cukup andal untuk aplikasi ini, sesuai dengan **Solution Statement**.

3. **Peningkatan Deep Learning**:  
   - Arsitektur jaringan saraf memberikan hasil luar biasa dengan akurasi 97%, menunjukkan kemampuannya yang unggul dalam memodelkan hubungan kompleks dalam data kualitas air, sesuai dengan **Solution Statement**.

4. **Kerangka Evaluasi Komprehensif**:  
   - Pendekatan evaluasi multi-metrik (akurasi, presisi, recall, F1) memberikan penilaian holistik terhadap kinerja model, mengonfirmasi keunggulan jaringan saraf untuk studi kasis ini, sesuai dengan **Solution Statement**.


Berikut hasil visualisasi dari evaluasi training model deep learning:

![eval](https://github.com/user-attachments/assets/bdf40886-9681-4c6f-b310-7315d542c644)

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
