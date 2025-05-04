# **Laporan Proyek Machine Learning - Royan Rosyad**

## Project Overview

Proyek ini bertujuan untuk membangun **sistem rekomendasi film** yang cerdas dan relevan menggunakan dataset **MovieLens**. Dataset ini mencakup lebih dari 9,000 judul film yang dirilis hingga September 2018. Selain itu, terdapat lebih dari 100k data rating yang diberikan oleh 600 pengguna, masing-masing dalam skala 1 sampai 5.

Dalam era digital saat ini, layanan *streaming* seperti Netflix atau Amazon Prime menyediakan ribuan pilihan film dan serial. Namun, banyaknya konten justru bisa membuat pengguna kesulitan memilih. Sistem rekomendasi hadir sebagai solusi untuk membantu pengguna menemukan film sesuai preferensi mereka, sehingga pengalaman menonton menjadi lebih personal dan menyenangkan. Di sisi lain, sistem ini juga memberikan nilai tambah bagi penyedia layanan dengan meningkatkan retensi dan kepuasan pengguna.

Proyek ini menggabungkan dua pendekatan utama dalam sistem rekomendasi, yaitu *content-based filtering* dan *collaborative filtering*, untuk membangun model yang mampu memberikan rekomendasi yang lebih akurat dan efisien berdasarkan pola perilaku pengguna dan karakteristik film.

### Latar Belakang

Di tengah pertumbuhan pesat platform streaming digital, jumlah konten film yang tersedia semakin banyak dan beragam. Hal ini sering kali menyebabkan pengguna mengalami kesulitan dalam memilih film yang sesuai dengan minat mereka sebuah fenomena yang dikenal sebagai information overload . Tanpa adanya mekanisme penyaringan otomatis, pengguna cenderung merasa frustrasi atau bahkan beralih ke platform lain. Oleh karena itu, sistem rekomendasi menjadi solusi strategis untuk meningkatkan pengalaman pengguna sekaligus mendukung retensi dan pertumbuhan bisnis platform hiburan.

Salah satu pendekatan yang paling umum digunakan dalam sistem rekomendasi adalah content-based filtering , di mana rekomendasi dibuat berdasarkan kesamaan karakteristik antar item, seperti genre, deskripsi, aktor, atau sutradara. Pendekatan ini sangat efektif ketika metadata film tersedia secara lengkap dan terstruktur. `Sundus Ayyaz et al. (2018)` menunjukkan bahwa integrasi metode fuzzy logic dan conformal prediction ke dalam pendekatan berbasis konten dapat membantu meningkatkan kualitas rekomendasi dengan memberikan tingkat keyakinan (confidence ) terhadap setiap saran yang diberikan. Namun, metode ini memiliki keterbatasan dalam menangkap preferensi dinamis pengguna atau membantu mereka menjelajahi hal-hal baru di luar pola konsumsi sebelumnya.

Sebagai komplementer, collaborative filtering memanfaatkan interaksi pengguna lain untuk memberikan rekomendasi yang lebih personal. Berdasarkan prinsip bahwa pengguna dengan preferensi serupa akan menyukai item serupa, pendekatan ini telah terbukti efektif dalam berbagai domain, termasuk pariwisata `(Lin et al., 2022)`, musik `(Wang, 2022)`, buku `(Du et al., 2022)`, dan juga film. Penelitian terbaru seperti `Liu et al. (2025)` juga menunjukkan perkembangan dalam penggabungan Graph Neural Network (GNN) dan teknik attention untuk meningkatkan akurasi prediksi dalam skema collaborative filtering . Kombinasi kedua pendekatan ini, baik secara terpisah maupun dalam model hybrid , memberikan potensi besar untuk menciptakan sistem rekomendasi film yang lebih cerdas dan responsif terhadap perubahan selera pengguna.

## 💼 Business Understanding

Dalam pengembangan sistem rekomendasi film, pemahaman akan kebutuhan bisnis menjadi dasar penting untuk memastikan bahwa solusi yang dibangun tidak hanya efektif secara teknis, tetapi juga relevan dalam menjawab permasalahan nyata di dunia pengguna. Sistem rekomendasi bertujuan untuk meningkatkan pengalaman pengguna dengan memberikan saran film yang lebih personal dan akurat, sekaligus membantu platform *streaming* dalam meningkatkan retensi dan keterlibatan pengguna.

### Problem Statements

- Bagaimana cara memahami dan memperoleh informasi mengenai data yang digunakan dalam pembuatan model sistem rekomendasi?
- Bagaimana cara membangun model sistem rekomendasi dengan menggunakan pendekatan *content-based filtering*?
- Bagaimana cara mengembangkan model sistem rekomendasi dengan pendekatan *collaborative filtering*?
- Bagaimana cara menilai kinerja model sistem rekomendasi yang telah dikembangkan?

### Goals

- Melakukan eksplorasi data awal (*Exploratory Data Analysis / EDA*) dan visualisasi untuk memahami struktur serta karakteristik dataset.
- Membangun sistem rekomendasi film menggunakan pendekatan *content-based filtering*.
- Membangun sistem rekomendasi film menggunakan pendekatan *collaborative filtering*.
- Mengevaluasi kinerja model sistem rekomendasi yang telah dibuat menggunakan metrik yang sesuai.

### Solution Approach

Untuk memecahkan rumusan masalah dan mencapai tujuan yang telah ditentukan, diperlukan pendekatan solusi yang sistematis dan terstruktur sebagai berikut:

- **Melakukan eksplorasi data**:  
  Langkah awal dalam proyek ini adalah memahami dataset yang digunakan melalui *exploratory data analysis*. Proses ini mencakup analisis jumlah baris dan kolom, tipe data, distribusi nilai, serta visualisasi grafik yang membantu mengidentifikasi pola atau anomali dalam data.

- **Membangun sistem rekomendasi berbasis konten (*Content-Based Filtering*)**:  
  Pendekatan ini dimulai dengan proses *data cleaning*, termasuk:
  - Penghapusan duplikasi dan data kosong (*removal of duplicates and NaN data*)
  - Penghapusan kolom yang tidak relevan
  - Penanganan ketidakseimbangan data (*imbalanced data handling*)
  - Pemrosesan teks (*text processing*)

  Setelah itu dilanjutkan dengan *data transformation*, seperti vektorisasi teks menggunakan TF-IDF, lalu menghitung kesamaan (*cosine similarity*) antar film. Tahap akhir adalah pembuatan fungsi rekomendasi dan uji coba prediksi.

- **Membangun sistem rekomendasi berbasis kolaboratif (*Collaborative Filtering*)**:  
  Prosesnya juga dimulai dengan *data cleaning* dan persiapan data, termasuk:
  - Penghapusan duplikasi dan data tidak valid
  - Penyandian variabel kategorikal (*data encoding*)
  - Pembagian data menjadi set pelatihan dan pengujian (*train-test split*)

- **Evaluasi kinerja model**:  
  Evaluasi dilakukan untuk memastikan model memberikan hasil yang optimal:
  - Untuk model *content-based filtering*, digunakan metrik evaluasi seperti *Precision* untuk mengukur tingkat keakuratan rekomendasi.
  - Untuk model *collaborative filtering*, digunakan metrik seperti *Root Mean Squared Error (RMSE)* untuk mengukur kesalahan prediksi rating pengguna.

## 📊 Data Understanding

### 📁 Import Dataset

Dataset yang digunakan dalam proyek ini adalah **"MovieLens Latest Small Dataset"**, yang tersedia secara publik di situs web GroupLens. Dataset ini dapat diakses dan diunduh melalui tautan berikut: [https://files.grouplens.org/datasets/movielens/ml-latest-small.zip](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip).

Dataset ini terdiri dari beberapa file CSV yang saling terkait, yaitu:

- `tags.csv`: berisi tag subjektif yang diberikan oleh pengguna terhadap film.
- `ratings.csv`: berisi penilaian pengguna terhadap film.
- `movies.csv`: berisi informasi judul dan genre film.
- `links.csv`: berisi ID eksternal seperti IMDb dan TMDB untuk masing-masing film.

Untuk kebutuhan proyek ini, dua file utama yang akan digunakan adalah:

- `movies.csv`: untuk membangun model *content-based filtering* berdasarkan metadata film (judul dan genre).
- `ratings.csv`: untuk membangun model *collaborative filtering* berdasarkan riwayat rating pengguna.

### Struktur Variabel Dataset

#### File: `movies.csv`
| Kolom     | Deskripsi |
|-----------|-----------|
| `movieId` | Pengenal unik untuk setiap film |
| `title`   | Judul lengkap dari film beserta tahun rilisnya |
| `genres`  | Genre atau kategori film (dipisahkan dengan `|`) |

#### File: `ratings.csv`
| Kolom       | Deskripsi |
|-------------|-----------|
| `userId`    | Pengenal unik untuk pengguna |
| `movieId`   | Pengenal unik untuk film |
| `rating`    | Nilai rating yang diberikan pengguna (skala 0–5) |
| `timestamp` | Waktu ketika rating dicatat (format UNIX timestamp) |

### Proses Pengunduhan dan Pemuatan Dataset

Proses impor dataset dilakukan dengan langkah-langkah berikut:

1. **Mengunduh file ZIP dataset** dari URL resmi.
2. **Mengekstrak file ZIP** agar file CSV dapat diakses.
3. **Memuat file CSV** (`movies.csv` dan `ratings.csv`) menggunakan library `pandas`.

Berikut code Python untuk proses impor dataset:

```python
import os
import requests
import zipfile
import pandas as pd

# 1. Unduh dataset dari URL
URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
response = requests.get(URL)
open("ml-latest-small.zip", "wb").write(response.content)

# 2. Ekstrak file zip
with zipfile.ZipFile("ml-latest-small.zip", 'r') as zip_ref:
    zip_ref.extractall(".")

# 3. Pastikan folder telah diekstrak
dataset_dir = "ml-latest-small"
print(f"Isi direktori {dataset_dir}:", os.listdir(dataset_dir))

# 4. Muat file movies.csv dan ratings.csv
movies_path = os.path.join(dataset_dir, "movies.csv")
ratings_path = os.path.join(dataset_dir, "ratings.csv")

movies_df = pd.read_csv(movies_path)
ratings_df = pd.read_csv(ratings_path)
```

Dengan proses tersebut, dataset siap digunakan untuk tahap selanjutnya, yaitu eksplorasi data (*Exploratory Data Analysis / EDA*) dan pembangunan model sistem rekomendasi.

## 🔍 Exploratory Data Analysis

*Exploratory Data Analysis* atau EDA adalah tahap awal dalam proses analisis data untuk memahami karakteristik dataset, mengidentifikasi pola, menemukan anomali, serta memverifikasi asumsi-asumsi yang muncul dari data. Dalam proyek sistem rekomendasi ini, dataset utama yang digunakan adalah `movies.csv` dan `ratings.csv`.

### Analisis File `movies.csv`

Untuk mengetahui jumlah baris dan kolom pada `movies_df`:

```python
print('panjang baris dan kolom movies_df:', movies_df.shape)
print('movies_df kolom', movies_df.keys())
```

```python
panjang baris dan kolom movies_df: (9742, 3)
movies_df kolom Index(['movieId', 'title', 'genres'], dtype='object')
```

Dataset ini memiliki 9.742 film dengan tiga informasi utama: ID film, judul, dan genre.

Selanjutnya, dilakukan pengecekan informasi detail tiap kolom:

```python
print('informasi kolom movies_df:')
movies_df.info()
```

```python
informasi kolom movies_df:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 9742 entries, 0 to 9741
Data columns (total 3 columns):
 #   Column   Non-Null Count  Dtype 
---  ------   --------------  ----- 
 0   movieId  9742 non-null   int64 
 1   title    9742 non-null   object
 2   genres   9742 non-null   object
dtypes: int64(1), object(2)
memory usage: 228.5+ KB
```

Tidak ada nilai kosong (*missing value*) pada seluruh baris dan kolom.

Ringkasan statistik dari kolom numerik (`movieId`) ditampilkan sebagai berikut:

```python
print('informasi statistik movies_df:')
movies_df.describe()
```

| Statistik| movieId           |
|----------|-------------------|
| Count    | 9742.000000       |
| Mean     | 42200.353623      |
| Std      | 52160.494854      |
| Min      | 1.00000           |
| 25%      | 3248.250000       |
| 50%      | 7300.000000       |
| 75%      | 76232.000000      |
| Max      | 193609.000000     |

Lima baris pertama dari `movies_df`:

```python
movies_df.head()
```

| movieId | title                             | genres                                          |
|---------|-----------------------------------|-------------------------------------------------|
| 1       | Toy Story (1995)                  | Adventure\|Animation\|Children\|Comedy\|Fantasy |
| 2       | Jumanji (1995)                    | Adventure\|Children\|Fantasy                    |
| 3       | Grumpier Old Men (1995)           | Comedy\|Romance                                 |
| 4       | Waiting to Exhale (1995)          | Comedy\|Drama\|Romance                          |
| 5       | Father of the Bride Part II (1995)| Comedy                                          |

Karena kolom `genres` berisi beberapa genre yang dipisahkan dengan simbol `|`, dilakukan pemrosesan untuk hanya mengambil genre utama menggunakan split:

```python
movies_df['genres'] = movies_df['genres'].str.split('|').str[0]
```

Setelah proses tersebut, lima baris pertama dari `movies_df` menjadi:

```python
print('movies_df top 5 rows after genre processing')
movies_df.head()
```

| movieId | title                             | genres    |
|---------|-----------------------------------|-----------|
| 0       | Toy Story (1995)                  | Adventure |
| 1       | Jumanji (1995)                    | Adventure |
| 2       | Grumpier Old Men (1995)           | Comedy    |
| 3       | Waiting to Exhale (1995)          | Comedy    |
| 4       | Father of the Bride Part II (1995)| Comedy    |

Pengecekan jumlah `movieId` yang unik:

```python
print('Num of unique movieId: ', len(movies_df['movieId'].unique()))
```

```markdown
Num of unique movieId: 9742
```

Semua `movieId` bersifat unik, tidak ada duplikasi.

Jumlah genre unik dan daftar isinya:

```python
print('Num of unique genres:', len(movies_df['genres'].unique()))
list_of_genres = pd.DataFrame(movies_df['genres'].unique(), columns=['genres'])
list_of_genres
```

Num of unique genres: 19
|    | Genre              |
|----|--------------------|
| 0  | Adventure          |
| 1  | Comedy             |
| 2  | Action             |
| 3  | Drama              |
| 4  | Crime              |
| 5  | Children           |
| 6  | Mystery            |
| 7  | Animation          |
| 8  | Documentary        |
| 9  | Thriller           |
| 10 | Horror             |
| 11 | Fantasy            |
| 12 | Western            |
| 13 | Film-Noir          |
| 14 | Romance            |
| 15 | Sci-Fi             |
| 16 | Musical            |
| 17 | War                |
| 18 | (no genres listed) |

Distribusi jumlah film per genre:

```python
movies_df['genres'].value_counts()
```

| Genre              | Jumlah Film |
|--------------------|-------------|
| Comedy             | 2,779       |
| Drama              | 2,226       |
| Action             | 1,828       |
| Adventure          | 653         |
| Crime              | 537         |
| Horror             | 468         |
| Documentary        | 386         |
| Animation          | 298         |
| Children           | 197         |
| Thriller           | 84          |
| Sci-Fi             | 62          |
| Mystery            | 48          |
| Fantasy            | 42          |
| Romance            | 38          |
| (no genres listed) | 34          |
| Musical            | 23          |
| Western            | 23          |
| Film-Noir          | 12          |
| War                | 4           |


Data menunjukkan adanya ketidakseimbangan jumlah film di setiap genre, terutama antara genre seperti Drama dan Comedy dibandingkan genre minoritas.

Pengecekan nilai kosong dan duplikasi:

```python
print(movies_df.isna().any().any())
print(movies_df.duplicated().any())
```

```markdown
False
Empty DataFrame
Columns: [movieId, title, genres]
Index: []
```

Cek missing value pada movies_df
```python
missing_value = movies_df.isnull().sum()
missing_value
```

output:
```markdown
movieId    0
title      0
genres     0
dtype: int64
```

Cek missing value pada ratings_df
```python
missing_value = ratings_df.isnull().sum()
missing_value
```
output:
```markdown
userId       0
movieId      0
rating       0
timestamp    0
dtype: int64
```



### 📊 Analisis File `ratings.csv`

Untuk mengetahui jumlah baris dan kolom pada `ratings_df`:

```python
print('panjang baris dan kolom ratings_df:', ratings_df.shape)
print('ratings_df kolom:', ratings_df.keys())
```

```markdown
panjang baris dan kolom ratings_df: (100836, 4)
ratings_df kolom: Index(['userId', 'movieId', 'rating', 'timestamp'], dtype='object')
```
Dataset ini mencatat lebih dari 20 juta interaksi pengguna terhadap film.

Informasi detail tiap kolom:

```python
ratings_df.info()
```

```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 100836 entries, 0 to 100835
Data columns (total 4 columns):
 #   Column     Non-Null Count   Dtype  
---  ------     --------------   -----  
 0   userId     100836 non-null  int64  
 1   movieId    100836 non-null  int64  
 2   rating     100836 non-null  float64
 3   timestamp  100836 non-null  int64  
dtypes: float64(1), int64(3)
memory usage: 3.1 MB
```

Berdasarkan *output* kode diatas didapatkan informasi berikut:

*   Terdapat 3 kolom numerik dengan tipe data *int64*, yaitu: `userId`, `movieId` dan `timestamp`.
*   Terdapat 1 kolom numerik dengan tipe data *float64*, yaitu `rating`.

Ringkasan statistik dari kolom numerik:

```python
print('ratings_df statistic information')
ratings_df.describe()
```

| Statistik| userId        | movieId       | rating     | timestamp         |
|----------|---------------|---------------|------------|-------------------|
| count    | 100,836.000   | 100,836.000   | 100,836.000| 100,836.000       |
| mean     | 326.128       | 19,435.296    | 3.502      | 1,205,946,000     |
| std      | 182.618       | 35,530.987    | 1.043      | 216,261,000       |
| min      | 1.000         | 1.000         | 0.500      | 828,124,600       |
| 25%      | 177.000       | 1,199.000     | 3.000      | 1,019,124,000     |
| 50%      | 325.000       | 2,991.000     | 3.500      | 1,186,087,000     |
| 75%      | 477.000       | 8,122.000     | 4.000      | 1,435,994,000     |
| max      | 610.000       | 193,609.000   | 5.000      | 1,537,799,000     |

Sepuluh baris pertama dari `ratings_df`:

```python
ratings_df.head(10)
```

| Index | userId | movieId | rating | timestamp     |
|-------|--------|---------|--------|---------------|
| 0     | 1      | 1       | 4.0    | 964,982,703   |
| 1     | 1      | 3       | 4.0    | 964,981,247   |
| 2     | 1      | 6       | 4.0    | 964,982,224   |
| 3     | 1      | 47      | 5.0    | 964,983,815   |
| 4     | 1      | 50      | 5.0    | 964,982,931   |
| 5     | 1      | 70      | 3.0    | 964,982,400   |
| 6     | 1      | 101     | 5.0    | 964,980,868   |
| 7     | 1      | 110     | 4.0    | 964,982,176   |
| 8     | 1      | 151     | 5.0    | 964,984,041   |
| 9     | 1      | 157     | 5.0    | 964,984,100   |


Jumlah pengguna unik dan film yang telah dinilai:

```python
print('Num of unique users:', len(ratings_df['userId'].unique()))
print('Num of unique movie:', len(ratings_df['movieId'].unique()))
```
```python
Num of unique userId: 610
Num of unique movieId: 9724
```

Pengecekan nilai kosong dan duplikasi:

```python
print(ratings_df.isna().any().any())
print(ratings_df.duplicated().any())
```

output:
```python
False
Empty DataFrame
Columns: [userId, movieId, rating, timestamp]
Index: []
```

## Data Visualization
#### *Univariate Analysis*
> *Univariate analysis* merupakan metode analisis yang hanya melibatkan satu variabel (*feature*) pada satu waktu. Analisis ini bertujuan untuk mengeksplorasi distribusi, sifat, serta pola yang terdapat pada variabel tersebut.

Mendefinisikan fungsi untuk menghitung total dan persentase data dari setiap kategori.
```python
def CountAndPlot(df, feature):
  count = df[feature].value_counts()
  percent = 100*df[feature].value_counts(normalize=True)
  samples = pd.DataFrame({'Sample Count':count, 'Percentage':percent.round(1)})
  print(samples)
  count.plot(kind='bar', title=feature)
```

Fungsi ini akan menampilkan:
- Jumlah sampel (`Sample Count`) dari setiap kategori dalam suatu fitur.
- Persentase relatifnya terhadap total data.
- Grafik batang (*bar chart*) untuk memvisualisasikan distribusi tersebut.

### Distribusi Genre Film

Distribusi genre dilakukan untuk memahami seberapa banyak film tersedia di setiap kategori genre. Hal ini penting karena dapat memengaruhi hasil rekomendasi jika tidak diseimbangkan selama tahap *preprocessing*.

```python
CountAndPlot(movies_df, 'genres')
```

Kode di atas menghasilkan tabel distribusi dan grafik sebagai berikut:

| Genre              | Sample Count | Percentage (%) |
|--------------------|--------------|----------------|
| Comedy             | 2,779        | 28.5           |
| Drama              | 2,226        | 22.8           |
| Action             | 1,828        | 18.8           |
| Adventure          | 653          | 6.7            |
| Crime              | 537          | 5.5            |
| Horror             | 468          | 4.8            |
| Documentary        | 386          | 4.0            |
| Animation          | 298          | 3.1            |
| Children           | 197          | 2.0            |
| Thriller           | 84           | 0.9            |
| Sci-Fi             | 62           | 0.6            |
| Mystery            | 48           | 0.5            |
| Fantasy            | 42           | 0.4            |
| Romance            | 38           | 0.4            |
| (no genres listed) | 34           | 0.3            |
| Musical            | 23           | 0.2            |
| Western            | 23           | 0.2            |
| Film-Noir          | 12           | 0.1            |
| War                | 4            | 0.0            |

![image](https://github.com/user-attachments/assets/007a7c4b-2fa8-4fff-8d06-b7485ff2c25c)


🔍 **Insight:**
- Genre **Drama** merupakan genre yang paling banyak muncul dalam dataset, disusul oleh **Comedy** dan **Action**.
- Beberapa genre seperti **War**, dan **Film-Noir** memiliki jumlah sangat sedikit dan bisa dikategorikan sebagai *minority class* atau *rare categories*.
- Perlu pertimbangan tambahan saat membangun model rekomendasi agar tidak bias terhadap genre dominan.

---

### Distribusi Rating Pengguna

Rating yang diberikan pengguna menjadi dasar utama dalam sistem rekomendasi berbasis kolaboratif. Dengan memahami distribusi rating, kita bisa mengetahui kecenderungan penilaian pengguna secara keseluruhan.

```python
CountAndPlot(ratings_df, 'rating')
```

Hasilnya adalah tabel dan grafik berikut:

| Rating | Sample Count | Percentage (%) |
|--------|--------------|----------------|
| 4.0    | 26,818       | 26.6           |
| 3.0    | 20,047       | 19.9           |
| 5.0    | 13,211       | 13.1           |
| 3.5    | 13,136       | 13.0           |
| 4.5    | 8,551        | 8.5            |
| 2.0    | 7,551        | 7.5            |
| 2.5    | 5,550        | 5.5            |
| 1.0    | 2,811        | 2.8            |
| 1.5    | 1,791        | 1.8            |
| 0.5    | 1,370        | 1.4            |

![image](https://github.com/user-attachments/assets/50316e39-87ef-4554-ac35-357e5928d708)


🔍 **Insight:**
- Sebagian besar *user* memberikan rating sebesar `4.0` untuk film.
- Mayoritas pengguna cenderung memberikan rating positif (di atas 3), tetapi masih ada cukup banyak rating negatif (di bawah 2.5).
- Distribusi ini membantu dalam mengevaluasi apakah sistem perlu lebih sensitif terhadap rating rendah atau fokus pada preferensi mayoritas.

## 🧹 Data Preparation
Data preparation merupakan tahap krusial dalam proses pengembangan sistem rekomendasi. Tujuan utamanya adalah memastikan bahwa data yang akan digunakan untuk pelatihan model bersih, relevan, dan dalam format yang sesuai sehingga dapat diproses secara efektif oleh algoritma machine learning.

### *Data Cleaning*

*Data cleaning* merupakan langkah penting dalam proses pengembangan sistem rekomendasi. Tujuan utama dari tahap ini adalah mengidentifikasi dan membersihkan data yang tidak relevan, tidak lengkap, atau tidak konsisten agar dataset siap digunakan dalam analisis maupun pelatihan model.

#### Removal of Duplicates and NaN Values

Langkah pertama dalam proses *data cleaning* adalah menghapus baris duplikat dan nilai kosong (*NaN*) yang dapat mengganggu akurasi dan performa model. Hal ini dilakukan untuk memastikan bahwa setiap entri dalam dataset bersifat unik dan bebas dari inkonsistensi.

```python
clean_movies_df = movies_df.drop_duplicates().dropna()
clean_ratings_df = ratings_df.drop_duplicates().dropna()
```

Setelah proses tersebut, dilakukan pengecekan ulang terhadap data:

- Tidak ditemukan data duplikat pada `movies_df` maupun `ratings_df`.
- Tidak ada nilai kosong (*missing value*) pada kedua dataframe.
  
Dengan demikian, data dianggap bersih dan siap untuk diproses lebih lanjut.

#### Removal of Irrelevant Columns and Values

Selain membersihkan data duplikat dan NaN, beberapa kolom atau nilai juga dianggap tidak relevan dan perlu dihapus untuk fokus pada informasi yang benar-benar dibutuhkan dalam pengembangan model.

1. **Genre `(no genres listed)`**  
   Genre ini tidak memberikan informasi spesifik tentang film dan tidak berguna dalam rekomendasi. Oleh karena itu, semua film dengan genre tersebut dihapus dari `clean_movies_df`.

2. **Kolom `timestamp`**  
   Kolom ini tidak relevan dalam proses prediksi rekomendasi film, sehingga dihapus dari `clean_ratings_df`.

```python
clean_movies_df = clean_movies_df[clean_movies_df['genres'] != '(no genres listed)']
clean_ratings_df = clean_ratings_df.drop(columns=['timestamp'])
```

3. **Memastikan kesesuaian antara `movieId` di kedua dataset**  
   Dilakukan filtering untuk memastikan bahwa hanya `movieId` yang tersedia di `movies_df` saja yang digunakan dalam `ratings_df`, sehingga tidak terjadi ketidakkonsistenan saat integrasi nanti.

#### Handling Imbalanced Data

Setelah data melalui tahap *cleaning*, langkah selanjutnya adalah menangani ketidakseimbangan jumlah film di setiap genre. Beberapa genre seperti Drama dan Comedy memiliki jumlah jauh lebih banyak dibandingkan genre minoritas seperti War atau Film-Noir, yang dapat menyebabkan model cenderung bias terhadap genre dominan.

Sebagai langkah awal dalam proses ini, genre `'War'` dihapus dari dataset karena jumlahnya sangat sedikit dan tidak signifikan dibandingkan genre lain.

```python
clean_movies_df = clean_movies_df[clean_movies_df['genres'] != 'War']
print('Jumlah Genre War:', len(clean_movies_df[clean_movies_df['genres'] == 'War']))
print()
print(clean_movies_df['genres'].value_counts())
```

**Output:**
```
Jumlah Genre War: 0

genres
Comedy         2779
Drama          2226
Action         1828
Adventure       653
Crime           537
Horror          468
Documentary     386
Animation       298
Children        197
Thriller         84
Sci-Fi           62
Mystery          48
Fantasy          42
Romance          38
Western          23
Musical          23
Film-Noir        12
Name: count, dtype: int64
```

**Penjelasan:**  
Genre `'War'` dihapus untuk menghindari overfitting atau noise pada model akibat jumlah sampel yang sangat sedikit. Setelah itu, dilakukan *undersampling* pada jumlah rating per film agar distribusi antar film lebih seimbang.

**Alasan:**  
Genre dengan jumlah sangat kecil dapat memperberat proses pelatihan model tanpa memberikan kontribusi signifikan terhadap kualitas rekomendasi secara keseluruhan. Dengan menghilangkan genre-genre minoritas dan membatasi jumlah rating per film, kita bisa mengurangi bias dan meningkatkan efisiensi serta akurasi model.

**Proses undersampling:**

- Setiap film dibatasi maksimal hanya memiliki 3 rating.
- Jika suatu film memiliki kurang dari 3 rating, maka dilakukan *resample* (pengambilan acak dengan penggantian) hingga mencapai jumlah target.

```python
target_ratings = 3
clean_ratings_df = pd.concat(
    [
        group.sample(n=target_ratings, random_state=42) if len(group) > target_ratings else
        resample(group, replace=True, n_samples=target_ratings, random_state=42)
        for _, group in grouped_ratings
    ]
)
```
**Output:**

|         | userId  |movieId |rating | 
|---------|---------|--------|-------|
| 90256   | 587     | 1      | 5.0   |
| 98666   | 608     | 1      | 2.5   |
| 58965   | 385     | 1      | 4.0   |
| 75200   | 475     | 2      | 4.5   |
| 12731   | 82      | 2      | 3.0   |
| ...     | ...     | ...    | ...   |
| 27259   | 184     | 193587 | 3.5   |
| 27259   | 184     | 193587 | 3.5   |
| 51362   | 331     | 193609 | 4.0   |
| 51362   | 331     | 193609 | 4.0   |
| 51362   | 331     | 193609 | 4.0   |


Langkah ini membantu menjaga keseimbangan data sehingga model tidak hanya cenderung merekomendasikan film dari genre mayoritas, tetapi juga mampu memberikan saran yang bervariasi dan relevan.


#### Text Processing

Beberapa genre memiliki penulisan yang tidak seragam, seperti `'Film-Noir'` dan `'Sci-Fi'`. Untuk menjaga konsistensi, nama genre tersebut diseragamkan menjadi `'Filmnoir'` dan `'Scifi'`.

```python
clean_movies_df['genres'] = clean_movies_df['genres'].replace({'Film-Noir': 'Filmnoir', 'Sci-Fi': 'Scifi'})
print('Filmnoir Genres Count : ', len(clean_movies_df[clean_movies_df['genres'] == 'Filmnoir']))
print('Scifi Genres Count : ', len(clean_movies_df[clean_movies_df['genres'] == 'Scifi']))

print()

print('Example movie with genre Filmnoir')
print(clean_movies_df[clean_movies_df['genres'] == 'Filmnoir'].head())

print()

print('Example movie with genre Scifi')
print(clean_movies_df[clean_movies_df['genres'] == 'Scifi'].head())
```

Contoh hasil:

```python
Filmnoir Genres Count :  12
Scifi Genres Count :  62

Example movie with genre Filmnoir
      movieId                       title    genres
279       320               Suture (1993)  Filmnoir
695       913  Maltese Falcon, The (1941)  Filmnoir
711       930            Notorious (1946)  Filmnoir
913      1212       Third Man, The (1949)  Filmnoir
1531     2066      Out of the Past (1947)  Filmnoir

Example movie with genre Scifi
      movieId                                  title genres
668       880       Island of Dr. Moreau, The (1996)  Scifi
1320     1779                          Sphere (1998)  Scifi
1719     2311  2010: The Year We Make Contact (1984)  Scifi
1902     2526                          Meteor (1979)  Scifi
2000     2661        It Came from Outer Space (1953)  Scifi
```

Langkah ini memastikan bahwa model tidak salah menilai genre yang sebenarnya sama hanya karena perbedaan penulisan.

#### Data Transformation

Sebagai persiapan pembuatan model *content-based filtering*, dilakukan transformasi teks genre menggunakan teknik TF-IDF (*Term Frequency - Inverse Document Frequency*), yang mengubah genre film menjadi representasi numerik yang bisa diproses oleh model machine learning.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(clean_movies_df['genres'])
tfidf_matrix.shape
tfidf_matrix.todense()
```

output:
```python
(9742, 23)

matrix([[0., 1., 0., ..., 0., 0., 0.],
        [0., 1., 0., ..., 0., 0., 0.],
        [0., 0., 0., ..., 0., 0., 0.],
        ...,
        [0., 0., 0., ..., 0., 0., 0.],
        [1., 0., 0., ..., 0., 0., 0.],
        [0., 0., 0., ..., 0., 0., 0.]])
```


Hasil transformasi berupa matriks dengan ukuran `(jumlah_film, jumlah_genre_unik)`, yang kemudian digunakan untuk menghitung kesamaan antar film.

#### Calculating Cosine Similarity

Untuk merekomendasikan film berdasarkan kesamaan genre, digunakan metode *cosine similarity*. Metode ini mengukur sudut antar vektor dan menghasilkan nilai antara 0 sampai 1, di mana semakin mendekati 1 berarti semakin mirip.

```python
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(tfidf_matrix)
cosine
```
output:
```python
array([[1., 1., 0., ..., 0., 0., 0.],
       [1., 1., 0., ..., 0., 0., 0.],
       [0., 0., 1., ..., 0., 0., 1.],
       ...,
       [0., 0., 0., ..., 1., 0., 0.],
       [0., 0., 0., ..., 0., 1., 0.],
       [0., 0., 1., ..., 0., 0., 1.]])
```
Matriks hasil ini akan digunakan sebagai dasar dalam sistem rekomendasi berbasis konten (*content-based recommendation*).

#### Data Encoding

Agar model dapat memproses ID pengguna (`userId`) dan ID film (`movieId`) secara efektif, nilai-nilai tersebut diubah menjadi indeks numerik berurutan.

```python
Id_user = clean_ratings_df['userId'].unique().tolist()
print('list userId:', Id_user)

user_encoded = {x: i for i, x in enumerate(Id_user)}
print('encoded userId:', user_encoded)

decode_user = {i: x for i, x in enumerate(Id_user)}
print('decoded userId:', decode_user)
```

```python
movies_id = clean_ratings_df['movieId'].unique().tolist()
print('list movieId:', movies_id)

movie_encoded = {x: i for i, x in enumerate(movies_id)}
print('encoded movieId:', movie_encoded)

decode_movie = {i: x for i, x in enumerate(movies_id)}
print('decode movieId:', decode_movie)
```

```python
clean_ratings_df['user'] = clean_ratings_df['userId'].map(user_encoded)
clean_ratings_df['movie'] = clean_ratings_df['movieId'].map(movie_encoded)
clean_ratings_df
```
output:

|        | userid  | movieid|rating| user  | movie |
|--------|---------|--------|------|-------|-------|
| 90256  | 587     | 1      | 5.0  | 0     | 0     |
| 98666  | 608     | 1      | 2.5  | 1     | 0     |
| 58965  | 385     | 1      | 4.0  | 2     | 0     |
| 75200  | 475     | 2      | 4.5  | 3     | 1     |
| 12731  | 82      | 2      | 3.0  | 4     | 1     |
| ...    | ...     | ...    | ...  | ...   | ...   |
| 27259  | 184     | 193587 | 3.5  | 524   | 9722  |
| 27259  | 184     | 193587 | 3.5  | 524   | 9722  |
| 51362  | 331     | 193609 | 4.0  | 97    | 9723  |
| 51362  | 331     | 193609 | 4.0  | 97    | 9723  |
| 51362  | 331     | 193609 | 4.0  | 97    | 9723  |

29172 rows × 5 columns


Proses encoding ini sangat penting dalam pengembangan model *collaborative filtering*, terutama untuk pendekatan berbasis *matrix factorization*.

#### Train-Test Split

Akhir dari tahap *data preparation* adalah pembagian dataset menjadi dua bagian: data latih dan data uji. Pembagian dilakukan dengan rasio 80:20.

```python
clean_ratings_df = clean_ratings_df.sample(frac=1, random_state=42)
clean_ratings_df

x = clean_ratings_df[['user', 'movie']].values
y = clean_ratings_df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

train_indices = int(0.8 * clean_ratings_df.shape[0])
x_train, x_val = (
    x[:train_indices],
    x[train_indices:]
)
y_train, y_val = (
    y[:train_indices],
    y[train_indices:]
)

print(x, y)
```

output:
```python
[[  21 1331]
 [ 143 7653]
 [  84 2120]
 ...
 [  35  286]
 [ 110 5265]
 [ 114 7884]] [0.33333333 0.55555556 0.11111111 ... 0.33333333 0.66666667 0.55555556]
```

Dengan membagi data menjadi set latih dan set uji, kita dapat mengukur akurasi model dalam menghadapi contoh yang belum pernah dilihat sebelumnya, sehingga performa sebenarnya dari model dapat dievaluasi.

## 🧠 Modeling

### *Content-Based Filtering Model*

Teknik *content-based filtering* digunakan untuk merekomendasikan film berdasarkan kesamaan karakteristik antar film. Dalam pendekatan ini, sistem mempertimbangkan fitur seperti genre dari film yang sudah ditonton atau dinilai oleh pengguna, lalu mencari film lain dengan fitur serupa untuk direkomendasikan.

#### Kelebihan:
- **Personalisasi tinggi**: Rekomendasi dibuat berdasarkan preferensi spesifik pengguna.
- **Independen terhadap data pengguna lain**: Tidak memerlukan interaksi dari pengguna lain, cukup dari riwayat pengguna itu sendiri.
- **Mudah diperbarui**: Jika ada informasi baru mengenai suatu film, model bisa langsung disesuaikan tanpa perlu pelatihan ulang menyeluruh.

#### Kekurangan:
- **Keterbatasan variasi rekomendasi**: Hanya mampu menyarankan film yang mirip dengan yang sebelumnya disukai, sehingga kurang eksploratif.
- **Cold start problem pada item**: Film baru dengan metadata yang tidak lengkap akan sulit untuk direkomendasikan.
- **Terbatas pada kualitas deskripsi film**: Jika informasi metadata tidak akurat, maka hasil rekomendasi juga bisa menjadi kurang relevan.

#### Penerapan Model

Untuk membuat rekomendasi berbasis konten, kita menggunakan matriks *cosine similarity* yang dihitung dari representasi TF-IDF dari genre film. Fungsi berikut digunakan untuk mendapatkan daftar film yang paling mirip dengan judul tertentu:

```python
def movie_recommendations(movie_title, similarity_data = cosine_df, items=movies_df[['title', 'genres']], k=5):
    index = similarity_data.loc[:,movie_title].to_numpy().argpartition(range(-1, -k, -1))

    closest = similarity_data.columns[index[-1:-(k+2):-1]]

    closest = closest.drop(movie_title, errors='ignore')

    return pd.DataFrame(closest).merge(items).head(k)
```

Sebagai contoh, ketika film `'Interstellar (2014)'` dimasukkan sebagai input:

```python
movie_recommendations('Interstellar (2014)')
```

Hasil yang didapat adalah lima film dengan genre dan karakteristik serupa:

| Index | Title                                       | Genre  |
|-------|---------------------------------------------|--------|
| 0     | Timecrimes (Cronocrímenes, Los) (2007)      | Scifi  |
| 1     | Meteor (1979)                               | Scifi  |
| 2     | Sphere (1998)                               | Scifi  |
| 3     | Beginning of the End (1957)                 | Scifi  |
| 4     | Contagion (2011)       		                  | Scifi  |

Hasil ini menunjukkan bahwa sistem berhasil menemukan film-film dengan genre dominan Scifi, sesuai dengan film yang menjadi input awal.

---

### *Collaborative Filtering Model*

Teknik *collaborative filtering* bekerja dengan prinsip berbeda, yaitu memanfaatkan pola interaksi pengguna lain untuk memberikan rekomendasi. Teknik ini sangat efektif ketika pengguna memiliki preferensi yang mirip dengan kelompok pengguna lain.

#### Kelebihan:
- **Lebih eksploratif**: Mampu memberikan rekomendasi film yang belum pernah diketahui pengguna sebelumnya.
- **Tidak bergantung pada metadata film**: Cukup menggunakan pola rating dari pengguna lain.
- **Rekomendasi berbasis perilaku nyata**: Menggunakan data riil dari pengguna, bukan hanya deskripsi teks.

#### Kekurangan:
- **Cold start untuk pengguna baru**: Tidak bisa merekomendasikan jika belum ada riwayat penilaian dari pengguna tersebut.
- **Membutuhkan banyak data interaksi**: Semakin sedikit data, semakin rendah akurasi rekomendasi.
- **Komputasi intensif**: Membandingkan ribuan pengguna atau film dapat memakan waktu dan sumber daya yang besar.

#### Arsitektur Model

Model *collaborative filtering* dibangun menggunakan arsitektur neural network sederhana dengan layer embedding untuk pengguna dan film:

```python
class RecommenderSystem(tf.keras.Model):
  def __init__(self, user_number, movie_number, embedding_size, **kwargs):
    super(RecommenderSystem, self).__init__(**kwargs)
    self.user_number = user_number
    self.movie_number = movie_number
    self.embedding_size = embedding_size
    self.user_embedding = tf.keras.layers.Embedding(
        user_number,
        embedding_size,
        
        embeddings_initializer= 'he_normal',
        embeddings_regularizer = tf.keras.regularizers.l2(1e-6)
    )
    self.user_bias = tf.keras.layers.Embedding(user_number, 1)
    self.resto_embedding = tf.keras.layers.Embedding(
        movie_number,
        embedding_size,
        
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = tf.keras.regularizers.l2(1e-6)
    )
    self.resto_bias = tf.keras.layers.Embedding(movie_number, 1)

  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:, 0])
    user_bias = self.user_bias(inputs[:, 0])
    resto_vector = self.resto_embedding(inputs[:, 1])
    resto_bias = self.resto_bias(inputs[:, 1])

    dot_user_resto = tf.tensordot(user_vector, resto_vector, 2)

    x = dot_user_resto + user_bias + resto_bias

    return tf.nn.sigmoid(x)
```

#### Pelatihan Model

Model dilatih selama 30 *epoch* menggunakan optimizer Adam dan fungsi loss Binary Crossentropy, dengan metrik evaluasi RMSE (*Root Mean Squared Error*):

```python
model = RecommenderSystem(user_number, movie_number, 50)

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics = [tf.keras.metrics.RootMeanSquaredError()]
)

history = model.fit(
    x = x_train,
    y = y_train,
    batch_size = 8,
    epochs = 30,
    validation_data = (x_val, y_val)
)
```

**Output Training:**
```
Epoch 1/30
2918/2918 ━━━━━━━━━━━━━━━━━━━━ 9s 2ms/step - loss: 0.6748 - root_mean_squared_error: 0.2510 - val_loss: 0.6445 - val_root_mean_squared_error: 0.2188
Epoch 2/30
2918/2918 ━━━━━━━━━━━━━━━━━━━━ 4s 1ms/step - loss: 0.6357 - root_mean_squared_error: 0.2094 - val_loss: 0.6366 - val_root_mean_squared_error: 0.2099
Epoch 3/30
2918/2918 ━━━━━━━━━━━━━━━━━━━━ 4s 1ms/step - loss: 0.6219 - root_mean_squared_error: 0.1933 - val_loss: 0.6308 - val_root_mean_squared_error: 0.2035
Epoch 4/30
2918/2918 ━━━━━━━━━━━━━━━━━━━━ 4s 1ms/step - loss: 0.6130 - root_mean_squared_error: 0.1846 - val_loss: 0.6279 - val_root_mean_squared_error: 0.2000
Epoch 5/30
2918/2918 ━━━━━━━━━━━━━━━━━━━━ 4s 1ms/step - loss: 0.6083 - root_mean_squared_error: 0.1753 - val_loss: 0.6237 - val_root_mean_squared_error: 0.1952
Epoch 6/30
2918/2918 ━━━━━━━━━━━━━━━━━━━━ 4s 1ms/step - loss: 0.6008 - root_mean_squared_error: 0.1687 - val_loss: 0.6213 - val_root_mean_squared_error: 0.1925
Epoch 7/30
2918/2918 ━━━━━━━━━━━━━━━━━━━━ 4s 1ms/step - loss: 0.5995 - root_mean_squared_error: 0.1608 - val_loss: 0.6187 - val_root_mean_squared_error: 0.1894
Epoch 8/30
2918/2918 ━━━━━━━━━━━━━━━━━━━━ 4s 1ms/step - loss: 0.5940 - root_mean_squared_error: 0.1580 - val_loss: 0.6166 - val_root_mean_squared_error: 0.1871
Epoch 9/30
2918/2918 ━━━━━━━━━━━━━━━━━━━━ 4s 1ms/step - loss: 0.5901 - root_mean_squared_error: 0.1526 - val_loss: 0.6151 - val_root_mean_squared_error: 0.1853
Epoch 10/30
2918/2918 ━━━━━━━━━━━━━━━━━━━━ 4s 1ms/step - loss: 0.5864 - root_mean_squared_error: 0.1458 - val_loss: 0.6134 - val_root_mean_squared_error: 0.1835
Epoch 11/30
2918/2918 ━━━━━━━━━━━━━━━━━━━━ 4s 1ms/step - loss: 0.5837 - root_mean_squared_error: 0.1432 - val_loss: 0.6125 - val_root_mean_squared_error: 0.1824
Epoch 12/30
2918/2918 ━━━━━━━━━━━━━━━━━━━━ 4s 1ms/step - loss: 0.5819 - root_mean_squared_error: 0.1407 - val_loss: 0.6104 - val_root_mean_squared_error: 0.1801
Epoch 13/30
2918/2918 ━━━━━━━━━━━━━━━━━━━━ 4s 1ms/step - loss: 0.5787 - root_mean_squared_error: 0.1366 - val_loss: 0.6102 - val_root_mean_squared_error: 0.1799
...
Epoch 29/30
2918/2918 ━━━━━━━━━━━━━━━━━━━━ 4s 1ms/step - loss: 0.5614 - root_mean_squared_error: 0.1101 - val_loss: 0.6048 - val_root_mean_squared_error: 0.1746
Epoch 30/30
2918/2918 ━━━━━━━━━━━━━━━━━━━━ 4s 1ms/step - loss: 0.5613 - root_mean_squared_error: 0.1092 - val_loss: 0.6048 - val_root_mean_squared_error: 0.1746
```

#### Prediksi Rekomendasi

Setelah model dilatih, sistem dapat memberikan rekomendasi film berdasarkan riwayat rating pengguna. Berikut contoh output rekomendasi untuk salah satu pengguna:

```python
if user_movie_array.size > 0:
    ratings = model.predict(user_movie_array).flatten()

    top_ratings_indices = ratings.argsort()[-10:][::-1]
    recommended_movie_ids = [
        decode_movie.get(movie_not_watched[x][0]) for x in top_ratings_indices if decode_movie.get(movie_not_watched[x][0]) is not None
    ] # Ensure decoding exists

    print('Showing recommendations for users: {}'.format(user_id))
    print('===' * 9)
    print('Movie with high ratings from user')
    print('----' * 8)

    top_movie_user = (
        movie_watched_by_user.sort_values(
            by = 'rating',
            ascending=False
        )
        .head(5)
        .movieId.values
    )

    movie_df_rows = movies_df[movies_df['movieId'].isin(top_movie_user)]
    if not movie_df_rows.empty:
        for row in movie_df_rows.itertuples():
            print(row.title)
    else:
        print("No highly rated movies found for this user in the original movies_df.")


    print('----' * 8)
    print('Top 10 movie recommendation')
    print('----' * 8)

    recommended_movie = movies_df[movies_df['movieId'].isin(recommended_movie_ids)]
    if not recommended_movie.empty:
        for row in recommended_movie.itertuples():
            print(row.title)
    else:
        print("No recommendations found based on predicted ratings.")
else:
    print(f"Cannot generate recommendations for user {user_id} as no unwatched movies were found or user/movie encoding failed.")
```

**Berikut hasil nya:**

```
290/290 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step
Showing recommendations for users: 610
===========================
Movie with high ratings from user
--------------------------------
Opera (1987)
Dead or Alive: Hanzaisha (1999)
Project A 2 ('A' gai wak juk jap) (1987)
Enter the Void (2009)
Maniac Cop 2 (1990)
--------------------------------
Top 10 movie recommendation
--------------------------------
Vagabond (Sans toit ni loi) (1985)
The Big Bus (1976)
Battle Royale 2: Requiem (Batoru rowaiaru II: Chinkonka) (2003)
Watching the Detectives (2007)
PK (2014)
The Fox and the Hound 2 (2006)
Sherlock Holmes and Dr. Watson: Acquaintance (1979)
Tenchi Muyô! In Love (1996)
Loving Vincent (2017)
Blue Planet II (2017)
```

Dengan pendekatan ini, sistem mampu memberikan rekomendasi yang beragam, bahkan di luar genre utama yang biasa ditonton oleh pengguna, berdasarkan pola interaksi dari pengguna lain yang memiliki preferensi serupa.

## 📊 Evaluation

Setelah tahap *modelling* selesai, langkah selanjutnya adalah melakukan evaluasi terhadap kinerja masing-masing model. Pada proyek ini, dua pendekatan utama digunakan: **Content-Based Filtering** dan **Collaborative Filtering**. Untuk mengevaluasi performa kedua model tersebut, digunakan metrik berbeda yang sesuai dengan karakteristik teknik masing-masing:

- **Precision** untuk model *content-based filtering*
- **Root Mean Squared Error (RMSE)** untuk model *collaborative filtering*

### Evaluasi Model *Content-Based Filtering* (Metrik: Precision)

#### Apa itu Precision?

*Precision* mengukur kemampuan model dalam memberikan rekomendasi film yang benar-benar relevan. Secara sederhana, *precision* menunjukkan proporsi film yang direkomendasikan dan benar-benar cocok dengan preferensi pengguna dari keseluruhan rekomendasi.

#### Rumus Precision:
$$
\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
$$

- **True Positives (TP)**: Jumlah film yang direkomendasikan dan memiliki genre yang sama atau mirip dengan film input.
- **False Positives (FP)**: Jumlah film yang direkomendasikan tetapi tidak relevan dengan genre film input.

#### Cara Kerja Evaluasi

Untuk mengevaluasi model *content-based*, dibuat fungsi yang memeriksa apakah film hasil rekomendasi memiliki genre yang serupa dengan film input. Hasil dievaluasi sebagai relevan atau tidak berdasarkan kesamaan genre.

```python
def evaluate_content_based_filtering(movie_title, similarity_data=cosine_df, items=movies_df[['title', 'genres']], k=5):
    # Ensure the movie title exists in the similarity data index
    if movie_title not in similarity_data.index:
        print(f"Movie '{movie_title}' not found in similarity data index.")
        return None

    # Ensure the movie title exists in the items dataframe
    if not items[items['title'] == movie_title].empty:
        input_movie_genres_series = items[items['title'] == movie_title]['genres']
        if not input_movie_genres_series.empty and input_movie_genres_series.values[0] is not None:
             input_movie_genres = input_movie_genres_series.values[0].split('|')
        else:
             print(f"Genres not found or are null for movie '{movie_title}'.")
             return None
    else:
        print(f"Movie '{movie_title}' not found in items dataframe.")
        return None

    recommended_movies_df = movie_recommendations(movie_title, similarity_data, items, k)
    if recommended_movies_df is None or recommended_movies_df.empty:
         print(f"No recommendations generated for '{movie_title}'.")
         return None

    recommended_movies = recommended_movies_df['title'].values

    relevant = []
    for movie in recommended_movies:
        movie_genres_series = items[items['title'] == movie]['genres']
        if not movie_genres_series.empty and movie_genres_series.values[0] is not None:
            movie_genres = movie_genres_series.values[0].split('|')
            # Check if any recommended movie genre is in the input movie's genres
            is_relevant = 1 if any(genre in input_movie_genres for genre in movie_genres) else 0
            relevant.append(is_relevant)
        else:
            relevant.append(0)


    precision = np.mean(relevant) if relevant else 0

    relevant_movies = [movie for movie, is_relevant in zip(recommended_movies, relevant) if is_relevant == 1]
    irrelevant_movies = [movie for movie, is_relevant in zip(recommended_movies, relevant) if is_relevant == 0]

    relevant_count = len(relevant_movies)
    irrelevant_count = len(irrelevant_movies)

    evaluation_summary = {
        'Movie': movie_title,
        'Total Recommendations': k,
        'Relevant Recommendations': relevant_count,
        'Irrelevant Recommendations': irrelevant_count,
        'Precision': precision,
        'Relevant Movies': relevant_movies,
        'Irrelevant Movies': irrelevant_movies
    }

    return evaluation_summary

# Ensure the movie exists before evaluating
if not movies_df[movies_df.title.eq('Interstellar (2014)')].empty:
    evaluation = evaluate_content_based_filtering('Interstellar (2014)')
    if evaluation:
        print(f"Evaluation Summary for 'Interstellar (2014)':")
        print(f"Total Recommendations: {evaluation['Total Recommendations']}")
        print(f"Relevant Recommendations: {evaluation['Relevant Recommendations']}")
        print(f"Irrelevant Recommendations: {evaluation['Irrelevant Recommendations']}")
        print(f"Precision: {evaluation['Precision']:.4f}")
        # Check if lists are empty before joining
        relevant_str = ', '.join(evaluation['Relevant Movies']) if evaluation['Relevant Movies'] else "None"
        irrelevant_str = ', '.join(evaluation['Irrelevant Movies']) if evaluation['Irrelevant Movies'] else "None"
        print(f"Relevant Movies: {relevant_str}")
        print(f"Irrelevant Movies: {irrelevant_str}")
else:
    print("Movie 'Interstellar (2014)' not found in movies_df, skipping evaluation.")
```

#### Output Evaluasi

```
Evaluation Summary for 'Interstellar (2014)':
Total Recommendations: 5
Relevant Recommendations: 5
Irrelevant Recommendations: 0
Precision: 1.0000
Relevant Movies: Timecrimes (Cronocrímenes, Los) (2007), Meteor (1979), Sphere (1998), Beginning of the End (1957), Contagion (2011)
Irrelevant Movies: None
```

#### Interpretasi Hasil

Model berhasil memberikan lima rekomendasi film yang semuanya relevan dengan film input (*genre*: Drama). Nilai *precision* yang dicapai adalah **1.0**, artinya semua rekomendasi tepat sasaran dan tidak ada film yang tidak relevan direkomendasikan.

---

### Evaluasi Model *Collaborative Filtering* (Metrik: RMSE)

#### Apa itu Root Mean Squared Error (RMSE)?

RMSE adalah metrik evaluasi yang digunakan untuk mengukur rata-rata kesalahan antara nilai prediksi dan nilai aktual dalam skala yang sama. Semakin rendah nilai RMSE, semakin baik akurasi prediksi dari model.

#### Rumus RMSE:
  $` RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} `$

  - $`(y_i)`$ adalah rating sebenarnya, dan $`(\hat{y}_i)`$ adalah rating prediksi.

RMSE memberikan informasi tentang sejauh mana prediksi sistem menyimpang dari nilai aktual. Semakin kecil nilai RMSE, semakin akurat prediksi sistem.

#### Cara Kerja Evaluasi

Model *collaborative filtering* dilatih selama 30 *epoch*. Setiap *epoch* mencatat metrik RMSE pada data pelatihan dan validasi. Berikut adalah visualisasi perkembangan RMSE selama proses pelatihan:

```python
plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('Collaborative Filtering Model Evalutation')
plt.ylabel('RMSE')
plt.xlabel('epoch')
plt.legend(['train', 'val'])
plt.show()
```

![image](https://github.com/user-attachments/assets/13be0765-02bb-4343-8be4-21a9b30f7f1f)



Dari grafik di atas dapat dilihat bahwa nilai RMSE terus menurun selama pelatihan, baik pada data pelatihan maupun validasi. Pada akhir pelatihan, hasil evaluasi menunjukkan:

| Metrik                          | Nilai       |
|---------------------------------|-------------|
| Training Loss                   | 0.5613      |
| Validation Loss                 | 0.6048      |
| Training RMSE                   | 0.1092      |
| Validation RMSE                 | 0.1746      |

Nilai RMSE yang relatif rendah menunjukkan bahwa model sudah cukup akurat dalam memprediksi rating pengguna terhadap film-film tertentu.

---

### Ringkasan Capaian terhadap Problem Statement

#### 🔹 *Problem Statement 1*: Bagaimana cara memahami dan memperoleh informasi mengenai data yang digunakan dalam pembuatan model sistem rekomendasi?
- Proses eksplorasi data *(Exploratory Data Analysis / EDA)* telah dilakukan secara menyeluruh.
- Dataset yang digunakan (`movies.csv` dan `ratings.csv`) telah dianalisis karakteristiknya, termasuk jumlah baris, tipe data, distribusi genre, dan distribusi rating.
- Informasi penting seperti jumlah pengguna unik (610) dan film unik (9724) telah berhasil diekstraksi.

#### 🔹 *Problem Statement 2*: Bagaimana cara membangun model sistem rekomendasi dengan menggunakan pendekatan *content-based filtering*?
- Tahapan *data cleaning*, *transformation*, dan *similarity calculation* telah dilakukan.
- Model *content-based filtering* berhasil dibangun dengan menggunakan representasi TF-IDF dan *cosine similarity*.
- Rekomendasi film berbasis konten telah diimplementasikan dan diuji.

#### 🔹 *Problem Statement 3*: Bagaimana cara mengembangkan model sistem rekomendasi dengan pendekatan *collaborative filtering*?
- Model *collaborative filtering* dikembangkan dengan arsitektur neural network sederhana menggunakan *embedding layer* untuk pengguna dan film.
- Data telah melalui proses *encoding*, normalisasi, dan *train-test split* sebelum pelatihan model.
- Model berhasil dilatih selama 30 *epoch* dengan penurunan nilai RMSE yang stabil.

#### 🔹 *Problem Statement 4*: Bagaimana cara menilai kinerja model sistem rekomendasi yang telah dikembangkan?
- Kedua model telah dievaluasi menggunakan metrik yang sesuai: *Precision* untuk model *content-based filtering* dan *RMSE* untuk model *collaborative filtering*.
- Hasil evaluasi menunjukkan bahwa kedua model memberikan rekomendasi yang sangat relevan dan akurat.
- Baik dari segi akurasi maupun kecepatan konvergensi, model telah memenuhi tujuan awal pengembangan sistem rekomendasi.

---

### 💡 Kesimpulan Evaluasi

Proyek ini bertujuan untuk mengembangkan sistem rekomendasi film berbasis *machine learning* yang cerdas dan personal. Dengan menggunakan dataset MovieLens, dua pendekatan utama telah berhasil diimplementasikan:

- **Content-Based Filtering**: Menghasilkan rekomendasi film berdasarkan kesamaan genre, dengan nilai *precision* sempurna yaitu 1.0.
- **Collaborative Filtering**: Memberikan prediksi rating pengguna terhadap film yang belum ditonton, dengan RMSE akhir sebesar 0.1752 pada data validasi.

Hasil evaluasi membuktikan bahwa kedua model mampu memberikan rekomendasi yang akurat dan relevan. Sistem ini dapat menjadi solusi strategis bagi platform *streaming online* untuk meningkatkan retensi pengguna, keterlibatan, dan tentunya pendapatan bisnis, karena hingga 40% pendapatan platform berasal dari efektivitas sistem rekomendasi.

## 📚 Referensi

Daftar pustaka yang digunakan dalam pengembangan sistem rekomendasi film ini mencakup jurnal ilmiah dan sumber teoretis terkait penerapan teknik *content-based filtering* dan *collaborative filtering*. Berikut adalah daftar referensi lengkap:

1. **Xu, G., Jia, G., Shi, L., & Zhang, Z. (2021).** *Personalized Course Recommendation System Fusing with Knowledge Graph and Collaborative Filtering.* *Computational Intelligence and Neuroscience, 2021*, 9590502.  https://doi.org/10.1155/2021/9590502

2. **Lin, K., Yang, S., & Na, S. (2022).** *Collaborative Filtering Algorithm-Based Destination Recommendation and Marketing Model for Tourism Scenic Spots.* *Computational Intelligence and Neuroscience, 2022*, 1–7.  https://doi.org/10.1155/2022/7115627

3. **Wang, L. (2022).** *Collaborative Filtering Recommendation of Music MOOC Resources Based on Spark Architecture.* *Computational Intelligence and Neuroscience, 2022*, 1–8. https://doi.org/10.1155/2022/2117081

4. **Liu, M., Wang, M., Li, B., & Zhong, Q. (2025).** *Collaborative Filtering Based on GNN with Attribute Fusion and Broad Attention.* *PeerJ Computer Science.* https://doi.org/10.7717/peerj-cs.2706

5. **Ayyaz, S., Qamar, U., & Nawaz, R. (2018).** *HCF-CRS: A Hybrid Content Based Fuzzy Conformal Recommender System for Providing Recommendations with Confidence.* *PLOS ONE, 13*(10), e0204849.  https://doi.org/10.1371/journal.pone.0204849

6. **Du, Y., Peng, L., Dou, S., Su, X., & Ren, X. (2022).** *Research on Personalized Book Recommendation Based on Improved Similarity Calculation and Data Filling Collaborative Filtering Algorithm.* *Computational Intelligence and Neuroscience, 2022*, 1–11. https://doi.org/10.1155/2022/1900209

