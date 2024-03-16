# Laporan Proyek Machine Learning *Recommendation System* - Rizki Hidayat

## Domain Proyek

Tujuan dasar dari Sistem Rekomendasi adalah untuk memahami hubungan antara pengguna dan barang yang mereka konsumsi. Biasanya, hal ini dilakukan dengan memperkirakan pasangan *user/item* (atau *item/item*) yang kompatibel, dan menampilkannya sebagai rekomendasi.[1]. *Feedback* pengguna dalam sistem rekomendasi biasanya diklasifikasikan ke dalam dua kategori: *Explicit-Feedback* di mana pengguna secara langsung mengekspresikan preferensi mereka (misalnya, rating), dan *Implicit Feedback* di mana pengguna secara tidak langsung mengungkapkan minat mereka melalui tindakan (misalnya, klik). Kedua paradigma ini telah lama dipelajari sebagai dua topik yang terpisah dan teknik yang berbeda telah dikembangkan untuk mengatasi masing-masing sifat yang berbeda.

Di luar definisi sempit *Explicit-Feedback* versus *Implicit*, terdapat berbagai jenis umpan balik pengguna berlimpah di banyak sistem informasi dunia nyata. Sebagai contoh, tampilan, klik, pembelian, dan skor peringkat pengguna biasanya tersedia di platform *e-commerce*. Semua sinyal ini mencerminkan (atau menyiratkan) minat pengguna terhadap item dari berbagai perspektif. Meskipun telah ada beberapa penelitian yang mempertimbangkan hubungan antara interaksi *Implicit* dan *Explicit*, sebagian besar berfokus pada peningkatan prediksi peringkat numerik dengan memanfaatkan sinyal-sinyal lain sebagai informasi tambahan[2] 

Sistem rekomendasi item tradisional sering kali bergantung pada serangkaian teknik penyaringan kolaboratif untuk belajar dari *Explicit-Feedback* seperti skor penilaian. Teknik berbasis model yang umum digunakan adalah metode *Matriks Factorization* (MF) [3], yang berusaha mempelajari item dan penyematan pengguna dan menggunakan *inner product* untuk memperkirakan peringkat yang diamati. Karena memberikan *Explicit-Feedback* sering kali membutuhkan upaya kognitif tambahan [4], interaksi ini mungkin jarang atau tidak tersedia dalam skenario dunia nyata. Dalam kasus seperti itu, metode MF di atas dapat diperluas untuk memodelkan sinyal *Implicit* yang lebih melimpah yang diungkapkan melalui tindakan pengguna yang dapat diamati seperti klik dan pembelian [5]. Untuk mengatasi sifat satu kelas dari pengaturan ini (yaitu, hanya contoh positif yang dapat diamati), beberapa pendekatan termasuk metode peringkat berpasangan BPR [6] dan metode optimasi pointwise WRMF [7] telah diusulkan.

Pada project ini akan dikembangkan sistem rekomendasi dari *Implicit Feedback* pembelian game dan lama jam permainan pada data *Steam Store*, dengan pendekatan seperti menangani *Explicit-Feedback* pada umumnya dengan menggunakan algoritma *BPR: Bayesian Personalized Ranking*. Selain itu dibuatkan pula *Content Based Filtering* dengan menggunakan *Cosine Similarity*

## Business Understanding

Pengembangan sistem rekomendasi dari *Implicit Feedback* pembelian game dan lama jam permainan pada data *Steam Store* akan memiliki dampak dan manfaat sebagai salah satu pedoman pengambilan keputusan bagi *Steam*, serta bermanfaat pula bagi *user* yang enggan memberikan rating sebagai *Explicit-Feedback* untuk tetap medapatkan informasi rekomendasi berdasarkan riwayat pembelian dan lama permainan dari game game yang disukainya

### Problem Statement

Berdasarkan kondisi tersebut maka perlu dikembangkan sistem prediksi untuk dapat menjawab permasalahan tersebut:
- Dari data *Implicit Feedback* pada *Steam Store* Apakah dapat dibuatkan sistem rekomendasi yang cukup baik?
- Dari serangkaian data tersebut, *Game* apa yang peling banyak dimainkan oleh *user*?
- *Game* apa yang memilki *user* paling banyak?
- Apakah terdapat korelasi antara lama bermain dengan jumlah *user*?
- Apakah dari data tersebut dapat pula dikembangkan sistem rekomendasi *Content Based FIltering*?

### Goals

Untuk menjawab pertanyaan tersebut akan dibuatkan Sistem Rekomendasi dengan tujuan sebagai berikut :

- Membuat Sistem Rekomendasi *Collaborative Filtering* dari data *Implicit Feedback* pada *Steam Store*.
- Melakukan Eksplorasi pada data untuk mengetahui *game* apa saja yang paling banyak dimainkan.
- Melakukan Eksplorasi pada data untuk mengetahui *game* apa yang memiliki *user* paling banyak.
- Melakukan Eksplorasi pada data untuk mengetahui adanya korelasi antara lama bermain dengan jumlah *user*.
- Membuat sistem rekomendasi *Content Based Filtering* dari fitur data yang dipilih.

### Solution statements
- Untuk eksplorasi fitur dilakukan *Univariate Analysis*. *Univariate Analysis* dilakukan untuk mengeksplorasi distribusi suatu fitur yang dipilih dalam suatu dataset. Teknik yang digunakan adalah menggunakan visualiasi data menggunakan barplot dari fitur yang dipilih dan manipulasi dataset.
- Untuk dapat memberikan rekomendasi berdasarkan data *Implicit Feedback* pada *Steam Store*, akan dibuatkan model *Collaborative Filtering* dengan menggunakan algoritma *BPR: Bayesian Personalized Ranking*.
- Selain itu dibuatkan pula sistem rekomendasi *Content Based Filtering* dengan menggunakan *Cosine Similarity*.

## Data Understanding

Untuk proyek ini, digunakan dua dataset yang berbeda. Keduannya tersedia gratis di Kaggle dan data diekstrak dari Steam. Dataset diimport menggunakan *Kaggle API*. Dataset tersebut adalah *User Dataset* dan *Game Dataset*.

### a. *User Dataset*
*Dataset* yang pertama adalah *user dataset*. *Dataset* terdiri dari kolom:
- ***User*** : *user id* dalam format numerik,
- ***Game*** : Nama *game*,
- ***Purchase_play*** : Perilaku dari user terhadap game tersebut apakah *purchase* ketika *user* membenli *game* tersebut atau *play* ketika *user* memainkannya dan
- ***hrs*** : Jumlah nilai sesuai dengan perilaku. Jika perilaku *play* nilai ini merepresentasikan jumlah jam *game* tersebut dimainkan. Jika perilaku *purchase* nilai ini bernilai 1.

Setiap baris dari dataset mewakili perilaku pengguna terhadap sebuah game, baik '*play*' atau '*purchase*'. Jika perilakunya adalah '*play*', nilai yang terkait dengannya sesuai dengan jumlah jam bermain. Jika perilakunya adalah '*purchase*', nilai yang terkait dengannya adalah 1, yang berarti pengguna membeli game tersebut. Dalam kasus dataset pengguna ini, nilai yang terkait dengan '*purchase*' selalu 1.


*Link Dataset User* : [*user*](https://www.kaggle.com/datasets/tamber/steam-video-games)

Adapun gambaran awal dari *user dataset* adalah sebagai berikut:

Tabel 1. Kondisi awal *User Dataset*

|   |    user   |            game            | purchase_play |  hrs  | tmp |
|:-:|:---------:|:--------------------------:|:-------------:|:-----:|:---:|
| 0 | 151603712 | The Elder Scrolls V Skyrim |    purchase   |  1.0  |  0  |
| 1 | 151603712 | The Elder Scrolls V Skyrim |      play     | 273.0 |  0  |
| 2 | 151603712 |          Fallout 4         |    purchase   |  1.0  |  0  |
| 3 | 151603712 |          Fallout 4         |      play     |  87.0 |  0  |
| 4 | 151603712 |            Spore           |    purchase   |  1.0  |  0  |
| 5 | 151603712 |            Spore           |      play     |  14.9 |  0  |
| 6 | 151603712 |      Fallout New Vegas     |    purchase   |  1.0  |  0  |
| 7 | 151603712 |      Fallout New Vegas     |      play     |  12.1 |  0  |
| 8 | 151603712 |        Left 4 Dead 2       |    purchase   |  1.0  |  0  |
| 9 | 151603712 |        Left 4 Dead 2       |      play     |  8.9  |  0  |

*User dataset* berisi total 200.000 baris, termasuk 5.155 game dan 12.393 pengguna . Demi kenyamanan, sruktur *dataset* diformat ulang dengan memisahkan informasi yang tersimpan dalam kolom 'perilaku' menjadi dua kolom: '*purchase*' dan '*play*'. Untuk setiap baris, kolom '*play*' memiliki nilai 1 jika pengguna benar-benar memainkan game atau 0 jika pengguna tidak memiliki catatan jam bermain.

Adapun informasi statistik untuk dataset ini adalah:

Tabel 2. Informasi statistik *user dataset*

|       |         user |           hrs |      tmp |
|:-----:|-------------:|--------------:|---------:|
| count | 2.000000e+05 | 200000.000000 | 200000.0 |
|  mean | 1.036559e+08 |     17.874384 |      0.0 |
|  std  | 7.208074e+07 |    138.056952 |      0.0 |
|  min  | 5.250000e+03 |      0.100000 |      0.0 |
|  25%  | 4.738420e+07 |      1.000000 |      0.0 |
|  50%  | 8.691201e+07 |      1.000000 |      0.0 |
|  75%  | 1.542309e+08 |      1.300000 |      0.0 |
|  max  | 3.099031e+08 |  11754.000000 |      0.0 |

Dari info statistik pada colomn *hrs*, nilai maksimalnya adalah 11.754 jam dengan nilai minimal 1 jam dan nilai rata-rata 17.87 jam. Karena colomn *temp* hanya memiliki nilai 0, maka colomn ini akan di-*drop*

Dari Tabel 1. telihat bahwa untuk 1 *user* dan 1 *game* yang sama terdapat 2 baris untuk keterangan *purchase* yang menyatakan *user* membeli game tersebut, dan *play* yang menyatakan *user* memainkan *game* tersebut. Hal tersebut membuat duplikat data, sehingga perlu dilakukan format ulang, agar keterangan *purcahse* dan *play* tidak berada di baris data, melainkan berada pada kolom data. Hasil format ulang data terdapat pada tabel dibawah ini.


Tabel 3. *User dataset* yang telah diformat ulang

|       |    user   |                       game |   hrs | purchase | play |
|------:|:---------:|---------------------------:|------:|:--------:|:----:|
| 95364 | 151603712 | The Elder Scrolls V Skyrim | 273.0 |        1 |    1 |
| 95339 | 151603712 |                  Fallout 4 |  87.0 |        1 |    1 |
| 95360 | 151603712 |                      Spore |  14.9 |        1 |    1 |
| 95340 | 151603712 |          Fallout New Vegas |  12.1 |        1 |    1 |
| 95353 | 151603712 |              Left 4 Dead 2 |   8.9 |        1 |    1 |

Setiap baris dalam *user dataset* yang diformat ulang mewakili sebuah interaksi pengguna yang unik. Total data setelah diformat ulang adalah 128.804 data.

Tabel 4. Informasi statistik *user dataset* yang telah diformat ulang

|       |     user     |           hrs |      purchase |          play |
|------:|:------------:|--------------:|--------------:|--------------:|
| count | 1.288040e+05 | 128804.000000 | 128804.000000 | 128804.000000 |
|  mean | 1.025343e+08 |     26.748904 |      1.005489 |      0.547258 |
|  std  | 7.242875e+07 |    171.390189 |      0.073884 |      0.497951 |
|  min  | 5.250000e+03 |      0.000000 |      1.000000 |      0.000000 |
|  25%  | 4.559264e+07 |      0.000000 |      1.000000 |      0.000000 |
|  50%  | 8.605570e+07 |      0.300000 |      1.000000 |      1.000000 |
|  75%  | 1.542307e+08 |      5.700000 |      1.000000 |      1.000000 |
|  max  | 3.099031e+08 |  11754.000000 |      2.000000 |      2.000000 |

Informasi ststistik dari dataset setelah diformat ulang, untuk kolom hrs rata rata jam bermain adalah 26.75 jam dengan maksmilam jam bermain masih sama yaitu 11.754 jam.


### b. Game Dataset
*Dataset* kedua adalah *game dataset*. Terdiri dari:
- ***url :*** *link url* (mengarah langsung ke *Steam store*)
- ***types :*** Tipe dari paket game (*app, bundle, other*)
- ***name:*** Judul *Game*
- ***desc_snippet:*** Deskripsi singkat dari *game*
- ***recent reviews:*** Ulasan terbaru
- ***all reviews:*** Seluruh ulasan
- ***release date:*** Tanggal *game* dirilis
- ***developer:*** Pengembang *game* / pembuat *game*
- ***publisher:*** Perusahaan penerbit *game*
- ***popular tags:*** *Tag* yang populer pada *game* (*Gore, Action, Shooter, PvP, Other*)
- ***game detail:*** Informasi detail dari *game* (*Multi-player, Single-player, Full controller support, other*)
- ***languages:*** Bahasa yang didukung oleh *game*
- ***achievements:*** Pencapaian yang terdapat dalam *game*
- ***genre:*** Jenis *genre* dari *game* (*Action, Adventure, RPG, Strategy, other*)
- ***game description:*** Deskripsi lengkap dari *game*
- ***description of mature content:*** Deskripsi dari konten dewasa dalam *game*
- ***minimum requirement to run the game:*** Spesifikasi minimal untuk menjalankan *game*
- ***recommended requirement:*** Spesifikasi rekomendasi untuk menjalankan *game* dalam kondisi optimalnya
- ***original price:*** Harga game asli
- ***price with discount:*** Harga game setelah *discount*

Secara total terdapat 51920 *game* dalam dataset.

*Link dataset Game* : [*game*](https://www.kaggle.com/datasets/trolukovich/steam-games-complete-dataset)

Adapun gambaran awal dari *game dataset* adalah sebagai berikut:

Tabel 5. Kondisi awal *Game dataset*

|   |                                               url |  types |                                       name |                                      desc_snippet |                   recent_reviews                  |                    all_reviews                    | release_date |           developer |                               publisher |                                      popular_tags |                                      game_details |                                         languages | achievements |                                             genre |                                 game_description |                                  mature_content |                minimum_requirements               |              recommended_requirements             | original_price | discount_price |
|--:|--------------------------------------------------:|-------:|-------------------------------------------:|--------------------------------------------------:|:-------------------------------------------------:|:-------------------------------------------------:|-------------:|--------------------:|----------------------------------------:|--------------------------------------------------:|--------------------------------------------------:|--------------------------------------------------:|-------------:|--------------------------------------------------:|-------------------------------------------------:|------------------------------------------------:|:-------------------------------------------------:|:-------------------------------------------------:|:--------------:|:--------------:|
| 0 |   https://store.steampowered.com/app/379720/DOOM/ |    app |                                       DOOM | Now includes all three premium DLC packs (Unto... | Very Positive,(554),- 89% of the 554 user revi... | Very Positive,(42,550),- 92% of the 42,550 use... | May 12, 2016 |         id Software |   Bethesda Softworks,Bethesda Softworks | FPS,Gore,Action,Demons,Shooter,First-Person,Gr... | Single-player,Multi-player,Co-op,Steam Achieve... | English,French,Italian,German,Spanish - Spain,... |         54.0 |                                            Action | About This Game Developed by id software, the... |                                             NaN | Minimum:,OS:,Windows 7/8.1/10 (64-bit versions... | Recommended:,OS:,Windows 7/8.1/10 (64-bit vers... |         $19.99 |         $14.99 |
| 1 | https://store.steampowered.com/app/578080/PLAY... |    app |              PLAYERUNKNOWN'S BATTLEGROUNDS | PLAYERUNKNOWN'S BATTLEGROUNDS is a battle roya... | Mixed,(6,214),- 49% of the 6,214 user reviews ... | Mixed,(836,608),- 49% of the 836,608 user revi... | Dec 21, 2017 |    PUBG Corporation |       PUBG Corporation,PUBG Corporation | Survival,Shooter,Multiplayer,Battle Royale,PvP... |            Multi-player,Online Multi-Player,Stats | English,Korean,Simplified Chinese,French,Germa... |         37.0 |            Action,Adventure,Massively Multiplayer |  About This Game PLAYERUNKNOWN'S BATTLEGROUND... | Mature Content Description The developers de... | Minimum:,Requires a 64-bit processor and opera... | Recommended:,Requires a 64-bit processor and o... |         $29.99 |            NaN |
| 2 | https://store.steampowered.com/app/637090/BATT... |    app |                                 BATTLETECH | Take command of your own mercenary outfit of '... | Mixed,(166),- 54% of the 166 user reviews in t... | Mostly Positive,(7,030),- 71% of the 7,030 use... | Apr 24, 2018 | Harebrained Schemes | Paradox Interactive,Paradox Interactive | Mechs,Strategy,Turn-Based,Turn-Based Tactics,S... | Single-player,Multi-player,Online Multi-Player... |                     English,French,German,Russian |        128.0 |                         Action,Adventure,Strategy |  About This Game From original BATTLETECH/Mec... |                                             NaN | Minimum:,Requires a 64-bit processor and opera... | Recommended:,Requires a 64-bit processor and o... |         $39.99 |            NaN |
| 3 |   https://store.steampowered.com/app/221100/DayZ/ |    app |                                       DayZ | The post-soviet country of Chernarus is struck... | Mixed,(932),- 57% of the 932 user reviews in t... | Mixed,(167,115),- 61% of the 167,115 user revi... | Dec 13, 2018 | Bohemia Interactive | Bohemia Interactive,Bohemia Interactive | Survival,Zombies,Open World,Multiplayer,PvP,Ma... | Multi-player,Online Multi-Player,Steam Worksho... | English,French,Italian,German,Spanish - Spain,... |          NaN |            Action,Adventure,Massively Multiplayer | About This Game The post-soviet country of Ch... |                                             NaN | Minimum:,OS:,Windows 7/8.1 64-bit,Processor:,I... | Recommended:,OS:,Windows 10 64-bit,Processor:,... |         $44.99 |            NaN |
| 4 | https://store.steampowered.com/app/8500/EVE_On... |    app |                                 EVE Online | EVE Online is a community-driven spaceship MMO... | Mixed,(287),- 54% of the 287 user reviews in t... | Mostly Positive,(11,481),- 74% of the 11,481 u... |  May 6, 2003 |                 CCP |                                 CCP,CCP | Space,Massively Multiplayer,Sci-fi,Sandbox,MMO... | Multi-player,Online Multi-Player,MMO,Co-op,Onl... |                     English,German,Russian,French |          NaN | Action,Free to Play,Massively Multiplayer,RPG,... |                                  About This Game |                                             NaN | Minimum:,OS:,Windows 7,Processor:,Intel Dual C... | Recommended:,OS:,Windows 10,Processor:,Intel i... |           Free |            NaN |
| 5 | https://store.steampowered.com/bundle/5699/Gra... | bundle | Grand Theft Auto V: Premium Online Edition | Grand Theft Auto V: Premium Online Edition bundle |                                               NaN |                                               NaN |          NaN |      Rockstar North |                          Rockstar Games |                                               NaN | Single-player,Multi-player,Downloadable Conten... | English, French, Italian, German, Spanish - Sp... |          NaN |                                  Action,Adventure |                                              NaN |                                             NaN |                                               NaN |                                               NaN |            NaN |         $35.18 |
| 6 | https://store.steampowered.com/app/601150/Devi... |    app |                            Devil May Cry 5 | The ultimate Devil Hunter is back in style, in... | Very Positive,(408),- 87% of the 408 user revi... | Very Positive,(9,645),- 92% of the 9,645 user ... |  Mar 7, 2019 |    CAPCOM Co., Ltd. |       CAPCOM Co., Ltd.,CAPCOM Co., Ltd. | Action,Hack and Slash,Great Soundtrack,Demons,... | Single-player,Online Multi-Player,Online Co-op... | English,French,Italian,German,Spanish - Spain,... |         51.0 |                                            Action | About This Game The Devil you know returns in... | Mature Content Description The developers de... | Minimum:,OS:,WINDOWS® 7, 8.1, 10 (64-BIT Requi... | Recommended:,OS:,WINDOWS® 7, 8.1, 10 (64-BIT R... |         $59.99 |         $70.42 |
| 7 | https://store.steampowered.com/app/477160/Huma... |    app |                           Human: Fall Flat | Human: Fall Flat is a quirky open-ended physic... | Very Positive,(629),- 91% of the 629 user revi... | Very Positive,(23,763),- 91% of the 23,763 use... | Jul 22, 2016 |     No Brakes Games |             Curve Digital,Curve Digital | Funny,Multiplayer,Co-op,Puzzle,Physics,Local C... | Single-player,Online Multi-Player,Local Co-op,... | English,French,German,Spanish - Spain,Russian,... |         55.0 |                                   Adventure,Indie |  About This Game ***NEW "DARK" LEVEL AVAILABL... |                                             NaN | Minimum:,OS:,Windows XP/Vista/7/8/8.1/10 x86 a... | Recommended:,OS:,Windows XP/Vista/7/8/8.1/10 x... |         $14.99 |         $17.58 |
| 8 | https://store.steampowered.com/app/644930/They... |    app |                          They Are Billions | They Are Billions is a Steampunk strategy game... | Very Positive,(192),- 83% of the 192 user revi... | Very Positive,(12,127),- 85% of the 12,127 use... | Dec 12, 2017 |     Numantian Games |         Numantian Games,Numantian Games | Early Access,Base Building,Strategy,Zombies,Su... | Single-player,Steam Achievements,Steam Trading... | English,Spanish - Spain,French,German,Japanese... |         34.0 |                             Strategy,Early Access |  About This Game They Are Billions is a strat... |                                             NaN | Minimum:,OS:,Windows 7, 8, 10 (32 and 64 bits)... |  Recommended:,OS:,Windows 7, 8, 10 (64 bits),P... |         $29.99 |            NaN |
| 9 | https://store.steampowered.com/app/774241/Warh... |    app |                       Warhammer: Chaosbane | In a world ravaged by war and dominated by mag... |                                               NaN | Mixed,(904),- 44% of the 904 user reviews for ... | May 31, 2019 |        Eko Software |   Bigben Interactive,Bigben Interactive | RPG,Adventure,Hack and Slash,Action,Action RPG... | Single-player,Multi-player,Co-op,Online Co-op,... | English,French,Italian,German,Spanish - Spain,... |         43.0 |                              Action,Adventure,RPG |  About This Game “Keep your eyes on this one,... | Mature Content Description The developers de... | Minimum:,Requires a 64-bit processor and opera... | Recommended:,Requires a 64-bit processor and o... |         $49.99 |            NaN |


## Referensi 
[1] A. Pathak, K. Gupta, and J. McAuley. Generating and Personalizing Bundle Recommendations on Steam. In SIGIR, 2017.

[2] M. Wan and J. McAuley. Item Recommendation on Monotonic Behavior Chains. In RecSys, 2018.

[3] T. Gurbanov and F. Ricci. Action prediction models for recommender systems based on collaborative filtering and sequence mining hybridization. In Proceedings of the Symposium on Applied Computing, 2017

[4] R. He, W.-C. Kang, and J. McAuley. Translation-based recommendation. In RecSys, 2017.

[5] Y. Hu, Y. Koren, and C. Volinsky. Collaborative filtering for implicit feedback datasets. In ICDM, 2008

[6] S. Rendle, C. Freudenthaler, Z. Gantner, and L. Schmidt-Thieme. Bpr: Bayesian personalized ranking from implicit feedback. In UAI, 2009.

[7] R. Pan, Y. Zhou, B. Cao, N. N. Liu, R. Lukose, M. Scholz, and Q. Yang. One-class collaborative filtering. In ICDM, 2008
