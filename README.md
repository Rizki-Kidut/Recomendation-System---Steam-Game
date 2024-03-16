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

Terdapat 40.817 game dalam dataset ini. Jika melihat jumlah non-null , dapat disimpulkan dataset ini memiliki banyak missing-values.

### Univariate Exploratory Data Analysis

### a. *User  Dataset*

Untuk mengetahui apakah *game* yang paling banyak dibeli sesuai dengan *game* yang paling banyak dimainkan. Untuk setiap *game*, jumlah total pengguna dan total waktu *game* dimainkan oleh semua pengguna dihitung. Hasilnya ditampilkan pada tabel di bawah ini dalam urutan menurun berdasarkan jumlah pengguna. Tabel dibuat untuk 20 *game* teratas dengan pengguna terbanyak.

Tabel 6. 20 *Game* dengan pengguna terbanyak

|    | index |                                         game | user |      hrs |
|---:|:-----:|---------------------------------------------:|:----:|---------:|
|  0 |  1336 |                                       Dota 2 | 4841 | 981684.6 |
|  1 |  4257 |                              Team Fortress 2 | 2323 | 173673.3 |
|  2 |  4788 |                                     Unturned | 1563 |  16096.4 |
|  3 |   981 |              Counter-Strike Global Offensive | 1412 | 322771.6 |
|  4 |  2074 |                       Half-Life 2 Lost Coast |  981 |    184.4 |
|  5 |   984 |                        Counter-Strike Source |  978 |  96075.5 |
|  6 |  2475 |                                Left 4 Dead 2 |  951 |  33596.7 |
|  7 |   978 |                               Counter-Strike |  856 | 134261.1 |
|  8 |  4899 |                                     Warframe |  847 |  27074.6 |
|  9 |  2071 |                       Half-Life 2 Deathmatch |  823 |   3712.9 |
| 10 |  1894 |                                  Garry's Mod |  731 |  49725.3 |
| 11 |  4364 |                   The Elder Scrolls V Skyrim |  717 |  70889.3 |
| 12 |  3562 |                                    Robocraft |  689 |   9096.6 |
| 13 |   980 | Counter-Strike Condition Zero Deleted Scenes |  679 |    418.2 |
| 14 |   979 |                Counter-Strike Condition Zero |  679 |   7950.0 |
| 15 |  2142 |                            Heroes & Generals |  658 |   3299.5 |
| 16 |  2070 |                                  Half-Life 2 |  639 |   4260.3 |
| 17 |  3825 |                   Sid Meier's Civilization V |  596 |  99821.3 |
| 18 |  4885 |                                  War Thunder |  590 |  14381.6 |
| 19 |  3222 |                                       Portal |  588 |   2282.8 |

Dari Tabel 6 terlihat bahwa, untuk beberapa *game*, ada hubungan antara yang paling banyak dimainkan dan yang paling banyak dibeli. Sebagai contoh, '*Dota 2*' tidak dapat disangkal lagi merupakan *game* yang paling populer, memiliki jumlah pengguna terbanyak dan total jam bermain terbanyak. Namun, hal ini tidak selalu terjadi, contoh yang menarik adalah '*Half-Life 2 Lost Coast*' yang memiliki jumlah pengguna yang tinggi (981 pengguna), tetapi total jam bermainnya cukup rendah (184,4 jam). Penjelasan yang mungkin untuk hal ini adalah karena game ini dibeli sebagai bagian dari bundel game.

Untuk memvisualisasikan hasil yang ditampilkan pada tabel di atas dengan lebih baik, *plot histogram* digunakan. Judul *game* diurutkan dalam urutan menurun berdasarkan jumlah pengguna. Gradien warna menunjukkan total jam bermain, dari yang paling banyak dimainkan hingga yang paling sedikit dimainkan.

![histogram-game-with-most-users](https://github.com/Rizki-Kidut/Recomendation-System---Steam-Game/assets/116653612/f5394a9d-cd8d-45db-9cfd-835ecaa1dc45)

Gambar 1. 20 game dengan Pengguna terbanyak

Dari Gambar 1 terlihat bahwa beberapa kasus, tidak ada hubungan antara jumlah total pengguna dan total jam yang dimainkan, yang berarti bahwa jumlah pengguna yang tinggi tidak merepresentasikan jumlah jam yang tinggi pula.

Jenis plot yang sama dibuat ulang, tetapi kali ini hanya mempertimbangkan pengguna yang benar-benar memainkan *game*. Jadi, untuk setiap game, pengguna yang membelinya tetapi tidak pernah memainkannya dihapus.

![histogram-game-with-most-users-play](https://github.com/Rizki-Kidut/Recomendation-System---Steam-Game/assets/116653612/3d8e1fcd-8e2f-45e7-9118-dea01986b580)

Gambar 2. 20 game dengan Pengguna terbanyak (Dimainkan)

Ketika membandingkan plot baru ini dengan plot sebelumnya, beberapa *game* jatuh dari 20 *game*teratas berdasarkan jumlah pengguna. Sebagai contoh '*Counter-Strike Condition Zero*', yang berada di posisi 15 teratas dalam plot dengan mempertimbangkan semua pengguna yang membeli *game* tersebut, tidak muncul di 20 *game* teratas dengan mempertimbangkan hanya pengguna yang benar-benar memainkan *game* tersebut. Contoh yang berlawanan adalah '*Terraria*' yang muncul di plot kedua sebagai 11 teratas sementara tidak terdaftar di plot pertama. Seperti yang telah disebutkan sebelumnya, penjelasan yang mungkin untuk perbedaan ini adalah beberapa *game* dibeli sebagai bagian dari bundel *game*.

### b. *Game Dataset*

Untuk memahami lebih baik bagaimana ulasan *game* didistribusikan, jumlah *game* dengan persentase ulasan positif masing-masing diplot.

Tabel 7. Ulasan pada *game dataset*

|       |                                              name | percentage_positive_review | review_qualification |                                       all_reviews |
|------:|--------------------------------------------------:|---------------------------:|---------------------:|--------------------------------------------------:|
|   0   |                                              DOOM |                         92 |        Very Positive | Very Positive,(42,550),- 92% of the 42,550 use... |
|   1   |                     PLAYERUNKNOWN'S BATTLEGROUNDS |                         49 |                Mixed | Mixed,(836,608),- 49% of the 836,608 user revi... |
|   2   |                                        BATTLETECH |                         71 |      Mostly Positive | Mostly Positive,(7,030),- 71% of the 7,030 use... |
|   3   |                                              DayZ |                         61 |                Mixed | Mixed,(167,115),- 61% of the 167,115 user revi... |
|   4   |                                        EVE Online |                         74 |      Mostly Positive | Mostly Positive,(11,481),- 74% of the 11,481 u... |
|  ...  |                                               ... |                        ... |                  ... |                                               ... |
| 40828 | Rocksmith® 2014 Edition – Remastered – Sabaton... |                         -1 |                  NaN |                                               NaN |
| 40829 | Rocksmith® 2014 Edition – Remastered – Stone T... |                         -1 |                  NaN |                                               NaN |
| 40830 | Fantasy Grounds - Quests of Doom 4: A Midnight... |                         -1 |                  NaN |                                               NaN |
| 40831 |                      Mega Man X5 Sound Collection |                         -1 |                  NaN |                                               NaN |
| 40832 |                                  Stories In Stone |                         -1 |                  NaN |                                               NaN |


![Distribution-of-Game-Reviews](https://github.com/Rizki-Kidut/Recomendation-System---Steam-Game/assets/116653612/1ea5dc0b-2a2d-4c86-8fbd-3f46944fb1cb)

Gambar 3. Distribusi ulasan game

Dari Gambar 3 diatas terlihat jelas bahwa nilai ulasan untuk *game* dalam *game dataset* terkonsentrasi pada nilai rating 60-100%.

Hal ini mengindikasikan bahwa kebanyakan *game* mendapatkan ulasan yang cukup baik dari pengguna.

Plot di bawah ini mencantumkan semua genre game yang tersedia di dataset game dengan jumlah game masing-masing.

![Recurrence-of-Genre](https://github.com/Rizki-Kidut/Recomendation-System---Steam-Game/assets/116653612/5fe130f0-7aa1-4720-b021-d450ee332c26)

Gambar 4. Genre *game* pada dataset

Berdasarkan Gambar 4 diatas, 5 genre terbanyak dalam *dataset* adalah *Indie, Action, Adventure, Casual,* dan *Simulation*. Hal ini mengindikasikan *game Action, Adventure* masih menjadi primadona dimata para pecinta *game* pada *Steam Store*

Plot serupa dibuat, menunjukkan 20 tag game terpopuler yang tersedia di dataset game dengan jumlah game masing-masing.

![Recurrence-of-Popular-Tags](https://github.com/Rizki-Kidut/Recomendation-System---Steam-Game/assets/116653612/98d6051a-2633-45d2-87bc-5d3758750bfb)

Gambar 5. Tag populer pada dataset

Hasil yang ditunjukkan pada Gambar 5 serupa dengan hasil pada plot *genre*. Dimana hasil 5 *popular tag* adalah *Indie, Action, Adventure, Casual, dan Simulation*


## Data Preparation

Untuk membangun sistem rekomendasi, data perlu dilakukan proses data preparation. Untuk melakukannya, pertama-tama perlu dilakukan praproses pada dataset game.

1. Memilih informasi yang berguna
2. Menangani *missing values*
3. Filter data dari kedua dataset

### 1. Pilih informasi yang berguna

Untuk menyiapkan data untuk rekomendasi berbasis konten, langkah pertama adalah memilih informasi yang paling berguna untuk menemukan game yang serupa. Kolom-kolom yang berguna dari dataset game dengan menggunakan kode berikut.

Tabel 8. Fitur yang dipilih pada *game dataset*

|   |                          name |                                             genre |                                      game_details |                                      popular_tags |
|--:|------------------------------:|--------------------------------------------------:|--------------------------------------------------:|--------------------------------------------------:|
| 0 |                          DOOM |                                            Action | Single-player,Multi-player,Co-op,Steam Achieve... | FPS,Gore,Action,Demons,Shooter,First-Person,Gr... |
| 1 | PLAYERUNKNOWN'S BATTLEGROUNDS |            Action,Adventure,Massively Multiplayer |            Multi-player,Online Multi-Player,Stats | Survival,Shooter,Multiplayer,Battle Royale,PvP... |
| 2 |                    BATTLETECH |                         Action,Adventure,Strategy | Single-player,Multi-player,Online Multi-Player... | Mechs,Strategy,Turn-Based,Turn-Based Tactics,S... |
| 3 |                          DayZ |            Action,Adventure,Massively Multiplayer | Multi-player,Online Multi-Player,Steam Worksho... | Survival,Zombies,Open World,Multiplayer,PvP,Ma... |
| 4 |                    EVE Online | Action,Free to Play,Massively Multiplayer,RPG,... | Multi-player,Online Multi-Player,MMO,Co-op,Onl... | Space,Massively Multiplayer,Sci-fi,Sandbox,MMO... |

Pada Tabel 8, terdapat 40.833 data

### 2. *Missing values*

Inspeksi *missing values* pada dataset Sebelum menangani *missing values*, *missing values* pada data perlu di inspeksi terlebih dahulu. *Missing values* pada dataset dapat dilihat dengan fungsi isnull(). Jumlah *missing values* pada dataset terdapat pada tabel di bawah ini.

Tabel 9. Jumlah *missing values* pada data

|     Fitur    | _Missing Values_ |
|:------------:|-----------------:|
|     name     |               16 |
|     genre    |              438 |
| game_details |              520 |
| popular_tags |             2945 |

Dari tabel 9 terlihat bahwa da banyak *missing values* dari semua kolom, kolom "*popular tag*" adalah kolom dengan nilai yang paling banyak hilang dengan 2.945 nilai yang hilang.

Dengan bantuan visualiasi matriks, tampak terlihat adanya data yang tidak lengkap seperti terlihat gambar dibawah ini:

![missing-data-matrix](https://github.com/Rizki-Kidut/Recomendation-System---Steam-Game/assets/116653612/ed3480b6-74b9-46fa-ac2f-19fbb12c5b3f)

Gambar 6. Matriks *missing values*


Terdapat tiga metode yang dapat digunakan untuk menangani *missing values* antara lain seperti berikut:

1. Dropping (metode yang dilakukan dengan cara menghapus sejumlah baris data)
2. Imputation (metode yang dilakukan dengan cara mengganti nilai yang "hilang" atau tidak tersedia dengan nilai tertentu yang bisa berupa median atau mean dari data)
3. Interpolation (metode menghasilkan titik-titik data baru dalam suatu jangkauan dari suatu data)

Metode yang dipilih untuk menangani masalah *missing values* ini adalah metode Dropping. Setelah *missing values* dihilangkan, jumlah *missing values* sudah menjadi 0 untuk setiap kolom pada tabel 10 dibawah ini

Tabel 10. Jumlah *missing values* setelah ditangani

|     Fitur    | _Missing Values_ |
|:------------:|-----------------:|
|     name     |                0 |
|     genre    |                0 |
| game_details |                0 |
| popular_tags |                0 |

### 3. Filter data dari kedua *Dataset*

Diputuskan untuk hanya menyimpan game yang ada di *game dataset* dan *user dataset*. Hal ini dipilih karena ada banyak game dalam dataset game yang belum pernah dimainkan atau dibeli oleh pengguna mana pun dalam dataset pengguna, sehingga tidak ada gunanya mempertimbangkannya dalam sistem rekomendasi. Selain itu, dataset game terlalu besar untuk membuat *matrix cosine similarity* karena membutuhkan terlalu banyak memori.

Untuk mencocokkan *game* dari kedua dataset secara bersamaan, ID untuk setiap *game* dibuat dengan menghapus semua simbol non-alfanumerik dan spasi, dan mengubah semua huruf kapital menjadi huruf kecil dengan menggunakan kode berikut ini, hal yang sama juga dilakukan untuk *game-game* yang ada di *user dataset*. Setelah itu, semua ID unik dari *user dataset* digunakan untuk memfilter baris di *game dataset*, dan menampilkan nilai ID yang sama dari kedua dataset.

Tabel 11. Data *game* yang telah disesuaikan dengan data *user*

|       |                     name |                                             genre |                                      game_details |                                      popular_tags |                 ID |
|------:|-------------------------:|--------------------------------------------------:|--------------------------------------------------:|--------------------------------------------------:|-------------------:|
|   3   |                     DayZ |            Action,Adventure,Massively Multiplayer | Multi-player,Online Multi-Player,Steam Worksho... | Survival,Zombies,Open World,Multiplayer,PvP,Ma... |               dayz |
|   4   |               EVE Online | Action,Free to Play,Massively Multiplayer,RPG,... | Multi-player,Online Multi-Player,MMO,Co-op,Onl... | Space,Massively Multiplayer,Sci-fi,Sandbox,MMO... |          eveonline |
|   12  |                     TERA | Action,Adventure,Free to Play,Massively Multip... | Multi-player,MMO,Co-op,Steam Trading Cards,Par... | Free to Play,MMORPG,Massively Multiplayer,RPG,... |               tera |
|   14  |              Stonehearth |                         Indie,Simulation,Strategy | Single-player,Multi-player,Online Multi-Player... | City Builder,Building,Sandbox,Strategy,Surviva... |        stonehearth |
|   20  | Call of Duty®: Black Ops |                                            Action | Single-player,Multi-player,Co-op,Steam Achieve... | Action,FPS,Zombies,Multiplayer,Shooter,Singlep... | callofdutyblackops |
|  ...  |                      ... |                                               ... |                                               ... |                                               ... |                ... |
| 34491 |              Particulars |                                             Indie | Single-player,Steam Achievements,Full controll... |                              Indie,Puzzle,Physics |        particulars |
| 34572 |                 Akuatica |                     Action,Adventure,Casual,Indie | Single-player,Steam Achievements,Steam Trading... |              Action,Indie,Casual,Adventure,Puzzle |           akuatica |
| 39625 |                 The Deer |                                   Adventure,Indie | Single-player,Steam is learning about this gam... | Gore,Adventure,Violent,Indie,Horror,Hunting,Fi... |            thedeer |
| 40711 |                 King-Dom |                                          Strategy | Online Multi-Player,Steam Achievements,Steam W... |                              Strategy,Chess,Indie |            kingdom |
| 40778 |                     GRID |            Action,Casual,Racing,Simulation,Sports |                 Single-player,Online Multi-Player |     Racing,Sports,Simulation,Action,Casual,Arcade |               grid |


Hasilnya, diperoleh 2.297 *game* dari *game dataset* yang cocok dengan 3.598 *game* dari *user dataset*.


## Modeling
Akan dibuatkan dua buah sistem rekomendasi menggunakan *Content Based Filtering* dan *Collaborative Based Filtering*. Algoritma yang digunakan untuk *Content Based Filtering* adalah *Cosine Similarity* dan untuk *Collaborative Based Filtering* adalah *BPR: Bayesian Personalized Ranking*

### 1. *Cosine Similarity* 

Dalam analisis data, *Cosine Similarity* adalah ukuran kemiripan antara dua vektor bukan nol yang didefinisikan dalam ruang hasil kali dalam. *Cosine Similarity* adalah *Cosine* dari sudut antara vektor; yaitu, hasil kali titik vektor dibagi dengan hasil kali panjangnya. Oleh karena itu, *Cosine Similarity* tidak bergantung pada besaran vektor, tetapi hanya pada sudutnya. *Cosine Similarity* selalu termasuk dalam interval.

Misalnya, dua vektor proporsional memiliki *Cosine Similarity* 1, dua vektor ortogonal memiliki kemiripan 0, dan dua vektor yang berlawanan memiliki kemiripan -1. Dalam beberapa konteks, nilai komponen vektor tidak boleh negatif, dalam hal ini *Cosine Similarity* dibatasi dalam [0,1]

Sebagai contoh, dalam pencarian informasi dan penggalian teks, setiap kata diberi koordinat yang berbeda dan dokumen diwakili oleh vektor jumlah kemunculan setiap kata dalam dokumen. *Cosine Similarity* kemudian memberikan ukuran yang berguna tentang seberapa mirip dua dokumen, dalam hal subjeknya, dan terlepas dari panjang dokumen.

Teknik ini juga digunakan untuk mengukur kohesi dalam kelompok dalam bidang penggalian data.

Salah satu keuntungan dari *Cosine Similarity* adalah kompleksitasnya yang rendah, terutama untuk vektor yang jarang: hanya koordinat yang bukan nol yang perlu dipertimbangkan.

Nama lain untuk *Cosine Similarity* termasuk Orchini similarity and *Tucker coefficient of congruence*; *the Otsuka–Ochiai similarity* adalah *Cosine Similarity* yang diterapkan pada data biner.

*Cosine Similarity* dari dua vektor yang tidak nol dapat diturunkan dengan menggunakan rumus dot product Euclidean:

$$ Cosine Similarity (A, B) = (A · B) / (||A|| * ||B||) $$ 

dimana: 
- (A·B)menyatakan produk titik dari vektor A dan B.
- ||A|| mewakili norma Euclidean (magnitudo) dari vektor A.
- ||B|| mewakili norma Euclidean (magnitudo) dari vektor B.

Tahapan yang dilakukan dalam pembuatan model *Content Based Filtering* dengan *Cosine Similarity* adalah sebagai berikut:

1.  Ekstrak fitur penting pada data dengan *TF-IDF Vectorizer*
2.  Hasil ekstraksi kemudian ditampilkan dalam bentuk matriks TF-IDF seperti tabel dibawah ini:

Tabel 12. Matriks TF-IDF dengan filter genre

    |                                                             | utilities | multiplayer |   sports | production | education |   free | video | strategy |    indie | access | ... | design | massively | illustration | software |   action | adventure |     to | animation | simulation | training |
|------------------------------------------------------------:|----------:|------------:|---------:|-----------:|----------:|-------:|------:|---------:|---------:|-------:|----:|-------:|----------:|-------------:|---------:|---------:|----------:|-------:|----------:|-----------:|---------:|
|                                                        name |           |             |          |            |           |        |       |          |          |        |     |        |           |              |          |          |           |        |           |            |          |
|                         Axiom Verge                         |       0.0 |    0.000000 | 0.000000 |        0.0 |       0.0 | 0.0000 |   0.0 | 0.000000 | 0.517948 |    0.0 | ... |    0.0 |  0.000000 |          0.0 |      0.0 | 0.531790 |  0.670022 | 0.0000 |       0.0 |   0.000000 |      0.0 |
| Adventures of Bertram Fiddle: Episode 1: A Dreadly Business |       0.0 |    0.000000 | 0.000000 |        0.0 |       0.0 | 0.0000 |   0.0 | 0.000000 | 0.436700 |    0.0 | ... |    0.0 |  0.000000 |          0.0 |      0.0 | 0.000000 |  0.564920 | 0.0000 |       0.0 |   0.000000 |      0.0 |
|                            GRID 2                           |       0.0 |    0.000000 | 0.727551 |        0.0 |       0.0 | 0.0000 |   0.0 | 0.000000 | 0.000000 |    0.0 | ... |    0.0 |  0.000000 |          0.0 |      0.0 | 0.000000 |  0.000000 | 0.0000 |       0.0 |   0.000000 |      0.0 |
|                           Collapse                          |       0.0 |    0.000000 | 0.000000 |        0.0 |       0.0 | 0.0000 |   0.0 | 0.000000 | 0.000000 |    0.0 | ... |    0.0 |  0.000000 |          0.0 |      0.0 | 0.621676 |  0.783274 | 0.0000 |       0.0 |   0.000000 |      0.0 |
|                          Planetbase                         |       0.0 |    0.000000 | 0.000000 |        0.0 |       0.0 | 0.0000 |   0.0 | 0.585946 | 0.418800 |    0.0 | ... |    0.0 |  0.000000 |          0.0 |      0.0 | 0.000000 |  0.000000 | 0.0000 |       0.0 |   0.693739 |      0.0 |
|                      Bob Came in Pieces                     |       0.0 |    0.000000 | 0.000000 |        0.0 |       0.0 | 0.0000 |   0.0 | 0.000000 | 0.611598 |    0.0 | ... |    0.0 |  0.000000 |          0.0 |      0.0 | 0.000000 |  0.791169 | 0.0000 |       0.0 |   0.000000 |      0.0 |
|                          Still Life                         |       0.0 |    0.000000 | 0.000000 |        0.0 |       0.0 | 0.0000 |   0.0 | 0.000000 | 0.000000 |    0.0 | ... |    0.0 |  0.000000 |          0.0 |      0.0 | 0.000000 |  1.000000 | 0.0000 |       0.0 |   0.000000 |      0.0 |
|       Castlevania: Lords of Shadow – Mirror of Fate HD      |       0.0 |    0.000000 | 0.000000 |        0.0 |       0.0 | 0.0000 |   0.0 | 0.000000 | 0.000000 |    0.0 | ... |    0.0 |  0.000000 |          0.0 |      0.0 | 0.621676 |  0.783274 | 0.0000 |       0.0 |   0.000000 |      0.0 |
|                      Grand Theft Auto V                     |       0.0 |    0.000000 | 0.000000 |        0.0 |       0.0 | 0.0000 |   0.0 | 0.000000 | 0.000000 |    0.0 | ... |    0.0 |  0.000000 |          0.0 |      0.0 | 0.621676 |  0.783274 | 0.0000 |       0.0 |   0.000000 |      0.0 |
|                    Dream Of Mirror Online                   |       0.0 |    0.466566 | 0.000000 |        0.0 |       0.0 | 0.3778 |   0.0 | 0.000000 | 0.000000 |    0.0 | ... |    0.0 |  0.466566 |          0.0 |      0.0 | 0.000000 |  0.234752 | 0.3778 |       0.0 |   0.000000 |      0.0 |



3. Menghitung *Cosine Similarity* dengan menggunakan Matriks TF-IDF sebagai inputannya

Tabel 13. *Cosine Similarity* dengan filter genre

|                           name | Axiom Verge | Adventures of Bertram Fiddle: Episode 1: A Dreadly Business | GRID 2 | Collapse | Planetbase | Bob Came in Pieces | Still Life | Castlevania: Lords of Shadow – Mirror of Fate HD | Grand Theft Auto V | Dream Of Mirror Online |   |   |   |   |   |   |   |   |   |   |   |
|-------------------------------:|------------:|------------------------------------------------------------:|-------:|---------:|-----------:|-------------------:|-----------:|-------------------------------------------------:|-------------------:|-----------------------:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
|                           name |             |                                                             |        |          |            |                    |            |                                                  |                    |                        |   |   |   |   |   |   |   |   |   |   |   |
| Call of Duty: United Offensive |    0.531790 |                                                    0.000000 |    0.0 | 0.621676 |   0.000000 |           0.000000 |   0.000000 |                                         0.621676 |           0.621676 |               0.000000 |   |   |   |   |   |   |   |   |   |   |   |
|  Aggression: Europe Under Fire |    0.000000 |                                                    0.000000 |    0.0 | 0.000000 |   0.585946 |           0.000000 |   0.000000 |                                         0.000000 |           0.000000 |               0.000000 |   |   |   |   |   |   |   |   |   |   |   |
|     Secret Files: Tunguska     |    0.670022 |                                                    0.564920 |    0.0 | 0.783274 |   0.000000 |           0.791169 |   1.000000 |                                         0.783274 |           0.783274 |               0.234752 |   |   |   |   |   |   |   |   |   |   |   |
|     Counter-Strike: Source     |    0.531790 |                                                    0.000000 |    0.0 | 0.621676 |   0.000000 |           0.000000 |   0.000000 |                                         0.621676 |           0.621676 |               0.000000 |   |   |   |   |   |   |   |   |   |   |   |
|            HuniePop            |    0.181196 |                                                    0.152773 |    0.0 | 0.000000 |   0.835325 |           0.213958 |   0.000000 |                                         0.000000 |           0.000000 |               0.156778 |   |   |   |   |   |   |   |   |   |   |   |
|           BattleSpace          |    0.000000 |                                                    0.000000 |    0.0 | 0.000000 |   0.154431 |           0.000000 |   0.000000 |                                         0.000000 |           0.000000 |               0.896426 |   |   |   |   |   |   |   |   |   |   |   |
|           Poly Bridge          |    0.267683 |                                                    0.225693 |    0.0 | 0.000000 |   0.810350 |           0.316082 |   0.000000 |                                         0.000000 |           0.000000 |               0.000000 |   |   |   |   |   |   |   |   |   |   |   |
|              Bleed             |    0.742341 |                                                    0.304696 |    0.0 | 0.445350 |   0.292206 |           0.426725 |   0.000000 |                                         0.445350 |           0.445350 |               0.000000 |   |   |   |   |   |   |   |   |   |   |   |
|           L.A. Noire           |    0.454866 |                                                    0.383514 |    0.0 | 0.531751 |   0.430229 |           0.537111 |   0.678882 |                                         0.531751 |           0.531751 |               0.159369 |   |   |   |   |   |   |   |   |   |   |   |
|  Ultimate General: Gettysburg  |    0.216916 |                                                    0.182890 |    0.0 | 0.000000 |   1.000000 |           0.256137 |   0.000000 |                                         0.000000 |           0.000000 |               0.000000 |   |   |   |   |   |   |   |   |   |   |   |


4. Terakhir dibuatkan rekomendasi game. Hasil dari rekomendasi game untuk game *Portal 2*

Tabel 14. Deskripsi game Portal 2

|     |     name |            genre |                                      game_details |                                      popular_tags |      ID |
|----:|---------:|-----------------:|--------------------------------------------------:|--------------------------------------------------:|--------:|
| 255 | Portal 2 | Action,Adventure | Single-player,Co-op,Steam Achievements,Full co... | Puzzle,Co-op,First-Person,Sci-fi,Comedy,Single... | portal2 |

Hasil rekomendasi dengan filter genre terdapat pada Tabel 15 dibawah ini

Tabel 15. Hasil rekomendasi *Content Base Filtering* dengan filter genre untuk *game* Portal 2

|   |                               name |            genre |
|--:|-----------------------------------:|-----------------:|
| 0 | Prince of Persia: The Two Thrones™ | Action,Adventure |
| 1 |                      Saints Row IV | Action,Adventure |
| 2 |                        Prototype 2 | Action,Adventure |
| 3 |                     Tomb Raider II | Action,Adventure |
| 4 |            Batman™: Arkham Origins | Action,Adventure |

Hasil rekomendasi untuk game *Portal 2* pada Tabel 15 di atas memiliki genre yang sama yaitu *Action* dan *Adventure*

Hasil rekomendasi dengan filter *popular tags* terdapat pada Tabel 16 dibawah ini.

Tabel 16. Hasil rekomendasi *Content Base Filtering* dengan filter *popular tag* untuk *game* Portal 2

|   |                               name |                                      popular_tags |
|--:|-----------------------------------:|--------------------------------------------------:|
| 0 |                      Space Farmers | Indie,Action,Co-op,Space,Multiplayer,Puzzle,On... |
| 1 | Artemis Spaceship Bridge Simulator | Simulation,Indie,Action,Space,Co-op,Multiplaye... |
| 2 |        Borderlands: The Pre-Sequel | Co-op,FPS,Action,Comedy,Loot,RPG,Open World,Mu... |
| 3 |                      Serious Sam 2 | Action,FPS,Co-op,Comedy,Multiplayer,Shooter,Fi... |
| 4 |    Rocketbirds: Hardboiled Chicken | Indie,Platformer,Action,Adventure,Great Soundt... |

Hasil rekomendasi untuk game Portal 2 pada Tabel 16 di atas memiliki *popular tags* yang sama yaitu *Puzzle,Co-op,First-Person,Sci-fi,Comedy*

Hasil rekomendasi dengan filter *game details* terdapat pada Tabel 17 dibawah ini.

|   |                                name |                                      game_details |
|--:|------------------------------------:|--------------------------------------------------:|
| 0 |                            Apollo4x | Single-player,Steam Trading Cards,Captions ava... |
| 1 |           Amnesia: The Dark Descent | Single-player,Steam Achievements,Full controll... |
| 2 |                           Gone Home | Single-player,Steam Achievements,Full controll... |
| 3 |                    Just Get Through | Single-player,Steam Achievements,Full controll... |
| 4 | Hegemony III: Clash of the Ancients | Single-player,Steam Achievements,Steam Trading... |

Tabel 17 di atas menunjukkan rekomendasi game berdasarkan game details. Hasil rekomendasi untuk game Portal 2 pada tabel di atas memiliki game details yang sama yaitu *Single-player dan Steam Achievements*

### 2. *Collaborative Filtering - BPR Bayesian Personalized Ranking*

Tugas utama dari Personalized Ranking adalah untuk memberikan daftar peringkat kepada pengguna

![Personalized](https://github.com/Rizki-Kidut/Recomendation-System---Steam-Game/assets/116653612/75020af6-3594-4b4c-ab07-2774127c95a9)

Gambar 7. Penanganan Data Implicit secara umum

Pendekatan yang biasa dilakukan adalah dengan memprediksi skor xui yang dipersonalisasi untuk sebuah item yang mencerminkan preferensi pengguna untuk item tersebut. Setelah itu, item akan diurutkan berdasarkan skor tersebut. Di sini, seperti yang Anda lihat pada gambar di atas, semua interaksi yang ada antara pengguna dan item diberi label sebagai kelas positif (1) dan interaksi lainnya diberi label sebagai kelas negatif (0).

Hal ini menyiratkan bahwa jika model kita cocok dengan data pelatihan, maka model akan memperlakukan semua interaksi yang tidak ada dalam data pelatihan dengan cara yang sama karena semuanya diberi label 0. Pendekatan ini tidak perlu mempertimbangkan peringkat dalam rekomendasi di masa mendatang.

Dalam pendekatan BPR, alih-alih mengambil satu item, pasangan item akan dianggap sebagai data pelatihan. Pengoptimalan akan dilakukan berdasarkan peringkat dari pasangan pengguna-item ini, bukan hanya berdasarkan interaksi pengguna-item. Dataset yang akan dipertimbangkan dirumuskan sebagai berikut

## Referensi 
[1] A. Pathak, K. Gupta, and J. McAuley. Generating and Personalizing Bundle Recommendations on Steam. In SIGIR, 2017.

[2] M. Wan and J. McAuley. Item Recommendation on Monotonic Behavior Chains. In RecSys, 2018.

[3] T. Gurbanov and F. Ricci. Action prediction models for recommender systems based on collaborative filtering and sequence mining hybridization. In Proceedings of the Symposium on Applied Computing, 2017

[4] R. He, W.-C. Kang, and J. McAuley. Translation-based recommendation. In RecSys, 2017.

[5] Y. Hu, Y. Koren, and C. Volinsky. Collaborative filtering for implicit feedback datasets. In ICDM, 2008

[6] S. Rendle, C. Freudenthaler, Z. Gantner, and L. Schmidt-Thieme. Bpr: Bayesian personalized ranking from implicit feedback. In UAI, 2009.

[7] R. Pan, Y. Zhou, B. Cao, N. N. Liu, R. Lukose, M. Scholz, and Q. Yang. One-class collaborative filtering. In ICDM, 2008
