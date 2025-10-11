# SOM_SITA2025 論文集 概要

このディレクトリには、Self-Organizing Maps (SOM) と気圧パターン分類に関する研究論文が収録されています。

## 主要論文一覧

### 1. 52-75Jiangetal2013.pdf

**ファイル名**: 52-75Jiangetal2013.pdf  
**タイトル**: Classification of synoptic circulation patterns with SOM and applications to climate and air quality data visualization  
**著者**: Ningbo Jiang, Kim N. Dirks, Kehui Luo  
**掲載誌**: Weather and Climate, 33, 52-75

**キーワード**: 自己組織化マップ（SOM）、総観天気型、局地気象、大気質、データ可視化、ニュージーランド

**概要**:  
ニュージーランド域のジオポテンシャル高度再解析データを用いて、SOM による新たな総観分類（25 タイプ）を構築。この分類は文献の典型的な総観型を再現するだけでなく、東進する総観システムの進化を可視化する。オークランドの局地気象・大気質（O3、NOx）データを SOM 平面に投影することで、総観型と大気質の関係性を可視化。低気圧型は換気強化と上層からの O3 下降輸送により日別 O3 増加に寄与し、ブロッキング時には外洋からの清浄空気移流が観測されることを明らかにした。

---

### 2. 2024*IPSJ 全国大会*髙須賀.pdf

**ファイル名**: 2024*IPSJ 全国大会*髙須賀.pdf  
**タイトル**: Self-Organizing Maps を用いた気圧配置のクラスタリングと 1km メッシュ気象データを用いた解析  
**著者**: 髙須賀拓海、高野雄樹、渡邉将太郎、雲井元道  
**発表**: 2024 年 IPSJ 全国大会

**キーワード**: Self-Organizing Maps (SOM)、気圧配置、クラスタリング、1km メッシュ気象データ、Batch SOM、PCA

**概要**:  
SOM を用いた気圧配置のクラスタリング手法を提案。2016 年 3 月～ 2024 年 10 月の毎日 9 時（JST）のデータ（3,193 日分）を使用。PCA により 20 次元に削減後、10×10 の Batch SOM（計 100 個のクラスタ）を構築。各クラスタについて、発生頻度、発生時期、気象的特徴を分析。1km メッシュの詳細気象データ（日照時間、日平均気温、日降水量）を用いて、各気圧配置パターンに対応する詳細な気象特性を把握。全体で 78%の分類精度を達成し、特に出現頻度が高いクラスタでは 100%の精度を示した。

---

### 3. Classification_of_synoptic_circulation_patterns_wi.pdf

**ファイル名**: Classification_of_synoptic_circulation_patterns_wi.pdf  
**タイトル**: Classification of synoptic circulation patterns with a two-stage clustering algorithm using the modified structural similarity index metric (SSIM)  
**著者**: Kristina Winderlich, Clementine Dalelane, Andreas Walter  
**掲載誌**: Earth Syst. Dynam., 15, 607–633, 2024  
**DOI**: 10.5194/esd-15-607-2024

**キーワード**: SSIM（構造類似度指標）、二段階クラスタリング、HAC（階層的凝集型クラスタリング）、k-medoids、総観循環パターン、気候モデル評価

**概要**:  
修正 SSIM を類似度指標として用い、HAC（階層的凝集型クラスタリング）と k-medoids を反復結合する新しい二段階クラスタリングアルゴリズムを提案。従来のユークリッド距離や k-means 法の欠点（構造の鈍感性、スノーボール効果）を克服。クラス中心にメドイド（実データ）を採用することで、代表性と頑健性を確保。ERA-Interim（1979-2018）の 500 hPa 面地衡高度に適用し、37 クラスの総観型を自動的に構築。クラス間分離の良好性、空間・時間的安定性、物理的妥当性を確認。CMIP6 モデル評価への応用例として、Jensen-Shannon 距離を用いた統計比較を示し、既存のスカラー指標を補完する診断ツールとしての有用性を提示。

---

### 4. gmd-14-2097-2021.pdf

**ファイル名**: gmd-14-2097-2021.pdf  
**タイトル**: S-SOM v1.0: a structural self-organizing map algorithm for weather typing  
**著者**: Quang-Van Doan, Hiroyuki Kusaka, Takuto Sato, Fei Chen  
**掲載誌**: Geosci. Model Dev., 14, 2097–2111, 2021  
**DOI**: 10.5194/gmd-14-2097-2021

**キーワード**: S-SOM（構造的 SOM）、S-SIM（構造的類似度）、天候型付け、BMU 探索、シルエット解析、トポロジ誤差

**概要**:  
空間・時間的構造を持つ入力データを扱える新規な構造的自己組織化マップ（S-SOM）アルゴリズムを提案。従来のユークリッド距離（ED）ではなく構造的類似度（S-SIM）指標に基づく BMU（Best Matching Unit）探索を実施。S-SIM は、高低気圧の位置などに関わる空間相関を考慮した分類を可能にする。日本域の ERA-Interim 海面更正気圧（SLP）データを用いた評価実験で、S-SOM は標準 SOM（ED-SOM）およびピアソン相関係数を用いる SOM（COR-SOM）よりも優れたクラスタリング品質（シルエット解析）とトポロジ保持（トポロジ誤差）を示した。計算時間は ED-SOM の 8-15 倍だが、全体で 1 分未満であり実用的。

---

### 5. advpub_2021-047.pdf

**ファイル名**: advpub_2021-047.pdf  
**タイトル**: Statistical Intercomparison of Similarity Metrics in Sea Level Pressure Pattern Classification  
**著者**: Takuto SATO, Hiroyuki KUSAKA  
**掲載誌**: Journal of the Meteorological Society of Japan  
**DOI**: 10.2151/jmsj.2021-047

**キーワード**: 類似度指標、相関係数（COR）、ユークリッド距離（EUC）、S1 スコア、SSIM、平均ハッシュ（aHash）、天気図分類

**概要**:  
SLP パターンを抽出するための代表的な 5 種類の類似度指標（COR、EUC、S1、SSIM、aHash）の精度を統計的に比較。日本海上の低気圧（CSoJ）パターン選択において、多数の教師データを用いて各指標の選択率を評価。結果、S1 スコアと SSIM が平均値・最大値の両面で最も高い精度を示し、ノイズを含む非理想的な教師データを用いた場合でも精度が低下しないことを確認。S1 と SSIM は、ユークリッド距離と比べて 2 枚の図の主観的な類似性をよりよく再現することを明らかにした。一方、ユークリッド距離はシグナルの中心位置の再現に寄与する。

---

### 6. 1990-Kohonen-PIEEE.pdf

**ファイル名**: 1990-Kohonen-PIEEE.pdf  
**ステータス**: 読み取り不可（PDF 自動テキスト抽出不可）

**キーワード**: Self-Organizing Maps、Kohonen、SOM の基礎理論

**概要**:  
SOM（自己組織化マップ）の創始者である Kohonen による基礎論文と推測される。PDF の内容を読み取ることができないため、詳細な解説は提供できない。

---

## その他の論文

ディレクトリには以下のような関連論文も含まれています：

- **82_323.pdf** - 気圧配置を対象とした事例ベース予報と SVM を用いた台風進路予測
- **99_2021-047.pdf** - 海面更正気圧パターン分類における類似度指標の統計相互比較（別バージョン）
- **7183018.pdf** - Taylor 図による多面的評価手法
- **19740022614.pdf** - データ分析関連
- **A_Computational_Approach_to_Edge_Detection-sz4.pdf** - エッジ検出に関する計算手法（Canny 法）
- **a062011-087.pdf** - 関連研究
- **dC.pdf** - 関連資料
- **github_com_JustGlowing_minisom.pdf** - MiniSOM（SOM の Python 実装）に関する資料
- **HaarPSI_preprint_v4.pdf** - HaarPSI（画像類似度評価指標）
- **marr-hildreth-edge-prsl1980.pdf** - Marr-Hildreth エッジ検出法
- **Multiscale_structural_similarity_for_image_quality.pdf** - MS-SSIM（マルチスケール構造類似度）
- **sethian.osher.88.pdf** - Level Set 法の基礎
- **tanimoto.pdf** - Tanimoto 係数（類似度指標）

---

## 主要な研究トピック

このディレクトリの論文群は、以下のトピックをカバーしています：

1. **Self-Organizing Maps (SOM)の手法論**

   - 標準 SOM vs 構造的 SOM (S-SOM)
   - Batch SOM vs 逐次学習 SOM
   - 初期化手法とパラメータ設定

2. **類似度指標の比較と選択**

   - ユークリッド距離（ED）
   - 相関係数（COR）
   - S1 スコア（圧力勾配ベース）
   - SSIM（構造類似度指標）
   - aHash（平均ハッシュ）

3. **気圧パターン分類への応用**

   - 総観天気型の自動分類
   - 日本域の気圧配置パターン
   - ヨーロッパ域の循環パターン
   - ニュージーランド域の天気型

4. **気候モデル評価**

   - CMIP6 モデルの評価
   - 再解析データとの比較
   - Jensen-Shannon 距離による統計的評価

5. **大気質・気象現象との関連**
   - オゾン（O3）濃度の総観支配
   - NOx 濃度と気圧パターンの関係
   - 極端現象（高温、多雨）との結びつき

---

## 技術的なポイント

### SOM の利点

- 高次元データの低次元マップへの写像
- 位相保持（類似パターンは近接配置）
- 視覚的な解釈が容易
- 教師なし学習

### 構造的類似度指標（SSIM）の利点

- 輝度、コントラスト、構造の 3 要素を考慮
- 人間の視覚的知覚に近い類似度評価
- 空間相関を保持
- 画像処理分野で広く使用

### 二段階クラスタリングの利点

- クラス数の自動決定
- 事前フィルタリング不要
- 稀な状況も含めた全データの分類
- 頑健性の向上（メドイド使用）

---

## 参考リンク

- ERA-Interim: https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-interim
- CMIP6: https://esgf-node.llnl.gov/search/cmip6/
- JRA-55: https://jra.kishou.go.jp/JRA-55/index_en.html
- MiniSOM (GitHub): https://github.com/JustGlowing/minisom/

---

## 更新履歴

- 2025/01/11: 初版作成。主要 5 論文の概要をまとめた。
