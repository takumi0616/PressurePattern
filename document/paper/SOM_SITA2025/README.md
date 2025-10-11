# SOM_SITA2025 論文集 概要

このディレクトリには、Self-Organizing Maps (SOM) と気圧パターン分類に関する研究論文が収録されています。

## 論文一覧

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

### 7. 82_323.pdf

**ファイル名**: 82_323.pdf  
**タイトル**: サポートベクターマシンによる気圧配置の検出：冬型気圧配置  
**著者**: 木村寛樹、川島秀行、日下博幸、北川裕之（筑波大学）  
**掲載誌**: 地理学評論 Series A, 82-4, 323–331, 2009

**キーワード**: 気圧配置、冬型、サポートベクターマシン（SVM）、地上天気図、パターン認識

**概要**:  
気候研究における気圧配置の自動検出を目的として、サポートベクターマシン（SVM）を用いた分類手法を提案。1981〜2000 年の JRA-25 データを用いて「冬型」と「非冬型」の気圧配置を分類し、90%以上の精度を達成した。従来の目視による膨大な地上天気図の走査を不要にし、長期データの自動判定を可能にする。RBF カーネルを使用し、次元削減を施した特徴空間での分類により、計算効率と精度の両立を実現。この手法により、気圧配置の頻度変化の評価など気候学的研究への応用が期待される。

---

### 8. 7183018.pdf

**ファイル名**: 7183018.pdf  
**タイトル**: SOM を用いた北部九州・中国地方で高潮災害を引き起こした気象場パターンの分類  
**著者**: 朝位孝二、西山浩司、白水元、丹羽晶大（山口大学、九州大学）  
**掲載誌**: Journal of Japan Society of Civil Engineers, Ser. B2 (Coastal Engineering), Vol. 76, No. 2, pp.I_1219-I_1224, 2020

**キーワード**: 高潮、台風、気象場、パターン認識、自己組織化マップ、防災

**概要**:  
1979 年から 2019 年までの 41 年間の気象データを自己組織化マップ（SOM）で分析し、北部九州・中国地方で高潮災害を引き起こした台風の気象場パターンを分類。20,172 個の気象場データを 696 個のユニット、45 個のグループに分類した結果、高潮災害を引き起こした台風は特定のパターン（グループ 45 のユニット 696）に分類されることを発見。4 つの気象要素（地表面気圧、風速成分、可降水量）を用いて 64 次元の入力データを構築し、NCEP/NCAR 再解析データを使用。台風 9918 号の時系列分析により、台風の進行に伴う SOM 上の挙動を追跡し、最も危険なパターンへの移行を可視化。全球数値モデル（GSM）やメソ数値予報モデル（MSM）と組み合わせることで、接近中の台風が過去の災害台風と類似していることを事前に把握し、早期の防災行動を促すシステムの構築が可能となる。

---

### 9. HaarPSI_preprint_v4.pdf

**ファイル名**: HaarPSI_preprint_v4.pdf  
**タイトル**: A Haar Wavelet-Based Perceptual Similarity Index for Image Quality Assessment  
**著者**: Rafael Reisenhofer, Sebastian Bosse, Gitta Kutyniok, Thomas Wiegand  
**掲載誌**: Signal Processing: Image Communication, 2018

**キーワード**: Haar ウェーブレット、知覚類似度指標（HaarPSI）、画像品質評価、フルリファレンス IQA、SSIM、FSIM

**概要**:  
Haar ウェーブレットに基づく新規で計算コストの低い画像品質評価指標（HaarPSI）を提案。Haar ウェーブレット分解から得られる係数を用いて、2 つの画像間の局所的な類似性および画像領域の相対的重要性を評価する。高周波 Haar ウェーブレット係数の大きさで局所類似度を定義し、低周波係数で重み付けを行う。4 つの大規模ベンチマークデータベース（LIVE、TID2008、TID2013、CSIQ）で検証した結果、SSIM、FSIM、VSI といった最先端のフルリファレンス類似度指標よりも高い相関を達成。わずか 6 つの離散 Haar ウェーブレットフィルタの応答から計算され、FSIM や VSI よりも大幅に高速（HaarPSI: 240ms、FSIM: 1420ms、VSI: 790ms）。簡潔な計算構造により、人間の視覚系の方位選択性と空間周波数選択性の基本的実装を提供。気圧パターン分類において、SSIM や S1 スコアと並んで有用な類似度指標として応用可能。

---

### 10. 19740022614.pdf

**ファイル名**: 19740022614.pdf  
**タイトル**: Monitoring Vegetation Systems in the Great Plains with ERTS  
**著者**: J. W. Rouse, Jr., R. H. Haas, J. A. Wells, D. W. Deering（テキサス A&M 大学）  
**出典**: Proceedings/Report（1973 年頃、ERTS-1 データによる研究報告）

**キーワード**: ERTS（ランドサット）、植生モニタリング、グレートプレーンズ、緑色バイオマス、バンド比パラメータ、リモートセンシング

**概要**:  
ERTS-1（初代ランドサット）MSS データを用いて、グレートプレーンズの広域植生状態を定量的に測定する方法を開発。太陽高度の影響を補正した ERTS-1 バンド 5 およびバンド 7 の輝度値からバンド比パラメータを計算し、これが草地における地上部の緑色バイオマスと相関することを示した。テキサス南部からノースダコタに至る 6 州にまたがる 10 の試験サイトで、4 シーズン分のデータを取得。季節的・気候的逆境に極めて脆弱な地域の農業活動を支援するための定量的な地域植生状態情報の提供に有望であることを実証。この研究は後の NDVI（正規化植生指数）などの植生指数の基礎となる概念を含む。

---

### 11. A_Computational_Approach_to_Edge_Detection-sz4.pdf

**ファイル名**: A_Computational_Approach_to_Edge_Detection-sz4.pdf  
**タイトル**: A Computational Approach to Edge Detection  
**著者**: John Francis Canny  
**掲載誌**: IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. PAMI-8, No. 6, November 1986  
**DOI**: 10.1109/TPAMI.1986.4767851

**キーワード**: Canny エッジ検出、エッジ検出アルゴリズム、画像処理、コンピュータビジョン、特徴抽出

**概要**:  
エッジ検出に対する計算論的アプローチを記述した画期的な論文。エッジ点の計算に対して包括的な目標集合を定義し、検出性能と位置特定性能、単一エッジへの単一応答という 3 つの基準を数学的に定式化。ガウス平滑化画像の勾配強度の極大でエッジを標示する実装を提案。検出と位置特定の間に自然な不確定性原理（トレードオフ）が存在することを示し、任意のスケールで最適となる単一の演算子形状を導出。複数幅の演算子を用いる拡張、および異なるスケールからの情報を粗から細へ統合する「特徴合成」手法を提示。Canny エッジ検出器として知られるこのアルゴリズムは、現在でも画像処理とコンピュータビジョンの基本的なツールとして広く使用されている。気圧パターン分析における強度変化の検出にも応用可能。

---

### 12. marr-hildreth-edge-prsl1980.pdf

**ファイル名**: marr-hildreth-edge-prsl1980.pdf  
**タイトル**: Theory of Edge Detection  
**著者**: D. Marr, E. Hildreth  
**掲載誌**: Proc. R. Soc. Lond. B 207, 187–217 (1980)

**キーワード**: エッジ検出理論、ラプラシアン・オブ・ガウシアン（LoG）、ゼロ交差、マルチスケール解析、視覚情報処理

**概要**:  
エッジ検出の包括的理論を提示した歴史的に重要な論文。自然画像における強度変化は広いスケール範囲にわたって起こるため、スケールごとに別個に検出する必要があることを示した。与えられたスケールでの最適なフィルタはガウス関数の二階微分（ラプラシアン）であることを数学的に証明。空間局在と周波数局在の両方を最適化する不確定性原理に基づき、ガウス分布が唯一の最適解であることを示した。画像をガウスフィルタで平滑化し、ラプラシアンを適用して得られるゼロ交差を検出することで、エッジを抽出する。複数のスケール（σ 値）で処理を行い、異なるスケールからの情報を統合する手法を提案。この理論は人間視覚系のモデルとしても提案され、単純細胞の機能との関連が議論された。気圧場の解析において、異なるスケールでの強度変化を検出する理論的基盤を提供。

---

### 13. Multiscale_structural_similarity_for_image_quality.pdf

**ファイル名**: Multiscale_structural_similarity_for_image_quality.pdf  
**タイトル**: Multi-scale structural similarity for image quality assessment  
**著者**: Zhou Wang, Eero P. Simoncelli, Alan C. Bovik  
**出典**: IEEE ACSSC 2003  
**DOI**: 10.1109/ACSSC.2003.1292216

**キーワード**: MS-SSIM、マルチスケール構造類似度、画像品質評価、SSIM 拡張、観視条件

**概要**:  
構造類似度（SSIM）をマルチスケールに拡張した MS-SSIM（Multi-Scale Structural Similarity）法を提案。単一スケール SSIM の短所である観視条件（表示解像度、観視距離等）への依存性を克服。参照・歪み画像にローパスフィルタと 2 倍間引きを反復適用し、複数の解像度スケールで輝度・コントラスト・構造の類似性を評価。画像合成ベースの新手法により各スケールの相対的重要度を較正するパラメータを決定。LIVE データベース（JPEG/JPEG2000 圧縮画像 344 枚）での実験評価により、単一スケール SSIM（最良のスケール選択を含む）および PSNR、Sarnoff JNDmetrix などの最先端指標を上回る性能を達成（相関係数 0.969）。異なるスケールの情報を統合することで、人間の視覚的知覚をより正確に予測。気圧パターンの類似度評価において、複数スケールでの構造的特徴を統合的に評価する手法として応用可能。

---

### 14. a062011-087.pdf

**ファイル名**: a062011-087.pdf  
**タイトル**: Performance Comparison of Self-Organizing Maps and k-means Clustering Techniques for Atmospheric Circulation Classification  
**著者**: Kostas Philippopoulos, Despina Deligiorgi, Georgios Kouroupetroglou  
**掲載誌**: International Journal of Energy, Environment and Economics, 2014 年 12 月  
**被引用数**: 10 回

**キーワード**: SOM、k-means、大気循環分類、クラスタリング比較、南東ヨーロッパ

**概要**:  
自己組織化マップ（SOM）と k-means クラスタリング手法による大気循環タイプ分類の性能を比較。南東ヨーロッパにおける 62 年間（1948-2009 年）の春季の平均日々海面更正気圧（MSLP）データを使用（格子間隔 2.5°×2.5°）。k-means は 10 タイプ、SOM は 20 タイプの循環パターンを生成。両手法は各 SOM 循環タイプの構成員（日）を各 k-means タイプへ分配して比較。特に高気圧系が大気循環を制御するパターンにおいて、高い類似性が観察された。SOM は非線形分類を生成できる能力、ならびに近縁の大気モードを隣接ニューロンで表現するマップを生成できる能力により優位であることを示した。圧力場の対応関係、月別頻度および累積頻度による詳細な比較分析を実施。二段階分類スキーム（SOM で多数の状態を生成後、k-means で実用的なカタログに統合）を今後の課題として提案。

---

### 15. github_com_JustGlowing_minisom.pdf

**ファイル名**: github_com_JustGlowing_minisom.pdf  
**タイトル**: MiniSom: minimalistic and NumPy-based implementation of the Self Organizing Map  
**著者**: Giuseppe Vettigli  
**種別**: GitHub リポジトリ（公開ソフトウェア）  
**URL**: https://github.com/JustGlowing/minisom/

**キーワード**: MiniSom、SOM 実装、Python、NumPy、自己組織化マップ、オープンソース

**概要**:  
自己組織化マップ（SOM）の最小実装を提供する Python ライブラリ。NumPy のみに依存し、機能面・依存関係・設計の面で可能な限り簡潔さを志向。コードはベクトル化されており、研究用途で手軽に使える設計。400 回以上引用されており、多くの研究で使用されている。pip で簡単にインストール可能（`pip install minisom`）。基本的な使い方は SOM の初期化、学習、BMU（Best Matching Unit）の取得、モデルの保存・読み込みなど。豊富なチュートリアルとサンプルコードが用意されており、色の量子化、外れ値検出、手書き数字のマッピングなど様々な応用例が提供されている。気圧パターン分類の実装において、簡潔で効率的な SOM ツールとして活用可能。

---

### 16. sethian.osher.88.pdf

**ファイル名**: sethian.osher.88.pdf  
**タイトル**: Fronts Propagating with Curvature Dependent Speed: Algorithms Based on Hamilton-Jacobi Formulations  
**著者**: Stanley Osher, James A. Sethian  
**掲載誌**: Journal of Computational Physics, 79, pp.12-49, 1988

**キーワード**: Level Set 法、Hamilton-Jacobi 方程式、曲率依存速度、フロント伝播、PSC アルゴリズム

**概要**:  
曲率依存速度で伝播するフロントを追跡するための新しい数値アルゴリズム（PSC アルゴリズム：Propagation of Surfaces under Curvature）を提案した歴史的に重要な論文。Hamilton-Jacobi 方程式に基づく定式化により、双曲型保存則の技術を用いて運動方程式を近似。様々な精度のオーダーの非振動スキームを使用し、移動フロントにおける急峻な勾配やカスプの形成を正確に捕捉。アルゴリズムはトポロジカルな合流・分離を自然に処理し、任意の空間次元で動作し、移動表面が関数として記述できなくても適用可能。結晶成長や炎の伝播など、物理現象における曲率依存速度を持つフロントの追跡に広く応用。Level Set 法の基礎となる重要な論文であり、画像処理、流体力学、気象学など多様な分野に影響を与えた。気圧場の等値線追跡や前線の移動解析への応用が期待される。

---

### 17. tanimoto.pdf

**ファイル名**: tanimoto.pdf  
**タイトル**: （判読不可）  
**ステータス**: PDF からテキスト抽出不可

**キーワード**: Tanimoto 係数、類似度指標

**概要**:  
PDF の内容を読み取ることができないため、詳細な解説は提供できない。ファイル名から推測すると、Tanimoto 係数（Tanimoto coefficient）または Jaccard 係数に関する論文と思われる。Tanimoto 係数は、2 つの集合の類似性を測定する指標であり、化学情報学における分子の類似性評価や、情報検索、パターン認識などで広く使用される。気圧パターンの類似度評価においても応用可能な類似度指標の一つ。

---

### 18. dC.pdf

**ファイル名**: dC.pdf  
**タイトル**: （判読不可）  
**ステータス**: PDF からテキスト抽出不可

**キーワード**: （不明）

**概要**:  
PDF の内容を読み取ることができないため、詳細な解説は提供できない。画像型 PDF の可能性があり、OCR 処理が必要と思われる。ファイル名からは内容を推測することが困難。

---

## 主要な研究トピック

このディレクトリの論文群は、以下のトピックをカバーしています：

1. **Self-Organizing Maps (SOM)の手法論**

   - 標準 SOM vs 構造的 SOM (S-SOM)
   - Batch SOM vs 逐次学習 SOM
   - SOM と k-means の性能比較
   - 初期化手法とパラメータ設定
   - MiniSom：Python による実装ライブラリ

2. **類似度指標の比較と選択**

   - ユークリッド距離（EUC）
   - 相関係数（COR）
   - S1 スコア（圧力勾配ベース）
   - SSIM（構造類似度指標）
   - MS-SSIM（マルチスケール SSIM）
   - HaarPSI（Haar ウェーブレットベース）
   - aHash（平均ハッシュ）
   - Tanimoto 係数

3. **クラスタリングアルゴリズム**

   - 二段階クラスタリング（HAC + k-medoids）
   - k-means クラスタリング
   - 階層的凝集型クラスタリング（HAC）
   - サポートベクターマシン（SVM）による分類

4. **気圧パターン分類への応用**

   - 総観天気型の自動分類
   - 日本域の気圧配置パターン（冬型、CSoJ など）
   - ヨーロッパ域の循環パターン（ERA-Interim）
   - ニュージーランド域の天気型
   - 南東ヨーロッパの大気循環（62 年間）

5. **気候モデル評価**

   - CMIP6 モデルの評価
   - 再解析データとの比較（JRA-25、JRA-55、ERA-Interim）
   - Jensen-Shannon 距離による統計的評価

6. **極端気象現象への応用**

   - 高潮災害を引き起こす気象場パターン
   - 台風進路予測と SOM
   - 降水・降雪パターンの分類
   - 極端現象（高温、多雨）との結びつき

7. **大気質・環境への応用**

   - オゾン（O3）濃度の総観支配
   - NOx 濃度と気圧パターンの関係
   - 大気質データの可視化

8. **画像処理・エッジ検出技術**

   - Canny エッジ検出アルゴリズム
   - Marr-Hildreth エッジ検出理論（ラプラシアン・オブ・ガウシアン）
   - Level Set 法（Hamilton-Jacobi 方程式）
   - マルチスケール解析

9. **リモートセンシング**
   - ERTS（ランドサット）による植生モニタリング
   - NDVI（正規化植生指数）の基礎概念
   - バンド比パラメータ

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
