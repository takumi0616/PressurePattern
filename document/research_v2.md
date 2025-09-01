# 自己組織化マップにおける構造類似度指標を用いた気圧配置パターン分類の高度化

_Enhancement of Synoptic Pattern Classification Using Structural Similarity Metrics in Self-Organizing Maps_

- 雲居玄道（長岡技術科学大学 / Nagaoka University of Technology, 1603-1 Kamitomioka-machi, Nagaoka, Niigata 940-2188, Japan）
- 高野雄紀（気象研究所 / Meteorological Research Institute, 1-1 Nagamine, Tsukuba, Ibaraki 305-0052, Japan）
- 渡邊正太郎（株式会社ウェザーマップ / Weather Map Co., Ltd., The HEXAGON 5F, 5-4-7 Akasaka, Minato-ku, Tokyo 107-0052, Japan）
- 高須賀匠（長岡技術科学大学 / Nagaoka University of Technology）

## Abstract

This study presents a comprehensive comparison of distance metrics for synoptic pattern classification using Self-Organizing Maps (SOM). We implemented Batch-SOM with four distance metrics: Euclidean (EUC), Structural Similarity Index with 5×5 moving window (SSIM5), Teweles-Wobus score (S1), and their fusion (S1+SSIM5). Using sea level pressure data from the broader Japan region (1991-1999 for training, 2000 for validation), we evaluated classification performance through Macro-averaged Recall and medoid representation quality. Results show that S1 achieved the highest training Macro-averaged Recall (0.3466), while S1+SSIM5 demonstrated superior generalization (0.3501 in validation). The SSIM5 with local windows consistently captured structural features effectively. We also introduced medoid and true-medoid representations to address the "blurring" effect of centroid averaging, enhancing interpretability of SOM output maps for operational use. This framework bridges recent advances in structural similarity metrics and contributes to future developments in pressure pattern classification.

## Keywords

Self-Organizing Map, Structural Similarity, Weather Pattern Classification, Distance Metrics, Medoid Representation

---

## 1. はじめに

総観規模の気圧配置パターン分類は，気象予報，気候システムの理解，防災・減災など，多様な応用の基盤技術である．従来の主観分類は専門家知見を直接反映できる一方で，労力・再現性・スケーラビリティに課題があるため，客観的・自動的な分類法が広く研究されてきた．教師ありでは，木村ほか [\cite{木村広希 2009 サポートベクターマシンを用いた気圧配置検出手法の提案}] が SVM により「冬型」「南高北低」「台風型」等の自動検出と検索システムを実装し，実用可能性を示した．しかし，ラベル付与のコストや主観ノイズに起因する学習データ品質の限界が指摘される．教師なしでは，SOM（Self-Organizing Map）が総観パターンの非線形構造を可視化・圧縮する手段として用いられ [\cite{philippopoulos2014performance}, \cite{jiang2013classification}]，国内では筆者であるタカスカほか [\cite{takasuka2024}] が 10×10 サイズの batchSOM により日本周辺の気圧配置をクラスタリングし，1km メッシュ天気との結び付けを示した．

一方，従来の多くの SOM やクラスタリングはユークリッド距離（EUC）を前提としており，勾配・形状・位置などの「構造」を評価しにくいという根源的制約がある．Wang and Bovik による信号比較の議論に呼応し，Doan らの S-SOM [\cite{doan2021s}] は BMU 探索に構造類似度指標（SSIM）を導入してシルエット係数やトポロジ保存性を改善した．加えて，Sato and Kusaka [\cite{SATOTakuto20212021-047}] は大規模検証により，勾配ベースの Teweles–Wobus スコア（S1）と SSIM が，EUC や単純相関よりも人間の主観的「似ている」をよく再現することを統計的に示した．さらに Winderlich ら [\cite{winderlich2024classification}] は，改良 SSIM を用いた HAC+k-medoids の二段階法と medoid 表現により，クラス分離性と代表性を両立し，Jensen–Shannon 距離で循環統計を総合評価する枠組みを提案している．

本研究は，これらの先行知見を SOM を基盤とする単一フレームワーク上で統合・比較可能な形に再編することを目的とする．具体的には，GPU 最適化 Batch-SOM を共通基盤とし，距離（類似度）指標として EUC，5×5 移動窓 SSIM（SSIM5），S1，および S1 と SSIM5 の単純融合（S1+SSIM5）を同一条件で厳密比較する．併せて，SOM の平均プロトタイプ（centroid）がもたらす「ぼけ」を回避するため，各ノードの代表として medoid を出力とし，解釈性と事例検索性を高める．対象は日本域の SLP で，学習（1991–1999）と独立検証（2000）を分離し，Macro Recall，ノード代表の一致性などで汎化性能を評価する．本論文の貢献は次の通りである．

- SSIM5・S1・EUC を SOM の BMU 探索で横並びに比較し，局所構造を評価する SSIM5 と勾配評価 S1 の相補性を定量化（融合指標の有効性を検証）．
- centroid の代替として medoid/true-medoid を SOM の標準出力に組み込み，代表パターンの鋭さと説明可能性を向上．
- 先行の S-SOM [\cite{doan2021s}] や類似度比較 [\cite{SATOTakuto20212021-047}]，HAC+k-medoids と medoid 表現 [\cite{winderlich2024classification}] の知見を，操作上の一貫性・比較可能性を担保した単一実装に橋渡し．

本枠組みは，気圧配置の運用的分類・検索への適用（例：[\cite{木村広希 2009 サポートベクターマシンを用いた気圧配置検出手法の提案}]）や，気圧配置と天気・空気質の関係可視化（例：[\cite{jiang2013classification}]）の双方に資する汎用プラットフォームを提供する．

## 2. 関連研究

### 2.1 総観パターン分類：主観・客観・ハイブリッド

ヨーロッパの主観分類（Lamb 型，Grosswetterlagen）から，客観的な相関法・和平方和法・クラスタリング・PCA に至るまで，総観分類には多様な系譜がある．自動分類では k-means や SOM が広く用いられ，Philippopoulos ら [\cite{philippopoulos2014performance}] は南東欧の春季 MSLP で SOM と k-means を比較し，SOM が非線形構造の把握と隣接ノードの位相的連続性に優れると報告した．ニュージーランド域では Jiang ら [\cite{jiang2013classification}] が 25 型 SOM 分類を構築し，SOM 平面上で局地気象・空気質（NOx, O$_3$）の空間分布を可視化して，総観—局地の連関理解に資することを示した．日本域ではタカスカほか [\cite{takasuka2024}] がバッチ型 SOM により気圧配置の代表パターンを抽出し，1km メッシュ天気と整合的な分布を提示している．教師あり分類の系譜として，木村ほか [\cite{木村広希 2009 サポートベクターマシンを用いた気圧配置検出手法の提案}] は SVM により主要 6 型の自動抽出と検索を実装したが，大量ラベリングのコストと主観ばらつきが課題として残る．

### 2.2 SOM と距離（類似度）指標

従来 SOM は BMU 探索に EUC を用いるが，格子場の比較において EUC や MSE は構造差（形状・位置・コントラスト）に鈍感である [\cite{doan2021s}]．Doan らの S-SOM [\cite{doan2021s}] は BMU に SSIM を導入し，四季・ノード数を跨いでシルエット係数とトポロジ誤差の改善を示した．一方，Sato and Kusaka [\cite{SATOTakuto20212021-047}] は，教師データにノイズが混入しても S1 と SSIM が平均・最大の選択率で優位であり，主観的な地上天気図の「似ている」を EUC より良く再現することを統計的に示した．このことは，勾配評価（S1）と構造評価（SSIM）を併用・融合する設計指針を支持する．

### 2.3 構造類似度と medoid 表現

Winderlich ら [\cite{winderlich2024classification}] は，(i) 混合符号データに適用可能な改良 SSIM，（ii）HAC+k-medoids の反復（二段階）クラスタリング，（iii）クラス中心の centroid ではなく medoid 表現，を組み合わせ，クラス分離性・時間安定性・空間解像度頑健性・物理解釈性を満たす天気型分類を構築した．さらに，CMIP6 歴史実験の循環表現を，頻度・遷移・持続の確率分布に対する Jensen–Shannon 距離で総合評価する枠組みを提示している．本研究は HAC ではなく SOM を基盤に据え，SSIM・S1・EUC 等の代替指標を SOM の BMU 探索に一貫実装し，SOM 特有のトポロジ保持と「地図化」の利点を保ちながら，medoid/true-medoid 出力で代表性の劣化（centroid のぼけ）を抑える点に特徴がある．

### 2.4 本研究の位置付けとギャップ

先行研究は，（a）SOM の BMU に SSIM を用いる効果 [\cite{doan2021s}]，（b）類似度指標の統計比較における S1・SSIM の優位 [\cite{SATOTakuto20212021-047}]，（c）改良 SSIM と medoid による高分離クラスタリング [\cite{winderlich2024classification}]，（d）SOM による総観—局地（天気・空気質）連関の可視化 [\cite{jiang2013classification}]，をそれぞれ個別に示してきた．しかし，SOM という同一基盤上で EUC/SSIM5/S1/融合を横並びに比較し，かつ medoid 表現を SOM の標準出力として実用化した研究は見当たらない．また，国内の最新 SOM 応用 [\cite{takasuka2024}] では距離指標の詳細比較や medoid 化は扱われていない．本研究は，GPU 最適化 Batch-SOM を共通プラットフォームとして，距離（類似度）指標の厳密比較と medoid 出力を同時に実現し，東アジア SLP に対する学習・独立検証で汎化性能を定量評価することで，これらのギャップを埋める．

## 3. 提案手法

### 3.1 GPU 最適化 Batch-SOM

本研究では，大規模データに対応可能な GPU 最適化 Batch-SOM を実装した．学習アルゴリズムは以下の通りである：

1. ミニバッチ単位で BMU（Best Matching Unit）を探索
2. 近傍関数 \( h\_{ij} \) を用いて各ノードの更新量を蓄積
3. エポック終了時に一括更新

近傍関数の幅 \( \sigma \) は反復回数に応じて漸減させ，最終的な収束を促進する．

### 3.2 距離指標の実装

以下の 4 種類の距離指標を実装し，BMU 探索に使用した：

#### 3.2.1 ユークリッド距離（EUC）

$$
d_{\mathrm{EUC}}(x,w) = \sqrt{\sum_{s\in\Omega} \bigl(x(s)-w(s)\bigr)^2 + \varepsilon}
$$

ここで， \( \Omega \) は格子領域， \( \varepsilon = 10^{-12} \) は数値安定化項である．

#### 3.2.2 5×5 移動窓 SSIM（SSIM5）

Doan et al. [\cite{doan2021s}] の仕様に準拠し，局所窓での評価を行う：

$$
\mathrm{SSIM}_{5\times5}(x,w) = \frac{1}{|\Omega|}\sum_{s\in\Omega} \mathrm{SSIM}_{\mathrm{loc}}(s)
$$

ここで， \( \mathrm{SSIM}\_{\mathrm{loc}}(s) \) は画素 \( s \) を中心とする 5×5 窓での局所 SSIM 値である．

$$
d_{\mathrm{SSIM5}}(x,w) = 1 - \mathrm{SSIM}_{5\times5}(x,w)
$$

#### 3.2.3 Teweles-Wobus スコア（S1）

水平勾配の差を評価する気象学的指標：

$$
S1(x,w) = 100 \times \frac{\sum_{(i,j)\in E_x \cup E_y} \left| \Delta x_{ij} - \Delta w_{ij} \right|}
{\sum_{(i,j)\in E_x \cup E_y} \max\!\left(\left|\Delta x_{ij}\right|, \left|\Delta w_{ij}\right|\right) + \varepsilon}
$$

#### 3.2.4 S1 と SSIM5 の融合（S1+SSIM5）

各距離を正規化後，等重み平均：

$$
d_{\mathrm{S1+SSIM5}} = 0.5 \times \tilde{d}_{\mathrm{S1}} + 0.5 \times \tilde{d}_{\mathrm{SSIM5}}
$$

### 3.3 Medoid 表現の導入

SOM の各ノードに対して，以下の 3 種類の代表を出力：

- Centroid：ノードに割り当てられたサンプルの平均
- Medoid：centroid に最も近い実サンプル
- True medoid：ノード内の全サンプルとの総距離が最小となる実サンプル

## 4. 実験設定

### 4.1 データと前処理

- 物理量：海面更正気圧（SLP）
- 領域：東アジア域（15–55°N, 115–155°E）
- 期間：学習 1991–1999 年，検証 2000 年（日次 09JST）
- ラベル：参考文献 [\cite{吉野 2002 日本の気候}] の付録 B「気圧配置ごよみ」準拠の 15 基本型
- 前処理：Pa→hPa 変換，空間平均の差し引き（偏差化）

### 4.2 SOM 設定

- マップサイズ：10×10
- 反復回数：1,000
- バッチサイズ：128
- 初期近傍幅：3.0
- 学習率：1.0

### 4.3 評価指標

- Macro Recall：基本ラベル（15 種）および複合ラベルでの平均再現率
- NodewiseMatchRate：ノード多数決ラベルと medoid ラベルの一致率

## 5. 実験結果

### 5.1 総合性能比較

表 1 に各距離指標の Macro Recall 結果を示す．

| 距離指標 |  学習:基本 |  学習:複合 |  検証:基本 |  検証:複合 |
| -------- | ---------: | ---------: | ---------: | ---------: |
| EUC      |     0.2544 |     0.1863 |     0.1958 |     0.1595 |
| SSIM5    |     0.2960 |     0.2060 |     0.2531 |     0.1917 |
| S1       | **0.3466** | **0.2329** |     0.2754 |     0.1850 |
| S1+SSIM5 |     0.3443 |     0.2361 | **0.3501** | **0.2157** |

学習期間では S1 が最高性能（0.3466）を示したが，検証期間では S1+SSIM5 が最良（0.3501）となった．SSIM5 は局所構造の評価により，EUC を上回る性能を示した．

### 5.2 ラベル別性能

主要なラベルの再現率を以下に示す：

- 高再現率：ラベル 1（冬型，0.878–0.951），3B（西風系，0.736–0.895）
- 中程度：4A（移動性高気圧，0.4–0.8，手法により変動）
- 低再現率：2B，2C，3C，3D，5（境界曖昧），6A–6C（サンプル希少）

### 5.3 代表性評価

NodewiseMatchRate の結果：

- S1：47.0%（47/100 ノード一致）
- S1+SSIM5：40.0%（40/100 ノード一致）
- SSIM5：30.3%（30/99 ノード一致）
- EUC：34.0%（34/100 ノード一致）

S1 系の指標で高い一致率を示し，medoid 表現の妥当性が確認された．

## 6. 考察

### 6.1 距離指標の特性

S1 は勾配情報を直接評価するため，前線帯や冬型など勾配構造が支配的なパターンで優れた性能を示した．一方，SSIM5 は局所的な構造類似性を評価し，台風など形状・位置が重要なパターンに有効であった．S1+SSIM5 の融合により，両者の長所を活かし，汎化性能が向上したと考えられる．

### 6.2 Medoid 表現の有効性

図 1 に示すように，centroid では平均化により気圧勾配が平滑化される傾向があったが，medoid および true-medoid では実際の気圧パターンの鋭い特徴が保持された．これにより，現業での解釈性と事例検索への応用可能性が高まった．

### 6.3 限界と今後の課題

本研究では単一変数（SLP）のみを用いたが，梅雨型や台風型など複雑なパターンの識別には，風ベクトルや温度場等の多変量化が必要である．また，季節正規化や面積重みの導入により，さらなる性能向上が期待される．

## 7. まとめ

本研究では，GPU 最適化 Batch-SOM を基盤として，4 種類の距離指標を厳密に比較する実験環境を構築した．主な成果は以下の通りである：

1. 検証期間において S1+SSIM5 が最高の Macro Recall（0.3501）を達成
2. SSIM5 が局所構造評価により有効性を確認
3. Medoid 表現により解釈性と代表性が向上

本研究は，Doan et al. [\cite{doan2021s}] の S-SOM，Sato and Kusaka [\cite{SATOTakuto20212021-047}] の類似度比較，Winderlich et al. [\cite{winderlich2024classification}] の medoid 表現の知見を，SOM を基盤とする統一フレームワークへと橋渡しするものである．今後は，多変量化や季節別学習，Jensen–Shannon 距離による循環表現評価等への拡張を予定している．

## 謝辞

本研究の一部は，JSPS 科研費の助成を受けて実施された．

---

参考文献は LaTeX の \`ref.bib\`（\`\\bibliography{ref}\`）に基づく引用キーを本文中に保持している（例：\`\\cite{...}\`）．必要に応じて文献リストを手動で追記すること．
