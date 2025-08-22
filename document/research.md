# 類似度指標とクラスタリング手法の比較を通じた気圧配置パターン分類の高度化（v3）

本稿は、総観場（海面更正気圧：SLP）の類似度指標とクラスタリング手法を比較し、気圧配置パターン分類の精度・頑健性・解釈性を同時に高めるための研究計画と実装設計（v3）をまとめたものである。v3 では、先行研究で示された SSIM/S1 類似度や medoid 表現、二段階クラスタリングの知見を踏まえ、GPU 最適化した Batch‑SOM に対して 5 種の距離（EUC/SSIM/SSIM5/S1/S1+SSIM）を同一条件で比較できる実験基盤を整備した。結果・結論・考察は本稿では扱わず、背景・目的・方法・実装・評価設計を記述する。

---

## 先行研究

### 1) 木村ほか (2009, DEIM Forum) ― SVM による自動分類

- 目的：目視で行っていた「西高東低」「谷」「移動性高気圧」「前線」「南高北低」「台風」を SVM で自動化。
- データ：JRA‑25 SLP（約 1.25°）、日本周辺、09JST、吉野（2002）のラベル。
- 手法：型ごとの二値 SVM（RBF）。TinySVM を使用。
- 含意：大量抽出の自動化は有効。多様性の大きい型（台風）やラベル主観・移行/複合型対応が課題。教師なし・構造敏感類似度の検討動機となる。

### 2) Takasuka et al. (2024) ― SOM と 1km 推計天気

- 目的：教師なし（Batch‑SOM）で気圧配置の代表パターンを抽出し、全国 1km 推計天気と対応付けて解釈。
- データ：GPV 全球 SLP（2016/3–2024/10・09JST）、15–55N/115–155E。PCA→SOM（10×10）。
- 含意：SOM は連続的な型空間を与え、地上天気との結び付きが視覚的に明瞭。一方、距離が EUC で構造差の捉え方に改善余地。台風の多様性に課題。

### 3) Philippopoulos et al. (2014) ― SOM と k‑means の比較（南東欧の春）

- 手法：PCA→k‑means と SOM（EUC）で MSLP を分類。SOM は非線形構造・トポロジ保存の可視化が強み。距離指標と型数固定の限界も示唆。

### 4) Jiang, Dirks & Luo (2013) ― SOM × 気象・大気質の可視化（NZ）

- 目的：Z1000 による 25 型 SOM と Auckland の局地気象/大気質の SOM 平面投影。
- 含意：SOM は「型 → 局地気象 → 大気質」の関係地図化に有効。ただし距離は EUC で、構造敏感な類似度への拡張余地。

### 5) Winderlich et al. (2024, ESD) ― 改良 SSIM と二段階クラスタリング

- 目的：改良 SSIM を類似度に用い、HAC→k‑medoids を反復してクラス数を閾値で自動決定。代表は centroid でなく medoid（実データ）。
- 含意：分離・安定性・代表性を満たし、希少型も保持。medoid 表現の解釈性が高い。JS 距離でモデルの循環表現を総合評価可能。

### 6) Doan et al. (2021, GMD) ― S‑SOM（BMU に SSIM）

- 目的：SOM の BMU 探索距離を ED から SSIM に置換。
- 含意：ED‑SOM よりシルエット係数が高く、トポロジ保存（TE）も改善。SSIM は 2 次元の構造的類似（形・位置・コントラスト）に敏感。

### 7) Sato & Kusaka (2021, JMSJ) ― 類似度指標の統計比較（SLP）

- 目的：COR/EUC/S1/SSIM/aHash の選別性能を網羅比較（CSoJ 抽出）。
- 含意：平均・最大の選択率で S1 と SSIM が最良。ノイズを含む教師にも頑健。EUC は構造差に弱い。

**まとめ（本研究への接続）**

- 構造を持つ SLP の類似は、L2/MSE より SSIM（構造）/S1（勾配）が人の判断との整合性・頑健性で優位。
- SOM の可視化力を活かしつつ、BMU 類似度を EUC/SSIM/S1 で比較し、分類精度と解釈性を両立。
- Winderlich の medoid 表現を可視化に導入し、平均化（centroid）の“ぼけ”を補う。

---

## 要旨（Abstract）

本稿は、総観場（海面更正気圧：SLP）の類似度指標とクラスタリング手法を体系的に比較し、気圧配置パターン分類の精度・頑健性・解釈性を同時に向上させるための研究計画と実装設計（v3）を提示する。先行研究が指摘した L2/MSE 距離の限界（構造無視）に対し、構造類似度 SSIM と勾配ベース S1 の優位性（Doan 2021; Sato & Kusaka 2021）、ならびに medoid 表現と SSIM 閾値ベースのクラスタリング安定性（Winderlich 2024）を取り入れた。GPU 最適化 Batch‑SOM 上で 5 種の BMU 距離（EUC/SSIM/SSIM5/S1/S1+SSIM 融合）を同一条件で比較可能な実験基盤を構築し、学習内・年別検証の双方において、分類性能（Macro Recall）、代表性（NodewiseMatchRate）、量子化誤差（QE）、ラベル分布可視化、centroid 対 medoid の比較などを多面的に評価する。結果・結論・考察は本稿では提示せず、背景、目的、関連研究、方法、実装、評価設計、再現性計画、限界と今後の拡張のみを記述する。

---

## 1. はじめに（Introduction）

気圧配置の客観分類は、気象研究・気候モデル評価・業務検索や事例抽出など幅広い応用に不可欠である。従来、教師あり学習（SVM 等）によるラベル予測（木村ほか, 2009）や、教師なし学習（SOM 等）による型空間の自動抽出（Jiang 2013; Philippopoulos 2014; 髙須賀ほか 2024）が実用的な選択肢として活用されてきた。しかし、以下の課題が残る。

- **距離指標の限界**：ユークリッド距離（MSE）はパターンの「形・位置・コントラスト」の差異に鈍感で、平均化により代表型が“ぼける”問題を助長する（Doan 2021; Wang & Bovik 2009）。
- **ラベル主観と多様性**：人手ラベルは主観や不一致に晒され、移行/複合型や台風のような多様パターンで誤分類が生じやすい（木村ほか 2009; 髙須賀ほか 2024）。
- **代表性の欠如**：クラスタ中心を平均（centroid）で表すと極端イベントの勾配や中心位置が弱まり、実務解釈が難しくなる（Winderlich 2024）。

これらの課題に対して、**構造的類似度 SSIM** と **勾配類似 S1** は、視覚/気象学的な意味で妥当な近さを与えることが示され（Doan 2021; Sato & Kusaka 2021）、さらに **medoid（最も代表的な実サンプル）** を用いたクラスタ代表は平均化の問題を避け、解釈性・安定性を高める（Winderlich 2024）。本研究では、これらの知見を統合し、GPU 最適化 Batch‑SOM を基盤に **5 種の距離指標** を比較することで、分類・代表性・可視化のバランスに優れた気圧配置分類フレームワークを設計する。

---

## 2. 目的および研究課題（Objectives & Research Questions）

### 2.1 目的

1. **距離指標の比較基盤の構築**：GPU 最適化 Batch‑SOM 上で、EUC/SSIM/SSIM5/S1/S1+SSIM の 5 距離を同一実装・同一データ前処理で比較する環境を整備する。
2. **多面的評価**：学習内と年別検証の双方で、分類性能（Macro Recall：基本/複合）、代表性（NodewiseMatchRate）、量子化誤差（QE）、ラベル分布、centroid vs medoid 可視化を用いて距離指標を評価する。
3. **代表性の強化**：medoid と true medoid（ノード内総距離最小）を併記し、SOM の平均化に伴う“ぼけ”を補完、現場解釈に資する可視化を標準出力とする。
4. **汎化評価手順の確立**：学習年に得た「ノード多数決（基本ラベル）」を検証年へ適用し、年跨ぎのラベル予測（年別推論）手順を確立する。

### 2.2 研究課題（RQs）と仮説（H）

- **RQ1**：EUC/SSIM/SSIM5/S1/S1+SSIM 間で、Macro Recall（基本/複合）はどのように異なるか？  
  **H1**：SSIM 系および S1 は EUC より高い Macro Recall を与える（Sato & Kusaka 2021）。
- **RQ2**：medoid（closest‑to‑centroid / true medoid）による代表性は、ノード多数決 raw ラベルとの一致率（NodewiseMatchRate）でどの程度担保されるか？  
  **H2**：非ユークリッド距離（SSIM/S1）では medoid が多数決と整合しやすく、centroid の“ぼけ”を補う（Winderlich 2024）。
- **RQ3**：学習年のノード代表（基本）を検証年に適用する際の汎化性能は距離により差が出るか？  
  **H3**：SSIM/S1 は構造・勾配を捉えるため、年別検証でも頑健性を示す（Doan 2021）。

---

## 3. 関連研究（Related Work）

- **教師あり**：木村ほか（2009）は SVM による 6 型抽出（RBF カーネル）を提案し、F 値の有用域と台風などの難しさを示した。
- **教師なし（SOM/k‑means）**：Philippopoulos（2014）は SOM の非線形・トポロジ保存の可視化優位を報告。Jiang（2013）は SOM 平面に局地気象・大気質を投影し、関係地図化の有効性を示した。
- **改良距離**：Doan（2021）は BMU に SSIM を導入（S‑SOM）し、ED よりクラスタ純度・トポロジ保存を改善。Sato & Kusaka（2021）は S1/SSIM が選別性能で最良と統計的に示した。
- **二段階クラスタリング**：Winderlich（2024）は改良 SSIM と HAC→k‑medoids により、medoid 代表・希少型保持・空間/時間安定性・ JS 距離によるモデル評価の枠組みを提示した。

---

## 4. データと前処理（Data & Pre‑processing）

- **物理量**：海面更正気圧（SLP, prmsl）
- **期間**：学習 1991‑01‑01〜1999‑12‑31、検証 2000‑01‑01〜2000‑12‑31（各日 09JST 相当）
- **領域**：115–155E, 15–55N（日本周辺）
- **ラベル**：吉野（2002）準拠の 15 基本ラベル  
  `1, 2A, 2B, 2C, 2D, 3A, 3B, 3C, 3D, 4A, 4B, 5, 6A, 6B, 6C`  
  （複合表記も許容。NFKC 正規化とトークナイズで基本成分を抽出）
- **前処理**：Pa→hPa、サンプル毎の空間平均差し引き（偏差化）。  
  ※季節正規化（平均/標準偏差）・面積重みは本比較では未適用（拡張項目）。

---

## 5. 提案枠組み（Methods）

### 5.1 GPU Batch‑SOM（minisom.MiniSom）

- **学習**：ミニバッチ蓄積 → 一括更新（BMU→ 近傍重み h→ 分子/分母を加算 → 更新）。総反復に対し σ は漸減（asymptotic/linear 切替可）。
- **BMU 距離（5 種）**：
  - **EUC**：ユークリッド距離（ベースライン）
  - **SSIM**：全体 1 窓の SSIM（C1=C2=1e‑8）
  - **SSIM5**：5×5 移動窓・C=0（Doan 2021 の仕様に近い）
  - **S1**：Teweles–Wobus（水平/南北勾配の差の正規化比）
  - **S1+SSIM**：S1 と SSIM5 をサンプル毎に min‑max 正規化後に等重み合成
- **medoid 置換**：任意間隔で各ノード重みを「距離的に最近傍の実サンプル」へ置換（距離一貫性・代表性を向上）。

### 5.2 可視化と代表性

- **centroid**（平均）と **medoid**（closest‑to‑centroid）を併記し、“ぼけ”を補う。
- **true medoid**（ノード内総距離最小）も出力し、代表性の頑健性を確認。
- **NodewiseMatchRate**：ノード多数決 raw と medoid raw の一致率を算出し、背景色（緑/赤）で可視化。

### 5.3 年別推論（Generalization）

- 学習年の「ノード多数決（基本ラベル）」を予測辞書として検証年に適用し、基本/複合の Macro Recall を算出する。

---

## 6. 実装（Implementation）

### 6.1 main_v4.py の構成

- **データ入出力**：`load_and_prepare_data_unified()`（時系列スライス・偏差化・ラベル/座標の取り扱い）
- **学習**：`run_one_method_learning()`（反復分割・QE/Recall/MatchRate の履歴保存、各種 PNG/CSV 出力）
- **検証**：`run_one_method_verification()`（年別推論・混同行列・per‑label 再現率・可視化出力）
- **ユーティリティ**：ラベル正規化、混同行列作成、メトリクス CSV/PNG、ノード詳細ログ、可視化器群

### 6.2 minisom.py の構成

- **距離**：`euclidean` / `ssim` / `ssim5` / `s1` / `s1ssim` の高速実装（ノード分割 `nodes_chunk` で VRAM 最適化）
- **学習**：`train_batch()`（σ 漸減、メドイド置換、固定評価サブセットで QE 安定化）
- **推論**：`predict()`（BMU 座標）
- **評価**：`quantization_error()`（距離タイプに依存）

### 6.3 ハイパーパラメータ（既定）

- SOM 10×10、反復数 `NUM_ITER`（分割評価 `SOM_EVAL_SEGMENTS`）、バッチ 256、`nodes_chunk` 2–4（VRAM 依存）
- 評価サンプル上限 4000、ログ間隔 10、GPU/CPU 自動選択、乱数シード固定

---

## 7. 評価設計（Evaluation Protocol）

1. **学習内評価**  
   反復分割ごとに QE、Macro Recall（基本/複合）、NodewiseMatchRate をロギング。centroid/medoid/true‑medoid を出力。
2. **年別検証**  
   学習年のノード多数決（基本）を用いて検証年の基本/複合 Macro Recall を算出。混同行列・ per‑label 棒グラフ・割当 CSV を出力。
3. **可視化**
   - ラベル分布ヒートマップ（全体/検証、基本 15 種）
   - centroid vs medoid（希少パターンや鋭い勾配の保持状況を比較）
   - ノード詳細（構成・純度・月別分布）
4. **統計的検定（任意）**  
   距離間の Macro Recall をノード/日付単位で対応づけ、ノンパラメトリック検定（例：Wilcoxon）で優劣を検討（Sato & Kusaka 2021 に準拠）。
5. **将来拡張の評価**
   - シルエット係数・トポロジカルエラー（Doan 2021）
   - Jensen–Shannon 距離（Winderlich 2024；頻度・遷移・持続）

> 本稿では評価設計のみを記し、数値結果・図表・考察は提示しない。

---

## 8. 再現性計画（Reproducibility Plan）

- **コード**：`main_v4.py` / `minisom.py` に集約。乱数・BLAS スレッド・PyTorch の deterministic を固定。
- **入出力**：NetCDF（ERA5 派生 prmsl）を `DATA_FILE` から読み込み。結果は `results_v4` 以下に自動保存（CSV/PNG/JSON/LOG）。
- **環境**：Python 3.x、PyTorch（CUDA/CPU 対応）、Cartopy/Matplotlib、xarray/pandas。VRAM 16–24GB を推奨（`nodes_chunk` で調整可）。
- **パラメータ**：SOM サイズ・反復・バッチ・距離タイプは定数で管理。ログに全設定を出力。
- **データ配布**：入力データの配布条件（ERA5/再解析のライセンス）を遵守し、前処理スクリプトを公開する。

---

## 9. 限界と想定される脅威（Limitations & Threats to Validity）

- **前処理**：季節正規化・面積重みを未適用のため、季節振幅/緯度面積の影響が残存しうる。
- **ラベル**：複合・主観・不一致の影響。基本/複合の両評価を設けるが、真の境界は曖昧。
- **汎化**：1990s→2000 年の年別推論に依存。年代の拡張・季節別学習の検討が必要。
- **距離の実装差**：SSIM の定数・窓設定、S1 の差分スキームなど実装差の影響に留意。
- **計算資源**：SSIM5/S1 は EUC より計算負荷が高い。`nodes_chunk` により緩和する。

---

## 10. 今後の拡張（Future Work）

- **前処理強化**：日別季節正規化・面積重み・高緯度安定化。
- **指標追加**：シルエット・TE・JS 距離の導入と報告テンプレート化。
- **自動クラスタ**：SSIM 閾値の HAC→k‑medoids（Winderlich）や SOM とのハイブリッド。
- **多変量化**：風ベクトル・相当温位・渦度傾度・地衡風・温度偏差の統合による梅雨・台風の識別力強化。
- **季節別/領域別運用**：季節別 SOM、東アジア/北半球への拡張と汎化検証。
- **オープンサイエンス**：コード・設定・前処理・出力テンプレを公開し、再現性パッケージ化。

---

## 出力成果物（例：メソッドごとに `results_v4/learning_result/{method}_som` 配下）

- `*_iteration_metrics.csv/png`：QE・MacroRecall・NodewiseMatchRate の学習推移。
- `*_assign_all.csv`：学習期間の BMU 割当（日時・BMU・raw ラベル）。
- `*_som_node_avg_all.png`：centroid（ノード平均）大図。
- `*_pernode_all/*.png`：ノード平均の個別図。
- `*_som_node_medoid_all.png`：medoid（closest‑to‑centroid）大図。
- `*_pernode_medoid_all/*.png`：medoid 個別図。
- `*_som_node_true_medoid_all.png`：true medoid（総距離最小）大図。
- `*_pernode_true_medoid_all/*.png`：true medoid 個別図。
- `*_node_medoids.csv` / `*_node_true_medoids.csv`：medoid メタ情報（ノード座標、日付、raw/基本ラベル、距離）。
- `*_nodewise_analysis_match.png`：多数決 raw vs medoid raw の一致可視化（緑/赤）。
- `*_confusion_matrix_all.csv`：学習データの混同行列（基本ラベル）。
- `*_label_dist_all/*.png`：ラベル分布ヒートマップ（全体/個別）。
- `node_majorities.json`：学習時ノード多数決（raw/基本）の辞書（検証用）。

**検証（`results_v4/verification_results/{method}_som` 配下）**

- `*_verification_confusion_matrix.csv`、`*_verification_per_label_recall.csv/png`：検証年の基本/複合再現率。
- `*_verification_assign.csv`：検証期間の BMU 割当（予測・正誤フラグ含む）。
- `*_verification_node_avg.png`：検証データの centroid 図。
- ラベル分布ヒートマップ（検証）。

---

## 今後の拡張（設計上のオプション）

- **前処理**：日別の季節正規化（平均・標準偏差）と面積重み（緯度依存）を導入し、季節振幅・緯度面積のバイアスを低減。
- **評価指標**：シルエット係数・トポロジカルエラー（Doan 2021）や Jensen–Shannon 距離（Winderlich 2024；頻度・遷移・持続）を追加。
- **自動クラスタ**：SSIM しきい値に基づく HAC→k‑medoids（Winderlich）の導入や、SOM とのハイブリッド運用。
- **多変量化**：風ベクトル・相当温位・渦度傾度・気温偏差などを併用し、梅雨・台風など多様パターンの識別力を強化。
- **季節別学習・領域拡張**：季節別 SOM、領域の可変化（国内/東アジア/北半球）による一般性の検証。

---

## 参考文献

1. 木村広希, 川島英之, 日下博幸, & 北川博之. (2009). サポートベクターマシンを用いた気圧配置検出手法の提案 ── 西高東低冬型を対象として ──. 地理学評論 Series A, 82(4), 323-331.

2. 高須賀 匠, 高野 雄紀, 渡邊 正太郎, & 雲居 玄道. (2024). 自己組織化マップを用いた気圧配置のクラスタリングと 1km メッシュ天気データによる分析. 情報処理学会全国大会.

3. Philippopoulos, K., Deligiorgi, D., & Kouroupetroglou, G. (2014). Performance comparison of self-organizing maps and k-means clustering techniques for atmospheric circulation classification. methods, 13, 14.

4. Jiang, N., Dirks, K. N., & Luo, K. (2013). Classification of synoptic weather types using the self-organising map and its application to climate and air quality data visualisation. Weather and Climate, 33, 52-75.

5. Winderlich, K., Dalelane, C., & Walter, A. (2024). Classification of synoptic circulation patterns with a two-stage clustering algorithm using the modified structural similarity index metric (SSIM). Earth System Dynamics, 15(3), 607-633.

6. Doan, Q. V., Kusaka, H., Sato, T., & Chen, F. (2021). S-SOM v1. 0: a structural self-organizing map algorithm for weather typing. Geoscientific Model Development, 14(4), 2097-2111.

7. Sato, T., and H. Kusaka, 2021: Statistical intercomparison of similarity metrics in sea level pressure pattern classification. J. Meteor. Soc. Japan, 99, 993–1001, doi:10.2151/jmsj.2021-047

**付録（用語）**

- **SSIM**：構造類似度。輝度・コントラスト・構造（共分散）を同時に比較。
- **S1**：Teweles–Wobus スコア。水平方向/南北方向の勾配差を正規化比で評価。
- **medoid**：クラスタ内の「最も代表的な実サンプル」。centroid と異なり平均化の影響を受けにくい。
