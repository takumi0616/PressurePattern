# 類似度指標とクラスタリング手法の比較を通じた気圧配置パターン分類の高度化

本稿は、総観場（海面更正気圧：SLP）の類似度指標とクラスタリング手法を比較し、気圧配置パターン分類の精度・頑健性・解釈性を同時に高めるための研究計画と実装設計（v3）をまとめたものである。v3 では、先行研究で示された SSIM/S1 類似度や medoid 表現、二段階クラスタリングの知見を踏まえ、GPU 最適化した Batch‑SOM に対して 5 種の距離（EUC/SSIM/SSIM5/S1/S1+SSIM5）を同一条件で比較できる実験基盤を整備した。本文では背景・目的・方法・実装・評価設計に加え、§9–§11 に主要な実験結果・考察・まとめを併記する。

---

## 先行研究

### 1) 木村ほか (2009, DEIM Forum) ― SVM による気圧配置の自動分類と検索

- 目的：研究者が目視で収集していた「西高東低冬型」「気圧の谷型」「移動性高気圧型」「前線型」「南高北低夏型」「台風型」の判別を、機械学習（SVM）により自動化。
- データ：JRA‑25 SLP（約 1.25°）、領域 15–60N/105–175E、日本時間 09JST、対象 1979–2006。気圧配置ラベルは吉野 (2002) に準拠。
- 手法：各型について「持つ/持たない」の二値 SVM を構築（RBF カーネル、TinySVM）。6 つの二値器の組合せで多クラスに対応。識別関数のしきい値をユーザが調整できる検索システムを開発（PHP/MySQL）。
- 評価：1981–2000 年を学習・検証に用い、型ごとの適合率/再現率/F 値を算出。F 値は型により差が大きく、おおむね 0.41〜0.86 の範囲。しきい値を上げると適合率 ↑・再現率 ↓ のトレードオフを確認。アンケート評価でも概ね有用との回答。
- 含意：自動抽出の実運用可能性を提示。一方で、台風の多様性・移行/複合型・ラベルの主観や誤ラベルの影響が課題。教師なし法や構造を意識した類似度の必要性を示唆。

### 2) Takasuka et al. (2024) ― SOM と全国 1km 推計天気の対応付け

- 目的：教師なし学習（Batch‑SOM）で日本周辺の気圧配置の代表パターン空間を構築し、全国 1km 推計気象分布（天気：晴/曇/雨・雪）と対応付けて解釈可能性を高める。
- データ：GPV 全球 SLP（2016/3–2024/10・09JST）、領域 15–55N/115–155E。ラベル解釈の補助に「日々の天気図」のキーワード日（冬/夏/梅雨/台風）を使用。外部評価は JRA‑3Q（1981–2000）から 100 日（各 25 例）を抽出。
- 手法：PCA（20 次元）→ SOM（10×10、近傍 3.0、学習率 1.0、反復 100,000）→ ノードごとに対応日の 1km 天気を多数決で集計。
- 評価：学習済み SOM での分類正解率（外部評価 100 日）は冬型 100%、夏型 84%、梅雨型 76%、台風 52%、平均 78%。SOM 平面上で隣接ノードほど類似パターンが並ぶトポロジ保存を確認。
- 含意：SOM は連続的な型空間と地上天気の関係を可視化する上で有効。距離がユークリッド（EUC）であるため、構造差（位置・形・コントラスト）への感度向上が次の課題。台風の多様性も未解決。

### 3) Philippopoulos et al. (2014) ― SOM と k‑means の比較（南東欧の春季）

- 目的：春季（1948–2009）の南東欧における MSLP を PCA→k‑means と SOM（EUC）で分類し、両者の性能と特性を比較。
- データ：NCEP/NCAR Reanalysis 1、2.5°、領域 30–60N/10W–37.5E、対象 5704 日。
- 手法：PCA で次元圧縮後、k‑means（10 型）と SOM（5×4=20 型）でクラスタリング。月別頻度や型対応で比較。
- 含意：SOM は非線形構造を捉え、隣接ノードに類似型が配列されるため、遷移や類縁関係の可視化に強み。k‑means は局所最適やセントロイド平均による「ぼけ」が課題。距離が EUC で構造差の取り扱いに限界。

### 4) Jiang, Dirks & Luo (2013, Weather & Climate) ― SOM × 局地気象/大気質の可視化（NZ・オークランド）

- 目的：ニュージーランド域の 1000 hPa geopotential height から SOM（25 型）で新たな客観分類を作成し、オークランドの局地気象（空港）・大気質（NO/NO₂/O₃）を SOM 平面に投影して関係を「地図化」。
- データ：NCEP/NCAR、0/12UTC、1958–2010、領域 25S–55S/155E–170W（195 グリッド）。二段階 Batch‑SOM を採用。局地データは 1962–2009（気象）、1999–2009/2004 の大気質。
- 主な知見：SOM 平面の左（高気圧/ブロッキング）〜右上（トラフ）へ自己組織化。O₃ はトラフ・西風卓越型で高く、ブロッキングでは海洋空気の流入で地上（Musick Point）は高/上層（Sky Tower）は低など、地点差も含む物理機構（下降混合・換気・海風 advection）を可視化。
- 含意：SOM は「型 → 局地気象 → 大気質」を俯瞰でき、環境アセスやヘルススタディにも応用可能。距離が EUC のため、構造敏感な指標への拡張余地。

### 5) Winderlich et al. (2024, Earth Syst. Dynam.) ― 改良 SSIM と HAC×k‑medoids の二段階クラスタリング

- 目的：伝統的な距離（MSE/EUC）では捉えにくい「構造的」類似性を重視し、改良 SSIM を用いた新しい気圧場分類と、CMIP6 モデル評価への展開。
- データ/領域：ERA‑Interim の Z500（1979–2018）、2°×3° サンプリング、欧州域。補助として NCEP1 と 32 モデル（CMIP6）。
- 手法：改良 SSIM（混合符号データ対応、c1=c2=1e‑8）で類似度を計算。クラスタリングは HAC（類似度しきい値 THmerge でマージ）→k‑medoids（プロトタイプは実データ）を反復。クラス数は THmerge により自動的に決定（例：TH=0.40→37 型、0.425→52 型、0.45→89 型）。
- 評価：一貫性（パラメタ摂動でのクラスの安定進化）、分離性（EV/DRATIO/SSIMRATIO）、時間・空間安定性（30–40 年・解像度 1°〜6° で頑健）、代表性（メドイドとセントロイドが強い SSIM=0.6 以上）を満たす。まれな型も保持。
- 応用：参照分類に対するモデル出力の「頻度・遷移・持続」確率分布の Jensen–Shannon 距離（JS）で総合評価。代替再解析（NCEP1）の JS=0.034 を下限目安に、CMIP6 は平均 ~0.11（最悪 ~0.137）など、循環表現の質を定量化。
- 含意：構造敏感な類似度（SSIM）＋メドイド表現＋二段階クラスタリングにより、「分離・安定・代表性」を両立。評価指標（JS）は従来のスカラ指標を補完。

### 6) Doan et al. (2021, GMD) ― S‑SOM（BMU 探索に SSIM を採用）

- 目的：SOM の BMU 検索距離を ED から SSIM に置換し、構造情報（形・位置・コントラスト）を反映できる SOM（S‑SOM）を提案。
- データ：ERA‑Interim MSLP（1979–2019）、日本域（20–50N/115–165E）。季節別に検証。
- 評価指標：クラスタ品質（シルエット係数）とトポロジ保存（Topological Error, TE）。
- 結果：S‑SOM は ED‑SOM より一貫して高いシルエット係数、低い（良い）TE を示し、COR‑SOM よりも概して良好。季節性の捉え方も S‑SIM/COR は ED より鋭敏。計算時間は ED 比 10–15 倍だが絶対時間は <1 分規模。
- 含意：SOM の距離置換だけで品質が向上。Doan の結果は「SOM×SSIM」の有効性の実証であり、SOM を本研究でも距離設計込みで再検討する動機となる。

### 7) Sato & Kusaka (2021, JMSJ) ― 類似度指標の統計比較（SLP・大規模教師を用いた頑健評価）

- 目的：CSoJ（日本海低気圧）を対象に、類似度指標 5 種（COR/EUC/S1/SSIM/aHash）の「抽出能力」を大量教師データで比較。
- データ：JRA‑55 SLP（2007–2016、6 時間毎、1.25°）。専門家 5 名の二段階ラベリングで CSoJ 328 枚、非 CSoJ 788 枚を作成。
- 手法：各 CSoJ を教師に、全候補との類似度でランキングし、上位 p 件に含まれる CSoJ 比率（Selection Rate）で性能比較。誤ラベル（非理想教師）を含む条件でも評価。
- 結果：平均/最大の選択率とも S1 と SSIM が最良（EUC/COR/aHash は劣後）。aHash（ハッシュ化）は勾配情報を失い不利。EUC は中心位置の一致には強いが、構造の主観的一致を再現しにくい。
- 含意：構造（SSIM）と勾配（S1）を重視する指標が、人間の判断との整合性・誤教師への頑健性で優位。SLP パターン類似度の設計方針を裏付け。

### 8) その他の関連動向（要点のみ）

- SOM/クラスタの気候応用：SOM による天候型の可視化とモデル評価（Cannon, 2012/2020；Fabiano et al., 2020；Hochman et al., 2021）では、型頻度・持続性の再現がモデル評価で重要視されている。
- 伝統的気圧配置カタログ：Hess–Brezowsky（GWL）、Lamb など主観分類は豊富な知見を持つが、領域依存・人手負担が課題。COST733 は客観分類の比較・整備を進め、評価指標（EV/DRATIO 等）も整理。

### まとめ（本研究への接続）

- 構造情報を持つ SLP/高度場の類似性は、L2/MSE/EUC よりも SSIM（構造）や S1（水平勾配）が人の判断・物理整合性に近く、誤ラベルにも頑健（Sato & Kusaka, 2021）。
- SOM は「連続的な型空間」と「局地気象・大気質」や「1km 天気」との結び付けの可視化に有効（Jiang, 2013；Takasuka, 2024）。Doan (2021) は BMU に SSIM を用いるだけで品質が向上することを示した。
- クラスタ代表の扱いはセントロイド平均よりメドイド（実データ）の方が解釈性と極端事例の保持で優れる（Winderlich, 2024）。また、二段階（HAC→k‑medoids）によりクラス数を自動決定しつつ、分離・安定・代表性を両立できる。
- モデル評価では、参照分類に対する「頻度・遷移・持続」の Jensen–Shannon 距離（JS）を導入すると、従来のスカラ評価を補完できる（Winderlich, 2024）。
- 以上を踏まえ本研究では、SOM の BMU 類似度を EUC/SSIM/S1 等で比較し、メドイド表現と二段階クラスタリングの利点を取り入れ、台風など多様性の大きい型や希少型を保持しつつ、分類精度・解釈性・可視化力の三立を目指す。

---

## 要旨（Abstract）

本研究は、総観規模の海面更正気圧（SLP）場に対する「類似度指標」と「クラスタリング表現」を同一基盤上で厳密比較し、気圧配置パターン分類の精度・頑健性・解釈性を同時に高めるための手法設計と実装を提示する。従来、教師ありの SVM による型判別（木村ほか, 2009）や、教師なしの SOM／k‑means による型空間抽出（Jiang, 2013；Philippopoulos et al., 2014；髙須賀ほか, 2024）が広く用いられてきたが、(i) ユークリッド距離（EUC）や MSE がパターンの形・位置・コントラスト差に鈍感であること、(ii) セントロイド（平均）表現が「代表型のぼけ」を生じること、(iii) まれな型や移行・複合型の保持・解釈が困難であることが課題として残っていた。近年、構造類似度 SSIM を用いた BMU 探索によって SOM のクラスタ品質とトポロジ保存性を向上できること（Doan et al., 2021）、ならびに SSIM や勾配ベース S1 が SLP の主観的一致度・誤教師への頑健性で優れること（Sato & Kusaka, 2021）が示されている。また、改良 SSIM を距離とし、HAC→k‑medoids によりクラス数を自動決定、代表を medoid で与える二段階クラスタリングは、分離・安定性・代表性・希少型の保持を両立し、さらに Jensen–Shannon 距離でモデルの循環表現を総合評価できる（Winderlich et al., 2024）。

本研究ではこれらの知見を統合し、GPU 最適化した Batch‑SOM を共通の足場として、BMU 距離を EUC／SSIM（1 窓）／SSIM5（5×5 移動窓）／S1／S1+SSIM5（等重み融合）の 5 方式で厳密比較できる実験基盤を構築した。前処理は Pa→hPa 変換・空間平均の差し引き（偏差化）とし、学習（1991–1999）・検証（2000）を東アジア域（15–55N, 115–155E）の日次 09JST 相当で実施する。評価は、(a) 学習内での量子化誤差（QE）・Macro Recall（基本/複合）・NodewiseMatchRate（ノード多数決 raw と medoid raw の一致率）、(b) 年別汎化（学習年のノード多数決を辞書として検証年へ適用した Macro Recall）、(c) centroid／medoid／true medoid（総距離最小）の対比可視化、(d) ラベル分布・混同行列を含む。SOM の平均化による情報劣化を補うため、出力の標準として medoid（closest‑to‑centroid）と true medoid の併記を採用した。コードは Python/PyTorch ベースで公開可能な形に整理し、ハイパーパラメータ・乱数シード・入出力仕様をログ化する等、再現性を重視した設計とした。

本稿の貢献は、(1) 構造・勾配・融合の 5 種距離を同条件・同実装で比較するはじめての SOM 実験基盤の提示、(2) medoid 表現を SOM 可視化のデフォルトに組み込み、代表性と解釈性を体系的に評価する枠組みの整備、(3) 学習内と年別汎化の二層評価プロトコルの策定である。数値結果や結論の提示は別稿に譲り、本稿では背景・目的・関連研究・方法・実装・評価設計・再現性計画・限界と今後の拡張を詳述する。これにより、Doan（2021）の S‑SOM、Sato & Kusaka（2021）の類似度比較、Winderlich（2024）の medoid/二段階クラスタリングの利点を、SOM を基礎とする統一フレームワークへと橋渡しする。

---

## 1. はじめに（Introduction）

総観規模の気圧配置は、降雪・寒波・梅雨・台風等の顕著現象から、地域気象・大気質・再エネ需給・防災業務に至るまで、広範な分野の基盤情報である。その体系的な“型”の抽出・記述・検索は、事例解析・データ駆動研究・運用の効率化に資するだけでなく、気候モデルの循環表現を評価する上でも不可欠である（Gleckler et al., 2008；Hannachi et al., 2017）。このため、客観分類の研究は、教師あり・教師なしの双方で発展してきた。

教師ありの系譜では、木村ほか（2009）が SVM により「西高東低」「谷」「移動性高気圧」「前線」「南高北低」「台風」の自動判別と検索を実装し、目視作業を大幅に省力化できることを示した。一方で、台風や移行・複合型など多様性の大きいパターンでは、ラベル主観・誤ラベルの影響や適合率–再現率のトレードオフが顕在化した。教師なしの系譜では、k‑means や SOM が広く使われ、PCA→k‑means による型群抽出（Philippopoulos et al., 2014）、SOM による自己組織化とトポロジ保存の可視化（Hewitson & Crane, 2002；Jiang, 2013）が有効性を示してきた。特に SOM は、型の連続空間を提供し、局地気象・大気質の平面投影（Jiang, 2013）や 1 km 推計天気との対応付け（髙須賀ほか, 2024）により、「型 → 地上現象」の関係を直観的に理解させる強みを持つ。

しかし、従来の多くの手法は、以下の 3 点で本質的な限界を抱える。（i）距離指標の限界：ユークリッド距離（EUC）や MSE は、格子状の画像（SLP/高度場）に内在する空間相関・構造を十分に反映できず、平均化（セントロイド）と相まって代表型の“ぼけ”や「スノーボール」クラス（異質要素の混入）を誘発する（Wang & Bovik, 2009；Winderlich et al., 2024）。（ii）代表性の欠如：セントロイドは実データではなく、極端な勾配や中心位置の情報が希釈され、実務解釈や極端事象の追跡が難しくなる（Winderlich et al., 2024）。（iii）希少型・複合型の取り扱い：固定クラス数・平均化前提では、頻度の低いが重要なパターン（まれな循環・極端事象起因型）や移行・複合的な構造を十分に保持できない。

これらの制約に対し、近年 3 つの有望な方向性が示されている。第一に、「構造」に敏感な類似度の導入である。S‑SOM（Doan et al., 2021）は SOM の BMU 探索に SSIM（Wang et al., 2004）を用い、EUC 比でクラスタ品質（シルエット）とトポロジ保存（TE）を一貫して改善した。Sato & Kusaka（2021）は、SSIM と気象学的勾配ベースの S1 が、SLP の主観的類似・誤教師への頑健性で最良であることを、教師データ多数の統計で示した。第二に、クラスタ代表を「平均」ではなく「medoid（実データの代表）」で与えることである。Winderlich et al.（2024）は、改良 SSIM を距離に用い、HAC→k‑medoids の二段階クラスタリングで medoid 代表を採用し、分離・安定性・代表性・希少型保持を両立させた。第三に、型の「頻度・遷移・持続」特性を Jensen–Shannon 距離で総合評価し、循環表現の良否を客観的に測る枠組みである（Winderlich et al., 2024）。これらは、EUC/MSE に起因する構造劣化や代表性の欠如を回避し、まれな型を損なわない分類・評価の道筋を与える。

本研究は、以上の知見を SOM を基盤とする統一フレームワークへと橋渡しすることを目標とする。具体的には、GPU 最適化した Batch‑SOM 上に、BMU 距離として EUC／SSIM（全体 1 窓）／SSIM5（5×5 移動窓）／S1／S1+SSIM5（等重み融合）の 5 方式を実装し、前処理・学習条件・可視化を揃えた厳密比較の実験環境を整備する。そのうえで、(a) 学習内の量子化誤差（QE）・Macro Recall（基本/複合）・NodewiseMatchRate（ノード多数決 raw と medoid raw の一致）による内的妥当性、(b) 学習年のノード多数決を辞書として検証年に適用する年別汎化性能、(c) centroid／medoid／true medoid の三者比較可視化による代表性の評価、(d) ラベル分布・混同行列等の診断を、同一プロトコルで実施する。SOM 出力の解釈性を高めるため、各ノードの代表はセントロイドに加え medoid（closest‑to‑centroid）と true medoid（総距離最小）を標準出力とし、平均化に伴う情報劣化を補う。データは日本周辺域（15–55N, 115–155E）の日次 09JST 相当 SLP とし、期間は学習 1991–1999・検証 2000、ラベルは吉野（2002）に準拠した 15 基本クラス（複合は基本成分へ分解）を用いる。コードは Python/PyTorch で実装し、乱数・BLAS スレッド・I/O・ハイパーパラメータをログ化、入出力・成果物の構造化保存（CSV/PNG/JSON/LOG）を徹底し、再現性と拡張性を担保する。

本稿の主な貢献は以下の 4 点である。

1. SOM を共通足場に、構造・勾配・融合の 5 距離を同一条件で比較する初の GPU 最適化実験基盤を提示する。
2. medoid／true medoid 表現を SOM 可視化のデフォルトに組み込み、代表性・解釈性を定量・定性的に評価する手順を確立する。
3. 学習内と年別汎化の二層評価（Macro Recall・QE・一致率・混同行列）を策定し、ラベル主観や季節変動に対する頑健性を検証できる枠組みを示す。
4. Doan（2021）・Sato & Kusaka（2021）・Winderlich（2024）の知見を SOM 文脈で統合し、将来的な二段階クラスタリング（HAC→k‑medoids）や Jensen–Shannon 距離による循環評価への発展を見据えた設計とする。

以降では、研究目的と問い、関連研究、データ・前処理、提案方法（GPU Batch‑SOM、距離実装、medoid 可視化、年別推論）、実装、評価設計、再現性計画、限界と今後の拡張を詳細に述べる。本稿は設計・実装・評価計画に焦点を当て、数値結果と考察は別稿にて報告する。

---

## 2. 目的および研究課題（Objectives & Research Questions）

### 2.1 目的

1. **距離指標の比較基盤の構築**：GPU 最適化 Batch‑SOM 上で、EUC/SSIM/SSIM5/S1/S1+SSIM5 の 5 距離を同一実装・同一データ前処理で比較する環境を整備する。
2. **多面的評価**：学習内と年別検証の双方で、分類性能（Macro Recall：基本/複合）、代表性（NodewiseMatchRate）、量子化誤差（QE）、ラベル分布、centroid vs medoid 可視化を用いて距離指標を評価する。
3. **代表性の強化**：medoid と true medoid（ノード内総距離最小）を併記し、SOM の平均化に伴う“ぼけ”を補完、現場解釈に資する可視化を標準出力とする。
4. **汎化評価手順の確立**：学習年に得た「ノード多数決（基本ラベル）」を検証年へ適用し、年跨ぎのラベル予測（年別推論）手順を確立する。

### 2.2 研究課題（RQs）と仮説（H）

- **RQ1**：EUC/SSIM/SSIM5/S1/S1+SSIM5 間で、Macro Recall（基本/複合）はどのように異なるか？  
  **H1**：SSIM 系および S1 は EUC より高い Macro Recall を与える（Sato & Kusaka 2021）。
- **RQ2**：medoid（closest‑to‑centroid / true medoid）による代表性は、ノード多数決 raw ラベルとの一致率（NodewiseMatchRate）でどの程度担保されるか？  
  **H2**：非ユークリッド距離（SSIM/S1）では medoid が多数決と整合しやすく、centroid の“ぼけ”を補う（Winderlich 2024）。
- **RQ3**：学習年のノード代表（基本）を検証年に適用する際の汎化性能は距離により差が出るか？  
  **H3**：SSIM/S1 は構造・勾配を捉えるため、年別検証でも頑健性を示す（Doan 2021）。

---

## 3. データと前処理（Data & Pre‑processing）

- **物理量**：海面更正気圧（SLP, prmsl）
- **期間**：学習 1991‑01‑01〜1999‑12‑31、検証 2000‑01‑01〜2000‑12‑31（各日 09JST 相当）
- **領域**：115–155E, 15–55N（日本周辺）
- **ラベル**：吉野（2002）準拠の 15 基本ラベル  
  `1, 2A, 2B, 2C, 2D, 3A, 3B, 3C, 3D, 4A, 4B, 5, 6A, 6B, 6C`  
  （複合表記も許容。NFKC 正規化とトークナイズで基本成分を抽出）
- **前処理**：Pa→hPa、サンプル毎の空間平均差し引き（偏差化）。  
  ※季節正規化（平均/標準偏差）・面積重みは本比較では未適用（拡張項目）。

---

## 4. 提案枠組み（Methods）

### 4.1 GPU Batch‑SOM（minisom.MiniSom）

- **学習**：ミニバッチ蓄積 → 一括更新（BMU→ 近傍重み h→ 分子/分母を加算 → 更新）。総反復に対し σ は漸減（asymptotic/linear 切替可）。
- **BMU 距離（5 種）**：
  - **EUC**：ユークリッド距離（ベースライン）
  - **SSIM**：全体 1 窓の SSIM（C1=C2=1e‑8）
  - **SSIM5**：5×5 移動窓の平均 SSIM（C1=C2=0，分母に ε を付与；Doan 2021 に準拠）
  - **S1**：Teweles–Wobus（水平/南北勾配の差の正規化比）
  - **S1+SSIM5**：S1 と SSIM5 をサンプル毎に min‑max 正規化後に等重み合成
- **medoid 置換**：任意間隔で各ノード重みを「距離的に最近傍の実サンプル」へ置換（距離一貫性・代表性を向上）。

### 4.2 可視化と代表性

- **centroid**（平均）と **medoid**（closest‑to‑centroid）を併記し、“ぼけ”を補う。
- **true medoid**（ノード内総距離最小）も出力し、代表性の頑健性を確認。
- **NodewiseMatchRate**：ノード多数決 raw と medoid raw の一致率を算出し、背景色（緑/赤）で可視化。

### 4.3 年別推論（Generalization）

- 学習年の「ノード多数決（基本ラベル）」を予測辞書として検証年に適用し、基本/複合の Macro Recall を算出する。

---

## 5. 実装（Implementation）

### 5.1 main_v4.py の構成

- **データ入出力**：`load_and_prepare_data_unified()`（時系列スライス・偏差化・ラベル/座標の取り扱い）
- **学習**：`run_one_method_learning()`（反復分割・QE/Recall/MatchRate の履歴保存、各種 PNG/CSV 出力）
- **検証**：`run_one_method_verification()`（年別推論・混同行列・per‑label 再現率・可視化出力）
- **ユーティリティ**：ラベル正規化、混同行列作成、メトリクス CSV/PNG、ノード詳細ログ、可視化器群

### 5.2 minisom.py の構成

- **距離**：`euclidean` / `ssim` / `ssim5` / `s1` / `s1ssim` の高速実装（ノード分割 `nodes_chunk` で VRAM 最適化）
- **学習**：`train_batch()`（σ 漸減、メドイド置換、固定評価サブセットで QE 安定化）
- **推論**：`predict()`（BMU 座標）
- **評価**：`quantization_error()`（距離タイプに依存）

### 5.3 ハイパーパラメータ（既定）

- SOM 10×10、反復数 `NUM_ITER`（分割評価 `SOM_EVAL_SEGMENTS`）、バッチ 128、`nodes_chunk` 2–4（VRAM 依存）
- 評価サンプル上限 4000、ログ間隔 10、GPU/CPU 自動選択、乱数シード固定

---

## 6. 評価設計（Evaluation Protocol）

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

> 評価設計に加え、§9–§11 に主要な数値結果・図表の要約と考察を示す（詳細な図表・追加実験は別稿／付録に委ねる）。

---

## 7. 再現性計画（Reproducibility Plan）

- **コード**：`main_v4.py` / `minisom.py` に集約。乱数・BLAS スレッド・PyTorch の deterministic を固定。
- **入出力**：NetCDF（ERA5 の prmsl；例：`prmsl_era5_all_data_seasonal_large.nc`）を `DATA_FILE` から読み込み。…
- **環境**：Python 3.x、PyTorch（CUDA/CPU 対応）、Cartopy/Matplotlib、xarray/pandas。VRAM 16–24GB を推奨（`nodes_chunk` で調整可）。
- **パラメータ**：SOM サイズ・反復・バッチ・距離タイプは定数で管理。ログに全設定を出力。
- **データ配布**：入力データの配布条件（ERA5/再解析のライセンス）を遵守し、前処理スクリプトを公開する。

---

## 8. 限界と想定される脅威（Limitations & Threats to Validity）

- **前処理**：季節正規化・面積重みを未適用のため、季節振幅/緯度面積の影響が残存しうる。
- **ラベル**：複合・主観・不一致の影響。基本/複合の両評価を設けるが、真の境界は曖昧。
- **汎化**：1990s→2000 年の年別推論に依存。年代の拡張・季節別学習の検討が必要。
- **距離の実装差**：SSIM の定数・窓設定、S1 の差分スキームなど実装差の影響に留意。
- **計算資源**：SSIM5/S1 は EUC より計算負荷が高い。`nodes_chunk` により緩和する。

---

## 9. 実験結果（Experimental Results）

本節では，GPU 最適化 Batch‑SOM を共通基盤に，BMU 距離を 5 方式（EUC／SSIM／SSIM5／S1／S1+SSIM5）で同条件比較した v3 実験の主要結果をまとめる．前処理は SLP を hPa に変換し，サンプル毎に空間平均を差し引いた偏差（anomaly）とした．学習期間は 1991–1999，検証は 2000 年である．評価は Macro Recall（基本ラベル 15／複合ラベル）を用いた．

### 9.1 距離方式別の総合指標（Macro Recall）

- 学習期間（1991–1999）の Macro Recall（基本／複合）

  - Euclidean: 0.2544／0.1863
  - SSIM（全体 1 窓）: 0.2459／0.1819
  - SSIM5（5×5・C=0）: 0.2960／0.2060
  - S1: 0.3466／0.2329
  - S1+SSIM5（等重み融合）: 0.3443／0.2361

- 検証期間（2000 年）の Macro Recall（基本／複合）
  - Euclidean: 0.1958／0.1595
  - SSIM（全体 1 窓）: 0.2135／0.1500
  - SSIM5（5×5・C=0）: 0.2531／0.1917
  - S1: 0.2754／0.1850
  - S1+SSIM5（等重み融合）: 0.3501／0.2157

→ 学習内では S1 が最良（0.3466），検証では S1+SSIM5 が最良（0.3501）で，SSIM5 は SSIM を一貫して上回った．EUC は両期間で最下位グループであった．

### 9.2 マップの画像

- Euclidean、マップ画像
  ![画像](./image/maps_10x10_euclidean.png)

- Euclidean、メドイドマップ画像
  ![画像](./image/maps_10×10_euclidean_true_medoid.png)

- SSIM、マップ画像
  ![画像](./image/maps_10x10_ssim.png)

- SSIM、メドイドマップ画像
  ![画像](./image/maps_10×10_ssim_true_medoid.png)

- SSIM5、マップ画像
  ![画像](./image/maps_10x10_ssim5.png)

- SSIM5、メドイドマップ画像
  ![画像](./image/maps_10×10_ssim5_true_medoid.png)

- S1、マップ画像
  ![画像](./image/maps_10x10_s1.png)

- S1、メドイドマップ画像
  ![画像](./image/maps_10×10_s1_true_medoid.png)

- S1+SSIM5、マップ画像
  ![画像](./image/maps_10x10_s1+ssim5.png)

- S1+SSIM5、メドイドマップ画像
  ![画像](./image/maps_10×10_s1+ssim5_true_medoid.png)

### 9.2 ラベル別の傾向（検証）

- 高再現のラベル
  - 1（冬型）: 0.878–0.951（方式間ばらつき小）
  - 3B（西風系）: 0.736–0.895（S1／S1+SSIM5 で顕著）
- 中位
  - 4A（移動性高気圧）: 0.4–0.8（方式により振れ，S1 系で高め）
- 低位（いずれの方式も低い）
  - 2B，2C，3C，3D，5（乱流的・境界が曖昧／形状多様）
  - 6A，6B，6C（サンプル希少で 0 付近）
- SSIM5 は SSIM 比で 2D，3A，4A，3B の一部で改善．
- S1+SSIM5 は S1 の水準を保ちつつ，2A／5 等の構造・位置のブレを併合で吸収し，総合で最良．

### 9.3 可視化・代表性（概況）

- ノード平均（centroid）に加え，medoid（closest‑to‑centroid）・true medoid（総距離最小）を出力した．ノードごとの多数決 raw と medoid raw の一致可視化では，S1／S1+SSIM で一致ノード（緑）が相対的に多く，EUC／SSIM では不一致（赤）が目立つ領域が残存した（とくに多峰性・遷移帯）．
- ラベル分布ヒートマップでは，S1／S1+SSIM5 で 1，3B，4A の分化が明瞭で，SSIM5 は SSIM 比で局所勾配・コントラストの表現力が向上した．

---

## 10. 実験考察（Discussion）

1. **S1 と SSIM の補完性**  
   S1 は水平方向・南北方向の勾配差を直接比較するため，前線帯や西風卓越（3B），冬型（1）など勾配構造が支配的な型で強い．一方，台風や移行型のように中心位置や形状が重要な場合は，構造類似度（SSIM）が有効である（Sato & Kusaka, 2021；Doan et al., 2021）．本実験でも SSIM5 が SSIM を上回り，局所窓の導入が有効だった．S1+SSIM（等重み）は，両者の長所を取り入れ検証 Macro を最大化した（0.3501）．

2. **EUC の限界と medoid の効用**  
   EUC は平均化に伴う代表型の“ぼけ”やスノーボール化を招きやすく（Wang & Bovik, 2009；Winderlich et al., 2024），本実験でも Macro が最下位グループにとどまった．一方，medoid/true‑medoid 表現は，極値勾配や中心位置を保持し，ノード多数決 raw と整合する可視化を提供した．これにより，運用・事例検索での解釈性が向上する（Winderlich の指摘と整合）．

3. **難ラベル（2B, 2C, 3C, 3D, 5, 6x）**  
   これらは（i）多様性が大きく境界が曖昧，（ii）サンプル希少，（iii）単一変数 SLP では判別根拠が弱い，の複合要因がある．SOM 単独では分離が難しく，風ベクトル・温位・渦度等の多変量化，季節正規化・面積重み等の前処理強化が必要である（Takasuka et al., 2024；Jiang, 2013）．

4. **SSIM の窓とコントラスト**  
   SSIM5 は局所窓により等圧線の曲率・コアのコントラストを反映でき，SSIM を上回った．Doan（2021）の S‑SOM の結果とも整合的であり，BMU 探索における「構造の扱い」がクラスタ品質に直結する．

5. **汎化と頑健性**  
   検証年（2000 年）で S1+SSIM5 が最良となったのは，年変動に伴う勾配強度・中心位置の揺らぎを両者の併合が吸収したためと解釈できる．S1 単独は学習内では最良だが，年別では構造のドリフトに敏感な傾向がある．

---

## 11. まとめ（Conclusions）

- GPU Batch‑SOM を共通基盤に，距離 5 方式を厳密比較した結果，**検証年の Macro Recall（基本）で S1+SSIM5 が最良（0.3501）**，学習では S1 が最良（0.3466）であった．SSIM5 は SSIM を一貫して上回り，局所窓の有効性を確認した．
- ラベル別には 1（冬型），3B（西風系）が高再現，4A（移動性高気圧）は方式により改善しうる一方，2B/2C/3C/3D/5 と 6x は依然低く，多変量化と前処理強化が必要である．
- centroid に加え medoid/true‑medoid を標準出力とする可視化は，代表性・解釈性を高め，運用上の検索・事例同定に有用である．

---

## 12. 今後の改善（Future Work）

- **前処理強化**：日別季節正規化（日付ごとの μ・σ による標準化），緯度面積重み（cosφ），高緯度の安定化を導入し，季節振幅・面積のバイアスを低減．
- **評価指標の拡充**：シルエット係数・トポロジカルエラー（Doan 2021）を追加し，SOM 配置の品質を定量化．参照分類に対する**Jensen–Shannon 距離（頻度・遷移・持続）**を導入し，循環表現の総合評価を実装（Winderlich 2024）．
- **自動クラスタとハイブリッド**：SSIM 閾値に基づく **HAC→k‑medoids** を組み込み，クラス数を自動決定しつつ希少型を保持．SOM とのハイブリッド（SOM で粗分割 →HAC/k‑medoids で洗練）を検討．
- **多変量化**：SLP に加え，10m 風ベクトル・相当温位・渦度傾度・地衡風・気温偏差等を統合した**多変量 S‑SOM**を実装し，梅雨・台風・前線型の識別力を補強．
- **季節別・領域別運用**：季節別 SOM（DJF/MAM/JJA/SON）で季節性を陽に分離し，東アジア／北半球への領域拡張と汎化検証を行う．
- **オープンサイエンスと再現性**：コード・設定・前処理・出力テンプレートを公開し，再現性パッケージ化．成果物（CSV/PNG/JSON/LOG）の自動カタログ化とドキュメント整備．

> 付記：本稿で用いた成果物（例）  
> `*_iteration_metrics.csv/png`（QE・MacroRecall・一致率の推移），`*_assign_all.csv`（BMU 割当），`*_som_node_avg_all.png`（centroid），`*_som_node_medoid_all.png`／`*_som_node_true_medoid_all.png`（代表パターン），`*_node_medoids.csv`／`*_node_true_medoids.csv`（代表メタ情報），`*_nodewise_analysis_match.png`（一致可視化），`*_confusion_matrix_all.csv`（学習混同行列），`*_label_dist_*.png`（分布ヒートマップ），`node_majorities.json`（学習ノード代表）．検証側は `*_verification_*` として同様に出力する．

---

## 参考文献

1. 木村 広希, 川島 英之, 北川 博之 (2009): サポートベクターマシンを用いた気圧配置の自動分類. DEIM Forum 2009, B6-1.（関連：木村・川島・北川 (2008) DEWS など）

2. 高須賀 匠, 高野 雄紀, 渡邊 正太郎, & 雲居 玄道. (2024). 自己組織化マップを用いた気圧配置のクラスタリングと 1km メッシュ天気データによる分析. 情報処理学会全国大会.

3. Philippopoulos, K., Deligiorgi, D., and Kouroupetroglou, G. (2014):Performance Comparison of Self-Organizing Maps and k-means Clustering Techniques for Atmospheric Circulation Classification. International Journal of Energy and Environment, 8, 171–180.

4. Jiang, N., Dirks, K. N., & Luo, K. (2013). Classification of synoptic weather types using the self-organising map and its application to climate and air quality data visualisation. Weather and Climate, 33, 52-75.

5. Winderlich, K., Dalelane, C., & Walter, A. (2024). Classification of synoptic circulation patterns with a two-stage clustering algorithm using the modified structural similarity index metric (SSIM). Earth System Dynamics, 15(3), 607-633.

6. Doan, Q. V., Kusaka, H., Sato, T., & Chen, F. (2021). S-SOM v1. 0: a structural self-organizing map algorithm for weather typing. Geoscientific Model Development, 14(4), 2097-2111.

7. Sato, T., and H. Kusaka, 2021: Statistical intercomparison of similarity metrics in sea level pressure pattern classification. J. Meteor. Soc. Japan, 99, 993–1001, doi:10.2151/jmsj.2021-047

## 付録（用語）

- **SSIM**：構造類似度。輝度・コントラスト・構造（共分散）を同時に比較。
- **S1**：Teweles–Wobus スコア。水平方向/南北方向の勾配差を正規化比で評価。
- **medoid**：クラスタ内の「最も代表的な実サンプル」。centroid と異なり平均化の影響を受けにくい。

## 付録（計算式：tex）

本付録では，本研究の実装（main_v4.py／minisom.py）で用いた 5 種の距離（EUC／SSIM／SSIM5／S1／S1+SSIM5）の計算式を，コードと一致する形で LaTeX 記法で整理する．
対象は海面更正気圧の偏差場（各サンプルで空間平均を差し引いた hPa の 2 次元配列）であり，以下の記号を用いる．

- 格子領域：Ω（画素数 |Ω| = H×W）
- サンプル（観測／入力）: x(s), プロトタイプ（SOM ノード重み）: w(s), s∈Ω
- 画素平均・分散・共分散：
  μ*x = (1/|Ω|)∑*{s∈Ω} x(s), σ*x^2 = (1/|Ω|)∑*{s∈Ω} (x(s)-μ*x)^2, Cov(x,w) = (1/|Ω|)∑*{s∈Ω} (x(s)-μ_x)(w(s)-μ_w)
- 数値安定化用の極小量：ε = 10^{-12}
- 5×5 平均フィルタ核：K\_{5×5} = (1/25)・全要素 1 の 5×5 カーネル（畳み込みは反射パディング）

以下，いずれも「距離」を返す関数 d(・,・) を定義する（SOM の BMU 探索やメドイド選定で使用）．SSIM 系は実装通り「1 − SSIM」を距離とする．

---

### 1) Euclidean（ユークリッド距離，EUC）

実装: minisom.\_euclidean_distance_batch/\_euclidean_to_ref

二乗和の平方根（数値安定化のため sqrt 内に ε を付与）．
\[
d*{\mathrm{EUC}}(x,w)
= \left( \sum*{s\in\Omega} \bigl(x(s)-w(s)\bigr)^2 \right)^{1/2}
= \sqrt{ \sum\_{s\in\Omega} \bigl(x(s)-w(s)\bigr)^2 + \varepsilon }.
\]

---

### 2) SSIM（全体 1 窓，C1=C2=10^{-8}）

実装: minisom.\_ssim_distance_batch/\_ssim_global_to_ref（定数 c1=c2=1e-8）

中心化量を
\(
\tilde{x}(s)=x(s)-\mu*x,\ \tilde{w}(s)=w(s)-\mu_w
\)
とすると，
\[
\begin{aligned}
\mathrm{SSIM}*{\mathrm{global}}(x,w)
&= \frac{\bigl(2\,\mu*x \mu_w + C_1\bigr)\,\bigl(2\,\mathrm{Cov}(x,w) + C_2\bigr)}
{\bigl(\mu_x^2 + \mu_w^2 + C_1\bigr)\,\bigl(\sigma_x^2 + \sigma_w^2 + C_2\bigr) + \varepsilon},\\
d*{\mathrm{SSIM}}(x,w)
&= 1 - \mathrm{SSIM}_{\mathrm{global}}(x,w),
\end{aligned}
\]
ただし
\(
\mathrm{Cov}(x,w)=\frac{1}{|\Omega|}\sum_{s\in\Omega}\tilde{x}(s)\tilde{w}(s)
\),
\(C_1=C_2=10^{-8}\)．

注）本式は Wang et al. (2004) の SSIM を「1 窓＝画像全体」で評価した形に一致し，Doan et al. (2021) の S‑SOM で用いられる SSIM の基本形と整合する（本実装は Winderlich et al. (2024) が推奨する小さな定数を採用）．

---

### 3) SSIM5（5×5 移動窓 SSIM，C1=C2=0，分母に ε）

実装: minisom.\_ssim5_distance_batch/\_ssim5_to_ref（5×5 平均畳み込み，反射パディング，C1=C2=0，分母に ε，分散は 0 でクリップ）

各画素 s に対して，5×5 平均により
\[
\begin{aligned}
\mu*x(s) &= (K*{5\times5} _ x)(s),\quad
\mu*w(s) = (K*{5\times5} _ w)(s),\\
\mu*{x^2}(s) &= (K*{5\times5} _ x^2)(s),\quad
\mu*{w^2}(s) = (K*{5\times5} _ w^2)(s),\\
\mu*{xw}(s) &= (K*{5\times5} \* (x\cdot w))(s),
\end{aligned}
\]
とおき，
\[
\begin{aligned}
\sigma*x^2(s) &= \max\bigl\{\mu*{x^2}(s)-\mu*x(s)^2,\ 0\bigr\},\quad
\sigma_w^2(s) = \max\bigl\{\mu*{w^2}(s)-\mu*w(s)^2,\ 0\bigr\},\\
\mathrm{Cov}(x,w)(s) &= \mu*{xw}(s) - \mu*x(s)\mu_w(s).
\end{aligned}
\]
このとき局所 SSIM マップ（C1=C2=0）は
\[
\mathrm{SSIM}*{\mathrm{loc}}(s)
= \frac{\bigl(2\,\mu*x(s)\mu_w(s)\bigr)\,\bigl(2\,\mathrm{Cov}(x,w)(s)\bigr)}
{\bigl(\mu_x(s)^2+\mu_w(s)^2\bigr)\,\bigl(\sigma_x^2(s)+\sigma_w^2(s)\bigr) + \varepsilon}.
\]
画像全体の SSIM はその空間平均：
\[
\mathrm{SSIM}*{5\times5}(x,w) = \frac{1}{|\Omega|}\sum*{s\in\Omega} \mathrm{SSIM}*{\mathrm{loc}}(s),
\qquad
d*{\mathrm{SSIM5}}(x,w) = 1 - \mathrm{SSIM}*{5\times5}(x,w).
\]

注）Doan et al. (2021)（S‑SOM）は C1=C2=0 とし，分母に数値安定化項 ε を付与した移動窓 SSIM を採用しており，本実装はこれに一致する．

---

### 4) S1（Teweles–Wobus スコア）

実装: minisom.\_s1_distance_batch/\_s1_to_ref（前進差分，x/y 方向の格子「辺」上で和を評価，分母は成分ごとに max をとる，最後に 100 を乗ずる）

格子の経度・緯度方向の前進差分を
\[
\Delta*x x(i,j) = x(i,j+1)-x(i,j),\quad
\Delta_y x(i,j) = x(i+1,j)-x(i,j)
\]
（w も同様）と定義し，それぞれの全「辺」集合を \(E_x, E_y\) とする．すると，
\[
\begin{aligned}
\mathrm{num}\_x &= \sum*{(i,j)\in E*x} \left| \Delta_x x(i,j) - \Delta_x w(i,j) \right|,\quad
\mathrm{num}\_y = \sum*{(i,j)\in E*y} \left| \Delta_y x(i,j) - \Delta_y w(i,j) \right|,\\
\mathrm{den}\_x &= \sum*{(i,j)\in E*x} \max\!\left( \left|\Delta_x x(i,j)\right|,\ \left|\Delta_x w(i,j)\right| \right),\\
\mathrm{den}\_y &= \sum*{(i,j)\in E*y} \max\!\left( \left|\Delta_y x(i,j)\right|,\ \left|\Delta_y w(i,j)\right| \right).
\end{aligned}
\]
Teweles–Wobus の S1 は
\[
S1(x,w) = 100 \times \frac{\mathrm{num}\_x+\mathrm{num}\_y}{\mathrm{den}\_x+\mathrm{den}\_y + \varepsilon},
\qquad
d*{\mathrm{S1}}(x,w) = S1(x,w).
\]

注）S1 は水平方向の勾配差に基づく正規化指標であり，Sato & Kusaka (2021) が SLP パターン選択で高性能であることを示した．本実装はその標準形（TW 指数）に一致する．

---

### 5) S1+SSIM5（等重み融合距離；min–max 正規化後の平均）

実装: minisom.\_distance_batch における 's1ssim'（BMU 探索），および medoid 選定関数（compute_node_medoids_by_centroid／compute_node_true_medoids）での融合．

二つの距離
\(
d*{\mathrm{S1}}(x,w),\ d*{\mathrm{SSIM5}}(x,w)
\)
を「同一の比較集合内」で min–max 正規化したうえで等重み平均する．

- BMU 探索（1 サンプル x に対し，全ノード j=1,\dots,m の距離を並べる場合）：
  \[
  \begin{aligned}
  &d^{(j)}_1 = d_{\mathrm{S1}}(x,w*j),\quad
  d^{(j)}\_2 = d*{\mathrm{SSIM5}}(x,w*j),\\
  &\tilde d^{(j)}\_1 = \frac{d^{(j)}\_1 - \min*{k} d^{(k)}_1}{\max_{k} d^{(k)}_1 - \min_{k} d^{(k)}_1 + \varepsilon},\quad
  \tilde d^{(j)}\_2 = \frac{d^{(j)}\_2 - \min_{k} d^{(k)}_2}{\max_{k} d^{(k)}_2 - \min_{k} d^{(k)}_2 + \varepsilon},\\
  &d_{\mathrm{S1+SSIM5}}(x,w_j) = \alpha\,\tilde d^{(j)}\_1 + (1-\alpha)\,\tilde d^{(j)}\_2,\qquad \alpha=0.5.
  \end{aligned}
  \]

- ノード内 true‑medoid の行列融合（候補 i の行に対し，列 j 方向で正規化する場合）：
  \[
  \begin{aligned}
  &D*1(i,j)=d*{\mathrm{S1}}(x*i,x_j),\quad D_2(i,j)=d*{\mathrm{SSIM5}}(x*i,x_j),\\
  &\tilde D_1(i,j)=\frac{D_1(i,j)-\min*{j} D*1(i,j)}{\max*{j} D*1(i,j)-\min*{j} D*1(i,j)+\varepsilon},\quad
  \tilde D_2(i,j)=\frac{D_2(i,j)-\min*{j} D*2(i,j)}{\max*{j} D*2(i,j)-\min*{j} D_2(i,j)+\varepsilon},\\
  &D(i,j)=\alpha\,\tilde D_1(i,j) + (1-\alpha)\,\tilde D_2(i,j),\qquad \alpha=0.5.
  \end{aligned}
  \]
  （i 行の総和 \(\sum_j D(i,j)\) が最小の i を true‑medoid とする．）

注）融合は常に「比較集合内（BMU では全ノード，medoid では候補集合）」での min–max 正規化 → 等重み平均（α=0.5）であり，コードの実装と一致する．

---

【参考と整合性】

- SSIM（全体 1 窓）は Wang et al. (2004) の定義に基づき，Winderlich et al. (2024) 等で推奨される小定数（C1=C2=10^{-8}）を用いた実装に一致．
- SSIM5（5×5 移動窓）は Doan et al. (2021)（S‑SOM）の仕様（C1=C2=0，分母に ε）に準拠し，局所平均は反射パディングの 5×5 平均フィルタで計算（実装どおり分散は 0 でクリップ）．
- S1 は Teweles–Wobus 指数の標準形（差分の絶対値と分母の max 和，×100）を用い，Sato & Kusaka (2021) で高精度だった設定に一致．
- S1+SSIM5 は実装どおり sample（または行）内 min–max 正規化の等重み平均．
