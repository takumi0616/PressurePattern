# v4_1000 実験結果サマリ（SOM: EUC / SSIM5 / S1 / S1+SSIM5）

対象ディレクトリ: `src/PressurePattern/results_v4_1000`  
構成:

- learning_result/{euclidean_som, ssim5_som, s1_som, s1ssim_som}
  - 反復ごとの学習メトリクス（CSV, 図）
  - ノード代表（平均/medoid/true-medoid）、ラベル分布、ノード多数決ラベル
- verification_results/{各手法}
  - 検証期（2000 年）での割当、混同行列、ラベル別再現率、ノード平均図 等

本サマリは上記成果物を精査し、手法別の学習推移、検証性能、ラベル別傾向、代表性（medoid 整合性）を整理したものである。

---

## 1. 検証期（2000 年）の総合性能（Macro Recall）

ログより（基本ラベル / 複合ラベル）:

| 手法     | Macro Recall（基本） | Macro Recall（複合） | 出典ログ                                                        |
| -------- | -------------------: | -------------------: | --------------------------------------------------------------- |
| S1+SSIM5 |               0.3501 |               0.2157 | `verification_results/s1ssim_som/s1ssim_verification.log`       |
| S1       |               0.2754 |               0.1850 | `verification_results/s1_som/s1_verification.log`               |
| SSIM5    |               0.2531 |               0.1917 | `verification_results/ssim5_som/ssim5_verification.log`         |
| EUC      |               0.1958 |               0.1595 | `verification_results/euclidean_som/euclidean_verification.log` |

要点:

- 一般化性能（検証）では S1+SSIM5 が最良（基本 0.3501 / 複合 0.2157）。
- S1 は学習時ピークが高いが、検証では融合に及ばない。
- SSIM5 は局所構造に敏感で、一部ラベルで強みを示すが、総合では S1 系に劣後。
- EUC はベースラインとして最下位。

---

## 2. 学習フェーズの反復推移（主要ポイント）

CSV: `learning_result/*/*_iteration_metrics.csv`  
列: `iteration, MacroRecall_majority, MacroRecall_composite, QuantizationError, NodewiseMatchRate`

### 2.1 S1+SSIM5（s1ssim_som）

- MacroRecall_majority: 反復 720 で 0.3515、770 で 0.3548（ピーク）、その後わずかに低下し 1000 で 0.3366。
- MacroRecall_composite: 反復 770 で 0.2389 付近（ピーク）。終盤も 0.233〜0.235 台で高止まり。
- NodewiseMatchRate: ピーク約 0.55（反復 360）、終盤は 0.40 前後。
- 所見: 中盤〜後半でピーク。最終 1000 固定より「ベスト反復のチェックポイント」採用が有効な可能性。

### 2.2 S1（s1_som）

- MacroRecall_majority: 940〜960 で 0.3486〜0.3491（ピーク）、1000 で 0.3454。
- MacroRecall_composite: 反復 730 で 0.2337 付近（ピーク）、1000 で 0.2321。
- NodewiseMatchRate: 序盤から相対的に高く、ピーク〜0.55、1000 で 0.47。
- 所見: 勾配評価ベースで学習安定性・代表性（後述）が高い。

### 2.3 SSIM5（ssim5_som）

- MacroRecall_majority: 980 で 0.3068（ピーク）、1000 で 0.2830。
- MacroRecall_composite: 反復 710 で 0.2052 前後（ピーク）。
- NodewiseMatchRate: ピーク〜0.46（反復 670）、終盤は 0.30 前後。
- 所見: 局所構造の捉え方に強みがあるが、全体の Macro では S1 系に届かない。

### 2.4 EUC（euclidean_som）

- MacroRecall_majority: 570〜600 で〜0.259（ピーク）、1000 で 0.2544。
- MacroRecall_composite: 640 で 0.1928（ピーク）、1000 で 0.1860。
- NodewiseMatchRate: ピーク〜0.39、1000 で 0.34。
- 所見: ベースラインとして妥当だが、他手法に比べ弱い。

注意（QuantizationError について）:

- 各手法でスケールが異なる（例: S1 は~70→68 台、S1+SSIM は~0.036→0.020 台、SSIM5 は~0.81→0.73 台、EUC は~646→553 台）。距離定義の違いによるため、**絶対値の横比較は不可**。ただし、減少傾向＝収束挙動としては概ね良好。

---

## 3. ラベル別の傾向（検証）

ログ「Per-label recall」より抜粋（複合ラベルを中心に所見）:

- 高再現ラベル
  - 1（冬型）: 全手法で高再現。SSIM5/EUC で 0.7949、S1 で 0.7692、S1+SSIM5 で 0.7436。
  - 3B（西風系）: SSIM5 が 0.6379 で最良。S1/S1+SSIM5 は 0.6034、EUC は 0.5690。
  - 4A（移動性高気圧）: SSIM5 が 0.4912 で最良。EUC0.4211、S1+SSIM50.4035、S10.3860。
- 低再現ラベル（要対策）
  - 2B, 2C, 3C, 3D: 全手法で極めて低い（0〜0.08）。クラス境界の曖昧さ・サンプル数の希少性が主因と推察。
  - 4B, 5: 全体に低め。例）S1+SSIM5: 5 が 0.3061、S1: 5 が 0.3673（S1 がやや優勢）、SSIM5/EUC は低い。
  - 6A〜6C: N が小さく安定しない（例: 6A は S1+SSIM5 で 0.1786、他はほぼ 0）。

総括:

- SSIM5 は形状・位置の局所構造が支配的なラベル（3B/4A 等）に強み。
- S1 は勾配支配型（冬型など）に強み。
- 融合（S1+SSIM5）はラベル別の最高値でないこともあるが、**全体バランスが良く Macro 最大**。

---

## 4. Medoid の代表性（NodewiseMatchRate）

学習 CSV（最終反復近傍）より:

- S1: 約 47%（1000 反復時）
- S1+SSIM5: 約 40%
- SSIM5: 約 30.3%
- EUC: 約 34%

補足:

- S1 系でノード多数決ラベルと medoid ラベルの一致が高く、**medoid 代表の妥当性が高い**。
- S1+SSIM5 でも学習中盤に一致率ピーク（〜0.55）あり。チェックポイント選択で代表性向上余地。

---

## 5. 実運用/改良に向けた示唆

- 早期停止/最良チェックポイント選択
  - S1+SSIM5 で反復 720〜770 に Macro のピーク。最終固定より**最良反復採用が有利**な可能性。
- 難ラベル（2B/2C/3C/3D/4B/5/6x）対策
  - 入力特徴の多変量化（風ベクトル・温度場）
  - 季節正規化、面積重み（格子の緯度依存補正）
  - データ拡張/再ラベリング検討、クラス重み付け/再サンプリング
  - 類似度の季節別/ラベル別ハイパラ（SSIM 窓サイズ、S1/SSIM の重み等）最適化
- 評価設計
  - Macro 以外に、混同行列の特定誤分類パターンを踏まえた**運用目的別 KPI**設定（例：冬型検出優先 vs 4A/3B の識別重視）

---

## 6. 図表参照（例）

- 学習推移（各手法）:
  - `*_iteration_vs_MacroRecall_majority.png`
  - `*_iteration_vs_MacroRecall_composite.png`
  - `*_iteration_vs_NodewiseMatchRate.png`
- 検証期のラベル別再現率:
  - `verification_results/*/*_verification_per_label_recall_*.png`
- 出力マップの代表表現:
  - `*_som_node_avg_all.png`, `*_som_node_medoid_all.png`, `*_som_node_true_medoid_all.png`

---

## 7. research_v2.md 反映案（ドラフト）

- 「実験結果」節を実測値で差し替え:
  - 検証 Macro（基本/複合）:  
    S1+SSIM5=0.3501/0.2157、S1=0.2754/0.1850、SSIM5=0.2531/0.1917、EUC=0.1958/0.1595
  - NodewiseMatchRate（代表性）:  
    S1=47%、S1+SSIM5=40%、SSIM5=30.3%、EUC=34%（最終反復付近）
  - 学習曲線所見（S1+SSIM5 は中盤〜後半にピーク、ベスト反復選択で向上余地）
- 「ラベル別性能」節を拡充:
  - 1/3B/4A での手法差、2B/2C/3C/3D/4B/5/6x の課題
- 「考察」節を補強:
  - SSIM5（局所構造）と S1（勾配）の相補性、融合でバランス最良
  - QE 横比較の非妥当性（スケール差）と収束挙動の確認

以上を基に、研究ノート（research_v2.md）の該当節を実測値ベースで改稿可能。
