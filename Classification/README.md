# コマンド

```bash
nohup python main_v1.py > main_v1.log 2>&1 &

notify-run via-tml2 -- nohup python main_v1.py > main_v1.log 2>&1 &

pkill -f "main_v1.py"

pkill -f "run_backbones.py"

notify-run via-tml2 -- nohup python run_backbones.py --backbones resnet_small,deformable,convnext_tiny > run_backbones.log 2>&1 &
```

via-tml2→mac

```bash
rsync -avz --progress via-tml2:/home/s233319/docker_miniconda/src/PressurePattern/Classification/result_1 /Users/takumi0616/Develop/docker_miniconda/src/PressurePattern/Classification/result_via-tml2
```

## 研究概要（現状構成の要約）

- 目的: ERA5 由来の日本周辺の気圧配置を、吉野の基本型（15 サブタイプ）に基づきマルチラベルで自動分類する。
- 問題設定: 1 サンプルにつき複数ラベル（複合型/移行型）を許容するマルチラベル分類。
- 入力データ: SLP を含む多数の物理量（圧力面/単一レベル/派生量）を時刻同期して結合した NetCDF（daily 09 JST 相当）。
- モデル: シンプルな CNN（SimpleCNN）。ただし不均衡対策・学習安定化・推論時のしきい値最適化など、ベースライン CNN に各種テクニックを上乗せ。
- 評価: エポックごとに macro Average Precision（mAP）と検証損失を記録。最終的には検証集合で推定した「クラス別最適しきい値」で 2 値化し、classification_report を保存。

---

## データ取得・作成パイプライン

- スクリプト: `data/download_era5_large.py`
  - 期間: 1991–2000 年
  - 対象域: 15–55°N, 115–155°E（ERA5 矩形領域）
  - 時刻: 00:00 UTC（= 09:00 JST 相当）
  - 主な処理:
    - ERA5 SLP（単一レベル）/ 圧力面データ（500/700/850 hPa 等）/ 単一レベル諸量（TCWV, IVT, TCC, TP, 10m 風, SST ほか）を取得
    - 圧力面からの派生（例）
      - gh500, gh1000, thickness thk_1000_500, grad_thk_1000_500
      - u500/v500, u850/v850, vmag500/vmag850, shear_850_500
      - t850, q850, thetae850（Bolton 1980 近似）, grad_thetae850
      - vo500/vo850, d850(div850), w500/w700
      - mfc850（-∇·(qV)）
    - 単一レベルからの派生（例）
      - ivt（ベクトル合成）, ivt_dir（arctan2）, vmag10（10m 風速）
    - SLP の時間/空間派生
      - msl_dt24（24h 変化）, grad_msl（水平勾配）, lap_msl（簡易ラプラシアン）
    - 季節特徴（天文学的季節に基づく連続特徴）
      - f1_season = cos(2π t), f2_season = cos(4π t)
    - ラベル付与: `data/label.py` の `data_label_dict` により日付 → ラベル文字列をアサイン
    - 結合と検証:
      - 時刻・緯度経度の一致を担保して merge
      - NaN/inf の排除を検査
    - 出力: `./data/nc/era5_all_data_pp.nc`（LZ4 圧縮, h5netcdf, 時間チャンク=1）

参考: 生成される主な変数は `main_v1_config.py` の `VAR_CANDIDATES` と `result/cnn_v1_torch_norm.json` の `channel_names` を参照。

---

## 学習用データ構成と前処理

- データ読込: `main_v1.py`
  - エンジン: `xarray`（h5netcdf 優先, LZ4 対応）
  - 時刻処理: `valid_time` を UTC→`Asia/Tokyo` に変換（JST 想定）
- 変数選択:
  - `SELECTED_VARIABLES = ["ALL"]` により、`VAR_CANDIDATES` のうち NetCDF に存在するものを自動全採用
  - 例（実際に採用されたチャネルの一例）: `msl, gh500, t850, u500, v500, u850, v850, r700, r850, vo850, vo500, gh1000, thk_1000_500, grad_thk_1000_500, q850, thetae850, grad_thetae850, div850, w500, w700, vmag500, vmag850, shear_850_500, mfc850, tcwv, ivt, ivt_dir, tcc, tp, u10, v10, vmag10, tclw, tciw, tcrw, tcsw, sst, msl_dt24, grad_msl, lap_msl, f1_season, f2_season`
- 前処理:
  - `SLP_TO_HPA = True` により `msl` を Pa→hPa に変換（領域平均の差し引きは不実施）
  - 空間ダウンサンプリング: `COARSEN_FACTOR=1`（変更で coarsen による間引き可能）
  - 非有限値（NaN, ±inf）は 0 へ置換（数値安定性）
  - 正規化: 学習集合のチャネル別 `mean/std` で標準化（検証集合にも適用）
- 配列形状:
  - xarray → NumPy (N, H, W, C) → PyTorch (N, C, H, W)
- データ分割:
  - 学習年: 1991–1997
  - 検証年: 1998–2000
  - ラベル欠損（全 0）は除外

---

## ラベル体系とマルチラベル化

- 基本型（15 クラス）:`["1","2A","2B","2C","2D","3A","3B","3C","3D","4A","4B","5","6A","6B","6C"]`
- 複合/移行ラベルの処理:
  - `'+'` および `'-'` で分割し、含まれる基本型の和集合としてマルチホット化（例: `"3B+4B"` → `3B=1, 4B=1`、`"2A-1"` → `2A=1, 1=1`）
- 学習時の損失は各クラスを独立に 0/1 回帰する形式（sigmoid 前のロジットを出力）

---

## モデル

- 実装: `CNN.py`
- 既定モデル: `SimpleCNN`（`build_cnn()` は SimpleCNN を返す）
  - 特徴抽出: Conv(32) → ReLU → MaxPool → Dropout 0.25 → Conv(64) → ReLU → MaxPool → Dropout 0.25 → Conv(64) → ReLU → MaxPool → Dropout 0.25 → AdaptiveAvgPool2d(4×4)
  - 分類器: Flatten → Linear(64×4×4→512) → ReLU → Dropout 0.5 → Linear(512→15)（ロジット）
- 参考実装（オプション）:
  - `ResNetSmall`（SE 付き残差ブロック, GAP ヘッド）を同ファイル内に用意（現設定では未使用）

---

## 学習手法と「通常の CNN への上乗せテクニック」

設定は `main_v1_config.py` で一元管理。以下がベースライン CNN に追加している主な工夫。

- 不均衡対策
  - 陽性クラス重み（`pos_weight`）: `compute_pos_weights(y_train)` によりクラス頻度の逆比相当を `BCE/Focal` に注入
  - サンプル再重み: `WeightedRandomSampler`（サンプル重み = Σ 1/freq(label)）でミニバッチのクラス分布をバランス化
- 損失関数
  - `FocalLoss`（γ=2.0, `pos_weight` 併用）を既定で使用（`USE_FOCAL_LOSS=True`）。BCE に切替も可
- 最適化と正則化
  - オプティマイザ: `AdamW`（weight decay を decouple）
  - `weight_decay=1e-4`（L2 正則化）
- 学習率制御
  - `ReduceLROnPlateau`（`SELECTION_METRIC` に応じて mode を `max`/`min` 切替）
  - 監視指標（model selection）: 既定は `val_map`（しきい値に依存しない AP の macro 平均）
- しきい値最適化（推論時）
  - 検証集合の `precision_recall_curve` からクラス別に F1 最大点の閾値を推定・保存（`*_thresholds.json`）
  - 既定しきい値 `PREDICTION_THRESHOLD=0.7` はフォールバックとして使用
- その他
  - 並列 I/O ワーカー数の制御（省メモリのため既定は小さめ）
  - Matplotlib 非表示バッチ（`Agg`）で損失曲線 png を出力
  - 数値安定性のための NaN/inf 置換

---

## 学習・評価の設定（既定値）

- 乱数シード: `42`
- エポック: `50`
- バッチサイズ: `128`
- 学習率: `1e-3`
- 重み減衰: `1e-4`
- 監視/選択指標: `SELECTION_METRIC="val_map"`
- Focal γ: `2.0`
- Sampler / LR scheduler: 有効（`USE_WEIGHTED_SAMPLER=True`, `USE_LR_SCHEDULER=True`）

---

## 出力アーティファクト（`./result` 配下）

- 最終/ベスト重み
  - `{MODEL_NAME}.pt`（最終はベスト重みを再保存）
  - `{MODEL_NAME}.best.pt`
- 学習履歴: `{MODEL_NAME}_history.json`（train/val loss, val mAP, best など）
- 正規化統計: `{MODEL_NAME}_norm.json`（チャンネル名, mean/std, 年度, 指標 など）
- クラス別最適しきい値: `{MODEL_NAME}_thresholds.json`
- 検証レポート: `{MODEL_NAME}_val_report.json`（classification_report）
- 損失曲線: `{MODEL_NAME}_loss_curve.png`
- ログ: `main_v1.log`（例: nohup 実行時）

`MODEL_NAME` 既定値: `cnn_v1_torch`

---

## 実行手順（再掲）

1. 環境

- PyTorch + CUDA を想定（`environments_gpu/swinunet_env.yml` など）
- `~/.cdsapirc` 設定済み（データを再生成する場合）

2. データ

- 既存の `./data/nc/era5_all_data_pp.nc` を使用、または
- `python -m src.PressurePattern.Classification.data.download_era5_large`（時間/容量に注意）

3. 学習

- 本ディレクトリで:
  - バックグラウンド実行: 先頭の「コマンド」参照
  - フォアグラウンド実行: `python main_v1.py`

4. 結果確認

- `result/` 配下の JSON/PNG/pt を参照

---

## ディレクトリ/ファイル（抜粋）

- データ生成
  - `data/download_era5_large.py`: ERA5 取得・派生量作成・季節特徴付与・マージ・出力
- 学習
  - `main_v1_config.py`: 変数/年/ハイパラ/出力先の設定
  - `main_v1.py`: 学習・評価・しきい値最適化の本体
  - `CNN.py`: SimpleCNN（既定）/ ResNetSmall（参考）のモデル定義、`pos_weight` 計算
- 結果
  - `result/*.json, *.pt, *.png`

---

## 現状のテクニック一覧（通常の CNN に対する追加点の要約）

- マルチラベル化（複合/移行ラベルを和集合化）
- チャンネル拡張（圧力面/単一レベル/派生量/季節性）
- チャンネル別標準化（学習集合統計）
- 不均衡対策
  - `pos_weight` によるクラス重み
  - `WeightedRandomSampler` によるサンプル重み
  - `FocalLoss(γ=2)` による難例強調
- 学習率制御/正則化
  - `AdamW` + `ReduceLROnPlateau`
  - `weight_decay=1e-4`
- モデル選択
  - `val_map` 最大で選択（mAP はしきい値非依存）
- 推論の 2 値化改善
  - 検証集合由来の「クラス別最適しきい値」（F1 最大点）で 2 値化
- 数値安定化/省メモリの工夫
  - NaN/inf → 0 置換、DataLoader ワーカー抑制、LZ4 圧縮, 時間チャンク=1

---

## 注意点と今後の拡張

- 早期終了（EarlyStopping）は未導入。`val_map` 停滞で打ち切ると計算効率/過学習抑制が見込める。
- `COARSEN_FACTOR>1` の検討（高速化/正則化）。
- `ResNetSmall` などアーキテクチャ拡張、EMA, AMP, OneCycleLR などの導入。
- 損失/再重みづけのアブレーション（`FocalLoss`/`BCE`/`ASL`、Sampler/pos_weight の組合せ）。
- しきい値の業務最適化（クラス別に Fβ で最適化、確率キャリブレーションの適用など）。
