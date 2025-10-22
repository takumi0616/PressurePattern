# -*- coding: utf-8 -*-
"""
main_v1 の設定値を集約するモジュール（PyTorch 版）

- データパス、使用する変数、前処理の有無、学習/検証の年、学習パラメータ
- 出力（モデル/履歴/正規化統計など）の保存先（要求により ./result 配下へ）

注意:
- 環境は environments_gpu/swinunet_env.yml を想定（PyTorch + CUDA）
- 使える変数は全て使用する方針（SELECTED_VARIABLES = ["ALL"]）
"""

import os

# この設定ファイルの場所（絶対パス）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 学習に用いる NetCDF ファイル（main_v1.py からはこのファイルの相対パスとして解決）
# ユーザ指定: ./data/nc/era5_all_data_pp.nc
DATA_PATH = "./data/nc/era5_all_data_pp.nc"

# 入力に使う物理量チャネル
# - 要求: 「使える変数は全て使う」
# - 実装: ["ALL"] と指定すると、NetCDF 内に存在する候補変数を自動で全採用
#   候補: "msl", "gh500", "t850", "u500", "v500", "u850", "v850", "r700", "r850", "vo850"
SELECTED_VARIABLES = ["ALL"]

# 季節特徴量（f1_season, f2_season）もチャネルとして使用（要求: 全て使う → True）
USE_SEASONAL_AS_CHANNELS = True

# 空間ダウンサンプリング係数（>1 で coarsen により緯度経度方向を間引き）
# メモリ不足(OOM)の回避や高速化のために 2 以上を推奨。1 で元解像度のまま。
COARSEN_FACTOR = 1

# 前処理
# - msl は Pa -> hPa に変換（領域平均の差し引きは行わない）
SLP_TO_HPA = True
# 互換性のためのエイリアス（旧名）。今後は SLP_TO_HPA を使用。
SLP_TO_HPA_AND_REMOVE_AREA_MEAN = SLP_TO_HPA

# 乱数シード
RANDOM_SEED = 42

# データ分割（年ベース）
TRAIN_YEARS = list(range(1991, 1998))  # 1991–1997
VAL_YEARS   = list(range(1998, 2001))  # 1998–2000

# 学習パラメータ
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# EMA/評価関連
USE_EMA = False           # AveragedModel を使う（無効にすると常に生モデルで評価）
EVAL_WITH_EMA = False     # 検証・最終保存はEMAを使用（Falseで生モデル）
EMA_UPDATE_BN = False     # 各エポック後にEMAのBN統計を再推定(update_bn)

# マルチラベルの閾値（0–1 の確率をラベル有無に変換するときのしきい値）
PREDICTION_THRESHOLD = 0.7


# クラス不均衡対策：陽性クラス重みを有効化（BCEWithLogitsLoss の pos_weight に反映）
USE_POSITIVE_CLASS_WEIGHTS = True

# 出力先（要求: ./result 配下に出力）
OUTPUT_DIR = "./result"
MODEL_NAME = "cnn_v1_torch"

# 保存パス
FINAL_MODEL_PATH      = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}.pt")   # PyTorch state_dict
BEST_WEIGHTS_PATH     = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}.best.pt")
HISTORY_JSON_PATH     = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_history.json")
NORM_STATS_JSON_PATH  = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_norm.json")
VAL_REPORT_JSON_PATH  = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_val_report.json")
TENSORBOARD_LOGS_DIR  = os.path.join(OUTPUT_DIR, "logs")  # 任意（使わない場合も可）

# ラベル体系（基本型 15 クラス）
# README 準拠：1, 2A, 2B, 2C, 2D, 3A, 3B, 3C, 3D, 4A, 4B, 5, 6A, 6B, 6C
BASE_LABELS = [
    "1",
    "2A", "2B", "2C", "2D",
    "3A", "3B", "3C", "3D",
    "4A", "4B",
    "5",
    "6A", "6B", "6C",
]

# 自動選択の候補（存在チェックして採用）
# 取得・派生する可能性のある物理量チャネル候補（NetCDF 内に存在するもののみ自動採用）
VAR_CANDIDATES = [
    # 既存
    "msl",
    "gh500",
    "t850",
    "u500", "v500",
    "u850", "v850",
    "r700", "r850",
    "vo850",
    # 追加（圧力面/派生）
    "vo500",               # 500hPa 相対渦度
    "gh1000",              # 1000hPa ジオポテンシャル高度（m）
    "thk_1000_500",        # 厚さ（1000-500hPa, m）
    "grad_thk_1000_500",   # 厚さの水平勾配（m m-1）
    "q850",                # 850hPa 比湿（kg kg-1）
    "thetae850",           # 850hPa 相当温位（K）
    "grad_thetae850",      # |∇θe|（K m-1）
    "div850",              # 850hPa 発散（s-1）
    "w500",                # 500hPa 鉛直速度（Pa s-1）
    "w700",                # 700hPa 鉛直速度（Pa s-1）
    "vmag500", "vmag850",  # 風速 |V|（m s-1）
    "shear_850_500",       # ベクトル鉛直風シア |V500-V850|（m s-1）
    "mfc850",              # 850hPa 水蒸気フラックス収束（-∇·qV, s-1）
    # 追加（単一レベル）
    "tcwv",                # 可降水量（kg m-2）
    "ivt",                 # 統合水蒸気輸送量の大きさ（kg m-1 s-1）
    "ivt_dir",             # IVT の方位（度）
    "tcc",                 # 総雲量（0-1）
    "tp",                  # 総降水量（m）
    "u10", "v10",          # 10m 風（m s-1）
    "vmag10",              # 10m 風速（m s-1）
    "tclw", "tciw",        # 全雲液水量/氷水量
    "tcrw", "tcsw",        # 全降水液水量/雪水量
    "sst",                 # 海面水温（K）
    # SLP の時間・空間派生
    "msl_dt24",            # 24h 変化量（Pa / 24h）
    "grad_msl",            # 水平勾配大きさ（Pa m-1）
    "lap_msl",             # ラプラシアン近似（Pa m-2）
]

# 以降は main_v1.py がこの設定を読み込み、学習/評価を実行します。
