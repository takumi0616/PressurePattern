import cdsapi
import os
from tqdm import tqdm

# ================================================================================
# 設定項目
# ================================================================================

# --- 出力ディレクトリ ---
# このディレクトリに年ごとのNetCDFファイルが保存されます
OUTPUT_DIR = "./nc/era5_msl_small"

# --- ダウンロードしたい年リスト ---
# 1940年から2024年まで（2025年はまだデータが完全ではないため除外推奨）
YEARS_TO_DOWNLOAD = [str(year) for year in range(1940, 2025)] 

# --- CDS APIリクエストの基本設定 ---
# 年以外の共通パラメータをここで定義
BASE_REQUEST = {
    "product_type": "reanalysis",
    "variable": "mean_sea_level_pressure",
    "month": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"],
    "day": [
        "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12",
        "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24",
        "25", "26", "27", "28", "29", "30", "31"
    ],
    "time": "00:00",  # 取得する時間（UTC）。ここでは毎日00:00のデータを取得
    "area": [
        45, 125, 25, 145  # [North, West, South, East] - 日本周辺 small版[45, 125, 25, 145]
    ],
    "format": "netcdf", # ★改善点: 'data_format'ではなく'format'が正しいパラメータ名
}

# ================================================================================
# メイン処理
# ================================================================================

def download_era5_data():
    """
    指定された年リストに基づき、ERA5の海面更正気圧データを年ごとにダウンロードする。
    """
    print("="*80)
    print("ERA5 海面更正気圧データ ダウンロードプログラム")
    print(f"対象期間: {YEARS_TO_DOWNLOAD[0]}年 ～ {YEARS_TO_DOWNLOAD[-1]}年")
    print(f"保存先ディレクトリ: {OUTPUT_DIR}")
    print("="*80 + "\n")

    # --- 1. CDS APIクライアントの初期化 ---
    try:
        client = cdsapi.Client()
    except Exception as e:
        print("❌ エラー: cdsapi.Clientの初期化に失敗しました。")
        print("~/.cdsapirc ファイルが正しく設定されているか確認してください。")
        print(f"詳細: {e}")
        return

    # --- 2. 出力ディレクトリの作成 ---
    if not os.path.exists(OUTPUT_DIR):
        print(f"出力ディレクトリ '{OUTPUT_DIR}' を作成します。")
        os.makedirs(OUTPUT_DIR)
    
    # --- 3. 年ごとにループしてダウンロード ---
    print("ダウンロード処理を開始します...")
    
    # tqdmを使って進捗バーを表示
    for year in tqdm(YEARS_TO_DOWNLOAD, desc="全体の進捗"):
        
        # --- ファイル名とパスを定義 ---
        output_filename = f"era5_msl_{year}.nc"
        output_filepath = os.path.join(OUTPUT_DIR, output_filename)

        # --- ★改善点: 再開機能 ---
        # すでにファイルが存在する場合は、その年のダウンロードをスキップ
        if os.path.exists(output_filepath):
            print(f"\n✅ スキップ: {output_filename} は既に存在します。")
            continue

        # --- リクエストを作成 ---
        request = BASE_REQUEST.copy()
        request['year'] = year

        # --- ★改善点: エラーハンドリング ---
        # APIリクエスト中にエラーが発生しても、全体が止まらないようにする
        try:
            print(f"\n🚀 ダウンロード開始: {year}年")
            client.retrieve(
                "reanalysis-era5-single-levels",
                request,
                output_filepath
            )
            print(f"🎉 成功: {output_filename} を保存しました。")

        except Exception as e:
            print(f"\n❌ エラー: {year}年のダウンロード中に問題が発生しました。")
            print(f"   詳細: {e}")
            print("   次の年の処理に進みます...")
            # エラーが発生した場合、中途半端なファイルが残らないように削除
            if os.path.exists(output_filepath):
                os.remove(output_filepath)

    print("\n" + "="*80)
    print("すべてのダウンロード処理が完了しました。")
    print("="*80)


if __name__ == '__main__':
    download_era5_data()