import os
import sys
import xarray as xr
import pandas as pd
import numpy as np
from tqdm import tqdm # 進捗表示のためにtqdmライブラリをインポート
import calendar # うるう年判定用

def inspect_single_era5_file(filepath):
    """
    単一のERA5 NetCDFファイルを詳細にチェックし、結果を文字列のリストとして返す。
    画面への出力は行わない。
    """
    report_lines = []
    report_lines.append("-" * 70)
    report_lines.append(f"📄 ファイル診断: {os.path.basename(filepath)}")
    report_lines.append("-" * 70)
    
    try:
        with xr.open_dataset(filepath) as ds:
            # --- 1. 基本構造 ---
            report_lines.append("\n[1. 基本構造]")
            required_vars = ['msl', 'latitude', 'longitude', 'valid_time']
            all_vars_exist = all(v in ds for v in required_vars)
            if all_vars_exist:
                report_lines.append("  ✅ 必須変数（msl, latitude, longitude, valid_time）が存在します。")
            else:
                missing = [v for v in required_vars if v not in ds]
                report_lines.append(f"  ❌ エラー: 必須変数 {missing} が見つかりません。")
                return report_lines # これ以上チェックできないので終了

            # --- 2. 時間情報 ---
            report_lines.append("\n[2. 時間情報]")
            time_coord = ds['valid_time']
            time_values = pd.to_datetime(time_coord.values)
            year_in_file = time_values[0].year
            
            expected_year = int(os.path.basename(filepath).split('_')[-1].replace('.nc', ''))
            if year_in_file == expected_year:
                report_lines.append(f"  ✅ ファイル名とデータ内の年 ({year_in_file}) が一致します。")
            else:
                report_lines.append(f"  ❌ 警告: ファイル名({expected_year})とデータ内の年({year_in_file})が一致しません。")

            expected_days = 366 if calendar.isleap(year_in_file) else 365
            if time_coord.size == expected_days:
                report_lines.append(f"  ✅ 時間ステップ数 ({time_coord.size}) が正常です（{expected_days}日）。")
            else:
                report_lines.append(f"  ❌ エラー: 時間ステップ数 ({time_coord.size}) が期待値 ({expected_days}日) と異なります。")

            # --- 3. 空間格子情報 ---
            report_lines.append("\n[3. 空間格子情報]")
            lat_coord = ds['latitude']
            report_lines.append(f"  - 緯度 格子点数: {lat_coord.size}")
            report_lines.append(f"  - 緯度 範囲: {lat_coord.min().item():.2f}° ～ {lat_coord.max().item():.2f}°")
            if lat_coord.size > 1:
                spacing = abs(lat_coord[1].item() - lat_coord[0].item())
                report_lines.append(f"  - 緯度 格子間隔: 約 {spacing:.2f}°")

            lon_coord = ds['longitude']
            report_lines.append(f"  - 経度 格子点数: {lon_coord.size}")
            report_lines.append(f"  - 経度 範囲: {lon_coord.min().item():.2f}° ～ {lon_coord.max().item():.2f}°")
            if lon_coord.size > 1:
                spacing = abs(lon_coord[1].item() - lon_coord[0].item())
                report_lines.append(f"  - 経度 格子間隔: 約 {spacing:.2f}°")

            # --- 4. 気圧データの品質チェック ---
            report_lines.append("\n[4. 気圧データ(msl)の品質]")
            msl_data = ds['msl']
            msl_data.load()
            
            # ★★★ 改善点1: 整数オーバーフロー対策 ★★★
            # 統計計算の前に、より安全な倍精度浮動小数点数(float64)にデータ型を変換する
            msl_data_float64 = msl_data.astype(np.float64)
            
            # NaN/inf チェック
            nan_count = msl_data_float64.isnull().sum().item()
            inf_count = np.isinf(msl_data_float64.values).sum()
            
            if nan_count == 0:
                report_lines.append("  ✅ NaN（非数）は含まれていません。")
            else:
                report_lines.append(f"  ❌ エラー: {nan_count}個のNaNが含まれています。")
                
            if inf_count == 0:
                report_lines.append("  ✅ inf（無限大）は含まれていません。")
            else:
                report_lines.append(f"  ❌ エラー: {inf_count}個のinfが含まれています。")

            # 統計値と物理的妥当性チェック (float64に変換したデータを使用)
            mean_val_hpa = msl_data_float64.mean().item() / 100
            min_val_hpa = msl_data_float64.min().item() / 100
            max_val_hpa = msl_data_float64.max().item() / 100
            std_val_hpa = msl_data_float64.std().item() / 100
            
            report_lines.append(f"  - 単位: {msl_data.attrs.get('units', 'N/A')}")
            report_lines.append(f"  - 統計情報:")
            report_lines.append(f"    - 平均値: {mean_val_hpa:.2f} hPa")
            report_lines.append(f"    - 最小値: {min_val_hpa:.2f} hPa")
            report_lines.append(f"    - 最大値: {max_val_hpa:.2f} hPa")
            report_lines.append(f"    - 標準偏差: {std_val_hpa:.2f} hPa")

            # 平均値が妥当な範囲にあるか
            if 950 < mean_val_hpa < 1050:
                report_lines.append("  ✅ 平均値は物理的に妥当な範囲です。")
            else:
                report_lines.append("  ❌ エラー: 平均値が物理的にありえない値です。ファイルが破損している可能性があります。")

    except Exception as e:
        report_lines.append(f"\n❌❌❌ ファイル '{os.path.basename(filepath)}' の処理中に致命的なエラーが発生しました。 ❌❌❌")
        report_lines.append(f"詳細: {e}")
    
    report_lines.append("-" * 70 + "\n")
    return report_lines


def main():
    """
    メイン実行関数。期間内の全ファイルの存在を確認し、各ファイルを詳細に診断する。
    診断結果は最後にまとめて表示する。
    """
    # --- 設定 ---
    target_dir = './nc/era5_msl_small'
    start_year = 1940
    end_year = 2024

    # --- 1. 期間の網羅性チェック ---
    print("="*80)
    print(f"ERA5データファイル 網羅性チェック ({start_year}年～{end_year}年)")
    print(f"対象ディレクトリ: {target_dir}")
    print("="*80)

    if not os.path.isdir(target_dir):
        print(f"❌ エラー: ディレクトリ '{target_dir}' が存在しません。")
        return

    expected_years = range(start_year, end_year + 1)
    found_files = []
    missing_files = []

    for year in expected_years:
        filename = f"era5_msl_{year}.nc"
        filepath = os.path.join(target_dir, filename)
        if os.path.exists(filepath):
            found_files.append(filepath)
        else:
            missing_files.append(filename)

    print(f"\n結果: {len(expected_years)}年分のうち、{len(found_files)}個のファイルが見つかりました。")

    if missing_files:
        print(f"❌ 以下の {len(missing_files)} 個のファイルが見つかりません:")
        for f in missing_files:
            print(f"  - {f}")
    else:
        print("✅ 全ての年のファイルが揃っています。")

    # --- 2. 各ファイルの詳細診断 ---
    if not found_files:
        print("\n診断対象のファイルがないため、処理を終了します。")
        return

    print("\n" + "="*80)
    print("各ファイルの詳細診断を開始します")
    print("="*80)
    
    all_reports = []
    found_files.sort()
    
    # ★★★ 改善点2: 結果を一旦リストに格納し、tqdmで進捗を表示 ★★★
    for filepath in tqdm(found_files, desc="ファイルを診断中"):
        report_lines = inspect_single_era5_file(filepath)
        all_reports.extend(report_lines) # extendでリストを連結
        
    # --- 3. 診断結果の集約表示 ---
    print("\n" + "="*80)
    print("詳細診断結果レポート")
    print("="*80)
    
    for line in all_reports:
        print(line)

    print("すべての診断が完了しました。")

if __name__ == '__main__':
    main()