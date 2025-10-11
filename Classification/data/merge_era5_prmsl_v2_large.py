import os
import sys
import xarray as xr
import pandas as pd
import numpy as np
from tqdm import tqdm
import calendar
from datetime import datetime, date, timedelta
try:
    # Prefer package-relative import (works when this file is part of a package)
    from .label import data_label_dict  # type: ignore[import]
except Exception:
    # Fallback for running this file directly as a script
    import os as _os, sys as _sys
    _THIS_DIR = _os.path.dirname(_os.path.abspath(__file__))
    if _THIS_DIR not in _sys.path:
        _sys.path.insert(0, _THIS_DIR)
    from label import data_label_dict

### 改善点: 1940年～2024年の分点・至点の日付データ（日本標準時） ###
# 国立天文台の暦計算室のデータを参考に作成
solstice_equinox_dates_jst = {
    1940: {'spring': (3, 21), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 22)},
    1941: {'spring': (3, 21), 'summer': (6, 22), 'autumn': (9, 23), 'winter': (12, 22)},
    1942: {'spring': (3, 21), 'summer': (6, 22), 'autumn': (9, 24), 'winter': (12, 22)},
    1943: {'spring': (3, 21), 'summer': (6, 22), 'autumn': (9, 24), 'winter': (12, 23)},
    1944: {'spring': (3, 20), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 22)},
    1945: {'spring': (3, 21), 'summer': (6, 22), 'autumn': (9, 23), 'winter': (12, 22)},
    1946: {'spring': (3, 21), 'summer': (6, 22), 'autumn': (9, 24), 'winter': (12, 22)},
    1947: {'spring': (3, 21), 'summer': (6, 22), 'autumn': (9, 24), 'winter': (12, 23)},
    1948: {'spring': (3, 20), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 22)},
    1949: {'spring': (3, 21), 'summer': (6, 22), 'autumn': (9, 23), 'winter': (12, 22)},
    1950: {'spring': (3, 21), 'summer': (6, 22), 'autumn': (9, 23), 'winter': (12, 22)},
    1951: {'spring': (3, 21), 'summer': (6, 22), 'autumn': (9, 24), 'winter': (12, 23)},
    1952: {'spring': (3, 21), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 22)},
    1953: {'spring': (3, 21), 'summer': (6, 22), 'autumn': (9, 23), 'winter': (12, 22)},
    1954: {'spring': (3, 21), 'summer': (6, 22), 'autumn': (9, 23), 'winter': (12, 22)},
    1955: {'spring': (3, 21), 'summer': (6, 22), 'autumn': (9, 24), 'winter': (12, 23)},
    1956: {'spring': (3, 20), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 22)},
    1957: {'spring': (3, 21), 'summer': (6, 22), 'autumn': (9, 23), 'winter': (12, 22)},
    1958: {'spring': (3, 21), 'summer': (6, 22), 'autumn': (9, 23), 'winter': (12, 22)},
    1959: {'spring': (3, 21), 'summer': (6, 22), 'autumn': (9, 24), 'winter': (12, 23)},
    1960: {'spring': (3, 21), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 22)},
    1961: {'spring': (3, 21), 'summer': (6, 22), 'autumn': (9, 23), 'winter': (12, 22)},
    1962: {'spring': (3, 21), 'summer': (6, 22), 'autumn': (9, 23), 'winter': (12, 22)},
    1963: {'spring': (3, 21), 'summer': (6, 22), 'autumn': (9, 24), 'winter': (12, 23)},
    1964: {'spring': (3, 20), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 22)},
    1965: {'spring': (3, 21), 'summer': (6, 22), 'autumn': (9, 23), 'winter': (12, 22)},
    1966: {'spring': (3, 21), 'summer': (6, 22), 'autumn': (9, 23), 'winter': (12, 22)},
    1967: {'spring': (3, 21), 'summer': (6, 22), 'autumn': (9, 24), 'winter': (12, 22)},
    1968: {'spring': (3, 20), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 21)},
    1969: {'spring': (3, 21), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 22)},
    1970: {'spring': (3, 21), 'summer': (6, 22), 'autumn': (9, 23), 'winter': (12, 22)},
    1971: {'spring': (3, 21), 'summer': (6, 22), 'autumn': (9, 23), 'winter': (12, 22)},
    1972: {'spring': (3, 20), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 22)},
    1973: {'spring': (3, 21), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 22)},
    1974: {'spring': (3, 21), 'summer': (6, 22), 'autumn': (9, 23), 'winter': (12, 22)},
    1975: {'spring': (3, 21), 'summer': (6, 22), 'autumn': (9, 24), 'winter': (12, 22)},
    1976: {'spring': (3, 21), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 21)},
    1977: {'spring': (3, 21), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 22)},
    1978: {'spring': (3, 21), 'summer': (6, 22), 'autumn': (9, 23), 'winter': (12, 22)},
    1979: {'spring': (3, 21), 'summer': (6, 22), 'autumn': (9, 23), 'winter': (12, 22)},
    1980: {'spring': (3, 20), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 22)},
    1981: {'spring': (3, 21), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 22)},
    1982: {'spring': (3, 21), 'summer': (6, 22), 'autumn': (9, 23), 'winter': (12, 22)},
    1983: {'spring': (3, 21), 'summer': (6, 22), 'autumn': (9, 23), 'winter': (12, 22)},
    1984: {'spring': (3, 20), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 21)},
    1985: {'spring': (3, 21), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 22)},
    1986: {'spring': (3, 21), 'summer': (6, 22), 'autumn': (9, 23), 'winter': (12, 22)},
    1987: {'spring': (3, 21), 'summer': (6, 22), 'autumn': (9, 23), 'winter': (12, 22)},
    1988: {'spring': (3, 20), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 21)},
    1989: {'spring': (3, 21), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 22)},
    1990: {'spring': (3, 21), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 22)},
    1991: {'spring': (3, 21), 'summer': (6, 22), 'autumn': (9, 23), 'winter': (12, 22)},
    1992: {'spring': (3, 20), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 21)},
    1993: {'spring': (3, 20), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 22)},
    1994: {'spring': (3, 21), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 22)},
    1995: {'spring': (3, 21), 'summer': (6, 22), 'autumn': (9, 23), 'winter': (12, 22)},
    1996: {'spring': (3, 20), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 21)},
    1997: {'spring': (3, 20), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 22)},
    1998: {'spring': (3, 21), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 22)},
    1999: {'spring': (3, 21), 'summer': (6, 22), 'autumn': (9, 23), 'winter': (12, 22)},
    2000: {'spring': (3, 20), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 22)},
    2001: {'spring': (3, 20), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 22)},
    2002: {'spring': (3, 21), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 22)},
    2003: {'spring': (3, 21), 'summer': (6, 22), 'autumn': (9, 23), 'winter': (12, 22)},
    2004: {'spring': (3, 20), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 22)},
    2005: {'spring': (3, 20), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 22)},
    2006: {'spring': (3, 21), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 22)},
    2007: {'spring': (3, 21), 'summer': (6, 22), 'autumn': (9, 23), 'winter': (12, 22)},
    2008: {'spring': (3, 20), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 21)},
    2009: {'spring': (3, 20), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 22)},
    2010: {'spring': (3, 21), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 22)},
    2011: {'spring': (3, 21), 'summer': (6, 22), 'autumn': (9, 23), 'winter': (12, 22)},
    2012: {'spring': (3, 20), 'summer': (6, 21), 'autumn': (9, 22), 'winter': (12, 21)},
    2013: {'spring': (3, 20), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 22)},
    2014: {'spring': (3, 21), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 22)},
    2015: {'spring': (3, 21), 'summer': (6, 22), 'autumn': (9, 23), 'winter': (12, 22)},
    2016: {'spring': (3, 20), 'summer': (6, 21), 'autumn': (9, 22), 'winter': (12, 21)},
    2017: {'spring': (3, 20), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 22)},
    2018: {'spring': (3, 21), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 22)},
    2019: {'spring': (3, 21), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 22)},
    2020: {'spring': (3, 20), 'summer': (6, 21), 'autumn': (9, 22), 'winter': (12, 21)},
    2021: {'spring': (3, 20), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 22)},
    2022: {'spring': (3, 21), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 22)},
    2023: {'spring': (3, 21), 'summer': (6, 21), 'autumn': (9, 23), 'winter': (12, 22)},
    2024: {'spring': (3, 20), 'summer': (6, 21), 'autumn': (9, 22), 'winter': (12, 21)},
}


def add_astronomical_seasonal_encoding(ds: xr.Dataset) -> xr.Dataset:
    """
    データセットに天文準拠の季節性エンコーディング(f1, f2)を追加する。
    f1: 夏と冬を対比させる指標 (冬至:1, 夏至:-1, 春秋分:0)
    f2: 春と秋をグループ化する指標 (冬至・夏至:1, 春秋分:-1)
    """
    timestamps = ds['valid_time'].to_series()

    # 1. すべての冬至の日付オブジェクトのリストを作成
    winter_solstices = []
    years = range(min(solstice_equinox_dates_jst.keys()) - 1, max(solstice_equinox_dates_jst.keys()) + 2)
    for year in years:
        if year in solstice_equinox_dates_jst:
            month, day = solstice_equinox_dates_jst[year]['winter']
        else:
            if year + 1 in solstice_equinox_dates_jst:
                month, day = solstice_equinox_dates_jst[year + 1]['winter']
            elif year - 1 in solstice_equinox_dates_jst:
                month, day = solstice_equinox_dates_jst[year - 1]['winter']
            else:
                month, day = 12, 22
        winter_solstices.append(datetime(year, month, day))

    winter_solstices = sorted(list(set(winter_solstices)))
    ws_series = pd.Series(winter_solstices, index=winter_solstices)

    # 2. 各タイムスタンプに対応する「前の冬至」と「次の冬至」を検索
    df = pd.DataFrame(index=timestamps.index)
    df['ws_start'] = pd.merge_asof(
        left=df,
        right=ws_series.to_frame('ws_start'),
        left_index=True,
        right_index=True,
        direction='backward'
    )['ws_start']

    df['ws_end'] = pd.merge_asof(
        left=df,
        right=ws_series.to_frame('ws_end'),
        left_index=True,
        right_index=True,
        direction='forward'
    )['ws_end']

    # 3. ★★★ NaN発生の根本原因を解決 ★★★
    # merge_asof によって生じる可能性のある NaT (Not a Time) を完全に埋める
    df['ws_start'] = df['ws_start'].fillna(method='bfill') # 前方の日付で埋める
    df['ws_end'] = df['ws_end'].fillna(method='ffill')     # 後方の日付で埋める

    # NaTが残っていないか最終チェック (念のため)
    if df.isnull().values.any():
        raise ValueError("Seasonal encoding failed: NaT values still exist after fill.")

    # 4. 正規化された時間 `t` を計算
    time_passed = (df.index - df['ws_start']).dt.total_seconds()
    total_time = (df['ws_end'] - df['ws_start']).dt.total_seconds()

    # ゼロ除算を完全に回避するロジック
    # total_timeが0の場合、tは0とする（期間の開始点）
    t = np.divide(time_passed, total_time, out=np.zeros_like(time_passed, dtype=float), where=(total_time != 0))

    # 5. f1 と f2 を計算
    angle = 2 * np.pi * t
    f1_season = np.cos(angle)
    f2_season = np.cos(2 * angle)

    # 6. データセットに新しい変数を追加
    ds['f1_season'] = xr.DataArray(f1_season, dims=['valid_time'], coords={'valid_time': ds['valid_time']})
    ds['f2_season'] = xr.DataArray(f2_season, dims=['valid_time'], coords={'valid_time': ds['valid_time']})

    ds['f1_season'].attrs['long_name'] = 'Seasonal feature f1 (winter/summer contrast)'
    ds['f1_season'].attrs['description'] = 'cos(2*pi*t) where t is normalized time from winter solstice to the next. Winter=+1, Summer=-1.'
    ds['f2_season'].attrs['long_name'] = 'Seasonal feature f2 (equinox grouping)'
    ds['f2_season'].attrs['description'] = 'cos(4*pi*t) where t is normalized time from winter solstice to the next. Solstices=+1, Equinoxes=-1.'

    return ds


def inspect_single_era5_file(filepath):
    """
    単一のERA5 NetCDFファイルを詳細にチェックし、結果を文字列のリストとして返す。
    """
    report_lines = []
    report_lines.append("-" * 70)
    report_lines.append(f"📄 ファイル診断: {os.path.basename(filepath)}")
    report_lines.append("-" * 70)
    
    try:
        with xr.open_dataset(filepath) as ds:
            # [1] 基本構造
            report_lines.append("\n[1. 基本構造]")
            required_vars = ['msl', 'latitude', 'longitude', 'valid_time']
            all_vars_exist = all(v in ds for v in required_vars)
            if all_vars_exist:
                report_lines.append("  ✅ 必須変数（msl, latitude, longitude, valid_time）が存在します。")
            else:
                missing = [v for v in required_vars if v not in ds]
                report_lines.append(f"  ❌ エラー: 必須変数 {missing} が見つかりません。")
                return report_lines

            # [2] 時間情報
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

            # [3] 空間格子情報
            report_lines.append("\n[3. 空間格子情報]")
            lat_coord = ds['latitude']
            report_lines.append(f"  - 緯度 格子点数: {lat_coord.size}")
            report_lines.append(f"  - 緯度 範囲: {lat_coord.min().item():.2f}° ～ {lat_coord.max().item():.2f}°")
            
            lon_coord = ds['longitude']
            report_lines.append(f"  - 経度 格子点数: {lon_coord.size}")
            report_lines.append(f"  - 経度 範囲: {lon_coord.min().item():.2f}° ～ {lon_coord.max().item():.2f}°")

            # [4] 気圧データの品質チェック
            report_lines.append("\n[4. 気圧データ(msl)の品質]")
            msl_data = ds['msl']
            msl_data.load()
            
            msl_data_float64 = msl_data.astype(np.float64)
            
            nan_count = msl_data_float64.isnull().sum().item()
            inf_count = np.isinf(msl_data_float64.values).sum()
            
            report_lines.append("  ✅ NaN（非数）は含まれていません。" if nan_count == 0 else f"  ❌ エラー: {nan_count}個のNaNが含まれています。")
            report_lines.append("  ✅ inf（無限大）は含まれていません。" if inf_count == 0 else f"  ❌ エラー: {inf_count}個のinfが含まれています。")

            mean_val_hpa = msl_data_float64.mean().item() / 100
            min_val_hpa = msl_data_float64.min().item() / 100
            max_val_hpa = msl_data_float64.max().item() / 100
            
            report_lines.append(f"  - 単位: {msl_data.attrs.get('units', 'N/A')}")
            report_lines.append(f"  - 統計情報 (平均: {mean_val_hpa:.2f}, 最小: {min_val_hpa:.2f}, 最大: {max_val_hpa:.2f}) hPa")

            if 950 < mean_val_hpa < 1050:
                report_lines.append("  ✅ 平均値は物理的に妥当な範囲です。")
            else:
                report_lines.append("  ❌ エラー: 平均値が物理的にありえない値です。")

    except Exception as e:
        report_lines.append(f"\n❌❌❌ ファイル '{os.path.basename(filepath)}' の処理中に致命的なエラーが発生しました。 ❌❌❌")
        report_lines.append(f"詳細: {e}")
    
    report_lines.append("-" * 70 + "\n")
    return report_lines


def validate_combined_file(filepath):
    """
    生成された統合NetCDFファイルを検証する。
    """
    print("\n" + "="*80)
    print(f"最終生成ファイル '{filepath}' の検証")
    print("="*80)
    
    if not os.path.exists(filepath):
        print(f"❌ エラー: ファイル '{filepath}' が見つかりません。")
        return

    try:
        with xr.open_dataset(filepath) as ds:
            print("✅ ファイルを正常に読み込めました。")
            
            # --- 変数の存在チェック (修正) ---
            required_vars = ['msl', 'f1_season', 'f2_season', 'label']
            missing = [v for v in required_vars if v not in ds]
            if not missing:
                print(f"✅ 必須変数 {required_vars} がすべて存在します。")
            else:
                print(f"❌ エラー: 必須変数 {missing} が見つかりません。")
                return

            # --- 各変数の品質チェック ---
            # 1. msl (海面更正気圧)
            print("\n[msl の品質チェック]")
            msl_data = ds['msl']
            nan_count = msl_data.isnull().sum().item()
            inf_count = np.isinf(msl_data.values).sum()
            print(f"  - NaN（非数）の数: {nan_count}")
            print(f"  - inf（無限大）の数: {inf_count}")
            if nan_count == 0 and inf_count == 0:
                print("  ✅ 欠損値はありません。")
            else:
                print("  ❌ 欠損値が含まれています。")

            # 2. 季節性エンコーディング (修正)
            print("\n[季節性エンコーディング変数の品質チェック]")
            for var_name in ['f1_season', 'f2_season']:
                var_data = ds[var_name]
                nan_count = var_data.isnull().sum().item()
                min_val, max_val = var_data.min().item(), var_data.max().item()
                print(f"  - {var_name}:")
                print(f"    - NaNの数: {nan_count}")
                print(f"    - 範囲: {min_val:.4f} ～ {max_val:.4f}")
                if nan_count == 0 and -1.0001 <= min_val and max_val <= 1.0001: # 浮動小数点誤差を考慮
                    print("    ✅ 正常です。")
                else:
                    print("    ❌ 異常な値または欠損値が含まれています。")

            # 3. label (ラベル情報)
            print("\n[ラベル情報の品質チェック]")
            label_data = ds['label']
            total_labels = label_data.size
            non_na_labels = np.sum(label_data.values != 'N/A').item()
            print(f"  - 総データ点数: {total_labels}")
            print(f"  - ラベルが付与されたデータ数 ('N/A'以外): {non_na_labels}")
            
            expected_label_count = len(data_label_dict)
            if non_na_labels == expected_label_count:
                print(f"  ✅ 付与されたラベル数が期待値 ({expected_label_count}) と一致します。")
            else:
                print(f"  ❌ 付与されたラベル数 ({non_na_labels}) が期待値 ({expected_label_count}) と異なります。")

            # ラベル期間のチェック
            label_dates = [k for k, v in data_label_dict.items() if v != 'N/A']
            if label_dates:
                start_label_date = min(label_dates).strftime('%Y-%m-%d')
                end_label_date = max(label_dates).strftime('%Y-%m-%d')
                label_subset = ds.sel(valid_time=slice(start_label_date, end_label_date))
                non_na_in_period = np.sum(label_subset['label'].values != 'N/A').item()
                if non_na_in_period == expected_label_count:
                     print(f"  ✅ ラベルはすべて期待される期間内 ({start_label_date}～{end_label_date}) に存在します。")
                else:
                     print("  ❌ ラベルが期待される期間外に存在するか、期間内に不足しています。")

    except Exception as e:
        print(f"\n❌❌❌ ファイル '{filepath}' の検証中にエラーが発生しました。 ❌❌❌")
        print(f"詳細: {e}")


def main():
    """
    メイン実行関数。
    """
    # --- 設定 ---
    target_dir = './nc/era5_msl_large'
    start_year = 1940
    end_year = 2024
    output_filename = "prmsl_era5_all_data_seasonal_large.nc"

    # =========================================================================
    # Phase 1: ファイルの網羅性チェック
    # =========================================================================
    print("="*80)
    print(f"Phase 1: ERA5データファイル 網羅性チェック ({start_year}年～{end_year}年)")
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
        print("必要なファイルが不足しているため、処理を中断します。")
        return
    else:
        print("✅ 全ての年のファイルが揃っています。")
    
    found_files.sort()

    # =========================================================================
    # Phase 2: 各ファイルの詳細診断
    # =========================================================================
    print("\n" + "="*80)
    print("Phase 2: 各ファイルの詳細診断")
    print("="*80)
    
    all_reports = []
    for filepath in tqdm(found_files, desc="ファイルを診断中"):
        report_lines = inspect_single_era5_file(filepath)
        all_reports.extend(report_lines)
        
    print("\n--- 個別ファイル診断レポート ---")
    for line in all_reports:
        print(line, end='')
    print("--- レポート終了 ---\n")

    # =========================================================================
    # Phase 3: 全データの結合と前処理
    # =========================================================================
    print("\n" + "="*80)
    print("Phase 3: 全データの結合と前処理")
    print("="*80)

    try:
        print("1. 全NetCDFファイルを結合しています...")
        combined_ds = xr.open_mfdataset(
            found_files, 
            combine='by_coords',
            parallel=True
        )
        print(f"  ✅ 全 {len(found_files)} 個のファイルを結合しました。総時間ステップ: {combined_ds.dims['valid_time']}")

        print("2. 天文準拠の季節性エンコーディングを追加しています...")
        combined_ds = add_astronomical_seasonal_encoding(combined_ds)
        print("  ✅ f1_season (夏冬対比), f2_season (春秋グルーピング) 変数を追加しました。")


        print("3. ラベル情報を付与しています...")
        time_coord = combined_ds['valid_time']
        dates = pd.to_datetime(time_coord.values).date
        labels = [data_label_dict.get(d, 'N/A') for d in dates]
        
        combined_ds['label'] = xr.DataArray(labels, dims=['valid_time'], coords={'valid_time': time_coord})
        print(f"  ✅ label 変数を追加しました。")
        print(f"  - 付与されたラベル数: {len([l for l in labels if l != 'N/A'])}")
        
    except Exception as e:
        print(f"\n❌❌❌ データの結合または前処理中にエラーが発生しました。 ❌❌❌")
        print(f"詳細: {e}")
        return

    # =========================================================================
    # Phase 4: 統合ファイルの保存
    # =========================================================================
    print("\n" + "="*80)
    print(f"Phase 4: 統合ファイルの保存")
    print("="*80)
    
    try:
        print(f"処理済みデータを '{output_filename}' として保存しています...")
        
        # ★★★ 改善 (変数名変更) ★★★
        # 数値変数には圧縮を適用し、文字列変数('label')には適用しないようにエンコーディング設定を定義
        encoding_settings = {
            'msl': {'zlib': True, 'complevel': 5},
            'f1_season': {'zlib': True, 'complevel': 5},
            'f2_season': {'zlib': True, 'complevel': 5},
            # 'label'変数は、圧縮フィルタを適用しないため、ここには含めない
        }
        
        # 定義したエンコーディング設定でファイルを保存
        combined_ds.to_netcdf(output_filename, encoding=encoding_settings)
        
        print(f"✅ ファイルの保存が完了しました: {output_filename}")
    except Exception as e:
        print(f"\n❌❌❌ ファイルの保存中にエラーが発生しました。 ❌❌❌")
        print(f"詳細: {e}")
        return
        
    # =========================================================================
    # Phase 5: 最終ファイルの検証
    # =========================================================================
    validate_combined_file(output_filename)

    print("\nすべての処理が正常に完了しました。")

if __name__ == '__main__':
    main()
