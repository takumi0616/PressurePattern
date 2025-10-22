import os
import sys
import glob
import cdsapi
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import hdf5plugin

# ================================================================================
# Dask/chunked DataArray 対応のユーティリティ（xarray.apply_ufunc を並列化）
# ================================================================================
def _da_log(x: xr.DataArray) -> xr.DataArray:
    return xr.apply_ufunc(np.log, x, dask="parallelized", output_dtypes=[np.float64])

def _da_exp(x: xr.DataArray) -> xr.DataArray:
    return xr.apply_ufunc(np.exp, x, dask="parallelized", output_dtypes=[np.float64])

def _da_arctan2(y: xr.DataArray, x: xr.DataArray) -> xr.DataArray:
    return xr.apply_ufunc(np.arctan2, y, x, dask="parallelized", output_dtypes=[np.float64])

def _da_degrees(x: xr.DataArray) -> xr.DataArray:
    return xr.apply_ufunc(np.degrees, x, dask="parallelized", output_dtypes=[np.float64])

# ================================================================================
# 参照/パス設定（このファイルの場所基準で解決）
# ================================================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # .../src/PressurePattern/Classification/data
DATA_DIR = THIS_DIR
OUTPUT_DIR_MSL = os.path.join(DATA_DIR, "nc/era5_msl_large")
OUTPUT_DIR_PL  = os.path.join(DATA_DIR, "nc/era5_pl_large")
OUTPUT_DIR_SL  = os.path.join(DATA_DIR, "nc/era5_single_large")
OUTPUT_ALL_FILE = os.path.join(DATA_DIR, "nc", "era5_all_data_pp.nc")

# ラベル辞書読み込み（パッケージ/スクリプト直実行の両対応）
try:
    from .label import data_label_dict  # type: ignore[import]
except Exception:
    import os as _os, sys as _sys
    _THIS_DIR = _os.path.dirname(_os.path.abspath(__file__))
    if _THIS_DIR not in _sys.path:
        _sys.path.insert(0, _THIS_DIR)
    from label import data_label_dict

# ================================================================================
# ERA5 ダウンロード設定
# ================================================================================
# ダウンロードする年（1991-2000）
YEARS_TO_DOWNLOAD = [str(year) for year in range(1991, 2001)]

# 共通: 月・日・時間・領域・フォーマット
COMMON_DATE_AREA = {
    "month": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"],
    "day": [
        "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12",
        "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24",
        "25", "26", "27", "28", "29", "30", "31"
    ],
    "time": "00:00",  # UTC 00:00
    "area": [55, 115, 15, 155],  # [North, West, South, East]
    "format": "netcdf",
    # "grid": [0.25, 0.25],
}

# 単一レベル（SLP）のベースリクエスト
BASE_REQUEST_SINGLE = {
    "product_type": "reanalysis",
    "variable": "mean_sea_level_pressure",
    **COMMON_DATE_AREA,
}

# 圧力面データのタスクリスト（CDSの正式キーを使用）
PL_TASKS = [
    {"name": "z",     "variables": ["geopotential"],                                 "levels": ["500", "1000"], "filename_tpl": "era5_z_500_1000_{year}.nc"},
    {"name": "t850",  "variables": ["temperature"],                                  "levels": ["850"],         "filename_tpl": "era5_t_850_{year}.nc"},
    {"name": "uv500", "variables": ["u_component_of_wind", "v_component_of_wind"],   "levels": ["500"],         "filename_tpl": "era5_uv_500_{year}.nc"},
    {"name": "uv850", "variables": ["u_component_of_wind", "v_component_of_wind"],   "levels": ["850"],         "filename_tpl": "era5_uv_850_{year}.nc"},
    {"name": "rh7085","variables": ["relative_humidity"],                            "levels": ["700", "850"],  "filename_tpl": "era5_rh_700_850_{year}.nc"},
    {"name": "vo850", "variables": ["vorticity"],                                    "levels": ["850"],         "filename_tpl": "era5_vo_850_{year}.nc"},
    {"name": "vo500", "variables": ["vorticity"],                                    "levels": ["500"],         "filename_tpl": "era5_vo_500_{year}.nc"},
    {"name": "q850",  "variables": ["specific_humidity"],                            "levels": ["850"],         "filename_tpl": "era5_q_850_{year}.nc"},
    {"name": "w500",  "variables": ["vertical_velocity"],                            "levels": ["500"],         "filename_tpl": "era5_w_500_{year}.nc"},
    {"name": "w700",  "variables": ["vertical_velocity"],                            "levels": ["700"],         "filename_tpl": "era5_w_700_{year}.nc"},
    {"name": "d850",  "variables": ["divergence"],                                   "levels": ["850"],         "filename_tpl": "era5_d_850_{year}.nc"},
]

# 追加: 単一レベルの有用変数（IVT/TCWV/CAPE/CIN/TCC/TP）
SINGLE_TASKS = [
    {"name": "tcwv",     "variables": ["total_column_water_vapour"],                                                                      "filename_tpl": "era5_tcwv_{year}.nc"},
    {"name": "ivt_flux", "variables": ["vertical_integral_of_eastward_water_vapour_flux", "vertical_integral_of_northward_water_vapour_flux"], "filename_tpl": "era5_ivt_flux_{year}.nc"},
    {"name": "cape_cin", "variables": ["convective_available_potential_energy", "convective_inhibition"],                                 "filename_tpl": "era5_cape_cin_{year}.nc"},
    {"name": "tcc",      "variables": ["total_cloud_cover"],                                                                              "filename_tpl": "era5_tcc_{year}.nc"},
    {"name": "tp",       "variables": ["total_precipitation"],                                                                            "filename_tpl": "era5_tp_{year}.nc"},
    {"name": "u10v10",   "variables": ["10m_u_component_of_wind", "10m_v_component_of_wind"],                                             "filename_tpl": "era5_u10_v10_{year}.nc"},
    {"name": "sst",      "variables": ["sea_surface_temperature"],                                                                        "filename_tpl": "era5_sst_{year}.nc"},
    {"name": "tclw",     "variables": ["total_column_cloud_liquid_water"],                                                                "filename_tpl": "era5_tclw_{year}.nc"},
    {"name": "tciw",     "variables": ["total_column_cloud_ice_water"],                                                                   "filename_tpl": "era5_tciw_{year}.nc"},
    {"name": "tcrw",     "variables": ["total_column_rain_water"],                                                                        "filename_tpl": "era5_tcrw_{year}.nc"},
    {"name": "tcsw",     "variables": ["total_column_snow_water"],                                                                        "filename_tpl": "era5_tcsw_{year}.nc"},
]

def ensure_dirs():
    for d in [OUTPUT_DIR_MSL, OUTPUT_DIR_PL, OUTPUT_DIR_SL, os.path.join(DATA_DIR, "nc")]:
        os.makedirs(d, exist_ok=True)

def retrieve_single_level_year(client: cdsapi.Client, year: str):
    fn = f"era5_msl_{year}.nc"
    fp = os.path.join(OUTPUT_DIR_MSL, fn)
    if os.path.exists(fp):
        print(f"✅ スキップ: {fn} は既に存在します。")
        return
    req = BASE_REQUEST_SINGLE.copy()
    req["year"] = year
    try:
        print(f"🚀 ダウンロード開始 (SLP): {year}年")
        client.retrieve("reanalysis-era5-single-levels", req, fp)
        print(f"🎉 成功: {fn} を保存しました。")
    except Exception as e:
        print(f"❌ エラー: {year}年 SLP のダウンロードに失敗: {e}")
        if os.path.exists(fp):
            os.remove(fp)

def retrieve_single_misc_year(client: cdsapi.Client, year: str):
    for task in SINGLE_TASKS:
        filename = task["filename_tpl"].format(year=year)
        filepath = os.path.join(OUTPUT_DIR_SL, filename)
        if os.path.exists(filepath):
            print(f"✅ スキップ: {filename} は既に存在します。")
            continue
        req = BASE_REQUEST_SINGLE.copy()
        req["year"] = year
        req["variable"] = task["variables"]
        try:
            print(f"🚀 ダウンロード開始 (SL:{task['name']}): {year}年 変数={task['variables']}")
            client.retrieve("reanalysis-era5-single-levels", req, filepath)
            print(f"🎉 成功: {filename} を保存しました。")
        except Exception as e:
            print(f"❌ エラー: {year}年 {task['name']} のダウンロードに失敗: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)

def retrieve_pressure_levels_year(client: cdsapi.Client, year: str):
    for task in PL_TASKS:
        filename = task["filename_tpl"].format(year=year)
        filepath = os.path.join(OUTPUT_DIR_PL, filename)
        if os.path.exists(filepath):
            print(f"✅ スキップ: {filename} は既に存在します。")
            continue
        req = {
            "product_type": "reanalysis",
            "variable": task["variables"],
            "pressure_level": task["levels"],
            "year": year,
            **COMMON_DATE_AREA,
        }
        try:
            print(f"🚀 ダウンロード開始 (PL:{task['name']}): {year}年 変数={task['variables']} レベル={task['levels']}")
            client.retrieve("reanalysis-era5-pressure-levels", req, filepath)
            print(f"🎉 成功: {filename} を保存しました。")
        except Exception as e:
            print(f"❌ エラー: {year}年 {task['name']} のダウンロードに失敗: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)

def download_era5_data():
    print("="*80)
    print("ERA5 データ ダウンロード（SLP + 圧力面）")
    print(f"対象期間: {YEARS_TO_DOWNLOAD[0]}年 ～ {YEARS_TO_DOWNLOAD[-1]}年")
    print(f"保存先 (SLP): {OUTPUT_DIR_MSL}")
    print(f"保存先 (PL) : {OUTPUT_DIR_PL}")
    print(f"保存先 (SL) : {OUTPUT_DIR_SL}")
    print("="*80)
    try:
        client = cdsapi.Client()
    except Exception as e:
        print("❌ エラー: cdsapi.Client の初期化に失敗（~/.cdsapirc を確認）。")
        print(f"詳細: {e}")
        return
    ensure_dirs()
    print("ダウンロード処理を開始します...")
    for year in tqdm(YEARS_TO_DOWNLOAD, desc="全体の進捗"):
        retrieve_single_level_year(client, year)
        retrieve_single_misc_year(client, year)
        retrieve_pressure_levels_year(client, year)
    print("すべてのダウンロード処理が完了しました。")

# ================================================================================
# 季節エンコーディング用定数（変更禁止）
# ================================================================================
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

# ================================================================================
# 季節エンコーディング
# ================================================================================
def add_astronomical_seasonal_encoding(ds: xr.Dataset) -> xr.Dataset:
    timestamps = ds['valid_time'].to_series()
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

    df = pd.DataFrame(index=timestamps.index)
    df['ws_start'] = pd.merge_asof(left=df, right=ws_series.to_frame('ws_start'), left_index=True, right_index=True, direction='backward')['ws_start']
    df['ws_end'] = pd.merge_asof(left=df, right=ws_series.to_frame('ws_end'), left_index=True, right_index=True, direction='forward')['ws_end']
    df['ws_start'] = df['ws_start'].bfill()
    df['ws_end'] = df['ws_end'].ffill()
    if df.isnull().values.any():
        raise ValueError("Seasonal encoding failed: NaT values still exist after fill.")
    time_passed = (df.index - df['ws_start']).dt.total_seconds()
    total_time = (df['ws_end'] - df['ws_start']).dt.total_seconds()
    t = np.divide(time_passed, total_time, out=np.zeros_like(time_passed, dtype=float), where=(total_time != 0))
    angle = 2 * np.pi * t
    f1_season = np.cos(angle)
    f2_season = np.cos(2 * angle)

    ds['f1_season'] = xr.DataArray(f1_season, dims=['valid_time'], coords={'valid_time': ds['valid_time']})
    ds['f2_season'] = xr.DataArray(f2_season, dims=['valid_time'], coords={'valid_time': ds['valid_time']})
    ds['f1_season'].attrs['long_name'] = 'Seasonal feature f1 (winter/summer contrast)'
    ds['f1_season'].attrs['description'] = 'cos(2*pi*t) where t is normalized time from winter solstice to the next. Winter=+1, Summer=-1.'
    ds['f2_season'].attrs['long_name'] = 'Seasonal feature f2 (equinox grouping)'
    ds['f2_season'].attrs['description'] = 'cos(4*pi*t) where t is normalized time from winter solstice to the next. Solstices=+1, Equinoxes=-1.'
    return ds

# ================================================================================
# マージ/整形ユーティリティ
# ================================================================================
def standardize_time(ds: xr.Dataset) -> xr.Dataset:
    if 'valid_time' in ds.coords:
        return ds
    if 'time' in ds.coords:
        ds = ds.rename({'time': 'valid_time'})
    return ds

def open_msl_files(msl_dir: str):
    pattern = os.path.join(msl_dir, "era5_msl_*.nc")
    found_files = sorted(glob.glob(pattern))
    if not found_files:
        raise FileNotFoundError(f"SLPファイルが見つかりません: {msl_dir}")
    years = []
    for f in found_files:
        base = os.path.basename(f)
        try:
            y = int(base.replace("era5_msl_", "").replace(".nc", ""))
            years.append(y)
        except Exception:
            pass
    if years:
        print(f"SLPファイル {len(found_files)} 個を検出（対象年: {min(years)}～{max(years)}）")
    else:
        print(f"SLPファイル {len(found_files)} 個を検出")
    print("SLPファイルを結合中...")
    ds = xr.open_mfdataset(
        found_files,
        combine='by_coords',
        parallel=True,
        join='outer',
        compat='no_conflicts',
        chunks="auto",
        data_vars='minimal',
        coords='minimal',
        combine_attrs='drop_conflicts'
    )
    ds = standardize_time(ds)
    if 'msl' not in ds:
        raise ValueError("msl 変数が見つかりません。")
    return ds

def open_pl_files(pl_dir: str):
    pattern = os.path.join(pl_dir, "*.nc")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"圧力面ファイルが見つかりません: {pl_dir}")
    print(f"圧力面ファイル {len(files)} 個を結合中...")
    ds = xr.open_mfdataset(
        files,
        combine='by_coords',
        parallel=True,
        join='outer',
        compat='no_conflicts',
        chunks="auto",
        data_vars='minimal',
        coords='minimal',
        combine_attrs='drop_conflicts'
    )
    ds = standardize_time(ds)
    if 'level' in ds.dims and 'pressure_level' not in ds.dims:
        ds = ds.rename({'level': 'pressure_level'})
    expected_any = ['z', 't', 'u', 'v', 'r', 'vo', 'q', 'w', 'd']
    if not any(v in ds.variables for v in expected_any):
        raise ValueError("圧力面データに必要な変数が見つかりません。")
    return ds

def open_single_misc_files(sl_dir: str):
    pattern = os.path.join(sl_dir, "*.nc")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"単一レベル追加変数ファイルが見つかりません: {sl_dir}")
        return None
    print(f"単一レベル（追加）ファイル {len(files)} 個を結合中...")
    ds = xr.open_mfdataset(
        files,
        combine='by_coords',
        parallel=True,
        join='outer',
        compat='no_conflicts',
        chunks="auto",
        data_vars='minimal',
        coords='minimal',
        combine_attrs='drop_conflicts'
    )
    ds = standardize_time(ds)
    return ds

def derive_fields_from_pl(pl_ds: xr.Dataset) -> xr.Dataset:
    if 'pressure_level' not in pl_ds.dims:
        if 'level' in pl_ds.dims:
            pl_ds = pl_ds.rename({'level': 'pressure_level'})
        else:
            raise ValueError("圧力面データに 'pressure_level' 次元がありません。")
    g0 = 9.80665
    out = xr.Dataset()
    # 基本層の抽出
    if 'z' in pl_ds:
        z500 = pl_ds['z'].sel(pressure_level=500, drop=True) / g0
        out['gh500'] = z500.rename('gh500').assign_attrs(long_name='Geopotential height at 500 hPa', units='m')
    if 't' in pl_ds:
        t850 = pl_ds['t'].sel(pressure_level=850, drop=True).rename('t850')
        t850.attrs['long_name'] = 'Temperature at 850 hPa'; t850.attrs['units'] = 'K'
        out['t850'] = t850
    if 'u' in pl_ds:
        u500 = pl_ds['u'].sel(pressure_level=500, drop=True).rename('u500'); u500.attrs['long_name'] = 'U wind at 500 hPa'; u500.attrs['units'] = 'm s-1'; out['u500'] = u500
        u850 = pl_ds['u'].sel(pressure_level=850, drop=True).rename('u850'); u850.attrs['long_name'] = 'U wind at 850 hPa'; u850.attrs['units'] = 'm s-1'; out['u850'] = u850
    if 'v' in pl_ds:
        v500 = pl_ds['v'].sel(pressure_level=500, drop=True).rename('v500'); v500.attrs['long_name'] = 'V wind at 500 hPa'; v500.attrs['units'] = 'm s-1'; out['v500'] = v500
        v850 = pl_ds['v'].sel(pressure_level=850, drop=True).rename('v850'); v850.attrs['long_name'] = 'V wind at 850 hPa'; v850.attrs['units'] = 'm s-1'; out['v850'] = v850
    if 'r' in pl_ds:
        r700 = pl_ds['r'].sel(pressure_level=700, drop=True).rename('r700'); r700.attrs['long_name'] = 'Relative humidity at 700 hPa'; r700.attrs['units'] = '%'; out['r700'] = r700
        r850 = pl_ds['r'].sel(pressure_level=850, drop=True).rename('r850'); r850.attrs['long_name'] = 'Relative humidity at 850 hPa'; r850.attrs['units'] = '%'; out['r850'] = r850
    if 'vo' in pl_ds:
        vo850 = pl_ds['vo'].sel(pressure_level=850, drop=True).rename('vo850'); vo850.attrs['long_name'] = 'Relative vorticity at 850 hPa'; vo850.attrs['units'] = 's-1'; out['vo850'] = vo850
        # 500hPa の相対渦度
        try:
            vo500 = pl_ds['vo'].sel(pressure_level=500, drop=True).rename('vo500')
            vo500.attrs['long_name'] = 'Relative vorticity at 500 hPa'; vo500.attrs['units'] = 's-1'
            out['vo500'] = vo500
        except Exception:
            pass
    # 転置（valid_time, lat, lon）
    out = standardize_time(out)
    for v in out.data_vars:
        da = out[v]; dims = list(da.dims)
        for key in ['valid_time', 'latitude', 'longitude']:
            if key in dims:
                dims.remove(key)
        new_order = ['valid_time', 'latitude', 'longitude'] + dims
        try:
            out[v] = da.transpose(*new_order)
        except Exception:
            pass
    # 追加派生
    try:
        # 厚さ（1000-500 hPa）
        if 'z' in pl_ds:
            if 1000 in pl_ds['pressure_level']:
                z1000 = pl_ds['z'].sel(pressure_level=1000, drop=True) / g0
                z1000 = z1000.rename('gh1000').assign_attrs(long_name='Geopotential height at 1000 hPa', units='m')
                out['gh1000'] = z1000
            if ('gh500' in out) and ('gh1000' in out):
                thk = (out['gh500'] - out['gh1000']).rename('thk_1000_500')
                thk.attrs['long_name'] = 'Thickness (1000-500 hPa)'
                thk.attrs['units'] = 'm'
                out['thk_1000_500'] = thk
                # 厚さの水平勾配
                R = 6371000.0
                lat_rad = np.deg2rad(out['thk_1000_500']['latitude'])
                dlat = out['thk_1000_500'].differentiate('latitude') * (np.pi/180.0) / R
                dlon = out['thk_1000_500'].differentiate('longitude') * (np.pi/180.0) / (R * np.cos(lat_rad))
                gthk = np.sqrt(dlat**2 + dlon**2).rename('grad_thk_1000_500')
                gthk.attrs['long_name'] = 'Gradient magnitude of thickness (1000-500 hPa)'
                gthk.attrs['units'] = 'm m-1'
                out['grad_thk_1000_500'] = gthk
        # q850
        if 'q' in pl_ds:
            q850 = pl_ds['q'].sel(pressure_level=850, drop=True).rename('q850')
            q850.attrs['long_name'] = 'Specific humidity at 850 hPa'; q850.attrs['units'] = 'kg kg-1'
            out['q850'] = q850
        # 風速・シア
        if all(k in out for k in ['u500','v500']):
            vmag500 = np.sqrt(out['u500']**2 + out['v500']**2).rename('vmag500')
            vmag500.attrs['long_name'] = 'Wind speed at 500 hPa'; vmag500.attrs['units'] = 'm s-1'
            out['vmag500'] = vmag500
        if all(k in out for k in ['u850','v850']):
            vmag850 = np.sqrt(out['u850']**2 + out['v850']**2).rename('vmag850')
            vmag850.attrs['long_name'] = 'Wind speed at 850 hPa'; vmag850.attrs['units'] = 'm s-1'
            out['vmag850'] = vmag850
        if all(k in out for k in ['u500','v500','u850','v850']):
            shear = np.sqrt((out['u500'] - out['u850'])**2 + (out['v500'] - out['v850'])**2).rename('shear_850_500')
            shear.attrs['long_name'] = 'Vector wind shear magnitude (500-850 hPa)'; shear.attrs['units'] = 'm s-1'
            out['shear_850_500'] = shear
        # θe850の計算（Bolton 1980 近似）とその勾配
        if ('t850' in out) and ('q850' in out):
            p = 85000.0
            T = out['t850']
            q = out['q850']
            r = q / (1.0 - q)
            e = (r / (0.622 + r)) * p
            logT = _da_log(T.astype(np.float64))
            loge = _da_log(e.clip(min=1.0).astype(np.float64))
            Tl   = 2840.0 / (3.5 * logT - loge - 4.805) + 55.0
            theta = T.astype(np.float64) * (100000.0 / p) ** (0.2854 * (1.0 - 0.28 * r.astype(np.float64)))
            expterm = _da_exp((3.376 / Tl - 0.00254) * r.astype(np.float64) * (1.0 + 0.81 * r.astype(np.float64)))
            theta_e = (theta * expterm)
            theta_e = theta_e.rename('thetae850')
            theta_e.attrs['long_name'] = 'Equivalent potential temperature at 850 hPa'
            theta_e.attrs['units'] = 'K'
            out['thetae850'] = theta_e
            R = 6371000.0
            lat_rad = np.deg2rad(out['thetae850']['latitude'])
            dlat = out['thetae850'].differentiate('latitude') * (np.pi/180.0) / R
            dlon = out['thetae850'].differentiate('longitude') * (np.pi/180.0) / (R * np.cos(lat_rad))
            grad = np.sqrt(dlat**2 + dlon**2)
            grad = grad.rename('grad_thetae850')
            grad.attrs['long_name'] = 'Horizontal gradient magnitude of thetae (850 hPa)'
            grad.attrs['units'] = 'K m-1'
            out['grad_thetae850'] = grad
        # 発散と鉛直流
        if 'd' in pl_ds:
            div850 = pl_ds['d'].sel(pressure_level=850, drop=True).rename('div850')
            div850.attrs['long_name'] = 'Divergence at 850 hPa'; div850.attrs['units'] = 's-1'
            out['div850'] = div850
        if 'w' in pl_ds:
            w500 = pl_ds['w'].sel(pressure_level=500, drop=True).rename('w500')
            w500.attrs['long_name'] = 'Vertical velocity at 500 hPa'; w500.attrs['units'] = 'Pa s-1'
            out['w500'] = w500
            try:
                w700 = pl_ds['w'].sel(pressure_level=700, drop=True).rename('w700')
                w700.attrs['long_name'] = 'Vertical velocity at 700 hPa'; w700.attrs['units'] = 'Pa s-1'
                out['w700'] = w700
            except Exception:
                pass
        # Moisture flux convergence at 850 hPa: -∇·(q V)
        try:
            if all(k in out for k in ['q850', 'u850', 'v850']):
                q = out['q850']; u = out['u850']; v = out['v850']
                R = 6371000.0
                lat_rad = np.deg2rad(q['latitude'])
                dlon = (np.pi/180.0) / (R * np.cos(lat_rad))
                dlat = (np.pi/180.0) / R
                div_qv = ( (q*u).differentiate('longitude') * dlon + (q*v).differentiate('latitude') * dlat )
                mfc = (-div_qv).rename('mfc850')
                mfc.attrs['long_name'] = 'Moisture flux convergence at 850 hPa'
                mfc.attrs['units'] = 's-1'
                out['mfc850'] = mfc
        except Exception as _e:
            print("⚠ mfc850 計算で例外:", _e)
    except Exception as _e:
        print("⚠ 追加派生計算で例外:", _e)
    return out

def derive_fields_from_single(sl_ds: xr.Dataset) -> xr.Dataset:
    if sl_ds is None:
        return xr.Dataset()
    out = xr.Dataset()
    sl_ds = standardize_time(sl_ds)
    # そのまま通す候補
    passthrough = [
        'tcwv', 'cape', 'cin', 'tcc', 'tp',
        'u10', 'v10',
        'tclw', 'tciw', 'tcrw', 'tcsw', 'sst',
        'viwve', 'viwvn',
        'vertical_integral_of_eastward_water_vapour_flux', 'vertical_integral_of_northward_water_vapour_flux',
        'total_column_cloud_liquid_water', 'total_column_cloud_ice_water',
        'total_column_rain_water', 'total_column_snow_water', 'sea_surface_temperature'
    ]
    for cand in passthrough:
        if cand in sl_ds:
            out[cand if cand in ['tcwv','cape','cin','tcc','tp','u10','v10','viwve','viwvn'] else cand] = sl_ds[cand]
    # 別名の正規化（長名→短名）
    alias_pairs = [
        ('total_column_cloud_liquid_water', 'tclw'),
        ('total_column_cloud_ice_water', 'tciw'),
        ('total_column_rain_water', 'tcrw'),
        ('total_column_snow_water', 'tcsw'),
        ('sea_surface_temperature', 'sst'),
    ]
    for long_name, short in alias_pairs:
        if long_name in sl_ds and short not in out:
            out[short] = sl_ds[long_name].rename(short)
    # IVT 大きさと方位
    if ('viwve' in out) and ('viwvn' in out):
        ivt = np.sqrt(out['viwve']**2 + out['viwvn']**2).rename('ivt')
        ivt.attrs['long_name'] = 'Integrated vapor transport magnitude'; ivt.attrs['units'] = 'kg m-1 s-1'
        out['ivt'] = ivt
        angle = _da_arctan2(out['viwvn'].astype(np.float64), out['viwve'].astype(np.float64))
        ivt_dir = _da_degrees(angle).rename('ivt_dir')
        ivt_dir.attrs['long_name'] = 'IVT direction (meteorological angle, arctan2(north,east))'
        ivt_dir.attrs['units'] = 'degree'
        out['ivt_dir'] = ivt_dir
    elif ('vertical_integral_of_eastward_water_vapour_flux' in out) and ('vertical_integral_of_northward_water_vapour_flux' in out):
        uflx = out['vertical_integral_of_eastward_water_vapour_flux']
        vflx = out['vertical_integral_of_northward_water_vapour_flux']
        ivt = np.sqrt(uflx**2 + vflx**2).rename('ivt')
        ivt.attrs['long_name'] = 'Integrated vapor transport magnitude'; ivt.attrs['units'] = 'kg m-1 s-1'
        out['ivt'] = ivt
        angle = _da_arctan2(vflx.astype(np.float64), uflx.astype(np.float64))
        ivt_dir = _da_degrees(angle).rename('ivt_dir')
        ivt_dir.attrs['long_name'] = 'IVT direction (meteorological angle, arctan2(north,east))'
        ivt_dir.attrs['units'] = 'degree'
        out['ivt_dir'] = ivt_dir
    # 10m 風速
    if ('u10' in out) and ('v10' in out):
        vmag10 = np.sqrt(out['u10']**2 + out['v10']**2).rename('vmag10')
        vmag10.attrs['long_name'] = '10 m wind speed'; vmag10.attrs['units'] = 'm s-1'
        out['vmag10'] = vmag10
    out = standardize_time(out)
    for v in out.data_vars:
        try:
            out[v] = out[v].transpose('valid_time', 'latitude', 'longitude')
        except Exception:
            pass
    return out

def validate_final(ds: xr.Dataset):
    required = ['msl', 'gh500', 't850', 'u500', 'v500', 'u850', 'v850', 'r700', 'r850', 'vo850', 'f1_season', 'f2_season', 'label']
    missing = [v for v in required if v not in ds]
    if missing:
        print(f"❌ エラー: 必須変数が不足しています: {missing}")
        return False
    for v in required:
        if v == 'label':
            continue
        arr = ds[v]
        if arr.isnull().any():
            print(f"❌ {v}: NaNを含みます。")
            return False
        if np.isinf(arr.values).any():
            print(f"❌ {v}: infを含みます。")
            return False
    print("✅ 必須変数はすべて存在し、欠損やinfは検出されませんでした。")
    return True

def clean_cfgrib_artifacts(ds: xr.Dataset) -> xr.Dataset:
    drop_vars = [v for v in ['number', 'expver'] if v in ds.variables]
    if drop_vars:
        ds = ds.drop_vars(drop_vars, errors='ignore')
    for v in ds.data_vars:
        if 'coordinates' in ds[v].attrs:
            try:
                del ds[v].attrs['coordinates']
            except Exception:
                pass
    return ds

# ================================================================================
# マージ + LZ4圧縮出力
# ================================================================================
def merge_and_save():
    print("="*80)
    print("Phase 1: 入力ファイルの存在チェックと読み込み（自動検出）")
    print("="*80)
    print(f"msl_dir: {OUTPUT_DIR_MSL}")
    print(f"pl_dir : {OUTPUT_DIR_PL}")
    print(f"sl_dir : {OUTPUT_DIR_SL}")
    if not os.path.isdir(OUTPUT_DIR_MSL):
        print(f"❌ エラー: ディレクトリ '{OUTPUT_DIR_MSL}' が存在しません。")
        return
    if not os.path.isdir(OUTPUT_DIR_PL):
        print(f"❌ エラー: ディレクトリ '{OUTPUT_DIR_PL}' が存在しません。")
        return

    # SLP/PL 読み込み
    try:
        msl_ds = open_msl_files(OUTPUT_DIR_MSL)
    except Exception as e:
        print(f"❌ SLP読み込みでエラー: {e}")
        return
    try:
        pl_ds = open_pl_files(OUTPUT_DIR_PL)
    except Exception as e:
        print(f"❌ 圧力面読み込みでエラー: {e}")
        return
    try:
        sl_ds = open_single_misc_files(OUTPUT_DIR_SL)
    except Exception as e:
        print(f"⚠ 単一レベル（追加）読み込みでエラー: {e}")
        sl_ds = None

    print("\n" + "="*80)
    print("Phase 2: 圧力面データから必要変数の生成")
    print("="*80)
    try:
        pl_fields = derive_fields_from_pl(pl_ds)
        print(f"  ✅ 生成した変数(PL): {list(pl_fields.data_vars)}")
        sl_fields = derive_fields_from_single(sl_ds) if sl_ds is not None else xr.Dataset()
        if sl_fields.data_vars:
            print(f"  ✅ 生成した変数(SL): {list(sl_fields.data_vars)}")
    except Exception as e:
        print(f"❌ 派生変数生成でエラー: {e}")
        return

    print("\n" + "="*80)
    print("Phase 3: 時間・空間の整合と結合、季節・ラベル付与")
    print("="*80)
    try:
        msl_ds = standardize_time(msl_ds)
        pl_fields = standardize_time(pl_fields)

        common_times = np.intersect1d(msl_ds['valid_time'].values, pl_fields['valid_time'].values)
        if sl_ds is not None and sl_fields is not None and sl_fields.data_vars:
            common_times = np.intersect1d(common_times, sl_fields['valid_time'].values)
        if len(common_times) == 0:
            raise ValueError("msl / PL / SL の時間が一致しません。")
        msl_ds = msl_ds.sel(valid_time=common_times)
        pl_fields = pl_fields.sel(valid_time=common_times)
        if sl_ds is not None and sl_fields is not None and sl_fields.data_vars:
            sl_fields = sl_fields.sel(valid_time=common_times)

        for coord in ['latitude', 'longitude']:
            if not np.array_equal(msl_ds[coord].values, pl_fields[coord].values):
                pl_fields = pl_fields.reindex({coord: msl_ds[coord].values}, method=None)
            if sl_ds is not None and sl_fields is not None and sl_fields.data_vars:
                if (coord in sl_fields.coords) and (not np.array_equal(msl_ds[coord].values, sl_fields[coord].values)):
                    sl_fields = sl_fields.reindex({coord: msl_ds[coord].values}, method=None)

        if sl_ds is not None and sl_fields is not None and sl_fields.data_vars:
            combined_ds = xr.merge([msl_ds, pl_fields, sl_fields], compat='no_conflicts', join='inner')
        else:
            combined_ds = xr.merge([msl_ds, pl_fields], compat='no_conflicts', join='inner')
        combined_ds = add_astronomical_seasonal_encoding(combined_ds)
        # 24h SLP傾度（前日との差分; 先頭は0で埋める）と空間勾配・ラプラシアン
        try:
            diff = combined_ds['msl'] - combined_ds['msl'].shift(valid_time=1)
            combined_ds['msl_dt24'] = diff.fillna(0.0)
            combined_ds['msl_dt24'].attrs['long_name'] = '24h MSL change'
            combined_ds['msl_dt24'].attrs['units'] = 'Pa per 24h'
            # 水平勾配とラプラシアン（平面近似の球面メトリック補正）
            R = 6371000.0
            lat_rad = np.deg2rad(combined_ds['latitude'])
            dlat = combined_ds['msl'].differentiate('latitude') * (np.pi/180.0) / R
            dlon = combined_ds['msl'].differentiate('longitude') * (np.pi/180.0) / (R * np.cos(lat_rad))
            grad = np.sqrt(dlat**2 + dlon**2).rename('grad_msl')
            grad.attrs['long_name'] = 'Horizontal gradient magnitude of MSLP'
            grad.attrs['units'] = 'Pa m-1'
            combined_ds['grad_msl'] = grad
            # 二階微分（簡易ラプラシアン）
            d2lat = combined_ds['msl'].differentiate('latitude').differentiate('latitude') * ((np.pi/180.0)/R)**2
            # cosφ の緯度依存は無視した近似
            d2lon = combined_ds['msl'].differentiate('longitude').differentiate('longitude') * ((np.pi/180.0)/(R*np.cos(lat_rad)))**2
            lap = (d2lat + d2lon).rename('lap_msl')
            lap.attrs['long_name'] = 'Approx. Laplacian of MSLP'
            lap.attrs['units'] = 'Pa m-2'
            combined_ds['lap_msl'] = lap
        except Exception as _e:
            print("⚠ msl_dt24/grad/lap 計算に失敗:", _e)

        time_coord = combined_ds['valid_time']
        dates = pd.to_datetime(time_coord.values).date
        labels = [data_label_dict.get(d, 'N/A') for d in dates]
        combined_ds['label'] = xr.DataArray(labels, dims=['valid_time'], coords={'valid_time': time_coord})

        combined_ds = clean_cfgrib_artifacts(combined_ds)
        print("  ✅ 季節性特徴とラベルを付与しました。")
    except Exception as e:
        print(f"❌ 結合・付与処理でエラー: {e}")
        return

    print("\n" + "="*80)
    print("Phase 4: LZ4圧縮 + チャンク(valid_time=1) で出力")
    print("="*80)
    try:
        # LZ4圧縮 + 時間チャンク(valid_time)=1（h5netcdf エンジン）
        encoding_settings = {}
        for var in combined_ds.data_vars:
            if var == 'label':
                continue  # 可変長文字列は圧縮・チャンク設定しない
            da = combined_ds[var]
            if 'valid_time' in da.dims:
                if da.ndim == 3:
                    lat_len = int(da.sizes.get('latitude', da.shape[1]))
                    lon_len = int(da.sizes.get('longitude', da.shape[2]))
                    encoding_settings[var] = {**hdf5plugin.LZ4(), 'chunksizes': (1, lat_len, lon_len)}
                elif da.ndim == 1:
                    encoding_settings[var] = {**hdf5plugin.LZ4(), 'chunksizes': (1,)}
                else:
                    encoding_settings[var] = {**hdf5plugin.LZ4()}
            else:
                encoding_settings[var] = {**hdf5plugin.LZ4()}

        os.makedirs(os.path.dirname(OUTPUT_ALL_FILE), exist_ok=True)
        tmp_path = OUTPUT_ALL_FILE + ".part"
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

        combined_ds.to_netcdf(tmp_path, encoding=encoding_settings, engine="h5netcdf")
        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            print(f"❌ 予期せぬエラー: 一時出力ファイルが見つからないかサイズ0です: {tmp_path}")
            return
        try:
            with open(tmp_path, "rb") as _f:
                os.fsync(_f.fileno())
        except Exception:
            pass
        os.replace(tmp_path, OUTPUT_ALL_FILE)
        print(f"✅ ファイルの保存が完了しました: {OUTPUT_ALL_FILE}")
    except Exception as e:
        print(f"❌ 保存中にエラー: {e}")
        return

    print("\n" + "="*80)
    print("Phase 5: 最終ファイルの検証")
    print("="*80)
    try:
        if not os.path.exists(OUTPUT_ALL_FILE):
            print(f"❌ 検証用ファイルが存在しません: {OUTPUT_ALL_FILE}")
            return
        with xr.open_dataset(OUTPUT_ALL_FILE) as ds:
            ok = validate_final(ds)
            if not ok:
                print("❌ 検証で問題が見つかりました。")
                return
            print("✅ 最終ファイルの検証に成功しました。")
    except Exception as e:
        print(f"❌ 検証中にエラー: {e}")
        return

    print("\nすべての処理が正常に完了しました。")

# ================================================================================
# メイン処理（ダウンロード + マージ + LZ4圧縮出力）
# ================================================================================
def main():
    ensure_dirs()
    skip_dl = os.environ.get("SKIP_DOWNLOAD", "").lower() in ("1", "true", "yes", "y")
    if skip_dl:
        print("環境変数 SKIP_DOWNLOAD が設定されています。download_era5_data() をスキップします。")
    else:
        download_era5_data()
    merge_and_save()

if __name__ == '__main__':
    main()
