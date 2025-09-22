# -*- coding: utf-8 -*-
"""
./prmsl_era5_all_data_seasonal_large.nc を読み込み、
1991-01-01 〜 2000-12-31 の海面更正気圧（msl）の「領域平均を引いた偏差[hPa]」画像を ./image に保存。

可視化フォーマットは src/PressurePattern/main_v7.py の図と整合（色、等値線、海岸線、描画領域など）。
- カラーマップ: RdBu_r
- レベル: -40 .. 40 (21段)
- 海岸線: 50m, linewidth=0.8, edgecolor='black'
- 等値線: 黒, linewidth=0.3
- 描画範囲: [115, 155, 15, 55]
- Colorbar ラベル: 'Sea Level Pressure Anomaly (hPa)'
- 軸目盛りは非表示

実行は不要（スクリプトのみ作成）。
"""

import os
import sys
import argparse
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import xarray as xr

# Matplotlib (サーバ上で保存できるように Agg)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# cartopy: 可視化に使用（main_v7 に合わせる）
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ========== ユーザ指定/既定 ==========
DATA_FILE = './prmsl_era5_all_data_seasonal_large.nc'
START_DATE = '1991-01-01'
END_DATE   = '2000-12-31'
OUT_DIR    = './image'

# 可視化パラメータ（main_v7 と整合）
V_MIN, V_MAX = -40, 40
LEVELS = np.linspace(V_MIN, V_MAX, 21)
CMAP = 'RdBu_r'
COAST_SCALE = '50m'
COAST_LW = 0.8
COAST_EDGE = 'black'
CONTOUR_COLOR = 'k'
CONTOUR_LW = 0.3
EXTENT = [115, 155, 15, 55]  # lon_min, lon_max, lat_min, lat_max
CBAR_LABEL = 'Sea Level Pressure Anomaly (hPa)'

# 基本ラベル（15）: main_v7 と同じ
BASE_LABELS = [
    '1', '2A', '2B', '2C', '2D', '3A', '3B', '3C', '3D', '4A', '4B', '5', '6A', '6B', '6C'
]

def _normalize_to_base_candidate(label_str: Optional[str]) -> Optional[str]:
    """main_v7 と同等の正規化: 全角→半角、英数大文字化、+/- 正規化、英数字以外除去"""
    import unicodedata, re
    if label_str is None:
        return None
    s = str(label_str)
    s = unicodedata.normalize('NFKC', s)
    s = s.upper().strip()
    s = s.replace('＋', '+').replace('－', '-').replace('−', '-')
    s = re.sub(r'[^0-9A-Z\+\-]', '', s)
    return s if s != '' else None

def basic_label_or_none(label_str: Optional[str], base_labels: List[str]) -> Optional[str]:
    """main_v7.basic_label_or_none を踏襲（先頭一致＋残部に英数字が無ければ該当基本ラベルとみなす）"""
    import re
    cand = _normalize_to_base_candidate(label_str)
    if cand is None:
        return None
    if cand in base_labels:
        return cand
    for bl in base_labels:
        if cand == bl:
            return bl
        if cand.startswith(bl):
            rest = cand[len(bl):]
            if re.search(r'[0-9A-Z]', rest) is None:
                return bl
    return None

# ========== main_v7 の処理を踏襲したユーティリティ ==========
def format_date_yyyymmdd(ts_val) -> str:
    """YYYY/MM/DD 形式に整形（main_v7 相当）。"""
    if ts_val is None:
        return ''
    try:
        ts = pd.to_datetime(ts_val)
        if pd.isna(ts):
            return ''
        return ts.strftime('%Y/%m/%d')
    except Exception:
        try:
            s = str(ts_val)
            if 'T' in s:
                s = s.split('T')[0]
            s = s.replace('-', '/')
            if len(s) >= 10:
                return s[:10]
            return s
        except Exception:
            return str(ts_val)


def load_and_prepare_data_for_images(
    filepath: str,
    start_date: Optional[str],
    end_date: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    main_v7.load_and_prepare_data_unified の「msl[hPa] と anomaly[hPa] を返す」部分を純numpyで再実装。
    戻り値:
      - msl_hpa:     (N, H, W)  hPa
      - anomaly_hpa: (N, H, W)  hPa（各サンプルから空間平均を差し引いた偏差）
      - lat:         (H,)
      - lon:         (W,)
      - times:       (N,)  numpy datetime64 など
    """
    ds = xr.open_dataset(filepath, decode_times=True)

    # time 座標名の検出（main_v7 と同様）
    if 'valid_time' in ds:
        time_coord = 'valid_time'
    elif 'time' in ds:
        time_coord = 'time'
    else:
        raise ValueError('No time coordinate named "valid_time" or "time".')

    # 期間指定
    sub = ds.sel({time_coord: slice(start_date, end_date)}) if (start_date or end_date) else ds

    if 'msl' not in sub:
        raise ValueError('Variable "msl" not found in dataset.')

    msl = sub['msl'].astype('float32')  # (time, lat, lon) など

    # 次元名の標準化
    lat_name = 'latitude'
    lon_name = 'longitude'
    for dn in msl.dims:
        if 'lat' in dn.lower():
            lat_name = dn
        if 'lon' in dn.lower():
            lon_name = dn

    # (N,H,W)
    msl = msl.transpose(time_coord, lat_name, lon_name)
    ntime = msl.sizes[time_coord]
    nlat = msl.sizes[lat_name]
    nlon = msl.sizes[lon_name]

    arr = msl.values  # Pa, shape (N,H,W)
    arr2 = arr.reshape(ntime, nlat * nlon)

    # NaN を含むサンプルを除外
    valid_mask = ~np.isnan(arr2).any(axis=1)
    arr2 = arr2[valid_mask]
    times = msl[time_coord].values[valid_mask]
    lat = sub[lat_name].values
    lon = sub[lon_name].values

    # hPa へ換算 -> 空間平均を引いて偏差に
    msl_hpa_flat = (arr2 / 100.0).astype(np.float32)                    # (N, D)
    mean_per_sample = np.nanmean(msl_hpa_flat, axis=1, keepdims=True)   # (N, 1)
    anomaly_flat = msl_hpa_flat - mean_per_sample                       # (N, D)

    # 3D へ
    N = anomaly_flat.shape[0]
    msl_hpa = msl_hpa_flat.reshape(N, nlat, nlon)
    anomaly_hpa = anomaly_flat.reshape(N, nlat, nlon)

    # 後片付け
    try:
        ds.close()
    except Exception:
        pass

    return msl_hpa, anomaly_hpa, lat, lon, times


def load_labels_aligned(
    filepath: str,
    start_date: Optional[str],
    end_date: Optional[str],
) -> List[Optional[str]]:
    """
    NetCDF から 'label' を読み出し、main_v7 と同様の期間抽出・NaN行除外（msl基準の valid_mask）に整合させた
    ラベル配列（長さN、要素は str または None）を返す。
    """
    ds = xr.open_dataset(filepath, decode_times=True)

    # time 座標名の検出
    if 'valid_time' in ds:
        time_coord = 'valid_time'
    elif 'time' in ds:
        time_coord = 'time'
    else:
        raise ValueError('No time coordinate named "valid_time" or "time".')

    sub = ds.sel({time_coord: slice(start_date, end_date)}) if (start_date or end_date) else ds

    if 'msl' not in sub:
        # ラベルだけを返す（msl が無いケースは想定薄だが保険）
        if 'label' in sub.variables:
            raw_all = sub['label'].values.reshape(-1)
            labels = []
            for v in raw_all:
                try:
                    s = v.decode('utf-8', errors='ignore').strip() if isinstance(v, (bytes, bytearray)) else str(v).strip()
                except Exception:
                    s = ''
                labels.append(s if s != '' else None)
        else:
            labels = []
        try:
            ds.close()
        except Exception:
            pass
        return labels

    # msl に基づき valid_mask を作る（main_v7 と同様）
    msl = sub['msl'].astype('float32')
    lat_name = 'latitude'
    lon_name = 'longitude'
    for dn in msl.dims:
        if 'lat' in dn.lower():
            lat_name = dn
        if 'lon' in dn.lower():
            lon_name = dn
    msl = msl.transpose(time_coord, lat_name, lon_name)
    ntime = msl.sizes[time_coord]
    nlat = msl.sizes[lat_name]
    nlon = msl.sizes[lon_name]
    arr = msl.values  # (N,H,W)
    arr2 = arr.reshape(ntime, nlat * nlon)
    valid_mask = ~np.isnan(arr2).any(axis=1)

    labels: List[Optional[str]] = []
    if 'label' in sub.variables:
        raw = sub['label'].values
        raw = raw[valid_mask]
        for v in raw:
            try:
                s = v.decode('utf-8', errors='ignore').strip() if isinstance(v, (bytes, bytearray)) else str(v).strip()
            except Exception:
                s = ''
            labels.append(s if s != '' else None)
    else:
        labels = [None] * int(valid_mask.sum())

    try:
        ds.close()
    except Exception:
        pass
    return labels


# ========== 可視化 ==========
def save_anomaly_image(anom2d: np.ndarray, lat: np.ndarray, lon: np.ndarray, ts, out_path: str,
                       label_raw: Optional[str] = None, label_base: Optional[str] = None) -> None:
    """
    偏差[hPa] 2D を main_v7 の描画スタイルで保存（単枚）。
    """
    fig = plt.figure(figsize=(4, 3))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # 等値図
    cf = ax.contourf(lon, lat, anom2d, levels=LEVELS, cmap=CMAP, transform=ccrs.PlateCarree(), extend='both')
    ax.contour(lon, lat, anom2d, levels=LEVELS, colors=CONTOUR_COLOR, linewidths=CONTOUR_LW, transform=ccrs.PlateCarree())

    # 海岸線・範囲
    ax.add_feature(cfeature.COASTLINE.with_scale(COAST_SCALE), linewidth=COAST_LW, edgecolor=COAST_EDGE)
    ax.set_extent(EXTENT, ccrs.PlateCarree())

    # 軸目盛りを消す
    ax.set_xticks([])
    ax.set_yticks([])

    # ラベル表示（raw/base）。main_v7 のテイストに合わせて白背景の小枠で左上表示
    texts = []
    if label_raw is not None and str(label_raw).strip() != '':
        texts.append(f'Label: {label_raw}')
    if label_base is not None and str(label_base).strip() != '':
        texts.append(f'Base: {label_base}')
    if len(texts) > 0:
        ax.text(0.02, 0.96, "\n".join(texts), transform=ax.transAxes,
                fontsize=10, va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # カラーバー（ラベルは main_v7 と同一文言）
    cb = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label=CBAR_LABEL)
    cb.ax.tick_params(labelsize=10)
    cb.set_label(CBAR_LABEL, fontsize=11)

    # タイトルは YYYY/MM/DD のみ（整形は main_v7 の関数準拠）
    ax.set_title(format_date_yyyymmdd(ts), fontsize=11)

    plt.tight_layout(pad=0.5)
    plt.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.25)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='ERA5 msl anomaly image exporter (format aligned with main_v7).')
    parser.add_argument('--nc', type=str, default=DATA_FILE, help='NetCDF file path (default: ./prmsl_era5_all_data_seasonal_large.nc)')
    parser.add_argument('--start', type=str, default=START_DATE, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=END_DATE, help='End date (YYYY-MM-DD)')
    parser.add_argument('--out', type=str, default=OUT_DIR, help='Output directory (default: ./image)')
    parser.add_argument('--stride', type=int, default=1, help='Temporal stride to thin outputs (e.g., 7 for weekly). 1 = all.')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # データ読み込み（main_v7 と同じ定義に忠実：hPa化 -> 空間平均を差し引いた偏差）
    msl_hpa, anomaly_hpa, lat, lon, times = load_and_prepare_data_for_images(
        filepath=args.nc,
        start_date=args.start,
        end_date=args.end
    )
    labels = load_labels_aligned(args.nc, args.start, args.end)

    N = anomaly_hpa.shape[0]
    # 画像出力
    for i in range(0, N, max(1, int(args.stride))):
        ts = pd.to_datetime(times[i])
        date_tag = ts.strftime('%Y%m%d') if not pd.isna(ts) else f'{i:06d}'
        out_path = os.path.join(args.out, f'slp_anom_{date_tag}.png')
        lraw = labels[i] if labels is not None and i < len(labels) else None
        lbase = basic_label_or_none(lraw, BASE_LABELS) if lraw is not None else None
        save_anomaly_image(anomaly_hpa[i], lat, lon, times[i], out_path, label_raw=lraw, label_base=lbase)


if __name__ == '__main__':
    # 本スクリプトは「実行しなくてよい」要件だが、直接呼ばれた場合に備えエントリを用意
    main()
