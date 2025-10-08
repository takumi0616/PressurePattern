#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
低気圧中心検出・追跡・可視化（ERA5 海面更正気圧, Cartopy 図化, GIF生成）

要求仕様（ユーザ要望の要点を満たす機能を実装）:
- 入力: ./prmsl_era5_all_data_seasonal_large.nc（msl 変数のみ使用, 単位 Pa）
  次元:
    valid_time: CFTime(秒 since 1970-01-01)
    latitude: 161（55 → 15, 降順）
    longitude: 161（115 → 155）
- 低気圧中心検出を2系統で実装:
  1) 「ユーザ版」(low_surface_locate_v5 に準拠)
     - NaN→+inf 置換 → ガウシアン平滑（σ）→ footprint（中心除外, 正方形 2r+1）で minimum_filter
     - 厳密局所最小: s[i,j] < min(近傍) かつ 外周1セルは除外
  2) 「storm_tracking版」（ノートブック実装に準拠）
     - 平滑なし → minimum_filter(size=nsize, mode='nearest')
     - 等号判定: data == filtered を満たす点を候補（plateauも拾う）

- 可視化:
  - それぞれの検出結果について、範囲 [lon: 115〜155E, lat: 15〜55N] の天気図を
    全時刻で作成（msl 等圧線 + 海岸線 + 検出中心を赤いバツ印）
- 追跡（storm_tracking 相当）:
  - 初期時刻の全低気圧中心から開始し、以後の時刻において Shapely の buffer(半径R度)の
    交差面積が最大の候補を連結（しきい値 ε 超で継続）。結果をトラックとして作図。
  - トラック集合図（全トラック）と「規定以上継続（例: 5ステップ以上）」図を作成
  - 最長トラックについて、各時刻の MSLP + 軌跡のフレームPNGを作成し、GIF化

- 出力: src/PressurePattern/low_surface_locate_result/ 以下に全成果物を保存
  - algo_user/png/ ... ユーザ法による検出図
  - algo_storm/png/ ... storm法による検出図
  - tracking/ ... トラック集合図、24h（またはNステップ）以上図
  - anim/ ... 最長トラックの MSLP+追跡 GIF
  - txt/ ... minima 一覧、tracks 一覧

注意:
- 全時刻処理は非常に重い/大量出力になります。--limit や --skip を用いて確認しながらの実行を推奨します。
- 依存: xarray, numpy, scipy, shapely, matplotlib, cartopy, pandas, tqdm, Pillow

実行例:
  nohup python low_surface_locate.py --years 1980 1990 --do-gif > low_surface_locate.out 2>&1 &

"""

import os
import sys
import math
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from scipy.ndimage import gaussian_filter, minimum_filter, maximum_filter
from shapely.geometry import Point
from PIL import Image
from tqdm import tqdm


# =========================
# 設定（デフォルト）
# =========================

# 入力 NetCDF の既定パス（プロジェクトルート基準）
DEFAULT_INPUT = "./prmsl_era5_all_data_seasonal_large.nc"

# 出力ルート（スクリプトからの相対配置）
SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_ROOT = (SCRIPT_DIR / "low_surface_locate_result").resolve()

# ドメイン（ユーザ指定の範囲）
LON_W, LON_E = 115.0, 155.0
LAT_S, LAT_N = 15.0, 55.0

# 作図レベル等（適宜調整可）
MSLP_LEVELS = np.arange(900, 1072, 2)  # hPa 等圧線
# 陰影（圧力偏差）の塗りつぶし設定（低=青, 高=赤）
ANOM_LEVELS = np.arange(-40, 42, 2)  # hPa
ANOM_CMAP = "RdBu_r"
ANOM_ALPHA = 0.8

# ユーザ法パラメータ（初期値）
USER_RADIUS = 6          # 近傍半径 r → footprint (2r+1)^2
USER_SIGMA = 2.0         # ガウシアン σ (grid 単位)

# storm 法パラメータ（初期値）
STORM_NSIZE = 25         # minimum_filter の size

# 追跡パラメータ
# バッファ半径（度）: データが 1日刻みの場合は大きめが必要。dt_時間[h] × 0.15deg/h などでスケール可
BASE_BUFFER_DEG_PER_HOUR = 0.15
MIN_BUFFER_DEG = 2.0
MAX_BUFFER_DEG = 6.0
AREA_EPS = 1e-3          # 交差面積の下限（度^2）
MIN_TRACK_LEN = 5        # 「長いトラック」図に載せる最小長（ステップ数）

# GIF 生成の間隔（ファイル数抑制用, None=全フレーム）
GIF_EVERY = None


# =========================
# ユーティリティ
# =========================

def ensure_dirs() -> Dict[str, Path]:
    out = {
        "root": RESULT_ROOT,
        "algo_user": RESULT_ROOT / "algo_user" / "png",
        "algo_storm": RESULT_ROOT / "algo_storm" / "png",
        "compare": RESULT_ROOT / "compare" / "png",
        "tracking": RESULT_ROOT / "tracking",
        "anim": RESULT_ROOT / "anim",
        "txt": RESULT_ROOT / "txt",
    }
    for p in out.values():
        p.mkdir(parents=True, exist_ok=True)
    return out


def open_dataset_lazy(nc_path: Path) -> xr.Dataset:
    """
    入力 NetCDF を遅延ロードで開く。
    msl のみ利用。時間座標 valid_time はエポック秒なので pandas で変換。
    """
    ds = xr.open_dataset(nc_path)
    if "msl" not in ds:
        raise RuntimeError("msl 変数が見つかりません: " + str(nc_path))

    # 単位 Pa → hPa に変換
    # Note: 読み込み時は slice 時に部分読み込みを行う（遅延アクセス）
    return ds


def get_time_index_and_datetime(ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    valid_time の CFTime(秒 since 1970-01-01) を pandas.Timestamp に変換した配列を返す
    """
    if "valid_time" in ds:
        vt = ds["valid_time"].values
        # numpy datetime64[s] になっている場合もあるためユニット指定が必要なケースに対応
        try:
            times = pd.to_datetime(vt)
        except Exception:
            times = pd.to_datetime(vt, unit="s", origin="unix")
    else:
        # msl 変数に time 座標名が異なる形でついている可能性にフォールバック
        for cand in ("time", "valid_time"):
            if cand in ds["msl"].coords:
                vt = ds["msl"][cand].values
                try:
                    times = pd.to_datetime(vt)
                except Exception:
                    times = pd.to_datetime(vt, unit="s", origin="unix")
                break
        else:
            raise RuntimeError("時間座標(valid_time/time)が見つかりません")
    time_index = np.arange(len(times))
    return time_index, times.values


def select_time_indices(times: np.ndarray,
                        start_year: Optional[int] = None,
                        end_year: Optional[int] = None,
                        years_list: Optional[List[int]] = None) -> np.ndarray:
    """
    指定年で処理対象の時間インデックスを返す。
    - start_year/end_year を指定した場合は範囲（両端含む）
    - years_list を指定した場合はリスト（複数年）
    - どちらも None の場合は全インデックス
    """
    t = pd.to_datetime(times)
    years = t.year

    mask = np.ones_like(years, dtype=bool)

    if start_year is not None:
        mask &= (years >= start_year)
    if end_year is not None:
        mask &= (years <= end_year)
    if years_list is not None and len(years_list) > 0:
        years_set = set(int(y) for y in years_list)
        mask &= np.array([yy in years_set for yy in years], dtype=bool)

    indices = np.where(mask)[0]
    return indices


def compute_dt_hours(times: np.ndarray) -> float:
    """
    与えられた times 配列（np.datetime64 等）から代表的な時間間隔（時間）を返す。
    1要素または時間差が取れない場合は 6.0 を返す。
    """
    if times is None or len(times) < 2:
        return 6.0
    t = pd.to_datetime(times)
    # numpy の timedelta を時間に換算
    diffs = (t[1:] - t[:-1]) / np.timedelta64(1, 'h')
    # 不規則な場合もあり得るのでメディアンで代表化
    try:
        return float(np.median(diffs))
    except Exception:
        return 6.0


def subset_domain(data2d: np.ndarray, lats: np.ndarray, lons: np.ndarray,
                  lat_s=LAT_S, lat_n=LAT_N, lon_w=LON_W, lon_e=LON_E) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    指定範囲でのサブセットを返す（lat は降順が想定されるため slice は lat_n → lat_s）
    """
    # lats: (lat,), lons: (lon,)
    # Xarray の sel を使う方が簡便だが、本関数は Numpy 配列想定のため index を作る
    lat_mask = (lats <= lat_n + 1e-9) & (lats >= lat_s - 1e-9)
    lon_mask = (lons >= lon_w - 1e-9) & (lons <= lon_e + 1e-9)

    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    if len(lat_idx) == 0 or len(lon_idx) == 0:
        return data2d, lats, lons

    data_sub = data2d[np.ix_(lat_idx, lon_idx)]
    lats_sub = lats[lat_idx]
    lons_sub = lons[lon_idx]
    return data_sub, lats_sub, lons_sub


def to_hpa(pa_array: np.ndarray) -> np.ndarray:
    return pa_array / 100.0


def plot_msl_with_centers(save_path: Path,
                          lon: np.ndarray,
                          lat: np.ndarray,
                          msl_hpa_2d: np.ndarray,
                          centers_mask: Optional[np.ndarray],
                          title: str,
                          red_x_size: int = 8):
    """
    MSLP 等圧線 + 海岸線 + 低気圧中心(赤いバツ印) を描画して保存
    """
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=proj)
    ax.set_extent([LON_W, LON_E, LAT_S, LAT_N], crs=proj)
    ax.coastlines(resolution="10m", color="gray")
    ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="gray", alpha=0.7)
    ax.add_feature(cfeature.LAKES, alpha=0.4)
    ax.add_feature(cfeature.RIVERS, alpha=0.5)

    # 圧力偏差（領域平均を差し引き）で塗りつぶし（低=青, 高=赤）
    try:
        anom = msl_hpa_2d - np.nanmean(msl_hpa_2d)
        cf = ax.contourf(lon, lat, anom, levels=ANOM_LEVELS, cmap=ANOM_CMAP,
                         extend="both", alpha=ANOM_ALPHA, transform=proj)
        cb = fig.colorbar(cf, orientation="horizontal", aspect=65, shrink=0.75, pad=0.05, extendrect=True)
        cb.set_label("MSLP anomaly (hPa)", size="large")
    except Exception:
        pass

    # 等圧線
    try:
        cs = ax.contour(lon, lat, msl_hpa_2d, levels=MSLP_LEVELS, colors="black", linewidths=0.8, transform=proj)
        ax.clabel(cs, fmt="%4.0f", fontsize=8)
    except Exception:
        # 等値線が引けないケース（領域外や NaN 過多）を許容
        pass

    # 低気圧中心（赤いバツ印）
    if centers_mask is not None and centers_mask.shape == msl_hpa_2d.shape:
        ys, xs = np.where(centers_mask)
        if len(xs) > 0:
            lon_points = lon[xs]
            lat_points = lat[ys]
            ax.plot(lon_points, lat_points, "rx", markersize=red_x_size, transform=proj)

    ax.set_title(title, fontsize=12)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_msl_panel(ax,
                   lon: np.ndarray,
                   lat: np.ndarray,
                   msl_hpa_2d: np.ndarray,
                   centers_mask: Optional[np.ndarray],
                   panel_title: str,
                   add_colorbar: bool = False):
    """
    1枚の軸(ax)に対して、圧力偏差の塗りつぶし＋等圧線＋低気圧中心(赤×)を描画する。
    戻り値: contourf の返り値（共有カラーバー用に使用可能）
    """
    proj = ccrs.PlateCarree()
    ax.set_extent([LON_W, LON_E, LAT_S, LAT_N], crs=proj)
    ax.coastlines(resolution="10m", color="gray")
    ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="gray", alpha=0.7)
    ax.add_feature(cfeature.LAKES, alpha=0.4)
    ax.add_feature(cfeature.RIVERS, alpha=0.5)

    cf = None
    try:
        anom = msl_hpa_2d - np.nanmean(msl_hpa_2d)
        cf = ax.contourf(lon, lat, anom, levels=ANOM_LEVELS, cmap=ANOM_CMAP,
                         extend="both", alpha=ANOM_ALPHA, transform=proj)
        if add_colorbar and cf is not None:
            fig = ax.get_figure()
            cb = fig.colorbar(cf, orientation="horizontal", aspect=65, shrink=0.75, pad=0.05, extendrect=True)
            cb.set_label("MSLP anomaly (hPa)", size="large")
    except Exception:
        pass

    try:
        cs = ax.contour(lon, lat, msl_hpa_2d, levels=MSLP_LEVELS, colors="black", linewidths=0.8, transform=proj)
        ax.clabel(cs, fmt="%4.0f", fontsize=8)
    except Exception:
        pass

    if centers_mask is not None and centers_mask.shape == msl_hpa_2d.shape:
        ys, xs = np.where(centers_mask)
        if len(xs) > 0:
            lon_points = lon[xs]
            lat_points = lat[ys]
            ax.plot(lon_points, lat_points, "rx", markersize=8, transform=proj)

    ax.set_title(panel_title, fontsize=12)
    return cf


def draw_msl_panel_dual(ax,
                        lon: np.ndarray,
                        lat: np.ndarray,
                        msl_hpa_2d: np.ndarray,
                        lows_mask: Optional[np.ndarray],
                        highs_mask: Optional[np.ndarray],
                        panel_title: str,
                        add_colorbar: bool = False):
    """
    ユーザ法（CLパラメータ準拠）で検出した低気圧(青×)・高気圧(赤×)を同一パネルに可視化
    """
    proj = ccrs.PlateCarree()
    ax.set_extent([LON_W, LON_E, LAT_S, LAT_N], crs=proj)
    ax.coastlines(resolution="10m", color="gray")
    ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="gray", alpha=0.7)
    ax.add_feature(cfeature.LAKES, alpha=0.4)
    ax.add_feature(cfeature.RIVERS, alpha=0.5)

    cf = None
    try:
        anom = msl_hpa_2d - np.nanmean(msl_hpa_2d)
        cf = ax.contourf(lon, lat, anom, levels=ANOM_LEVELS, cmap=ANOM_CMAP,
                         extend="both", alpha=ANOM_ALPHA, transform=proj)
        if add_colorbar and cf is not None:
            fig = ax.get_figure()
            cb = fig.colorbar(cf, orientation="horizontal", aspect=65, shrink=0.75, pad=0.05, extendrect=True)
            cb.set_label("MSLP anomaly (hPa)", size="large")
    except Exception:
        pass

    try:
        cs = ax.contour(lon, lat, msl_hpa_2d, levels=MSLP_LEVELS, colors="black", linewidths=0.8, transform=proj)
        ax.clabel(cs, fmt="%4.0f", fontsize=8)
    except Exception:
        pass

    # 低(青×)
    if lows_mask is not None and lows_mask.shape == msl_hpa_2d.shape:
        ys, xs = np.where(lows_mask)
        if len(xs) > 0:
            ax.plot(lon[xs], lat[ys], "bx", markersize=8, transform=proj)
    # 高(赤×)
    if highs_mask is not None and highs_mask.shape == msl_hpa_2d.shape:
        ys, xs = np.where(highs_mask)
        if len(xs) > 0:
            ax.plot(lon[xs], lat[ys], "rx", markersize=8, transform=proj)

    ax.set_title(panel_title, fontsize=12)
    return cf


# =========================
# 低気圧中心検出（2実装）
# =========================

def detect_centers_user(slp_hpa_2d: np.ndarray, radius: int = USER_RADIUS, sigma: float = USER_SIGMA) -> np.ndarray:
    """
    ユーザ実装: ガウシアン平滑 → 厳密局所最小（中心を近傍から除外）→ 外周1セル除外
    入力は hPa（ただし単位は問わない）
    戻り値: bool mask (True=低気圧中心)
    """
    data = slp_hpa_2d.astype(np.float32)
    data_filled = np.where(np.isnan(data), np.inf, data)
    smoothed = gaussian_filter(data_filled, sigma=sigma)

    # footprint: (2r+1)^2 の正方形で中心セルは False
    footprint = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    footprint[radius, radius] = False

    neigh_min = minimum_filter(smoothed, footprint=footprint, mode="constant", cval=np.inf)
    local_min = smoothed < neigh_min

    # 外周 1 セルは除外
    if local_min.shape[0] > 2 and local_min.shape[1] > 2:
        local_min[0, :] = False
        local_min[-1, :] = False
        local_min[:, 0] = False
        local_min[:, -1] = False

    return local_min


def detect_centers_user_high(slp_hpa_2d: np.ndarray, radius: int = USER_RADIUS, sigma: float = USER_SIGMA) -> np.ndarray:
    """
    ユーザ実装（高気圧版）: ガウシアン平滑 → 厳密局所最大（中心除外）→ 外周1セル除外
    戻り値: bool mask (True=高気圧中心)
    """
    data = slp_hpa_2d.astype(np.float32)
    data_filled = np.where(np.isnan(data), -np.inf, data)
    smoothed = gaussian_filter(data_filled, sigma=sigma)

    # footprint: (2r+1)^2 の正方形で中心セルは False
    footprint = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    footprint[radius, radius] = False

    neigh_max = maximum_filter(smoothed, footprint=footprint, mode="constant", cval=-np.inf)
    local_max = smoothed > neigh_max

    # 外周 1 セルは除外
    if local_max.shape[0] > 2 and local_max.shape[1] > 2:
        local_max[0, :] = False
        local_max[-1, :] = False
        local_max[:, 0] = False
        local_max[:, -1] = False

    return local_max


def detect_centers_storm(slp_hpa_2d: np.ndarray, nsize: int = STORM_NSIZE) -> np.ndarray:
    """
    storm_tracking 実装: 平滑なし → minimum_filter(size=nsize) と一致するセルを抽出（等号判定）
    戻り値: bool mask
    """
    data = slp_hpa_2d.astype(np.float32)
    data_filled = np.where(np.isnan(data), np.inf, data)
    data_ext = minimum_filter(data_filled, size=nsize, mode="nearest")
    # equality で plateau も拾う
    local_min = data_filled == data_ext
    return local_min


# =========================
# storm_tracking 追跡
# =========================

def default_buffer_radius_deg(dt_hours: float) -> float:
    # 時間差に比例してバッファを拡大（上限下限あり）
    r = max(MIN_BUFFER_DEG, min(MAX_BUFFER_DEG, BASE_BUFFER_DEG_PER_HOUR * dt_hours))
    return r


def build_minima_list_for_all_times(ds: xr.Dataset,
                                    times: np.ndarray,
                                    lats: np.ndarray,
                                    lons: np.ndarray,
                                    method: str = "storm",
                                    user_radius: int = USER_RADIUS,
                                    user_sigma: float = USER_SIGMA,
                                    storm_nsize: int = STORM_NSIZE,
                                    indices: Optional[np.ndarray] = None,
                                    extent_clip: bool = True) -> List[List[Tuple[float, float, float]]]:
    """
    指定インデックス列（indices）が与えられた時刻について、minima の一覧を作る。
    戻り値: minima_per_t[k] = [(lon, lat, pres_hPa), ...] （k は indices の走査順に対応）
    """
    msl = ds["msl"]  # Pa

    if "latitude" in msl.coords:
        ds_lats = msl["latitude"].values
    else:
        ds_lats = ds["latitude"].values
    if "longitude" in msl.coords:
        ds_lons = msl["longitude"].values
    else:
        ds_lons = ds["longitude"].values

    if indices is None:
        indices = np.arange(len(times))

    minima_per_t: List[List[Tuple[float, float, float]]] = []

    for ti in tqdm(indices, desc=f"Detect minima ({method})", unit="t"):
        # 有効時刻のスライス → Pa → hPa
        slp_pa = msl.isel(valid_time=ti).values
        slp_hpa = to_hpa(slp_pa)

        # 範囲サブセット
        slp_sub, lat_sub, lon_sub = subset_domain(slp_hpa, ds_lats, ds_lons, LAT_S, LAT_N, LON_W, LON_E)

        if method == "user":
            mask = detect_centers_user(slp_sub, radius=user_radius, sigma=user_sigma)
        elif method == "storm":
            mask = detect_centers_storm(slp_sub, nsize=storm_nsize)
        else:
            raise ValueError("method must be 'user' or 'storm'")

        ys, xs = np.where(mask)
        items: List[Tuple[float, float, float]] = []
        for y, x in zip(ys, xs):
            lon_v = float(lon_sub[x])
            lat_v = float(lat_sub[y])
            pres_v = float(slp_sub[y, x])
            if extent_clip:
                if not (LON_W <= lon_v <= LON_E and LAT_S <= lat_v <= LAT_N):
                    continue
            items.append((lon_v, lat_v, pres_v))
        minima_per_t.append(items)

    return minima_per_t


def track_from_initial_time(minima_per_t: List[List[Tuple[float, float, float]]],
                            times: np.ndarray,
                            dt_hours: float,
                            buffer_deg: Optional[float] = None,
                            area_eps: float = AREA_EPS) -> List[List[Tuple[float, float, float, np.datetime64]]]:
    """
    初期時刻(最初の要素)の minima を全て起点として、それぞれに最長のトラックを構築する。
    - 点は (lon, lat, pres) とし、トラックには時刻も保存する。
    - 隣接時刻の候補点に対し、Shapely の buffer(R度) の交差面積が最大の点を採用（面積>area_eps の場合のみ継続）

    戻り値: tracks = [[(lon,lat,pres,time), ...], ...]
    """
    if len(minima_per_t) == 0:
        return []

    if buffer_deg is None:
        buffer_deg = default_buffer_radius_deg(dt_hours)

    tracks: List[List[Tuple[float, float, float, np.datetime64]]] = []

    initial_points = minima_per_t[0]  # t0
    for p0 in initial_points:
        lon0, lat0, pres0 = p0
        tr = [(lon0, lat0, pres0, times[0])]
        last_point = Point(lon0, lat0)

        for k in range(1, len(minima_per_t)):
            cand = minima_per_t[k]
            if len(cand) == 0:
                break

            # 交差面積最大の候補を選ぶ
            buf_last = last_point.buffer(buffer_deg)
            areas = []
            for (lon1, lat1, pres1) in cand:
                area = buf_last.intersection(Point(lon1, lat1).buffer(buffer_deg)).area
                areas.append(area)
            areas = np.array(areas, dtype=float)
            if np.max(areas) > area_eps:
                idx = int(np.argmax(areas))
                lon1, lat1, pres1 = cand[idx]
                tr.append((lon1, lat1, pres1, times[k]))
                last_point = Point(lon1, lat1)
            else:
                break

        tracks.append(tr)

    return tracks


def save_minima_txt(out_txt_path: Path,
                    times: np.ndarray,
                    minima_per_t: List[List[Tuple[float, float, float]]]):
    """
    「時刻 lon lat pres(hPa)」の行を連ねたテキストを書き出す（全時刻, 全最小）
    """
    with out_txt_path.open("w", encoding="utf-8") as f:
        for k, items in enumerate(minima_per_t):
            tstr = pd.to_datetime(times[k]).strftime("%Y-%m-%dT%H")
            for (lon, lat, pres) in items:
                f.write(f"{tstr} {lon:.3f} {lat:.3f} {int(round(pres))}\n")


def save_tracks_txt(out_tracks_path: Path,
                    tracks: List[List[Tuple[float, float, float, np.datetime64]]]):
    """
    # 区切りで複数トラックを連結
    各行: "<YYYY-MM-DDTHH> <lon> <lat> <pres>"
    """
    with out_tracks_path.open("w", encoding="utf-8") as f:
        for tr in tracks:
            f.write("#\n")
            for (lon, lat, pres, tstamp) in tr:
                tstr = pd.to_datetime(tstamp).strftime("%Y-%m-%dT%H")
                f.write(f"{tstr} {lon:.3f} {lat:.3f} {int(round(pres))}\n")


def plot_tracks_map(save_path: Path,
                    tracks: List[List[Tuple[float, float, float, np.datetime64]]],
                    title: str = "Tracks (initial time)",
                    color_by_index: bool = True):
    """
    全トラックの起点(緑)と終点(赤)、経路（線）を描画した地図を保存
    """
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(12, 9))
    ax = plt.axes(projection=proj)
    ax.set_extent([LON_W, LON_E, LAT_S, LAT_N], crs=proj)
    ax.coastlines(resolution="10m", color="gray")
    ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="gray", alpha=0.7)
    ax.gridlines(draw_labels=False, color="gray", alpha=0.6, linestyle="-")

    for idx, tr in enumerate(tracks):
        if len(tr) == 0:
            continue
        xs = [p[0] for p in tr]
        ys = [p[1] for p in tr]
        color = None
        if color_by_index:
            # インデックスで色を変える（簡易）
            color = plt.cm.tab20((idx % 20) / 19.0)
        ax.plot(xs, ys, "-", color=color, transform=proj, label=str(idx) if idx < 20 else None)
        ax.scatter(xs[0], ys[0], c="green", transform=proj)
        ax.scatter(xs[-1], ys[-1], c="red", transform=proj)

    ax.set_title(title, fontsize=14)
    # 凡例はトラック数が多い場合に冗長のため上位のみ
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=8, ncol=1)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_gif_from_pngs(png_dir: Path, gif_path: Path, duration_ms: int = 900, pattern: str = "*.png"):
    files = sorted(list(png_dir.glob(pattern)))
    if len(files) == 0:
        return
    images = [Image.open(fp) for fp in files]
    images[0].save(
        gif_path, save_all=True, append_images=images[1:],
        duration=duration_ms, loop=0
    )
    # 画像は残す（用途に応じて削除可）


def build_msl_tracking_frames_and_gif(out_dir_anim: Path,
                                      out_dir_png: Path,
                                      lon: np.ndarray,
                                      lat: np.ndarray,
                                      ds: xr.Dataset,
                                      track: List[Tuple[float, float, float, np.datetime64]],
                                      every: Optional[int] = GIF_EVERY):
    """
    最長トラックについて、各時刻における MSLP + 追跡経路のフレームを作成し GIF 化
    """
    out_dir_png.mkdir(parents=True, exist_ok=True)

    # track に含まれる時刻に対応する index を逆引き
    times_all = pd.to_datetime(ds["valid_time"].values, unit="s", origin="unix")
    time_to_index = {pd.to_datetime(tt).to_datetime64(): idx for idx, tt in enumerate(times_all)}

    proj = ccrs.PlateCarree()

    for i, (lon_i, lat_i, pres_i, t_i) in enumerate(tqdm(track, desc="Frames for longest track", unit="frame")):
        if every is not None and (i % every != 0):
            continue

        ti = time_to_index[pd.to_datetime(t_i).to_datetime64()]

        slp_hpa = to_hpa(ds["msl"].isel(valid_time=ti).values)
        slp_sub, lat_sub, lon_sub = subset_domain(slp_hpa, ds["latitude"].values, ds["longitude"].values,
                                                  LAT_S, LAT_N, LON_W, LON_E)

        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=proj)
        ax.set_extent([LON_W, LON_E, LAT_S, LAT_N], crs=proj)
        ax.coastlines(resolution="10m", color="gray")
        ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="gray", alpha=0.7)
        ax.gridlines(draw_labels=False, color="gray", alpha=0.6, linestyle="-")

        # 圧力偏差の塗りつぶし（低=青, 高=赤）
        try:
            anom = slp_sub - np.nanmean(slp_sub)
            cf = ax.contourf(lon_sub, lat_sub, anom, levels=ANOM_LEVELS, cmap=ANOM_CMAP,
                             extend="both", alpha=ANOM_ALPHA, transform=proj)
            cb = fig.colorbar(cf, orientation="horizontal", aspect=65, shrink=0.75, pad=0.05, extendrect=True)
            cb.set_label("MSLP anomaly (hPa)", size="large")
        except Exception:
            pass

        try:
            cs = ax.contour(lon_sub, lat_sub, slp_sub, levels=MSLP_LEVELS, colors="black", linewidths=0.8, transform=proj)
            ax.clabel(cs, fmt="%4.0f", fontsize=8)
        except Exception:
            pass

        # 今までの軌跡
        xs = [p[0] for p in track[:i + 1]]
        ys = [p[1] for p in track[:i + 1]]
        ax.plot(xs, ys, "r-+", transform=proj)
        ax.scatter(xs[-1], ys[-1], c="green", transform=proj)

        tstr_l = pd.to_datetime(track[0][3]).strftime("%Y-%m-%dT%H")
        tstr_r = pd.to_datetime(t_i).strftime("%Y-%m-%dT%H")
        ax.set_title(f"MSLP and tracking: {tstr_l} to {tstr_r}", fontsize=12)

        out_png = out_dir_png / f"MSL_tracking_{tstr_r}.png"
        fig.savefig(out_png, dpi=180, bbox_inches="tight")
        plt.close(fig)

    # GIF に結合
    gif_path = out_dir_anim / "MSL_tracking.gif"
    make_gif_from_pngs(out_dir_png, gif_path, duration_ms=900, pattern="*.png")


# =========================
# メイン処理
# =========================

def main():
    parser = argparse.ArgumentParser(description="ERA5 MSLP 低気圧中心の検出・追跡・可視化")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT,
                        help="入力 NetCDF パス（デフォルト: ./prmsl_era5_all_data_seasonal_large.nc）")
    parser.add_argument("--user-radius", type=int, default=USER_RADIUS, help="ユーザ法: 近傍半径 r")
    parser.add_argument("--user-sigma", type=float, default=USER_SIGMA, help="ユーザ法: ガウシアン σ")
    parser.add_argument("--storm-nsize", type=int, default=STORM_NSIZE, help="storm 法: minimum_filter のサイズ")
    # CL（ユーザ法）ハイパーパラメータ（SOM側と整合する名称で上書き可）
    parser.add_argument("--cl-radius", type=int, default=None, help="CL/user法: 近傍半径 r の上書き（未指定で --user-radius を使用）")
    parser.add_argument("--cl-sigma", type=float, default=None, help="CL/user法: ガウシアン σ の上書き（未指定で --user-sigma を使用）")
    parser.add_argument("--cl-topk", type=int, default=5, help="CL距離で使用するTop-K（可視化では未使用）")
    parser.add_argument("--start-year", type=int, default=1991, help="処理開始年（デフォルト: 1991）")
    parser.add_argument("--end-year", type=int, default=2000, help="処理終了年（デフォルト: 2000、指定時はその年まで含む）")
    parser.add_argument("--years", type=int, nargs="+", default=None, help="処理対象の年（複数指定可, 例: --years 1980 1981 1990）")
    parser.add_argument("--limit", type=int, default=None, help="最大時刻数（None で全時刻）")
    parser.add_argument("--skip", type=int, default=1, help="時間の間引き（例: 24 → 1日毎に）")
    parser.add_argument("--do-gif", action="store_true", help="最長トラックの GIF も作成する")
    args = parser.parse_args()

    out_dirs = ensure_dirs()

    # 入力パスの解決
    input_path = Path(args.input)
    if not input_path.exists():
        # スクリプトの2つ上（プロジェクト root）からの相対も試す
        root_guess = SCRIPT_DIR.parents[2] if len(SCRIPT_DIR.parents) >= 2 else SCRIPT_DIR
        alt = root_guess / "prmsl_era5_all_data_seasonal_large.nc"
        if alt.exists():
            input_path = alt
        else:
            raise FileNotFoundError(f"入力ファイルが見つかりません: {args.input}")

    print(f"[INFO] Open dataset: {input_path}")
    ds = open_dataset_lazy(input_path)

    # 座標配列
    ds_lats = ds["latitude"].values
    ds_lons = ds["longitude"].values
    time_index, times = get_time_index_and_datetime(ds)

    # 時間間隔（時間）を計算（追跡バッファ半径の基準）
    if len(times) >= 2:
        dt_hours = (pd.to_datetime(times[1]) - pd.to_datetime(times[0])).total_seconds() / 3600.0
    else:
        dt_hours = 6.0  # 仮

    # =============
    # 時間範囲の選択（年単位）
    # =============
    sel_idx = select_time_indices(times, start_year=args.start_year, end_year=args.end_year, years_list=args.years)
    if args.limit is not None:
        sel_idx = sel_idx[:args.limit]
    sel_idx = sel_idx[::max(1, args.skip)]
    sel_times = times[sel_idx]
    dt_hours_sel = compute_dt_hours(sel_times)

    # =============
    # 低圧中心検出（ユーザ法 vs storm法）を並列表示で保存
    # =============
    # 統計カウンタ（ユーザ法: 低/高の検出数を各時刻でカウント）
    det_stats = {"times": [], "low_counts": [], "high_counts": []}
    for ti in tqdm(sel_idx, desc="Plot (user vs storm)", unit="t"):
        slp_pa = ds["msl"].isel(valid_time=ti).values
        slp_hpa = to_hpa(slp_pa)
        slp_sub, lat_sub, lon_sub = subset_domain(slp_hpa, ds_lats, ds_lons, LAT_S, LAT_N, LON_W, LON_E)

        # CLパラメータ（ユーザ法）適用（上書きがあれば優先）
        used_r = args.cl_radius if args.cl_radius is not None else args.user_radius
        used_s = args.cl_sigma if args.cl_sigma is not None else args.user_sigma
        # ユーザ法: 低/高の両方を検出
        mask_user_low = detect_centers_user(slp_sub, radius=used_r, sigma=used_s)
        mask_user_high = detect_centers_user_high(slp_sub, radius=used_r, sigma=used_s)
        # 統計: 各時刻の検出数を記録
        low_n = int(np.count_nonzero(mask_user_low))
        high_n = int(np.count_nonzero(mask_user_high))
        det_stats["times"].append(pd.to_datetime(times[ti]))
        det_stats["low_counts"].append(low_n)
        det_stats["high_counts"].append(high_n)
        # storm 法（従来通り: 低のみ）
        mask_storm = detect_centers_storm(slp_sub, nsize=args.storm_nsize)

        tstr = pd.to_datetime(times[ti]).strftime("%Y-%m-%dT%H")

        # 2枚を横連結した1枚の画像として保存
        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=(18, 8))
        ax1 = plt.subplot(1, 2, 1, projection=proj)
        ax2 = plt.subplot(1, 2, 2, projection=proj)

        title1 = f"[User] Low(blue) & High(red) centers {tstr} (R={used_r}, sigma={used_s})"
        title2 = f"[Storm] Minima centers {tstr} (nsize={args.storm_nsize})"

        cf1 = draw_msl_panel_dual(ax1, lon_sub, lat_sub, slp_sub, mask_user_low, mask_user_high, title1, add_colorbar=False)
        _cf2 = draw_msl_panel(ax2, lon_sub, lat_sub, slp_sub, mask_storm, title2, add_colorbar=False)

        # 共有カラーバー（下部）
        try:
            if cf1 is not None:
                cax = fig.add_axes([0.25, 0.08, 0.5, 0.03])
                cb = fig.colorbar(cf1, cax=cax, orientation="horizontal", extend="both")
                cb.set_label("MSLP anomaly (hPa)")
        except Exception:
            pass

        out_png = out_dirs["compare"] / f"compare_{tstr}.png"
        fig.savefig(out_png, dpi=180, bbox_inches="tight")
        plt.close(fig)

    # =============
    # ユーザ法 検出統計（1991-01-01〜2000-12-31の要件に適合する既定範囲で集計）
    # =============
    try:
        total_steps = len(det_stats["times"])
        larr = np.array(det_stats["low_counts"], dtype=int) if total_steps > 0 else np.array([], dtype=int)
        harr = np.array(det_stats["high_counts"], dtype=int) if total_steps > 0 else np.array([], dtype=int)
        n_low_pos = int((larr > 0).sum()) if total_steps > 0 else 0
        n_high_pos = int((harr > 0).sum()) if total_steps > 0 else 0
        n_both_pos = int(((larr > 0) & (harr > 0)).sum()) if total_steps > 0 else 0
        n_none = int(((larr == 0) & (harr == 0)).sum()) if total_steps > 0 else 0
        avg_low = float(larr.mean()) if total_steps > 0 else float("nan")
        avg_high = float(harr.mean()) if total_steps > 0 else float("nan")

        # 期間表記
        if len(sel_times) > 0:
            date1 = pd.to_datetime(sel_times[0]).strftime("%Y-%m-%dT%H")
            dateN = pd.to_datetime(sel_times[-1]).strftime("%Y-%m-%dT%H")
        else:
            date1, dateN = "NA", "NA"

        # CSV（各時刻のカウント）
        df_counts = pd.DataFrame({
            "time": [pd.to_datetime(t).strftime("%Y-%m-%dT%H") for t in det_stats["times"]],
            "low_count": det_stats["low_counts"],
            "high_count": det_stats["high_counts"],
        })
        out_counts_csv = ensure_dirs()["txt"] / f"user_detect_counts_{date1}_{dateN}.csv"
        df_counts.to_csv(out_counts_csv, index=False, encoding="utf-8-sig")

        # サマリTXT
        out_summary_txt = ensure_dirs()["txt"] / f"user_detect_summary_{date1}_{dateN}.txt"
        with out_summary_txt.open("w", encoding="utf-8") as fsum:
            fsum.write(f"Detection summary (User method) for [{date1} .. {dateN}]\n")
            fsum.write(f"Total timesteps: {total_steps}\n")
            fsum.write(f"Low detected (>=1): {n_low_pos}  ({(n_low_pos/total_steps*100.0 if total_steps>0 else 0):.2f}%)\n")
            fsum.write(f"High detected(>=1): {n_high_pos}  ({(n_high_pos/total_steps*100.0 if total_steps>0 else 0):.2f}%)\n")
            fsum.write(f"Both detected     : {n_both_pos}  ({(n_both_pos/total_steps*100.0 if total_steps>0 else 0):.2f}%)\n")
            fsum.write(f"None detected     : {n_none}  ({(n_none/total_steps*100.0 if total_steps>0 else 0):.2f}%)\n")
            fsum.write(f"Avg # of lows per timestep : {avg_low:.3f}\n")
            fsum.write(f"Avg # of highs per timestep: {avg_high:.3f}\n")

        # コンソールにも出力
        print(f"[STATS] User method detection [{date1}..{dateN}]")
        print(f"  Steps={total_steps}, Low>=1={n_low_pos} ({(n_low_pos/total_steps*100.0 if total_steps>0 else 0):.2f}%), "
              f"High>=1={n_high_pos} ({(n_high_pos/total_steps*100.0 if total_steps>0 else 0):.2f}%), "
              f"Both={n_both_pos} ({(n_both_pos/total_steps*100.0 if total_steps>0 else 0):.2f}%), "
              f"None={n_none} ({(n_none/total_steps*100.0 if total_steps>0 else 0):.2f}%)")
        print(f"  Avg lows/t={avg_low:.3f}, Avg highs/t={avg_high:.3f}")
        print(f"  -> CSV: {out_counts_csv}")
        print(f"  -> Summary: {out_summary_txt}")
    except Exception as e:
        print(f"[WARN] Failed to build detection statistics: {e}")

    # =============
    # storm_tracking 互換の minima テキスト化・トラック作成・作図
    # =============
    # 1) minima 一覧（storm 法で検出）を全時刻分作成
    minima_per_t = build_minima_list_for_all_times(
        ds=ds, times=sel_times, lats=ds_lats, lons=ds_lons,
        method="storm", storm_nsize=args.storm_nsize,
        user_radius=args.user_radius, user_sigma=args.user_sigma,
        indices=sel_idx, extent_clip=True
    )
    date1 = pd.to_datetime(sel_times[0]).strftime("%Y-%m-%dT%H") if len(sel_times) > 0 else "NA"
    dateN = pd.to_datetime(sel_times[-1]).strftime("%Y-%m-%dT%H") if len(sel_times) > 0 else "NA"
    out_min_txt = out_dirs["txt"] / f"era5_minimums_{date1}_{dateN}.txt"
    save_minima_txt(out_min_txt, sel_times, minima_per_t)

    # 2) 追跡（初期時刻からの全系）
    buffer_deg = default_buffer_radius_deg(dt_hours_sel)
    tracks = track_from_initial_time(minima_per_t, sel_times, dt_hours_sel, buffer_deg=buffer_deg)
    out_trk_txt = out_dirs["txt"] / f"era5_tracks_{date1}.txt"
    save_tracks_txt(out_trk_txt, tracks)

    # 3) 全トラック図
    plot_tracks_map(out_dirs["tracking"] / f"tracks_{date1}.png", tracks,
                    title=f"Tracks of all systems detected at {date1} (buffer={buffer_deg:.1f}°)")

    # 4) 最小長（MIN_TRACK_LEN）以上のトラックのみの図
    long_tracks = [tr for tr in tracks if len(tr) >= MIN_TRACK_LEN]
    if len(long_tracks) > 0:
        plot_tracks_map(out_dirs["tracking"] / f"tracks_len_ge_{MIN_TRACK_LEN}_{date1}.png",
                        long_tracks,
                        title=f"Tracks (len ≥ {MIN_TRACK_LEN}) detected at {date1}")
    else:
        # なければ全トラック図をコピー
        pass

    # 5) 最長トラックについて GIF（オプション）
    if args.do_gif and len(tracks) > 0:
        longest = max(tracks, key=lambda tr: len(tr))
        anim_dir = out_dirs["anim"]
        png_dir = anim_dir / "frames_longest"
        build_msl_tracking_frames_and_gif(anim_dir, png_dir, ds_lons, ds_lats, ds, longest, every=GIF_EVERY)

    print("[DONE] All products saved under:", str(out_dirs["root"]))


if __name__ == "__main__":
    main()
