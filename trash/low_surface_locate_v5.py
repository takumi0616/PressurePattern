#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
【機能概要】
1. グリッドサーチで NEIGHBOR_RADIUS, GAUSSIAN_SIGMA の組み合わせを変えながら、
   「前線と重なっている低気圧中心数 / 低気圧中心数」の割合を算出する。  
2. すべてのパラメータについてスコアをランキング形式で表示し、
   最も重なっている数が多いパラメータを「最適パラメータ」として選択する。  
3. 最適パラメータを使い、2014年のデータを対象に低気圧中心検出＋前線重ね描画した結果を出力する。  

【追加改修概要】
・最適パラメータを用いて、"./128_128/nc_gsm6/" 以下のすべての "gsm*.nc" に対して  
  低気圧中心の2次元マスク(1/0)を"surface_low_center"という新変数として追加し、  
  新しいディレクトリ "./128_128/nc_gsm7/" に保存する。
  nohup python3 low_surface_locate_v5.py > out_low_surface_locate_v5.log 2>&1 &
"""


import os
import glob
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import gaussian_filter, minimum_filter
import matplotlib.colors as mcolors

# --------------------------------------------------------------------------------
# 前線データの設定
FRONT_NC_DIR = "./128_128/nc_0p5_bulge_v2/"
CLASS_COLORS = {
    0: '#FFFFFF',
    1: '#FF0000',   # 温暖前線
    2: '#0000FF',   # 寒冷前線
    3: '#008015',   # 停滞前線
    4: '#800080',   # 閉塞前線
    5: '#FFA500',   # 暖冷前線
}
FRONT_CMAP = mcolors.ListedColormap([CLASS_COLORS[i] for i in range(6)])
FRONT_BOUNDS = np.arange(-0.5, 6.5, 1)
FRONT_NORM = mcolors.BoundaryNorm(FRONT_BOUNDS, FRONT_CMAP.N)


def detect_low_centers(data, radius, sigma):
    """
    ガウシアンフィルタでノイズ除去後に局所的極小を探索。
    radius: 極小探索の近傍半径
    sigma : ガウシアン平滑化の標準偏差
    戻り値: 2次元のTrue/Falseマスク (Trueが低圧中心候補)
    """
    # NaNを∞に変換
    data_filled = np.where(np.isnan(data), np.inf, data)
    
    # ガウシアンフィルターで平滑化し、局所的雑音を抑制
    smoothed = gaussian_filter(data_filled, sigma=sigma)

    # footprint作成（自身以外を近傍として用意）
    footprint = np.ones((2*radius+1, 2*radius+1), dtype=bool)
    footprint[radius, radius] = False

    # minimum_filterで局所最小かを判断
    local_min = smoothed < minimum_filter(smoothed, footprint=footprint, mode='constant', cval=np.inf)

    # ─────────────────────────────────────────
    # 修正箇所: 端の1マスを除外
    # ─────────────────────────────────────────
    local_min[0, :] = False     # 上端1行を検出対象外
    local_min[-1, :] = False    # 下端1行を検出対象外
    local_min[:, 0] = False     # 左端1列を検出対象外
    local_min[:, -1] = False    # 右端1列を検出対象外

    return local_min


def get_front_class_map(time_dt):
    """
    指定時刻に対応する前線クラスマップを返す (2次元のint配列)
    0: 前線なし
    1,2,3,4,5: 各種前線
    """
    month_str = time_dt.strftime('%Y%m')
    front_file = os.path.join(FRONT_NC_DIR, f'{month_str}.nc')
    if not os.path.exists(front_file):
        # 前線ファイルが無い場合は None を返す
        return None
    ds_front = xr.open_dataset(front_file)
    # 指定時刻に最も近いインデックスを取得
    nearest_t_idx = int(np.abs(ds_front['time'] - np.datetime64(time_dt)).argmin())

    class_map = np.zeros((len(ds_front['lat']), len(ds_front['lon'])), dtype=int)
    front_names = ['warm', 'cold', 'stationary', 'occluded', 'warm_cold']
    for cid, front_name in enumerate(front_names, 1):
        if front_name in ds_front.variables:
            mask = ds_front[front_name].isel(time=nearest_t_idx).values
            class_map[mask == 1] = cid

    ds_front.close()
    return class_map


def evaluate_param(input_dir, radius, sigma):
    """
    指定された radius, sigma で
    ./128_128/nc_gsm6/ 内の gsm*.nc 全てを対象に
    「前線と重なっている低気圧中心数 / 低気圧中心数」の割合や数を算出
    戻り値: (ratio, total_low_centers, overlapped_low_centers)
    """
    file_list = sorted(glob.glob(os.path.join(input_dir, 'gsm*.nc')))
    if not file_list:
        return 0.0, 0, 0  # ファイルが無い場合は0で返す

    total_low_centers = 0
    overlapped_low_centers = 0

    for file_path in file_list:
        ds = xr.open_dataset(file_path)
        if 'surface_prmsl' not in ds:
            ds.close()
            continue

        slp_data = ds['surface_prmsl']

        for t in range(slp_data.shape[0]):
            time_val = ds['time'].values[t]
            time_dt = pd.to_datetime(str(time_val))

            slp = slp_data.isel(time=t).values
            # 領域平均・気圧偏差
            area_mean = np.nanmean(slp)
            pressure_dev = slp - area_mean

            # 低気圧中心抽出 (ガウシアン＋局所極小)
            low_mask = detect_low_centers(slp, radius, sigma)

            # 低気圧中心の位置
            low_points = np.argwhere(low_mask)
            if len(low_points) == 0:
                continue

            # 前線データ取得
            front_class_map = get_front_class_map(time_dt)
            if front_class_map is None:
                continue

            # 全低気圧中心数を加算
            total_low_centers += len(low_points)

            # 前線と重なる低気圧中心数をカウント
            for y, x in low_points:
                if front_class_map[y, x] > 0:
                    overlapped_low_centers += 1

        ds.close()

    ratio = 0.0
    if total_low_centers > 0:
        ratio = overlapped_low_centers / total_low_centers

    return ratio, total_low_centers, overlapped_low_centers


def process_file_2014(file_path, output_dir, radius, sigma):
    """
    2014年用: 最適パラメータ (radius, sigma) を使って
    画像を生成（前線と低気圧中心の重ね描画）
    """
    print(f'処理中のファイル: {file_path}')
    ds = xr.open_dataset(file_path)
    if 'surface_prmsl' not in ds:
        ds.close()
        return

    slp_data = ds['surface_prmsl']
    lat = ds['lat'].values
    lon = ds['lon'].values
    Lon, Lat = np.meshgrid(lon, lat)

    for t in range(slp_data.shape[0]):
        time_val = ds['time'].values[t]
        time_dt = pd.to_datetime(str(time_val))
        t_str = time_dt.strftime('%Y%m%d%H%M')
        print(f'解析対象時刻: {t_str}')

        slp = slp_data.isel(time=t).values
        area_mean = np.nanmean(slp)
        pressure_dev = slp - area_mean

        # 低気圧中心抽出
        low_mask = detect_low_centers(slp, radius, sigma)

        # 低気圧中心の経緯度
        low_lats = lat[np.where(low_mask)[0]]
        low_lons = lon[np.where(low_mask)[1]]

        # 前線データ取得
        front_class_map = get_front_class_map(time_dt)
        if front_class_map is None:
            continue

        # プロット開始
        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()])
        ax.coastlines(resolution='10m')
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        ax.add_feature(cfeature.RIVERS)

        # 気圧偏差コンター
        cf = ax.contourf(Lon, Lat, pressure_dev,
                         levels=np.linspace(-40, 40, 21),
                         cmap='RdBu_r', extend='both')
        plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.02,
                     label='海面更正気圧偏差 (hPa)')

        # 前線データ重ねて表示
        ax.pcolormesh(Lon, Lat, front_class_map,
                      cmap=FRONT_CMAP, norm=FRONT_NORM,
                      shading='auto', alpha=0.6)

        # 低気圧中心表示
        ax.plot(low_lons, low_lats, 'rx', markersize=8)

        ax.set_title(f'海面更正気圧＋前線 ({t_str}) [R={radius}, Sigma={sigma}]')
        plt.savefig(os.path.join(output_dir, f'slp_{t_str}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    ds.close()


def add_low_center_variable(input_dir, output_dir, radius, sigma):
    """
    最適パラメータ (radius, sigma) を使って
    ./128_128/nc_gsm6/ 以下の gsm*.nc 全ファイルに
    新たに "surface_low_center"(time, lat, lon) を追加したファイルを
    ./128_128/nc_gsm7/ に保存する。
    ※ 値は 0 or 1 (低気圧中心なら1, それ以外は0)
    """
    os.makedirs(output_dir, exist_ok=True)

    file_list = sorted(glob.glob(os.path.join(input_dir, 'gsm*.nc')))
    if not file_list:
        print('[add_low_center_variable] gsm*.nc ファイルが見つかりません。')
        return

    for file_path in file_list:
        ds = xr.open_dataset(file_path)
        if 'surface_prmsl' not in ds:
            ds.close()
            continue

        slp_data = ds['surface_prmsl']  # (time, lat, lon)
        t_len = slp_data.shape[0]
        lat_len = slp_data.shape[1]
        lon_len = slp_data.shape[2]

        # 新しいデータ変数を0で初期化 (time, lat, lon)
        low_center_data = np.zeros((t_len, lat_len, lon_len), dtype=np.int8)

        for t in range(t_len):
            slp = slp_data.isel(time=t).values
            area_mean = np.nanmean(slp)
            # pressure_dev = slp - area_mean  # (今回は使用しない)

            # 局所極小だけを抽出
            low_mask = detect_low_centers(slp, radius, sigma)

            # True⇒1, False⇒0 に変換
            low_center_data[t, :, :] = low_mask.astype(np.int8)

        # xarray の DataArray としてセット
        da_low_center = xr.DataArray(
            low_center_data,
            dims=('time', 'lat', 'lon'),
            coords={
                'time': ds['time'],
                'lat': ds['lat'],
                'lon': ds['lon']
            },
            name='surface_low_center'
        )
        da_low_center.attrs['long_name'] = 'Low pressure center mask'
        da_low_center.attrs['units'] = '1 or 0'
        da_low_center.attrs['description'] = f'LOW center detection with radius={radius}, sigma={sigma}'

        # Datasetに追加
        ds['surface_low_center'] = da_low_center

        # 保存用パスを設定
        base_name = os.path.basename(file_path)  # 例: "gsm201409.nc" 等
        save_path = os.path.join(output_dir, base_name)  # "./128_128/nc_gsm7/gsm201409.nc" 等

        # 新形式で保存
        ds.to_netcdf(save_path)
        ds.close()

        print(f'[add_low_center_variable] 低気圧中心を追加して保存完了: {save_path}')


def main():
    input_dir = './128_128/nc_gsm6/'         # グリッドサーチの対象ディレクトリ
    output_dir = './surface_result_6_v5/'    # 2014年分の画像出力ディレクトリ
    new_dir = './128_128/nc_gsm7/'          # 低気圧中心を追加したファイルを保存するディレクトリ

    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------------------------------
    # 1. グリッドサーチのパラメータ範囲を設定
    radius_candidates = [6]
    sigma_candidates = [2.0]

    # 評価結果を保存するリスト: [(R, Sigma, ratio, total, overlapped), ... ]
    results = []

    # 全ファイル(gsm*.nc)を使って評価
    for r in radius_candidates:
        for s in sigma_candidates:
            ratio, total_cnt, overlap_cnt = evaluate_param(input_dir, r, s)
            print(f'[Check] NEIGHBOR_RADIUS={r}, GAUSSIAN_SIGMA={s} => '
                  f'重なり割合={ratio:.3f}, 全低気圧中心数={total_cnt}, 重なり数={overlap_cnt}')
            results.append((r, s, ratio, total_cnt, overlap_cnt))

    # すべてのパラメータを「前線と重なっている数(overlapped)」で降順ソート
    results_sorted = sorted(results, key=lambda x: x[4], reverse=True)

    # 最適パラメータは重なり数が最大のもの
    best_radius, best_sigma, best_ratio, best_total, best_overlapped = results_sorted[0]

    # ランキング結果を書き出すファイル
    log_filename = os.path.join(output_dir, 'parameter_ranking.log')
    with open(log_filename, 'w', encoding='utf-8') as logf:
        logf.write('■■ 全パラメータのランキング（重なり数降順） ■■\n')
        for rank, (rr, ss, rt, tot, ov) in enumerate(results_sorted, start=1):
            log_line = (f'{rank}位: R={rr}, Sigma={ss}, '
                        f'重なり割合={rt:.3f}, 全低気圧中心数={tot}, 重なり数={ov}\n')
            logf.write(log_line)

        logf.write('--------------------------------\n')
        logf.write(f'最適パラメータ: R={best_radius}, Sigma={best_sigma}, '
                   f'重なり数={best_overlapped}, (割合={best_ratio:.3f}, 全低気圧中心数={best_total})\n')
        logf.write('--------------------------------\n')

    # ランキング出力（画面にも表示）
    print('\n■■ 全パラメータのランキング（重なり数降順） ■■')
    for rank, (rr, ss, rt, tot, ov) in enumerate(results_sorted, start=1):
        print(f'{rank}位: R={rr}, Sigma={ss}, 重なり割合={rt:.3f}, '
              f'全低気圧中心数={tot}, 重なり数={ov}')
    print('--------------------------------')
    print(f'最適パラメータ: R={best_radius}, Sigma={best_sigma}, '
          f'重なり数={best_overlapped}, (割合={best_ratio:.3f}, 全低気圧中心数={best_total})')
    print('--------------------------------')

    # --------------------------------------------------------------------------------
    # (追加処理)
    # 2. 最適パラメータを用いて、"gsm*.nc" 全ファイルに "surface_low_center" を追加
    #    新しいディレクトリ (nc_gsm7) に保存する
    add_low_center_variable(input_dir, new_dir, best_radius, best_sigma)

    # --------------------------------------------------------------------------------
    # 3. 上で見つけた最適パラメータを使って、2014年分のみ画像を生成 (既存処理)
    file_list_2014 = sorted(glob.glob(os.path.join(input_dir, 'gsm2014*.nc')))
    if not file_list_2014:
        print('2014年のGSMファイルが見つかりません。')
        return

    for file_path in file_list_2014:
        process_file_2014(file_path, output_dir, best_radius, best_sigma)

    print('=== 全ての処理が完了しました ===')


if __name__ == '__main__':
    main()