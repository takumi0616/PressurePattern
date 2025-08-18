# main_v2.py
# -*- coding: utf-8 -*-
import os
import sys
import json
import logging
import time
from typing import Optional, List, Dict, Tuple
from collections import Counter

import numpy as np
import pandas as pd
import xarray as xr
import torch

import matplotlib
matplotlib.use('Agg')  # サーバ上でも保存できるように
import matplotlib.pyplot as plt

# cartopy はクラスタリング側で必須（元コード準拠）
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize

# s1クラスタリングのユーティリティ
from s1_clustering import (
    two_stage_clustering,
    evaluate_clusters_only_base,
    analyze_cluster_distribution,
    plot_s1_distribution_histogram,
    plot_final_distribution_summary,
    plot_final_clusters_medoids,
    save_daily_maps_per_cluster,
    save_metrics_history_to_csv,
    plot_iteration_metrics,
    summarize_cluster_info,
    basic_label_or_none,
    primary_base_label,
)

# 3type版 SOM（Euclidean/SSIM/S1対応）
from minisom import MiniSom as MultiDistMiniSom  # 3type_som/minisom.py
# s1版 SOM（メドイド可視化用：S1距離のみ）
from s1_minisom import MiniSom as S1MiniSom, grid_auto_size


# ============== ユーザ調整パラメータ（クラスタリング側） ==============
SEED = 42                 # 乱数シード（再現性）
TH_MERGE = 77             # HACのマージ停止しきい値（S1スコア, 小さいほど類似）

# 2段階クラスタリング用 行×列チャンク
CLUS_ROW_BATCH = 16
CLUS_COL_CHUNK = 1024

# SOM学習・推論（メドイド用SOM）のバッチサイズ
SOM_BATCH_SIZE = 32

# S1分布ヒストグラム設定
S1_DISTRIBUTION_MAX_SAMPLES = 3653
S1_HIST_ROW_BATCH = 16
S1_HIST_COL_CHUNK = 1024

# 日次マップ出力の1クラスタあたり上限(Noneで無制限)
DAILY_MAPS_PER_CLUSTER_LIMIT: Optional[int] = None

# メドイドSOM学習のイテレーション上限
SOM_MAX_ITERS_CAP = 300

# ============== 3種類のbatchSOM（全期間版）設定（3type_som側） ==============
SOM_X, SOM_Y = 10, 10
NUM_ITER = 300
BATCH_SIZE = 512
NODES_CHUNK = 16
LOG_INTERVAL = 10
EVAL_SAMPLE_LIMIT = 2048

# ============== 固定：実験条件・パス等 ==============
DATA_FILE = './prmsl_era5_all_data_seasonal_large.nc'

# 期間指定（Noneでファイル全期間、日付文字列で範囲指定）
TIME_START = '1991-01-01'   # 例: None にすれば全期間
TIME_END   = '2000-12-31'   # 例: None にすれば全期間

# 出力先（元コードに合わせて保持）
RESULT_DIR = './results_v1'
ONLY_BASE_DIR = os.path.join(RESULT_DIR, 'only_base_label')
SOM_DIR = os.path.join(RESULT_DIR, 'som')  # クラスタリング側のSOM（メドイド用）
OUTPUT_ROOT = './outputs_som_fullperiod'   # 3type_som側の出力（名称は元のまま）

# 基本ラベル（15）
BASE_LABELS = [
    '1', '2A', '2B', '2C', '2D', '3A', '3B', '3C', '3D', '4A', '4B', '5', '6A', '6B', '6C'
]
# 3type_som側で使う複合まとめラベル
COMPOSITE_LABEL = 'COMPOSITE'


# =====================================================
# 再現性・ログ
# =====================================================
def set_reproducibility(seed: int = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def setup_logging_v2():
    for d in [RESULT_DIR, ONLY_BASE_DIR, SOM_DIR, OUTPUT_ROOT]:
        os.makedirs(d, exist_ok=True)
    log_path = os.path.join(RESULT_DIR, 'run_v2.log')
    if os.path.exists(log_path):
        os.remove(log_path)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )
    logging.info("ログ初期化完了（run_v2.log）。")


# =====================================================
# データ読み込み ＆ 前処理（共通化：hPa偏差で統一）
# =====================================================
def load_and_prepare_data_unified(filepath: str,
                                  start_date: Optional[str],
                                  end_date: Optional[str],
                                  device: str = 'cpu'):
    """
    - ファイルからmsl[Pa]を読み込み
    - 必要なら期間スライス
    - NaN行を除去
    - hPaへ換算
    - サンプルごとに空間平均を差し引いた偏差[hPa]を作成
    戻り値:
      X_for_s1 (torch.Tensor, N x D, 偏差[hPa]),
      msl_hpa (np.ndarray, N x H x W),
      anomaly_hpa (np.ndarray, N x H x W),
      lat (np.ndarray), lon (np.ndarray), H(int), W(int),
      times (np.ndarray), labels (List[str] or None)
    """
    logging.info(f"データ読み込み: {filepath}")
    ds = xr.open_dataset(filepath, decode_times=True)

    # time座標名の検出
    if 'valid_time' in ds:
        time_coord = 'valid_time'
    elif 'time' in ds:
        time_coord = 'time'
    else:
        raise ValueError('No time coordinate named "valid_time" or "time".')

    # 期間指定
    if (start_date is not None) or (end_date is not None):
        sub = ds.sel({time_coord: slice(start_date, end_date)})
    else:
        sub = ds

    if 'msl' not in sub:
        raise ValueError('Variable "msl" not found in dataset.')

    msl = sub['msl'].astype('float32')

    # 次元名の標準化
    lat_name = 'latitude'
    lon_name = 'longitude'
    for dn in msl.dims:
        if 'lat' in dn.lower(): lat_name = dn
        if 'lon' in dn.lower(): lon_name = dn

    msl = msl.transpose(time_coord, lat_name, lon_name)  # (N,H,W)
    ntime = msl.sizes[time_coord]
    nlat = msl.sizes[lat_name]
    nlon = msl.sizes[lon_name]

    arr = msl.values  # (N,H,W) in Pa
    arr2 = arr.reshape(ntime, nlat*nlon)  # (N,D)

    # NaN行除外
    valid_mask = ~np.isnan(arr2).any(axis=1)
    arr2 = arr2[valid_mask]
    times = msl[time_coord].values[valid_mask]
    lat = sub[lat_name].values
    lon = sub[lon_name].values

    # ラベル（あれば）
    labels = None
    if 'label' in sub.variables:
        raw = sub['label'].values
        raw = raw[valid_mask]
        labels = [v.decode('utf-8') if isinstance(v, (bytes, bytearray)) else str(v) for v in raw]
        logging.info("ラベルを読み込みました。")

    # hPaへ換算
    msl_hpa_flat = (arr2 / 100.0).astype(np.float32)  # (N,D)
    mean_per_sample = np.nanmean(msl_hpa_flat, axis=1, keepdims=True)
    anomaly_flat = msl_hpa_flat - mean_per_sample  # (N,D)
    X_for_s1 = torch.from_numpy(anomaly_flat).to(device=device, dtype=torch.float32)

    # 3D形状に戻す
    n = anomaly_flat.shape[0]
    msl_hpa = msl_hpa_flat.reshape(n, nlat, nlon)
    anomaly_hpa = anomaly_flat.reshape(n, nlat, nlon)

    logging.info(f"期間: {str(times.min()) if len(times)>0 else '?'} 〜 {str(times.max()) if len(times)>0 else '?'}")
    logging.info(f"サンプル数={n}, 解像度={nlat}x{nlon}")
    return X_for_s1, msl_hpa, anomaly_hpa, lat, lon, nlat, nlon, times, labels


# =====================================================
# 3type_som側のユーティリティ（ログ・ラベル処理・可視化 等）
# =====================================================
class Logger:
    def __init__(self, path):
        self.path = path
        self.f = open(path, 'w', encoding='utf-8')
    def write(self, s):
        sys.stdout.write(s)
        self.f.write(s)
        self.f.flush()
    def close(self):
        self.f.close()

def normalize_label(l):
    if l is None:
        return None
    s = str(l).strip()
    s = s.replace('＋','+').replace('－','-').replace('−','-').replace('　','').replace(' ','')
    return s

def map_to_base_or_composite(label_str):
    """基本ラベルならそのまま、複合はCOMPOSITE、その他はCOMPOSITE扱い"""
    s = normalize_label(label_str)
    if not s:
        return None
    if s in BASE_LABELS:
        return s
    if ('+' in s) or ('-' in s):
        return COMPOSITE_LABEL
    return COMPOSITE_LABEL

def winners_to_clusters(winners_xy, som_shape):
    clusters = [[] for _ in range(som_shape[0]*som_shape[1])]
    for i,(ix,iy) in enumerate(winners_xy):
        k = ix*som_shape[1] + iy
        clusters[k].append(i)
    return clusters

def choose_representative_labels_train(clusters, labels_mapped, base_labels, forbid_label):
    rep = {}
    for k, idxs in enumerate(clusters):
        if len(idxs)==0:
            rep[k] = None
            continue
        from collections import Counter
        c = Counter()
        for j in idxs:
            lab = labels_mapped[j]
            if lab is not None:
                c[lab] += 1
        if len(c)==0:
            rep[k] = None
            continue
        max_lbl = None
        max_cnt = -1
        for bl in base_labels:
            if c[bl] > max_cnt:
                max_cnt = c[bl]
                max_lbl = bl
        if max_cnt > 0:
            rep[k] = max_lbl
        else:
            rep[k] = None
    return rep

def macro_recall_train(clusters, labels_mapped, rep_labels, base_labels):
    from collections import Counter
    label_indices = {bl: [] for bl in base_labels}
    for i,lab in enumerate(labels_mapped):
        if lab in base_labels:
            label_indices[lab].append(i)

    recalls = []
    for bl in base_labels:
        total = len(label_indices[bl])
        if total == 0:
            continue
        correct = 0
        for j in label_indices[bl]:
            found_k = None
            for k,idxs in enumerate(clusters):
                if j in idxs:
                    found_k = k
                    break
            if found_k is None:
                continue
            if rep_labels.get(found_k, None) == bl:
                correct += 1
        recalls.append(correct/total)
    macro = float(np.mean(recalls)) if len(recalls)>0 else 0.0
    return macro, recalls

def accuracy_on_dataset(winners_xy, labels_mapped, rep_labels, base_labels, som_shape):
    correct = 0
    total   = 0
    per_label_total = {bl:0 for bl in base_labels}
    per_label_correct = {bl:0 for bl in base_labels}

    for i,(ix,iy) in enumerate(winners_xy):
        true_lab = labels_mapped[i]
        if true_lab not in base_labels:
            continue
        node_k = ix*som_shape[1] + iy
        pred_lab = rep_labels.get(node_k, None)
        per_label_total[true_lab] += 1
        total += 1
        if (pred_lab is not None) and (pred_lab == true_lab):
            correct += 1
            per_label_correct[true_lab] += 1

    acc = correct/max(1,total)
    per_label_acc = {bl:(per_label_correct[bl]/per_label_total[bl] if per_label_total[bl]>0 else None)
                     for bl in base_labels}
    return acc, total, per_label_acc

def plot_som_node_average_patterns(data_flat, winners_xy, lat, lon, som_shape, save_path, title):
    H, W = len(lat), len(lon)
    X2 = data_flat.reshape(-1, H, W)  # 偏差[hPa]

    map_x, map_y = som_shape
    mean_patterns = np.full((map_x, map_y, H, W), np.nan, dtype=np.float32)
    counts = np.zeros((map_x, map_y), dtype=int)
    for ix in range(map_x):
        for iy in range(map_y):
            mask = (winners_xy[:,0]==ix) & (winners_xy[:,1]==iy)
            idxs = np.where(mask)[0]
            counts[ix,iy] = len(idxs)
            if len(idxs)>0:
                mean_patterns[ix,iy] = np.nanmean(X2[idxs], axis=0)

    absmax = np.nanmax(np.abs(mean_patterns))
    if not np.isfinite(absmax) or absmax==0:
        absmax = 1.0
    vmin,vmax = -absmax, absmax
    levels = np.linspace(vmin, vmax, 21)

    # 並べ方（(x,y)→画像は列優先で回転）
    nrows, ncols = som_shape[1], som_shape[0]
    figsize=(ncols*2.6, nrows*2.6)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
                             subplot_kw={'projection': ccrs.PlateCarree()})
    axes = np.atleast_2d(axes)
    axes = axes.T[::-1,:]

    last_cf=None
    for ix in range(map_x):
        for iy in range(map_y):
            ax = axes[ix,iy]
            mp = mean_patterns[ix,iy]
            if np.isnan(mp).all():
                ax.set_axis_off(); continue
            cf = ax.contourf(lon, lat, mp, levels=levels, cmap='RdBu_r', extend='both', transform=ccrs.PlateCarree())
            ax.contour(lon, lat, mp, levels=levels, colors='k', linewidth=0.3, transform=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='black', linewidth=0.4)
            ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], ccrs.PlateCarree())
            ax.text(0.02,0.96,f'({ix},{iy}) N={counts[ix,iy]}', transform=ax.transAxes,
                    fontsize=7, va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            ax.set_xticks([]); ax.set_yticks([])
            last_cf=cf
    if last_cf is not None:
        fig.subplots_adjust(right=0.88, top=0.94)
        cax = fig.add_axes([0.90, 0.12, 0.02, 0.72])
        fig.colorbar(last_cf, cax=cax, label='SLP Anomaly (hPa)')
    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0,0,0.88,0.94])
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def save_each_node_mean_image(data_flat, winners_xy, lat, lon, som_shape, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    H, W = len(lat), len(lon)
    X2 = data_flat.reshape(-1, H, W)  # 偏差[hPa]
    map_x, map_y = som_shape
    for ix in range(map_x):
        for iy in range(map_y):
            mask = (winners_xy[:,0]==ix) & (winners_xy[:,1]==iy)
            idxs = np.where(mask)[0]
            if len(idxs)>0:
                mean_img = np.nanmean(X2[idxs], axis=0)
            else:
                mean_img = np.full((H,W), np.nan, dtype=np.float32)
            vmax = np.nanmax(np.abs(mean_img))
            if not np.isfinite(vmax) or vmax == 0:
                vmax = 1.0
            vmin = -vmax

            fig = plt.figure(figsize=(4,3))
            ax = plt.axes(projection=ccrs.PlateCarree())
            cf = ax.contourf(lon, lat, mean_img, 21, cmap='RdBu_r', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.4)
            ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], ccrs.PlateCarree())
            plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label='SLP Anomaly (hPa)')
            ax.set_title(f'({ix},{iy}) N={len(idxs)}')
            ax.set_xticks([]); ax.set_yticks([])
            fpath = os.path.join(out_dir, f'{prefix}_node_{ix}_{iy}.png')
            plt.tight_layout()
            plt.savefig(fpath, dpi=180)
            plt.close(fig)

def plot_label_distributions(winners_xy, labels_mapped, label_list, som_shape, save_dir, title_prefix):
    os.makedirs(save_dir, exist_ok=True)
    node_counts = {lbl: np.zeros((som_shape[0], som_shape[1]), dtype=int) for lbl in label_list}
    for i,(ix,iy) in enumerate(winners_xy):
        lab = labels_mapped[i]
        if lab in node_counts:
            node_counts[lab][ix,iy] += 1
    cols = 5
    rows = int(np.ceil(len(label_list)/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.6, rows*2.6))
    axes = np.atleast_2d(axes)
    for idx,lbl in enumerate(label_list):
        r = idx//cols; c=idx%cols
        ax = axes[r,c]
        im = ax.imshow(node_counts[lbl].T[::-1,:], cmap='viridis')
        ax.set_title(lbl); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for k in range(len(label_list), rows*cols):
        r = k//cols; c=k%cols
        axes[r,c].axis('off')
    plt.suptitle(f'{title_prefix} Label Distributions on SOM nodes', fontsize=14)
    plt.tight_layout(rect=[0,0,1,0.95])
    fpath = os.path.join(save_dir, f'{title_prefix}_label_distributions.png')
    plt.savefig(fpath, dpi=250)
    plt.close(fig)


# =====================================================
# 3種類のbatchSOM（全期間・1方式分）
# =====================================================
def run_one_method(method_name, activation_distance, data_all, labels_all, times_all,
                   field_shape, lat, lon, out_dir):
    """
    method_name: 'euclidean' | 'ssim' | 's1'
    activation_distance: 同上
    data_all は「空間平均を引いた偏差[hPa]」（N,D）
    """
    os.makedirs(out_dir, exist_ok=True)
    log = Logger(os.path.join(out_dir, f'{method_name}_results.log'))
    log.write(f'=== {method_name} SOM (Full period) ===\n')
    log.write(f'Device CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}\n')
    log.write(f'SOM size: {SOM_X} x {SOM_Y}, iter={NUM_ITER}, batch={BATCH_SIZE}, nodes_chunk={NODES_CHUNK}\n')
    log.write(f'All samples: {data_all.shape[0]}\n')
    log.write('Input representation: SLP anomaly [hPa], spatial-mean removed per sample\n')
    if len(times_all) > 0:
        tmin = pd.to_datetime(times_all.min()).strftime('%Y-%m-%d')
        tmax = pd.to_datetime(times_all.max()).strftime('%Y-%m-%d')
        log.write(f'Period: {tmin} to {tmax}\n')

    # SOM構築（3距離対応版）
    som = MultiDistMiniSom(
        x=SOM_X, y=SOM_Y, input_len=data_all.shape[1],
        sigma=2.5, learning_rate=0.5,
        neighborhood_function='gaussian',
        topology='rectangular',
        activation_distance=activation_distance,             # 'euclidean'/'ssim'/'s1'
        random_seed=SEED,
        sigma_decay='asymptotic_decay',
        s1_field_shape=field_shape,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        dtype=torch.float32,
        nodes_chunk=NODES_CHUNK
    )
    som.random_weights_init(data_all)

    # 学習
    qhist = som.train_batch(
        data_all, num_iteration=NUM_ITER,
        batch_size=BATCH_SIZE, verbose=True, log_interval=LOG_INTERVAL,
        update_per_iteration=False, shuffle=True
    )
    qe_all = som.quantization_error(data_all, sample_limit=EVAL_SAMPLE_LIMIT, batch_size=max(32,BATCH_SIZE))
    log.write(f'Final quantization_error (mean distance) on ALL: {qe_all:.6f}\n')

    # BMU（全期間）
    winners_all = som.predict(data_all, batch_size=max(64,BATCH_SIZE))

    # 割当CSV
    assign_csv_all = os.path.join(out_dir, f'{method_name}_assign_all.csv')
    pd.DataFrame({
        'time': times_all,
        'bmu_x': winners_all[:,0], 'bmu_y': winners_all[:,1],
        'label_raw': labels_all if labels_all is not None else ['']*len(winners_all)
    }).to_csv(assign_csv_all, index=False, encoding='utf-8-sig')
    log.write(f'Assigned BMU (all) -> {assign_csv_all}\n')

    # ノード平均パターン（偏差[hPa]）
    bigmap_all = os.path.join(out_dir, f'{method_name}_som_node_avg_all.png')
    plot_som_node_average_patterns(
        data_all, winners_all, lat, lon, (SOM_X,SOM_Y),
        save_path=bigmap_all,
        title=f'{method_name.upper()} SOM Node Avg SLP Anomaly (All)'
    )
    log.write(f'Node average patterns (all) -> {bigmap_all}\n')

    # 各ノード平均画像（個別）
    pernode_dir_all = os.path.join(out_dir, f'{method_name}_pernode_all')
    save_each_node_mean_image(data_all, winners_all, lat, lon, (SOM_X,SOM_Y),
                              out_dir=pernode_dir_all, prefix='all')
    log.write(f'Per-node mean images (all) -> {pernode_dir_all}\n')

    # 評価（ラベルがあれば）
    if labels_all is not None:
        labels_all_mapped = [map_to_base_or_composite(l) for l in labels_all]
        clusters_all = winners_to_clusters(winners_all, (SOM_X,SOM_Y))
        rep_labels = choose_representative_labels_train(
            clusters=clusters_all, labels_mapped=labels_all_mapped,
            base_labels=BASE_LABELS, forbid_label=COMPOSITE_LABEL
        )
        rep_json = os.path.join(out_dir, f'{method_name}_representative_labels.json')
        with open(rep_json, 'w', encoding='utf-8') as f:
            json.dump({str(k): rep_labels[k] for k in sorted(rep_labels.keys())}, f, ensure_ascii=False, indent=2)
        log.write(f'Representative labels per node (saved) -> {rep_json}\n')

        macro, per_label_recalls = macro_recall_train(clusters_all, labels_all_mapped, rep_labels, BASE_LABELS)
        log.write('\n[All period] Macro Recall (base only): {:.4f}\n'.format(macro))
        for bl,rc in zip(BASE_LABELS, per_label_recalls):
            log.write(f'  Recall[{bl}]: {rc:.4f}\n')

        acc, total, per_label_acc = accuracy_on_dataset(
            winners_all, labels_all_mapped, rep_labels, BASE_LABELS, (SOM_X,SOM_Y)
        )
        log.write('\n[All period] Accuracy (base only): {:.4f}  [N={}]\n'.format(acc, total))
        for bl in BASE_LABELS:
            v = per_label_acc[bl]
            if v is None:
                log.write(f'  Acc[{bl}]: None (no sample)\n')
            else:
                log.write(f'  Acc[{bl}]: {v:.4f}\n')

        # 混同行列（基本ラベル vs 代表基本ラベル）
        conf_cols = BASE_LABELS + ['None']
        conf_mat = pd.DataFrame(0, index=BASE_LABELS, columns=conf_cols, dtype=int)
        for i,(ix,iy) in enumerate(winners_all):
            t = labels_all_mapped[i]
            if t not in BASE_LABELS:
                continue
            k = ix*SOM_Y + iy
            p = rep_labels.get(k, None)
            p_col = p if p in BASE_LABELS else 'None'
            conf_mat.loc[t, p_col] += 1
        conf_csv = os.path.join(out_dir, f'{method_name}_confusion_matrix_all.csv')
        conf_mat.to_csv(conf_csv, encoding='utf-8-sig')
        log.write(f'Confusion matrix (base vs predicted) -> {conf_csv}\n')

        # ラベル分布ヒートマップ
        dist_dir_all = os.path.join(out_dir, f'{method_name}_label_dist_all')
        plot_label_distributions(winners_all, labels_all_mapped, BASE_LABELS+[COMPOSITE_LABEL],
                                 (SOM_X,SOM_Y), dist_dir_all, title_prefix='All')
        log.write(f'Label-distribution heatmaps (all) -> {dist_dir_all}\n')
    else:
        log.write('Labels not found; skip evaluation.\n')

    log.write('\n=== Done (full period) ===\n')
    log.close()


# =====================================================
# クラスタリング側の補助（代表ラベル多数決）
# =====================================================
def compute_cluster_majorities(clusters: List[List[int]],
                               labels: List[Optional[str]],
                               base_labels: List[str]) -> Dict[int, Optional[str]]:
    rep_map: Dict[int, Optional[str]] = {}
    for i, idxs in enumerate(clusters):
        c = Counter()
        for j in idxs:
            bl = basic_label_or_none(labels[j], base_labels)
            if bl is not None:
                c[bl] += 1
        rep = c.most_common(1)[0][0] if c else None
        rep_map[i] = rep
    return rep_map


# =====================================================
# メイン
# =====================================================
def main():
    set_reproducibility(SEED)
    setup_logging_v2()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"使用デバイス: {device.upper()}")
    logging.info(f"SEED={SEED}, TH_MERGE={TH_MERGE}")
    logging.info(f"CLUS_ROW_BATCH={CLUS_ROW_BATCH}, CLUS_COL_CHUNK={CLUS_COL_CHUNK}, SOM_BATCH_SIZE={SOM_BATCH_SIZE}")
    logging.info(f"SOM(3type): size={SOM_X}x{SOM_Y}, iters={NUM_ITER}, batch={BATCH_SIZE}, nodes_chunk={NODES_CHUNK}")

    # 共通の前処理（hPa偏差）
    X_for_s1, X_original_hpa, X_anomaly_hpa, lat, lon, d_lat, d_lon, ts, labels = load_and_prepare_data_unified(
        DATA_FILE, TIME_START, TIME_END, device
    )
    # 3type_SOM用の2次元表現（numpy）
    data_all = X_anomaly_hpa.reshape(X_anomaly_hpa.shape[0], -1).astype(np.float32)
    field_shape = (d_lat, d_lon)

    # ============== まず 3種類のbatchSOM（全期間）を実行 ==============
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    methods = [
        ('euclidean', 'euclidean'),
        ('ssim',      'ssim'),
        ('s1',        's1')
    ]
    for mname, adist in methods:
        out_dir = os.path.join(OUTPUT_ROOT, f'{mname}_som')
        run_one_method(
            method_name=mname, activation_distance=adist,
            data_all=data_all, labels_all=labels, times_all=ts,
            field_shape=field_shape, lat=lat, lon=lon, out_dir=out_dir
        )

    # ============== S1スコア分布（偏差[hPa]） ==============
    s1_hist_path = os.path.join(RESULT_DIR, 's1_score_distribution_anomaly_hpa.png')
    plot_s1_distribution_histogram(
        X_for_s1, d_lat, d_lon, s1_hist_path,
        row_batch_size=S1_HIST_ROW_BATCH,
        col_chunk_size=S1_HIST_COL_CHUNK,
        max_samples=S1_DISTRIBUTION_MAX_SAMPLES
    )
    logging.info(f"S1分布ヒストグラム保存: {s1_hist_path}")
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    # ============== 二段階クラスタリング（S1, 偏差[hPa]） ==============
    history_only_base: Dict[str, List[float]] = {
        'iteration': [],
        'num_clusters': [],
        'MacroRecall_majority': [],
        # MicroAccuracy / ARI / NMI は記録しない（削除指定）
        'MedoidMajorityMatchRate': []
    }

    def eval_callback(iter_idx: int, clusters: List[List[int]], medoids: List[Optional[int]]):
        # 戻り値に他の指標が含まれても、ここでは記録しない（削除指定に従う）
        m = evaluate_clusters_only_base(
            clusters=clusters,
            all_labels=labels,
            base_labels=BASE_LABELS,
            title=f"反復 {iter_idx} 評価（基本ラベル）",
            medoids=medoids
        )
        if m is not None:
            history_only_base['iteration'].append(iter_idx)
            history_only_base['num_clusters'].append(len(clusters))
            history_only_base['MacroRecall_majority'].append(m.get('MacroRecall_majority', np.nan))
            history_only_base['MedoidMajorityMatchRate'].append(m.get('MedoidMajorityMatchRate', np.nan))

    logging.info("二段階クラスタリング開始...")
    t0 = time.time()
    clusters, medoids = two_stage_clustering(
        X_for_s1, TH_MERGE, labels, ts, BASE_LABELS,
        row_batch_size=CLUS_ROW_BATCH, col_chunk_size=CLUS_COL_CHUNK,
        d_lat=d_lat, d_lon=d_lon, eval_callback=eval_callback
    )
    elapsed = time.time() - t0
    logging.info("二段階クラスタリング完了")
    logging.info(f"総計算時間: {elapsed:.2f} 秒 ({elapsed/60:.2f} 分)")
    logging.info(f"最終クラスタ数: {len(clusters)}")

    # 最終状態の分析・評価
    analyze_cluster_distribution(clusters, labels, ts, BASE_LABELS, title="最終クラスタ分布分析（基本ラベル）")
    _ = evaluate_clusters_only_base(clusters, labels, BASE_LABELS, title="最終評価（基本ラベル）", medoids=medoids)

    # 結果保存
    results_path = os.path.join(RESULT_DIR, f'clustering_result_th{TH_MERGE}_v1.pt')
    torch.save({'clusters': clusters, 'medoids': medoids}, results_path)
    logging.info(f"クラスタリング結果保存: {results_path}")

    # 評価推移の保存（Micro/ARI/NMIは含めない）
    hist_csv = os.path.join(ONLY_BASE_DIR, 'iteration_metrics_only_base.csv')
    save_metrics_history_to_csv(history_only_base, hist_csv)
    hist_plot = os.path.join(ONLY_BASE_DIR, 'iteration_vs_metrics_only_base.png')
    plot_iteration_metrics(history_only_base, hist_plot)

    # 分布サマリ・可視化
    dist_img = os.path.join(ONLY_BASE_DIR, 'final_distribution_summary_only_base.png')
    plot_final_distribution_summary(clusters, labels, ts, BASE_LABELS, dist_img)
    logging.info(f"分布サマリ保存: {dist_img}")

    # メドイド・クラスタの空間偏差図
    final_clusters_img = os.path.join(ONLY_BASE_DIR, f'final_clusters_anomaly_th{TH_MERGE}_only_base.png')
    plot_final_clusters_medoids(medoids, clusters, X_anomaly_hpa, lat, lon, ts, labels, BASE_LABELS, final_clusters_img)
    logging.info(f"メドイド図保存: {final_clusters_img}")

    # 日次マップ出力
    daily_dir = os.path.join(ONLY_BASE_DIR, 'daily_anomaly_maps_only_base')
    save_daily_maps_per_cluster(clusters, X_anomaly_hpa, lat, lon, ts, labels, BASE_LABELS, daily_dir, per_cluster_limit=DAILY_MAPS_PER_CLUSTER_LIMIT)
    logging.info(f"日次マップ保存先: {daily_dir}")

    # クラスタ情報をCSVで保存
    info_df = summarize_cluster_info(clusters, medoids, labels, BASE_LABELS, ts)
    info_csv = os.path.join(RESULT_DIR, 'cluster_summary.csv')
    info_df.to_csv(info_csv, index=False)
    logging.info(f"クラスタサマリCSV: {info_csv}")

    # ============== S1-batchSOM（メドイドのみで可視化：s1_minisom使用） ==============
    medoid_indices = [m for m in medoids if m is not None]
    if len(medoid_indices) == 0:
        logging.warning("SOM学習のための有効メドイドがありません。SOMをスキップします。")
        return

    # 学習データ: メドイドの空間偏差[hPa]
    medoid_data = X_anomaly_hpa[medoid_indices].reshape(len(medoid_indices), d_lat * d_lon).astype(np.float32)

    # SOMサイズ自動選択（クラス数を覆う最小正方格子）
    som_x, som_y = grid_auto_size(len(medoid_indices))
    som_nodes = som_x * som_y
    logging.info(f"SOMサイズ自動選択: {som_x} x {som_y}（ノード数={som_nodes}、クラスタ={len(medoid_indices)}）")

    sigma0 = max(som_x, som_y) / 2.0
    nodes_chunk = max(8, min(64, SOM_BATCH_SIZE * 2))

    som = S1MiniSom(som_x, som_y, d_lat * d_lon, sigma=sigma0,
                    s1_field_shape=(d_lat, d_lon),
                    device=device, random_seed=SEED, nodes_chunk=nodes_chunk)

    som.random_weights_init(medoid_data)

    som_iters = min(SOM_MAX_ITERS_CAP, max(300, 20 * som_nodes))
    som_log_interval = max(10, som_iters // 10)
    logging.info(f"SOM学習: iters={som_iters}, batch={SOM_BATCH_SIZE}, sigma0={sigma0:.2f}, nodes_chunk={nodes_chunk}")
    qhist = som.train_batch(
        medoid_data,
        num_iteration=som_iters,
        batch_size=SOM_BATCH_SIZE,
        verbose=True,
        log_interval=som_log_interval,
        update_per_iteration=True,
        shuffle=True
    )

    # 量子化誤差履歴をプロット
    qe_plot = os.path.join(SOM_DIR, f'som_qe_history_{som_x}x{som_y}.png')
    try:
        it_ticks = np.linspace(0, som_iters - 1, num=len(qhist))
        plt.figure(figsize=(8, 4))
        plt.plot(it_ticks, qhist, marker='o', color='tab:blue')
        plt.title(f"SOM Quantization Error (size={som_x}x{som_y})")
        plt.xlabel("Iteration")
        plt.ylabel("Quantization Error (S1)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(qe_plot)
        plt.close()
        logging.info(f"SOM量子化誤差履歴を保存: {qe_plot}")
    except Exception as e:
        logging.warning(f"SOM量子化誤差履歴の保存に失敗: {e}")

    # メドイドBMU（重複なし割当）
    bmu_xy = som.predict_unique(medoid_data, batch_size=SOM_BATCH_SIZE)  # (M, 2)

    # medoid -> cluster_idのマッピング
    medoid_to_cluster = []
    for ci, m in enumerate(medoids):
        if m is not None:
            medoid_to_cluster.append((ci, m))
    cluster_ids_in_order = []
    for mi in medoid_indices:
        for ci, m in medoid_to_cluster:
            if m == mi:
                cluster_ids_in_order.append(ci)
                break

    # BMU散布図
    som_scatter = os.path.join(SOM_DIR, f'som_bmu_scatter_{som_x}x{som_y}.png')
    plt.figure(figsize=(max(6, som_y), max(6, som_x)))
    plt.title(f"SOM BMU Scatter (medoids)  size={som_x}x{som_y}")
    plt.scatter(bmu_xy[:, 1], bmu_xy[:, 0], c='tab:blue', s=60, alpha=0.8, edgecolors='k')
    for k, (x, y) in enumerate(bmu_xy):
        cid = cluster_ids_in_order[k] + 1
        plt.text(y + 0.1, x + 0.1, f"C{cid}", fontsize=8, color='black')
    plt.gca().invert_yaxis()
    plt.xticks(range(som_y))
    plt.yticks(range(som_x))
    plt.grid(True, alpha=0.3)
    plt.xlabel('y')
    plt.ylabel('x')
    plt.tight_layout()
    plt.savefig(som_scatter)
    plt.close()
    logging.info(f"SOM BMU散布図保存: {som_scatter}")

    # BMU使用頻度ヒートマップ
    usage = np.zeros((som_x, som_y), dtype=np.int32)
    for (x, y) in bmu_xy:
        usage[int(x), int(y)] += 1
    som_usage_img = os.path.join(SOM_DIR, f'som_bmu_usage_{som_x}x{som_y}.png')
    plt.figure(figsize=(max(6, som_y), max(6, som_x)))
    plt.title(f"SOM BMU Usage Heatmap (medoids)  size={som_x}x{som_y}")
    plt.imshow(usage, cmap='viridis', origin='upper')
    plt.colorbar(label='Assigned Medoids')
    plt.xticks(range(som_y))
    plt.yticks(range(som_x))
    plt.xlabel('y')
    plt.ylabel('x')
    plt.tight_layout()
    plt.savefig(som_usage_img)
    plt.close()
    logging.info(f"SOM BMU使用頻度ヒートマップ保存: {som_usage_img}")

    # クラスタ代表（多数決）
    rep_map = compute_cluster_majorities(clusters, labels, BASE_LABELS)  # {cluster_idx: rep or None}

    # ノードごとの割当メドイド情報（注釈・CSV）
    node_to_items: Dict[Tuple[int, int], List[Dict[str, str]]] = {}
    for k, (x, y) in enumerate(bmu_xy):
        x = int(x); y = int(y)
        cid0 = cluster_ids_in_order[k]
        cid = cid0 + 1
        mi = medoid_indices[k]
        date_str = pd.to_datetime(str(ts[mi])).strftime('%Y-%m-%d')
        raw_lbl = labels[mi] if labels else None
        base_lbl = primary_base_label(raw_lbl, BASE_LABELS)
        rep_lbl = rep_map.get(cid0, None)
        match = (rep_lbl == base_lbl) if (rep_lbl is not None and base_lbl is not None) else None
        node_to_items.setdefault((x, y), []).append({
            'cluster_id': f"C{cid}",
            'cluster_id0': cid0,
            'medoid_index': str(mi),
            'date': date_str,
            'label_raw': str(raw_lbl),
            'label_base': str(base_lbl) if base_lbl is not None else "-",
            'rep_label': str(rep_lbl) if rep_lbl is not None else "-",
            'match': '-' if match is None else str(bool(match))
        })

    # 割当CSV
    csv_rows = []
    for (x, y), items in node_to_items.items():
        for it in items:
            csv_rows.append({
                'node_x': x, 'node_y': y, 'node_flat': x * som_y + y,
                'cluster': it['cluster_id'],
                'medoid_index': it['medoid_index'],
                'date': it['date'],
                'label_base': it['label_base'],
                'label_raw': it['label_raw'],
                'cluster_majority_label': it['rep_label'],
                'majority_medoid_match': it['match']
            })
    assign_csv = os.path.join(SOM_DIR, f'som_node_assignments_{som_x}x{som_y}.csv')
    pd.DataFrame(csv_rows).to_csv(assign_csv, index=False)
    logging.info(f"SOMノード割当CSV保存: {assign_csv}")

    # 詳細ログ出力
    logging.info("\n--- SOM配置詳細ログ ---")
    logging.info(f"SOMグリッド: {som_x} x {som_y}（ノード数={som_nodes}）")
    for ix in range(som_x):
        for iy in range(som_y):
            items = node_to_items.get((ix, iy), [])
            if len(items) == 0:
                continue
            labels_here = [it['rep_label'] for it in items if it['rep_label'] != '-']
            rep_count = Counter(labels_here)
            rep_str = ", ".join([f"{k}:{v}" for k, v in rep_count.most_common()])
            clist = ", ".join([f"{it['cluster_id']}({it['rep_label']})" for it in items])
            logging.info(f"ノード({ix},{iy}) n={len(items)} | 代表ラベル内訳: [{rep_str}] | {clist}")

    logging.info("\n[SOMノード純度]")
    for (ix, iy), items in sorted(node_to_items.items()):
        labels_here = [it['rep_label'] for it in items if it['rep_label'] != '-']
        purity = 0.0
        if labels_here:
            c = Counter(labels_here)
            purity = c.most_common(1)[0][1] / len(items)
        logging.info(f"  ノード({ix},{iy}) n={len(items)} 純度={purity:.2f}")

    logging.info("\n[代表ラベルごとのBMUノード分布]")
    rep_to_nodes: Dict[str, Dict[Tuple[int, int], List[str]]] = {}
    for (ix, iy), items in node_to_items.items():
        for it in items:
            rep = it['rep_label']
            if rep == '-':
                continue
            rep_to_nodes.setdefault(rep, {}).setdefault((ix, iy), []).append(it['cluster_id'])
    for rep, nd in rep_to_nodes.items():
        num_nodes = len(nd)
        num_clusters_rep = sum(len(set(v)) for v in nd.values())
        node_desc = "; ".join([f"({ix},{iy}):{sorted(list(set(v)))}" for (ix, iy), v in sorted(nd.items())])
        logging.info(f"  代表={rep}: ノード数={num_nodes}, クラスタ数={num_clusters_rep} | {node_desc}")

    logging.info("\n[各クラスタのBMUノード（メドイド基準）]")
    cid_to_node: Dict[int, Tuple[int, int]] = {}
    for (ix, iy), items in node_to_items.items():
        for it in items:
            cid0 = it['cluster_id0']
            cid_to_node[cid0] = (ix, iy)
    for cid0 in range(len(clusters)):
        node = cid_to_node.get(cid0, None)
        rep_lbl = rep_map.get(cid0, None)
        if node is None:
            logging.info(f"  Cluster_{cid0+1}: ノード未割当（メドイドなし）")
        else:
            (ix, iy) = node
            logging.info(f"  Cluster_{cid0+1}: BMU=({ix},{iy}) 代表={rep_lbl}")

    logging.info("--- SOM配置詳細ログ 終了 ---\n")

    # SOMコードブック（地理図）タイル表示（割当なしノードは空白）
    som_codebook_img = os.path.join(SOM_DIR, f'som_codebook_maps_{som_x}x{som_y}.png')
    weights_grid = som.get_weights()  # (x, y, D)
    fig, axes = plt.subplots(som_x, som_y, figsize=(som_y * 3, som_x * 3), subplot_kw={"projection": ccrs.PlateCarree()})
    axes = np.atleast_2d(axes)
    cmap = plt.get_cmap('RdBu_r')
    vmin, vmax = -40, 40
    norm = Normalize(vmin=vmin, vmax=vmax)
    levels = np.linspace(vmin, vmax, 21)
    last_cont = None
    for ix in range(som_x):
        for iy in range(som_y):
            ax = axes[ix, iy]
            items = node_to_items.get((ix, iy), [])
            if len(items) == 0:
                ax.axis("off")
                continue
            pat = weights_grid[ix, iy].reshape(d_lat, d_lon)
            last_cont = ax.contourf(lon, lat, pat, levels=levels, cmap=cmap, norm=norm, extend='both', transform=ccrs.PlateCarree())
            ax.contour(lon, lat, pat, colors='k', linewidth=0.3, levels=levels, transform=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="black", linewidth=0.4)
            ax.set_extent([120, 150, 20, 50], crs=ccrs.PlateCarree())
            ax.set_title(f"({ix},{iy})", fontsize=8)
    fig.tight_layout(rect=[0, 0, 0.9, 0.95])
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    if last_cont is not None:
        fig.colorbar(last_cont, cax=cbar_ax, label="Sea Level Pressure Anomaly (hPa)")
    fig.suptitle(f"SOM Codebook Maps (size={som_x}x{som_y})", fontsize=14)
    plt.savefig(som_codebook_img)
    plt.close()
    logging.info(f"SOMコードブック（地理図）保存: {som_codebook_img}")

    # 注釈付きコードブック図（割当なしは注釈のみ）
    som_codebook_annotated_img = os.path.join(SOM_DIR, f'som_codebook_maps_{som_x}x{som_y}_annotated.png')
    fig, axes = plt.subplots(som_x, som_y, figsize=(som_y * 3.2, som_x * 3.2), subplot_kw={"projection": ccrs.PlateCarree()})
    axes = np.atleast_2d(axes)
    last_cont = None
    for ix in range(som_x):
        for iy in range(som_y):
            ax = axes[ix, iy]
            items = node_to_items.get((ix, iy), [])
            if len(items) == 0:
                ax.set_extent([120, 150, 20, 50], crs=ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="black", linewidth=0.3)
                txt = f"({ix},{iy}) | n=0\nNo medoids"
                ax.text(0.02, 0.02, txt, transform=ax.transAxes, ha='left', va='bottom',
                        fontsize=6, color='black',
                        bbox=dict(facecolor='white', edgecolor='black', alpha=0.7, boxstyle='round,pad=0.2'))
                continue

            pat = weights_grid[ix, iy].reshape(d_lat, d_lon)
            last_cont = ax.contourf(lon, lat, pat, levels=levels, cmap=cmap, norm=norm, extend='both', transform=ccrs.PlateCarree())
            ax.contour(lon, lat, pat, colors='k', linewidth=0.3, levels=levels, transform=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="black", linewidth=0.4)
            ax.set_extent([120, 150, 20, 50], crs=ccrs.PlateCarree())
            lines = [f"({ix},{iy}) | n={len(items)}"]
            for it in items:
                lines.append(f"{it['cluster_id']} | rep={it['rep_label']} | med={it['date']} | raw={it['label_raw']} | base={it['label_base']}")
            txt = "\n".join(lines)
            ax.text(0.02, 0.02, txt, transform=ax.transAxes, ha='left', va='bottom',
                    fontsize=6, color='black',
                    bbox=dict(facecolor='white', edgecolor='black', alpha=0.7, boxstyle='round,pad=0.2'))
    fig.tight_layout(rect=[0, 0, 0.9, 0.95])
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    if last_cont is not None:
        fig.colorbar(last_cont, cax=cbar_ax, label="Sea Level Pressure Anomaly (hPa)")
    fig.suptitle(f"SOM Codebook Maps Annotated (size={som_x}x{som_y})", fontsize=14)
    plt.savefig(som_codebook_annotated_img)
    plt.close()
    logging.info(f"SOMコードブック（注釈付き）保存: {som_codebook_annotated_img}")

    logging.info("全処理完了。")


if __name__ == '__main__':
    main()