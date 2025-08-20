# main_v3.py
# -*- coding: utf-8 -*-
import os
import sys
import json
import logging
import time
from typing import Optional, List, Dict, Tuple
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import xarray as xr
import torch

import matplotlib
matplotlib.use('Agg')  # サーバ上でも保存できるように
import matplotlib.pyplot as plt

# cartopy: 可視化に使用
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize

# 3type版 SOM（Euclidean/SSIM/S1対応, batchSOM）
from minisom import MiniSom as MultiDistMiniSom

# =====================================================
# ユーザ調整パラメータ
# =====================================================
SEED = 1

# SOM学習・推論（全期間版：3方式）
SOM_X, SOM_Y = 10, 10
NUM_ITER = 10000
BATCH_SIZE = 512
NODES_CHUNK = 16
LOG_INTERVAL = 10
EVAL_SAMPLE_LIMIT = 2048
SOM_EVAL_SEGMENTS = 10  # NUM_ITER をこの個数の区間に分割して評価

# データ
DATA_FILE = './prmsl_era5_all_data_seasonal_large.nc'
TIME_START = '1991-01-01'
TIME_END   = '2000-12-31'

# 出力先（v3）
RESULT_DIR = './results_v3'
OUTPUT_ROOT = os.path.join(RESULT_DIR, 'outputs_som_fullperiod_v3')

# 基本ラベル（15）
BASE_LABELS = [
    '1', '2A', '2B', '2C', '2D', '3A', '3B', '3C', '3D', '4A', '4B', '5', '6A', '6B', '6C'
]

# SSIMの定数（ミニソムに合わせる）
SSIM_C1 = 1e-8
SSIM_C2 = 1e-8

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


def setup_logging_v3():
    for d in [RESULT_DIR, OUTPUT_ROOT]:
        os.makedirs(d, exist_ok=True)
    log_path = os.path.join(RESULT_DIR, 'run_v3.log')
    if os.path.exists(log_path):
        os.remove(log_path)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )
    logging.info("ログ初期化完了（run_v3.log）。")


# =====================================================
# データ読み込み ＆ 前処理（hPa偏差、空間平均差し引き）
# =====================================================
def load_and_prepare_data_unified(filepath: str,
                                  start_date: Optional[str],
                                  end_date: Optional[str],
                                  device: str = 'cpu'):
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

    # hPaへ換算 → 空間平均差し引き
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
# 評価ユーティリティ（s1_clustering.pyから必要部分を移植）
# =====================================================
def _normalize_to_base_candidate(label_str: Optional[str]) -> Optional[str]:
    import unicodedata, re
    if label_str is None:
        return None
    s = str(label_str)
    s = unicodedata.normalize('NFKC', s)
    s = s.upper().strip()
    s = s.replace('＋', '+').replace('－', '-').replace('−', '-')
    s = re.sub(r'[^0-9A-Z]', '', s)
    return s if s != '' else None


def basic_label_or_none(label_str: Optional[str], base_labels: List[str]) -> Optional[str]:
    import re
    cand = _normalize_to_base_candidate(label_str)
    if cand is None:
        return None
    # 完全一致を優先
    if cand in base_labels:
        return cand
    # '2A+' → '2A' のようなパターンを許容（残りに英数字がなければOK）
    for bl in base_labels:
        if cand == bl:
            return bl
        if cand.startswith(bl):
            rest = cand[len(bl):]
            if re.search(r'[0-9A-Z]', rest) is None:
                return bl
    return None


def extract_base_components(raw_label: Optional[str], base_labels: List[str]) -> List[str]:
    import unicodedata, re
    if raw_label is None:
        return []
    s = unicodedata.normalize('NFKC', str(raw_label)).upper().strip()
    s = s.replace('＋', '+').replace('－', '-').replace('−', '-')
    tokens = re.split(r'[^0-9A-Z]+', s)
    comps: List[str] = []
    for t in tokens:
        if t in base_labels and t not in comps:
            comps.append(t)
    return comps


def primary_base_label(raw_label: Optional[str], base_labels: List[str]) -> Optional[str]:
    parts = extract_base_components(raw_label, base_labels)
    return parts[0] if parts else None


def build_confusion_matrix_only_base(clusters: List[List[int]],
                                     all_labels: List[Optional[str]],
                                     base_labels: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    num_clusters = len(clusters)
    cluster_names = [f'Cluster_{i+1}' for i in range(num_clusters)]
    cm = pd.DataFrame(0, index=base_labels, columns=cluster_names, dtype=int)
    for i, idxs in enumerate(clusters):
        col = cluster_names[i]
        cnt = Counter()
        for j in idxs:
            lbl = basic_label_or_none(all_labels[j], base_labels)
            if lbl is not None:
                cnt[lbl] += 1
        for lbl, k in cnt.items():
            cm.loc[lbl, col] = k
    return cm, cluster_names


def evaluate_clusters_only_base(clusters: List[List[int]],
                                all_labels: List[Optional[str]],
                                base_labels: List[str],
                                title: str = "評価（基本ラベルのみ）") -> Optional[Dict[str, float]]:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    logging.info(f"\n--- {title} ---")
    if not all_labels:
        logging.warning("ラベル無しのため評価をスキップします。")
        return None

    cm, cluster_names = build_confusion_matrix_only_base(clusters, all_labels, base_labels)
    present_labels = [l for l in base_labels if cm.loc[l].sum() > 0]
    if len(present_labels) == 0:
        logging.warning("基本ラベルに該当するサンプルがありません。評価をスキップします。")
        return None

    logging.info("【混同行列（基本ラベルのみ）】\n" + "\n" + cm.loc[present_labels, :].to_string())

    # 各クラスタの多数決（代表ラベル）
    cluster_majority: Dict[int, Optional[str]] = {}
    logging.info("\n【各クラスタの多数決（代表ラベル）】")
    total_count = int(cm.values.sum())
    micro_correct_sum = 0
    for k in range(len(cluster_names)):
        col = cluster_names[k]
        col_counts = cm[col]
        col_sum = int(col_counts.sum())
        if col_sum == 0:
            cluster_majority[k] = None
            logging.info(f" - {col:<12}: 代表ラベル=None（基本ラベル出現なし）")
            continue
        top_label = col_counts.idxmax()
        top_count = int(col_counts.max())
        micro_correct_sum += top_count
        share = top_count / col_sum if col_sum > 0 else 0.0
        top3 = col_counts.sort_values(ascending=False)[:3]
        top3_str = ", ".join([f"{lbl}:{int(cnt)}" for lbl, cnt in top3.items()])
        logging.info(f" - {col:<12}: 代表={top_label:<3} 件数={top_count:4d} シェア={share:5.2f} | 上位: {top3_str}")
        cluster_majority[k] = top_label

    # Macro Recall (基本ラベル)
    logging.info("\n【各ラベルの再現率（代表クラスタ群ベース）】")
    per_label = {}
    for lbl in present_labels:
        row_sum = int(cm.loc[lbl, :].sum())
        cols_for_lbl = [cluster_names[k] for k in range(len(cluster_names)) if cluster_majority.get(k, None) == lbl]
        correct = int(cm.loc[lbl, cols_for_lbl].sum()) if cols_for_lbl else 0
        recall = correct / row_sum if row_sum > 0 else 0.0
        per_label[lbl] = {'N': row_sum, 'Correct': correct, 'Recall': recall}
        logging.info(f" - {lbl:<3}: N={row_sum:4d} Correct={correct:4d} Recall={recall:.4f} 代表={cols_for_lbl if cols_for_lbl else 'なし'}")
    macro_recall = float(np.mean([per_label[l]['Recall'] for l in present_labels]))
    
    # Micro Accuracy
    micro_accuracy = micro_correct_sum / total_count if total_count > 0 else 0.0

    # ARI/NMI
    n_samples = len(all_labels)
    sample_to_cluster = [-1] * n_samples
    for ci, idxs in enumerate(clusters):
        for j in idxs:
            sample_to_cluster[j] = ci

    y_true, y_pred = [], []
    for j in range(n_samples):
        lbl = basic_label_or_none(all_labels[j], base_labels)
        if lbl is None:
            continue
        ci = sample_to_cluster[j]
        if ci < 0:
            continue
        rep = cluster_majority.get(ci, None)
        if rep is None:
            continue
        y_true.append(lbl)
        y_pred.append(rep)

    metrics: Dict[str, float] = {
        'MacroRecall_majority': macro_recall,
        'MicroAccuracy_majority': micro_accuracy
    }
    if len(y_true) > 1:
        uniq_true = {l: i for i, l in enumerate(sorted(set(y_true)))}
        uniq_pred = {l: i for i, l in enumerate(sorted(set(y_pred)))}
        y_true_idx = [uniq_true[l] for l in y_true]
        y_pred_idx = [uniq_pred[l] for l in y_pred]
        ari = adjusted_rand_score(y_true_idx, y_pred_idx)
        nmi = normalized_mutual_info_score(y_true_idx, y_pred_idx)
        metrics['ARI_majority'] = float(ari)
        metrics['NMI_majority'] = float(nmi)

    # 複合ラベル考慮（基本+応用）
    logging.info("\n【複合ラベル考慮の再現率（基本+応用）】")
    composite_totals = Counter()
    for j, raw_label in enumerate(all_labels):
        components = extract_base_components(raw_label, base_labels)
        for comp in components:
            composite_totals[comp] += 1
    present_labels_composite = sorted([l for l in base_labels if composite_totals[l] > 0])

    macro_recall_composite = np.nan
    if present_labels_composite:
        composite_correct_recall = Counter()
        for j, raw_label in enumerate(all_labels):
            comps = extract_base_components(raw_label, base_labels)
            if not comps:
                continue
            ci = sample_to_cluster[j]
            if ci < 0: continue
            pred = cluster_majority.get(ci)
            if pred is None: continue
            if pred in comps:
                composite_correct_recall[pred] += 1

        recalls_composite = []
        for lbl in present_labels_composite:
            total = int(composite_totals[lbl])
            correct = int(composite_correct_recall[lbl])
            recall = correct / total if total > 0 else 0.0
            recalls_composite.append(recall)
            logging.info(f" - {lbl:<3}: N={total:4d} Correct={correct:4d} Recall={recall:.4f}")
        if recalls_composite:
            macro_recall_composite = float(np.mean(recalls_composite))
    else:
        logging.warning("複合ラベル考慮での評価対象ラベルがありません。")

    metrics['MacroRecall_composite'] = macro_recall_composite

    logging.info("\n【集計】")
    logging.info(f"Macro Recall (基本ラベル) = {macro_recall:.4f}")
    logging.info(f"Macro Recall (基本+応用) = {macro_recall_composite:.4f}")
    logging.info(f"Micro Accuracy (基本ラベル) = {micro_accuracy:.4f}")
    if 'ARI_majority' in metrics:
        logging.info(f"Adjusted Rand Index (基本ラベル) = {metrics['ARI_majority']:.4f}")
    if 'NMI_majority' in metrics:
        logging.info(f"Normalized Mutual Info (基本ラベル) = {metrics['NMI_majority']:.4f}")
    logging.info(f"--- {title} 終了 ---\n")
    return metrics


def plot_iteration_metrics(history: Dict[str, List[float]], save_path: str) -> None:
    iters = history.get('iteration', [])
    metrics_names = [k for k in history.keys() if k != 'iteration']
    n = len(metrics_names)
    n_cols = 2
    n_rows = (n + n_cols - 1) // n_cols if n > 0 else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    for idx, mname in enumerate(metrics_names):
        ax = axes[idx]
        ax.plot(iters, history.get(mname, []), marker='o')
        ax.set_title(mname)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(mname)
        ax.grid(True)
    for i in range(n, len(axes)):
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_metrics_history_to_csv(history: Dict[str, List[float]], out_csv: str) -> None:
    df = pd.DataFrame(history)
    df.to_csv(out_csv, index=False)


# =====================================================
# 3type_som側のユーティリティ（ログ・評価・可視化）
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


def winners_to_clusters(winners_xy, som_shape):
    clusters = [[] for _ in range(som_shape[0]*som_shape[1])]
    for i,(ix,iy) in enumerate(winners_xy):
        k = ix*som_shape[1] + iy
        clusters[k].append(i)
    return clusters


def plot_som_node_average_patterns(data_flat, winners_xy, lat, lon, som_shape, save_path, title):
    """
    ノード平均（セントロイド）マップ（偏差[hPa]）
    """
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

    vmin, vmax = -40, 40
    levels = np.linspace(vmin, vmax, 21)
    cmap = 'RdBu_r'

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
            cf = ax.contourf(lon, lat, mp, levels=levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree())
            ax.contour(lon, lat, mp, levels=levels, colors='k', linewidths=0.3, transform=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='black', linewidth=0.8)
            ax.set_extent([115, 155, 15, 55], ccrs.PlateCarree())
            ax.text(0.02,0.96,f'({ix},{iy}) N={counts[ix,iy]}', transform=ax.transAxes,
                    fontsize=7, va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            ax.set_xticks([]); ax.set_yticks([])
            last_cf=cf
    if last_cf is not None:
        fig.subplots_adjust(right=0.88, top=0.94)
        cax = fig.add_axes([0.90, 0.12, 0.02, 0.72])
        fig.colorbar(last_cf, cax=cax, label='Sea Level Pressure Anomaly (hPa)')
    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0,0,0.88,0.94])
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def save_each_node_mean_image(data_flat, winners_xy, lat, lon, som_shape, out_dir, prefix):
    """
    ノード平均（セントロイド）の個別図を保存
    """
    os.makedirs(out_dir, exist_ok=True)
    H, W = len(lat), len(lon)
    X2 = data_flat.reshape(-1, H, W)  # 偏差[hPa]
    map_x, map_y = som_shape

    vmin, vmax = -40, 40
    levels = np.linspace(vmin, vmax, 21)
    cmap = 'RdBu_r'

    for ix in range(map_x):
        for iy in range(map_y):
            mask = (winners_xy[:,0]==ix) & (winners_xy[:,1]==iy)
            idxs = np.where(mask)[0]
            if len(idxs)>0:
                mean_img = np.nanmean(X2[idxs], axis=0)
            else:
                mean_img = np.full((H,W), np.nan, dtype=np.float32)

            fig = plt.figure(figsize=(4,3))
            ax = plt.axes(projection=ccrs.PlateCarree())
            cf = ax.contourf(lon, lat, mean_img, levels=levels, cmap=cmap, transform=ccrs.PlateCarree(), extend='both')
            ax.contour(lon, lat, mean_img, levels=levels, colors='k', linewidths=0.3, transform=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8, edgecolor='black')
            ax.set_extent([115, 155, 15, 55], ccrs.PlateCarree())
            plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label='Sea Level Pressure Anomaly (hPa)')
            ax.set_title(f'({ix},{iy}) N={len(idxs)}')
            ax.set_xticks([]); ax.set_yticks([])
            fpath = os.path.join(out_dir, f'{prefix}_node_{ix}_{iy}.png')
            plt.tight_layout()
            plt.savefig(fpath, dpi=180)
            plt.close(fig)


def plot_label_distributions_base(winners_xy, labels_raw: List[Optional[str]],
                                  base_labels: List[str], som_shape: Tuple[int,int],
                                  save_dir: str, title_prefix: str):
    """
    基本ラベルのみの分布ヒートマップ
    """
    os.makedirs(save_dir, exist_ok=True)
    node_counts = {lbl: np.zeros((som_shape[0], som_shape[1]), dtype=int) for lbl in base_labels}
    for i,(ix,iy) in enumerate(winners_xy):
        lab = basic_label_or_none(labels_raw[i], base_labels)
        if lab in node_counts:
            node_counts[lab][ix,iy] += 1
    cols = 5
    rows = int(np.ceil(len(base_labels)/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.6, rows*2.6))
    axes = np.atleast_2d(axes)
    for idx,lbl in enumerate(base_labels):
        r = idx//cols; c=idx%cols
        ax = axes[r,c]
        im = ax.imshow(node_counts[lbl].T[::-1,:], cmap='viridis')
        ax.set_title(lbl); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for k in range(len(base_labels), rows*cols):
        r = k//cols; c=k%cols
        axes[r,c].axis('off')
    plt.suptitle(f'{title_prefix} Label Distributions on SOM nodes (Base only)', fontsize=14)
    plt.tight_layout(rect=[0,0,1,0.95])
    fpath = os.path.join(save_dir, f'{title_prefix}_label_distributions_base.png')
    plt.savefig(fpath, dpi=250)
    plt.close(fig)


def analyze_nodes_detail_to_log(clusters: List[List[int]],
                                labels: List[Optional[str]],
                                timestamps: np.ndarray,
                                base_labels: List[str],
                                som_shape: Tuple[int, int],
                                log: Logger,
                                title: str):
    """
    ノードごとの詳細（基本ラベル構成・月別分布・純度など）を results.log に追記。
    """
    log.write(f'\n--- {title} ---\n')
    for k, idxs in enumerate(clusters):
        ix, iy = k // som_shape[1], k % som_shape[1]
        n = len(idxs)
        if n == 0:
            continue
        log.write(f'\n[Node ({ix},{iy})] N={n}\n')
        # 基本ラベル構成
        cnt = Counter()
        for j in idxs:
            bl = basic_label_or_none(labels[j], base_labels)
            if bl is not None:
                cnt[bl] += 1
        if cnt:
            log.write('  - ラベル構成（基本ラベルのみ）:\n')
            for lbl, c in sorted(cnt.items(), key=lambda x: x[1], reverse=True):
                log.write(f'    {lbl:<3}: {c:4d} ({c/n*100:5.1f}%)\n')
            purity = max(cnt.values()) / n
            log.write(f'  - ノード純度: {purity:.3f}\n')
        # 月別分布
        if timestamps is not None and n > 0:
            months = pd.to_datetime(timestamps[idxs]).month
            mon_c = Counter(months)
            log.write('  - 月別分布:\n')
            for m in range(1, 13):
                c = mon_c.get(m, 0)
                log.write(f'    {m:2d}月: {c:4d} ({c/n*100:5.1f}%)\n')
    log.write(f'--- {title} 終了 ---\n')


def log_som_recall_by_label_with_nodes(
    log: Logger,
    winners_xy: np.ndarray,
    labels_all: List[Optional[str]],
    base_labels: List[str],
    som_shape: Tuple[int, int],
    section_title: str
):
    if labels_all is None or len(labels_all) == 0:
        log.write("\nラベルが無いため、代表ノード群ベースの再現率出力をスキップします。\n")
        return

    H_nodes, W_nodes = som_shape
    n_nodes = H_nodes * W_nodes
    node_index_arr = winners_xy[:, 0] * W_nodes + winners_xy[:, 1]  # (N,)

    # ノード毎の基本ラベル分布
    node_counters = [Counter() for _ in range(n_nodes)]
    for i, k in enumerate(node_index_arr):
        bl = basic_label_or_none(labels_all[i], base_labels)
        if bl is not None:
            node_counters[int(k)][bl] += 1

    # ノードの代表（多数決）ラベル
    node_majority: List[Optional[str]] = [None] * n_nodes
    for k in range(n_nodes):
        if len(node_counters[k]) > 0:
            node_majority[k] = node_counters[k].most_common(1)[0][0]

    # ラベル→代表ノード一覧
    label_to_nodes: Dict[str, List[Tuple[int, int]]] = {lbl: [] for lbl in base_labels}
    for k, rep in enumerate(node_majority):
        if rep is None:
            continue
        ix, iy = k // W_nodes, k % W_nodes
        label_to_nodes[rep].append((ix, iy))

    # 基本ラベルベースの再現率
    total_base = Counter()
    correct_base = Counter()
    for i, k in enumerate(node_index_arr):
        bl = basic_label_or_none(labels_all[i], base_labels)
        if bl is None:
            continue
        total_base[bl] += 1
        pred = node_majority[int(k)]
        if pred is None:
            continue
        if pred == bl:
            correct_base[bl] += 1

    # 複合ラベル考慮（基本+応用）
    total_comp = Counter()
    correct_comp = Counter()
    for i, k in enumerate(node_index_arr):
        comps = extract_base_components(labels_all[i], base_labels)
        if not comps:
            continue
        for c in comps:
            total_comp[c] += 1
        pred = node_majority[int(k)]
        if pred is None:
            continue
        if pred in comps:
            correct_comp[pred] += 1

    log.write(f"\n【{section_title}】\n")
    log.write("【各ラベルの再現率（代表ノード群ベース）】\n")
    recalls_base = []
    for lbl in base_labels:
        N = int(total_base[lbl])
        C = int(correct_base[lbl])
        rec = (C / N) if N > 0 else 0.0
        recalls_base.append(rec if N > 0 else np.nan)
        nodes_disp = label_to_nodes.get(lbl, [])
        rep_str = "なし" if len(nodes_disp) == 0 else "[" + ", ".join([f"({ix},{iy})" for ix, iy in nodes_disp]) + "]"
        log.write(f" - {lbl:<3}: N={N:4d} Correct={C:4d} Recall={rec:.4f} 代表={rep_str}\n")

    log.write("\n【複合ラベル考慮の再現率（基本+応用）】\n")
    recalls_comp = []
    for lbl in base_labels:
        Nt = int(total_comp[lbl])
        Ct = int(correct_comp[lbl])
        rec_t = (Ct / Nt) if Nt > 0 else 0.0
        if Nt > 0:
            recalls_comp.append(rec_t)
        log.write(f" - {lbl:<3}: N={Nt:4d} Correct={Ct:4d} Recall={rec_t:.4f}\n")

    base_valid = [r for r, lbl in zip(recalls_base, base_labels) if not np.isnan(r) and total_base[lbl] > 0]
    macro_base = float(np.mean(base_valid)) if len(base_valid) > 0 else float('nan')
    macro_comp = float(np.mean(recalls_comp)) if len(recalls_comp) > 0 else float('nan')
    log.write(f"\n[Summary] Macro Recall (基本ラベル)   = {macro_base:.4f}\n")
    log.write(f"[Summary] Macro Recall (基本+応用) = {macro_comp:.4f}\n")


# =====================================================
# medoid（ノード重心に最も近いサンプル）を計算・保存する処理
# =====================================================
def _euclidean_dist_to_ref(Xb: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    # Xb: (B,H,W), ref: (H,W) -> (B,)
    diff = Xb - ref.view(1, *ref.shape)
    d2 = (diff*diff).sum(dim=(1,2))
    return torch.sqrt(d2 + 1e-12)


def _ssim_dist_to_ref(Xb: torch.Tensor, ref: torch.Tensor, c1: float = SSIM_C1, c2: float = SSIM_C2) -> torch.Tensor:
    # 1 - SSIM (全体1窓の簡略SSIM)
    B,H,W = Xb.shape
    mu_x = Xb.mean(dim=(1,2))              # (B,)
    xc = Xb - mu_x.view(B,1,1)
    var_x = (xc*xc).mean(dim=(1,2))        # (B,)
    mu_r = ref.mean()
    rc = ref - mu_r
    var_r = (rc*rc).mean()                 # scalar
    cov = (xc * rc.view(1,H,W)).mean(dim=(1,2))  # (B,)
    l_num = (2*mu_x*mu_r + c1)
    l_den = (mu_x**2 + mu_r**2 + c1)
    c_num = (2*cov + c2)
    c_den = (var_x + var_r + c2)
    ssim = (l_num*c_num) / (l_den*c_den + 1e-12)
    return 1.0 - ssim


def _s1_dist_to_ref(Xb: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    # Teweles–Wobus S1: 100 * sum|∇X-∇ref| / sum max(|∇X|,|∇ref|)
    dXdx = Xb[:,:,1:] - Xb[:,:,:-1]
    dXdy = Xb[:,1:,:] - Xb[:,:-1,:]
    dRdx = ref[:,1:] - ref[:,:-1]
    dRdy = ref[1:,:] - ref[:-1,:]
    num_dx = (torch.abs(dXdx - dRdx.view(1, *dRdx.shape))).sum(dim=(1,2))
    num_dy = (torch.abs(dXdy - dRdy.view(1, *dRdy.shape))).sum(dim=(1,2))
    den_dx = torch.maximum(torch.abs(dXdx), torch.abs(dRdx).view(1, *dRdx.shape)).sum(dim=(1,2))
    den_dy = torch.maximum(torch.abs(dXdy), torch.abs(dRdy).view(1, *dRdy.shape)).sum(dim=(1,2))
    s1 = 100.0 * (num_dx + num_dy) / (den_dx + den_dy + 1e-12)
    return s1


def compute_node_medoids_by_centroid(
    method_name: str,
    data_flat: np.ndarray,          # (N,D) hPa偏差（空間平均差し引き）
    winners_xy: np.ndarray,         # (N,2)
    som_shape: Tuple[int,int],
    field_shape: Tuple[int,int],
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[Dict[Tuple[int,int], int], Dict[Tuple[int,int], float]]:
    """
    各ノードの「重心（平均画像）」に最も近いサンプルを、方式別距離で選ぶ。
    戻り値:
      - node_to_medoid_idx: {(ix,iy): sample_index}
      - node_to_medoid_dist: {(ix,iy): distance_value}
    """
    H,W = field_shape
    X2 = data_flat.reshape(-1, H, W)
    X_t = torch.as_tensor(X2, device=device, dtype=torch.float32)

    node_to_medoid_idx: Dict[Tuple[int,int], int] = {}
    node_to_medoid_dist: Dict[Tuple[int,int], float] = {}

    for ix in range(som_shape[0]):
        for iy in range(som_shape[1]):
            mask = (winners_xy[:,0]==ix) & (winners_xy[:,1]==iy)
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                continue
            # 重心（平均画像）
            centroid = np.nanmean(X2[idxs], axis=0).astype(np.float32)  # (H,W)
            centroid_t = torch.as_tensor(centroid, device=device, dtype=torch.float32)
            # 距離
            Xb = X_t[idxs]  # (B,H,W)
            if method_name == 'euclidean':
                d = _euclidean_dist_to_ref(Xb, centroid_t)
            elif method_name == 'ssim':
                d = _ssim_dist_to_ref(Xb, centroid_t)
            elif method_name == 's1':
                d = _s1_dist_to_ref(Xb, centroid_t)
            else:
                raise ValueError(f'Unknown method_name: {method_name}')
            # 最小
            pos = int(torch.argmin(d).item())
            node_to_medoid_idx[(ix,iy)] = int(idxs[pos])
            node_to_medoid_dist[(ix,iy)] = float(d[pos].item())

    return node_to_medoid_idx, node_to_medoid_dist


def plot_som_node_medoid_patterns(
    data_flat: np.ndarray,
    node_to_medoid_idx: Dict[Tuple[int,int], int],
    lat: np.ndarray, lon: np.ndarray,
    som_shape: Tuple[int,int],
    save_path: str,
    title: str
):
    """
    ノードmedoid（重心に最も近いサンプル）の○×○マップを保存
    """
    H, W = len(lat), len(lon)

    vmin, vmax = -40, 40
    levels = np.linspace(vmin, vmax, 21)
    cmap = 'RdBu_r'

    map_x, map_y = som_shape
    nrows, ncols = map_y, map_x
    figsize=(ncols*2.6, nrows*2.6)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
                             subplot_kw={'projection': ccrs.PlateCarree()})
    axes = np.atleast_2d(axes)
    axes = axes.T[::-1,:]  # 表示並びの整合

    last_cf = None
    for ix in range(map_x):
        for iy in range(map_y):
            ax = axes[ix, iy]
            key = (ix, iy)
            if key not in node_to_medoid_idx:
                ax.set_axis_off()
                continue
            mi = node_to_medoid_idx[key]
            pat = data_flat[mi].reshape(H, W)
            cf = ax.contourf(lon, lat, pat, levels=levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree())
            ax.contour(lon, lat, pat, levels=levels, colors='k', linewidths=0.3, transform=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='black', linewidth=0.8)
            ax.set_extent([115, 155, 15, 55], ccrs.PlateCarree())
            ax.set_title(f'({ix},{iy})')
            ax.set_xticks([]); ax.set_yticks([])
            last_cf = cf

    if last_cf is not None:
        fig.subplots_adjust(right=0.88, top=0.94)
        cax = fig.add_axes([0.90, 0.12, 0.02, 0.72])
        fig.colorbar(last_cf, cax=cax, label='Sea Level Pressure Anomaly (hPa)')
    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0,0,0.88,0.94])
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def save_each_node_medoid_image(
    data_flat: np.ndarray,
    node_to_medoid_idx: Dict[Tuple[int,int], int],
    lat: np.ndarray, lon: np.ndarray,
    som_shape: Tuple[int,int],
    out_dir: str,
    prefix: str
):
    os.makedirs(out_dir, exist_ok=True)
    H, W = len(lat), len(lon)

    vmin, vmax = -40, 40
    levels = np.linspace(vmin, vmax, 21)
    cmap = 'RdBu_r'

    for ix in range(som_shape[0]):
        for iy in range(som_shape[1]):
            key = (ix, iy)
            if key not in node_to_medoid_idx:
                continue
            mi = node_to_medoid_idx[key]
            pat = data_flat[mi].reshape(H, W)
            fig = plt.figure(figsize=(4,3))
            ax = plt.axes(projection=ccrs.PlateCarree())
            cf = ax.contourf(lon, lat, pat, levels=levels, cmap=cmap, transform=ccrs.PlateCarree(), extend='both')
            ax.contour(lon, lat, pat, levels=levels, colors='k', linewidths=0.3, transform=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8, edgecolor='black')
            ax.set_extent([115, 155, 15, 55], ccrs.PlateCarree())
            plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label='Sea Level Pressure Anomaly (hPa)')
            ax.set_title(f'Medoid ({ix},{iy}) sample={mi}')
            ax.set_xticks([]); ax.set_yticks([])
            fpath = os.path.join(out_dir, f'{prefix}_node_{ix}_{iy}_medoid.png')
            plt.tight_layout()
            plt.savefig(fpath, dpi=180)
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
        activation_distance=activation_distance,                # 'euclidean'/'ssim'/'s1'
        random_seed=SEED,
        sigma_decay='asymptotic_decay',
        s1_field_shape=field_shape,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        dtype=torch.float32,
        nodes_chunk=NODES_CHUNK
    )
    som.random_weights_init(data_all)

    # ====== 学習を区切って実施し、各区切りで評価（履歴プロット用） ======
    step = max(1, NUM_ITER // SOM_EVAL_SEGMENTS)
    iter_history: Dict[str, List[float]] = {
        'iteration': [], 'MacroRecall_majority': [], 'MacroRecall_composite': [],
        'MicroAccuracy_majority': [],
        'ARI_majority': [], 'NMI_majority': [], 'QuantizationError': []
    }

    current_iter = 0
    for seg in range(SOM_EVAL_SEGMENTS):
        n_it = min(step, NUM_ITER - current_iter)
        if n_it <= 0:
            break
        som.train_batch(
            data_all, num_iteration=n_it,
            batch_size=BATCH_SIZE, verbose=True, log_interval=LOG_INTERVAL,
            update_per_iteration=False, shuffle=True
        )
        current_iter += n_it

        # 量子化誤差
        qe_now = som.quantization_error(data_all, sample_limit=EVAL_SAMPLE_LIMIT, batch_size=max(32, BATCH_SIZE))

        # 評価
        winners_now = som.predict(data_all, batch_size=max(64, BATCH_SIZE))
        clusters_now = winners_to_clusters(winners_now, (SOM_X, SOM_Y))
        metrics = evaluate_clusters_only_base(
            clusters=clusters_now,
            all_labels=labels_all,
            base_labels=BASE_LABELS,
            title=f"[{method_name.upper()}] Iteration={current_iter} Evaluation (Base labels)"
        )
        # ログ（results.log）に集約指標を追記
        log.write(f'\n[Iteration {current_iter}] QuantizationError={qe_now:.6f}\n')
        if metrics is not None:
            metric_keys = ['MacroRecall_majority', 'MacroRecall_composite', 'MicroAccuracy_majority', 'ARI_majority', 'NMI_majority']
            for k in metric_keys:
                if k in metrics:
                    log.write(f'  {k} = {metrics[k]:.6f}\n')

        # 履歴に保存
        iter_history['iteration'].append(current_iter)
        iter_history['QuantizationError'].append(qe_now)
        if metrics is not None:
            metric_keys_for_history = ['MacroRecall_majority', 'MacroRecall_composite', 'MicroAccuracy_majority', 'ARI_majority', 'NMI_majority']
            for k in metric_keys_for_history:
                iter_history[k].append(metrics.get(k, np.nan))
        else:
            for k in ['MacroRecall_majority', 'MacroRecall_composite', 'MicroAccuracy_majority', 'ARI_majority', 'NMI_majority']:
                iter_history[k].append(np.nan)

    # イテレーション履歴の保存（CSV/PNG）
    iter_csv = os.path.join(out_dir, f'{method_name}_iteration_metrics.csv')
    save_metrics_history_to_csv(iter_history, iter_csv)
    iter_png = os.path.join(out_dir, f'{method_name}_iteration_vs_metrics.png')
    plot_iteration_metrics(iter_history, iter_png)
    log.write(f'\nIteration metrics saved: CSV={iter_csv}, PNG={iter_png}\n')

    # ====== 最終モデルでの割当 ======
    winners_all = som.predict(data_all, batch_size=max(64, BATCH_SIZE))

    # 割当CSV
    assign_csv_all = os.path.join(out_dir, f'{method_name}_assign_all.csv')
    pd.DataFrame({
        'time': times_all,
        'bmu_x': winners_all[:,0], 'bmu_y': winners_all[:,1],
        'label_raw': labels_all if labels_all is not None else ['']*len(winners_all)
    }).to_csv(assign_csv_all, index=False, encoding='utf-8-sig')
    log.write(f'\nAssigned BMU (all) -> {assign_csv_all}\n')

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

    # ===== 新規: ノードmedoid（重心に最も近いサンプル）マップ =====
    node_to_medoid_idx, node_to_medoid_dist = compute_node_medoids_by_centroid(
        method_name=method_name,
        data_flat=data_all, winners_xy=winners_all,
        som_shape=(SOM_X, SOM_Y), field_shape=field_shape
    )
    # ○×○マップ（medoid）
    medoid_bigmap = os.path.join(out_dir, f'{method_name}_som_node_medoid_all.png')
    plot_som_node_medoid_patterns(
        data_flat=data_all,
        node_to_medoid_idx=node_to_medoid_idx,
        lat=lat, lon=lon, som_shape=(SOM_X, SOM_Y),
        save_path=medoid_bigmap,
        title=f'{method_name.upper()} SOM Node Medoid (closest-to-centroid)'
    )
    log.write(f'Node medoid map (all) -> {medoid_bigmap}\n')

    # 各ノードmedoid個別図
    pernode_medoid_dir = os.path.join(out_dir, f'{method_name}_pernode_medoid_all')
    save_each_node_medoid_image(
        data_flat=data_all,
        node_to_medoid_idx=node_to_medoid_idx,
        lat=lat, lon=lon, som_shape=(SOM_X, SOM_Y),
        out_dir=pernode_medoid_dir, prefix='all'
    )
    log.write(f'Per-node medoid images (all) -> {pernode_medoid_dir}\n')

    # medoid選定結果のCSV
    rows = []
    for (ix, iy), mi in sorted(node_to_medoid_idx.items()):
        t = str(times_all[mi]) if len(times_all)>0 else ''
        raw = labels_all[mi] if labels_all is not None else ''
        base = basic_label_or_none(raw, BASE_LABELS)
        dist = node_to_medoid_dist.get((ix,iy), np.nan)
        rows.append({
            'node_x': ix, 'node_y': iy, 'node_flat': ix*SOM_Y+iy,
            'medoid_index': mi, 'time': t, 'label_raw': raw, 'label_base': base, 'distance': dist
        })
    medoid_csv = os.path.join(out_dir, f'{method_name}_node_medoids.csv')
    pd.DataFrame(rows).to_csv(medoid_csv, index=False, encoding='utf-8-sig')
    log.write(f'Node medoid selection CSV -> {medoid_csv}\n')

    # ログに概要（いくつか）
    log.write('\n[Node medoid summary (first 20)]\n')
    for r in rows[:20]:
        log.write(f"({r['node_x']},{r['node_y']}): idx={r['medoid_index']}, time={r['time']}, base={r['label_base']}, dist={r['distance']:.4f}\n")

    # ===== 評価（ラベルがあれば） =====
    if labels_all is not None:
        clusters_all = winners_to_clusters(winners_all, (SOM_X,SOM_Y))

        # 混同行列（基本ラベルのみ）を構築・保存
        cm, cluster_names = build_confusion_matrix_only_base(clusters_all, labels_all, BASE_LABELS)
        conf_csv = os.path.join(out_dir, f'{method_name}_confusion_matrix_all.csv')
        cm.to_csv(conf_csv, encoding='utf-8-sig')
        log.write(f'\nConfusion matrix (base vs clusters) -> {conf_csv}\n')

        # 集約指標（基本/複合）
        metrics = evaluate_clusters_only_base(
            clusters=clusters_all,
            all_labels=labels_all,
            base_labels=BASE_LABELS,
            title=f"[{method_name.upper()}] SOM Final Evaluation (Base labels)"
        )
        if metrics is not None:
            log.write('\n[Final Metrics]\n')
            metric_keys = ['MacroRecall_majority', 'MacroRecall_composite', 'MicroAccuracy_majority', 'ARI_majority', 'NMI_majority']
            for k in metric_keys:
                if k in metrics:
                    log.write(f'  {k} = {metrics[k]:.6f}\n')

        # ラベル分布ヒートマップ（基本ラベルのみ）
        dist_dir_all = os.path.join(out_dir, f'{method_name}_label_dist_all')
        plot_label_distributions_base(winners_all, labels_all, BASE_LABELS, (SOM_X,SOM_Y), dist_dir_all, title_prefix='All')
        log.write(f'Label-distribution heatmaps (base only) -> {dist_dir_all}\n')

        # ノード詳細（構成・月分布など）を results.log に
        analyze_nodes_detail_to_log(
            clusters_all, labels_all, times_all, BASE_LABELS, (SOM_X, SOM_Y),
            log, title=f'[{method_name.upper()}] SOM Node-wise Analysis'
        )

        # 代表ノード群ベースの再現率（基本/複合）と、各ラベルの代表ノード一覧を results.log に出力
        log_som_recall_by_label_with_nodes(
            log=log,
            winners_xy=winners_all,
            labels_all=labels_all,
            base_labels=BASE_LABELS,
            som_shape=(SOM_X, SOM_Y),
            section_title='SOM代表ノード群ベースの再現率（基本/複合）'
        )
    else:
        log.write('Labels not found; skip evaluation.\n')

    log.write('\n=== Done (full period) ===\n')
    log.close()


# =====================================================
# メイン
# =====================================================
def main():
    set_reproducibility(SEED)
    setup_logging_v3()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"使用デバイス: {device.upper()}")
    logging.info(f"SOM(3type): size={SOM_X}x{SOM_Y}, iters={NUM_ITER}, batch={BATCH_SIZE}, nodes_chunk={NODES_CHUNK}")

    # 共通の前処理（hPa偏差）
    X_for_s1, X_original_hpa, X_anomaly_hpa, lat, lon, d_lat, d_lon, ts, labels = load_and_prepare_data_unified(
        DATA_FILE, TIME_START, TIME_END, device
    )
    # 3type_SOM用の2次元表現（numpy）
    data_all = X_anomaly_hpa.reshape(X_anomaly_hpa.shape[0], -1).astype(np.float32)
    field_shape = (d_lat, d_lon)

    # 3種類のbatchSOM（全期間）を実行
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

    logging.info("全処理完了。")


if __name__ == '__main__':
    main()