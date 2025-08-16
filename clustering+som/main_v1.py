# main_v1.py
# -*- coding: utf-8 -*-
import os
import logging
import time
from typing import Optional, List

import numpy as np
import pandas as pd
import xarray as xr
import torch
import matplotlib.pyplot as plt

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
    summarize_cluster_info
)
from s1_minisom import MiniSom, grid_auto_size

import cartopy.crs as ccrs  # noqa: F401
import cartopy.feature as cfeature  # noqa: F401

# ============== ユーザ調整パラメータ（ここだけでOK） ==============
SEED = 42                 # 乱数シード（再現性）
TH_MERGE = 83             # HACのマージ停止しきい値（S1スコア, 小さいほど類似）

# 2段階クラスタリング用の行バッチと列チャンク
CLUS_ROW_BATCH = 4        # 行バッチ（B）。8でOOMが出たので2〜4程度を推奨
CLUS_COL_CHUNK = 256       # 列チャンク（Nc）。16〜64程度で調整

# SOM学習・推論用のバッチサイズ（メドイド数が少ないので大きめでもOK）
SOM_BATCH_SIZE = 32

# S1分布ヒストグラム用の設定（計算コスト削減のためのサンプル上限とチャンク）
S1_DISTRIBUTION_MAX_SAMPLES = 3653
S1_HIST_ROW_BATCH = 4
S1_HIST_COL_CHUNK = 256

# 日次マップ出力の1クラスタあたり上限(Noneで無制限)
DAILY_MAPS_PER_CLUSTER_LIMIT: Optional[int] = None

# SOM学習のイテレーション回数（自動設定: max(300, 20*ノード数), ただし2000上限）
SOM_MAX_ITERS_CAP = 2000

# ============== 固定：実験条件・パス等 ==============
DATA_FILE = './prmsl_era5_all_data_seasonal_large.nc'
RESULT_DIR = './results_v1'
ONLY_BASE_DIR = os.path.join(RESULT_DIR, 'only_base_label')
SOM_DIR = os.path.join(RESULT_DIR, 'som')

START_DATE = '1991-01-01'
END_DATE = '2000-12-31'

BASE_LABELS = [
    '1', '2A', '2B', '2C', '2D', '3A', '3B', '3C', '3D',
    '4A', '4B', '5', '6A', '6B', '6C'
]


def set_reproducibility(seed: int = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # optional: 断片化回避（環境に応じて）
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
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


def setup_logging():
    os.makedirs(RESULT_DIR, exist_ok=True)
    for d in [ONLY_BASE_DIR, SOM_DIR]:
        os.makedirs(d, exist_ok=True)
    log_path = os.path.join(RESULT_DIR, 'run_v1.log')
    if os.path.exists(log_path):
        os.remove(log_path)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )
    logging.info("ログ初期化完了。")


def load_and_prepare_data(filepath: str,
                          start_date: str,
                          end_date: str,
                          device: str = 'cpu'):
    logging.info(f"データ読み込み: {filepath}")
    ds = xr.open_dataset(filepath)
    sub = ds.sel(valid_time=slice(start_date, end_date))

    labels = None
    if 'label' in sub.variables:
        vals = sub['label'].values
        labels = [v.decode('utf-8') if isinstance(v, bytes) else str(v) for v in vals]
        logging.info("ラベルを読み込みました。")

    msl = sub['msl']  # (time, lat, lon) in Pa
    n, H, W = msl.shape
    msl_hpa = (msl.values / 100.0).astype(np.float32)
    spatial_mean = np.mean(msl_hpa, axis=(1, 2), keepdims=True)
    anomaly_hpa = msl_hpa - spatial_mean

    X_for_s1 = torch.from_numpy(anomaly_hpa.reshape(n, H * W)).to(device, dtype=torch.float32)

    lat = sub['latitude'].values
    lon = sub['longitude'].values
    ts = sub['valid_time'].values

    logging.info(f"期間: {start_date} 〜 {end_date}, サンプル数={n}, 解像度={H}x{W}")
    return X_for_s1, msl_hpa, anomaly_hpa, lat, lon, H, W, ts, labels


def main():
    set_reproducibility(SEED)
    setup_logging()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"使用デバイス: {device.upper()}")
    logging.info(f"SEED={SEED}, TH_MERGE={TH_MERGE}")
    logging.info(f"CLUS_ROW_BATCH={CLUS_ROW_BATCH}, CLUS_COL_CHUNK={CLUS_COL_CHUNK}, SOM_BATCH_SIZE={SOM_BATCH_SIZE}")

    X_for_s1, X_original_hpa, X_anomaly_hpa, lat, lon, d_lat, d_lon, ts, labels = load_and_prepare_data(
        DATA_FILE, START_DATE, END_DATE, device
    )

    # S1分布の可視化（チャンクでメモリ節約）
    s1_hist_path = os.path.join(RESULT_DIR, 's1_score_distribution_anomaly_hpa.png')
    plot_s1_distribution_histogram(
        X_for_s1, d_lat, d_lon, s1_hist_path,
        row_batch_size=S1_HIST_ROW_BATCH,
        col_chunk_size=S1_HIST_COL_CHUNK,
        max_samples=S1_DISTRIBUTION_MAX_SAMPLES
    )
    logging.info(f"S1分布ヒストグラム保存: {s1_hist_path}")
    # キャッシュを返却（任意）
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    # 反復毎の評価履歴
    history_only_base = {'iteration': [], 'MacroRecall_majority': [], 'MicroAccuracy_majority': [], 'ARI_majority': [], 'NMI_majority': []}

    def eval_callback(iter_idx: int, clusters: List[List[int]], medoids: List[Optional[int]]):
        m = evaluate_clusters_only_base(clusters, labels, BASE_LABELS, title=f"反復 {iter_idx} 評価（基本ラベル）")
        if m is not None:
            history_only_base['iteration'].append(iter_idx)
            history_only_base['MacroRecall_majority'].append(m.get('MacroRecall_majority', np.nan))
            history_only_base['MicroAccuracy_majority'].append(m.get('MicroAccuracy_majority', np.nan))
            history_only_base['ARI_majority'].append(m.get('ARI_majority', np.nan))
            history_only_base['NMI_majority'].append(m.get('NMI_majority', np.nan))

    # 二段階クラスタリング（行×列チャンクでメモリ節約）
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
    _ = evaluate_clusters_only_base(clusters, labels, BASE_LABELS, title="最終評価（基本ラベル）")

    # 結果保存（torch）
    results_path = os.path.join(RESULT_DIR, f'clustering_result_th{TH_MERGE}_v1.pt')
    torch.save({'clusters': clusters, 'medoids': medoids}, results_path)
    logging.info(f"クラスタリング結果保存: {results_path}")

    # 評価推移の保存
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

    # 日次マップ出力（必要に応じて上限可）
    daily_dir = os.path.join(ONLY_BASE_DIR, 'daily_anomaly_maps_only_base')
    save_daily_maps_per_cluster(clusters, X_anomaly_hpa, lat, lon, ts, labels, BASE_LABELS, daily_dir, per_cluster_limit=DAILY_MAPS_PER_CLUSTER_LIMIT)
    logging.info(f"日次マップ保存先: {daily_dir}")

    # クラスタ情報をCSVで保存
    info_df = summarize_cluster_info(clusters, medoids, labels, BASE_LABELS, ts)
    info_csv = os.path.join(RESULT_DIR, 'cluster_summary.csv')
    info_df.to_csv(info_csv, index=False)
    logging.info(f"クラスタサマリCSV: {info_csv}")

    # ============== S1-batchSOM（メドイドのみで可視化） ==============
    medoid_indices = [m for m in medoids if m is not None]
    if len(medoid_indices) == 0:
        logging.warning("SOM学習のための有効メドイドがありません。SOMをスキップします。")
        return

    medoid_data = X_anomaly_hpa[medoid_indices].reshape(len(medoid_indices), d_lat * d_lon).astype(np.float32)

    # SOMサイズ自動選択（クラス数を覆う最小正方格子）
    som_x, som_y = grid_auto_size(len(medoid_indices))
    som_nodes = som_x * som_y
    logging.info(f"SOMサイズ自動選択: {som_x} x {som_y}（ノード数={som_nodes}、クラスタ={len(medoid_indices)}）")

    sigma0 = max(som_x, som_y) / 2.0
    nodes_chunk = max(8, min(64, SOM_BATCH_SIZE * 2))

    som = MiniSom(som_x, som_y, d_lat * d_lon, sigma=sigma0,
                  s1_field_shape=(d_lat, d_lon),
                  device=device, random_seed=SEED, nodes_chunk=nodes_chunk)

    som.random_weights_init(medoid_data)

    som_iters = min(SOM_MAX_ITERS_CAP, max(300, 20 * som_nodes))
    logging.info(f"SOM学習: iters={som_iters}, batch={SOM_BATCH_SIZE}, sigma0={sigma0:.2f}, nodes_chunk={nodes_chunk}")
    _ = som.train_batch(medoid_data, num_iteration=som_iters, batch_size=SOM_BATCH_SIZE, verbose=True, log_interval=max(10, som_iters // 10))

    # メドイドのBMUを予測
    bmu_xy = som.predict(medoid_data, batch_size=SOM_BATCH_SIZE)  # (M, 2)

    # 各メドイドと元クラスタ番号の対応
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

    # BMU散布図を保存
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

    # SOMコードブック（各ノードのパターン）を地理図でタイル表示
    som_codebook_img = os.path.join(SOM_DIR, f'som_codebook_maps_{som_x}x{som_y}.png')
    weights_grid = som.get_weights()  # (x, y, D)
    from matplotlib.colors import Normalize

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

    logging.info("全処理完了。")


if __name__ == '__main__':
    main()