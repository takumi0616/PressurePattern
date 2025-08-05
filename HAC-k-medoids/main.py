import os
import logging
import torch
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize
from tqdm import tqdm
import time
from collections import Counter

# --- 1. 設定項目 ---

# ファイルパスとディレクトリ
DATA_FILE = './prmsl_era5_all_data_seasonal_small.nc'
RESULT_DIR = './two_stage_clustering_results'

# データ期間 (データセットに合わせて調整してください)
START_DATE = '1991-01-01'
END_DATE = '2000-12-31'

# 2段階クラスタリングアルゴリズムのパラメータ
TH_MERGE = 0.40

# 安定化定数
C1 = 1e-8
C2 = 1e-8

# 基本ラベルリスト
BASE_LABELS = [
    '1', '2A', '2B', '2C', '2D', '3A', '3B', '3C', '3D',
    '4A', '4B', '5', '6A', '6B', '6C'
]

# ★★★ パフォーマンス改善のためのバッチサイズ設定 ★★★
# ご使用のGPUメモリに応じて調整してください (例: 128, 256, 512)
CALCULATION_BATCH_SIZE = 256

# --- 2. 初期設定 (ロギング、結果ディレクトリ、デバイス) ---
os.makedirs(RESULT_DIR, exist_ok=True)
log_path = os.path.join(RESULT_DIR, 'clustering_analysis.log')
if os.path.exists(log_path): os.remove(log_path)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_path), logging.StreamHandler()])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f"使用デバイス: {device.upper()}")

# --- 3. データ準備関数 (変更なし) ---
def load_and_prepare_data(filepath, start_date, end_date, device):
    logging.info(f"データファイル '{filepath}' を読み込んでいます...")
    try:
        ds = xr.open_dataset(filepath)
    except FileNotFoundError:
        return None, None, None, None, None
    data_period = ds.sel(valid_time=slice(start_date, end_date))
    if data_period.valid_time.size == 0:
        return None, None, None, None, None
    labels = None
    if 'label' in data_period.variables:
        label_values = data_period['label'].values
        labels = [l.decode('utf-8') if isinstance(l, bytes) else str(l) for l in label_values]
        logging.info("ラベルデータを読み込みました。")
    msl_data = data_period['msl']
    n_samples, d_lat, d_lon = msl_data.shape
    flattened_data = msl_data.values.reshape(n_samples, d_lat * d_lon)
    mean = np.mean(flattened_data, axis=0, keepdims=True)
    std = np.std(flattened_data, axis=0, keepdims=True)
    std[std == 0] = 1
    normalized_data = (flattened_data - mean) / std
    X = torch.from_numpy(normalized_data).to(device, dtype=torch.float32)
    lat_coords = data_period['latitude'].values
    lon_coords = data_period['longitude'].values
    time_stamps = data_period.valid_time.values
    logging.info(f"データ期間: {START_DATE} から {END_DATE}")
    logging.info(f"データ形状: {X.shape[0]}サンプル, {X.shape[1]}次元 ({d_lat}x{d_lon})")
    return X, lat_coords, lon_coords, time_stamps, labels

# --- 4. SSIMおよびクラスタリング関連関数 (最適化版) ---

def calculate_ssim_pairwise_batch(x_batch, y_all, c1=C1, c2=C2):
    """
    【最適化】バッチ(B, D)と全体(N, D)のテンソル間でペアワイズSSIMを高速計算する。
    """
    B, D = x_batch.shape
    N, D = y_all.shape

    # 平均値の計算 (ブロードキャスト準備)
    mu_x = torch.mean(x_batch, dim=1).unsqueeze(1) # (B, 1)
    mu_y = torch.mean(y_all, dim=1).unsqueeze(0)   # (1, N)

    # 修正SSIMのための平均値 (ブロードキャストにより(B, N)になる)
    mu_x_prime = (mu_x + mu_y) / 2
    mu_y_prime = mu_x_prime + torch.abs(mu_x - mu_y)
    
    # 分散の計算
    sigma_x_sq = torch.var(x_batch, dim=1, unbiased=False).unsqueeze(1) # (B, 1)
    sigma_y_sq = torch.var(y_all, dim=1, unbiased=False).unsqueeze(0)   # (1, N)

    # 共分散の計算 (ブロードキャスト活用)
    x_expanded = x_batch.unsqueeze(1) # (B, 1, D)
    y_expanded = y_all.unsqueeze(0)   # (1, N, D)
    cov_xy = torch.mean((x_expanded - mu_x.unsqueeze(2)) * (y_expanded - mu_y.unsqueeze(2)), dim=2) # (B, N)

    # SSIM計算 (全ての演算が(B, N)行列で行われる)
    numerator = (2 * mu_x_prime * mu_y_prime + c1) * (2 * cov_xy + c2)
    denominator = (mu_x_prime**2 + mu_y_prime**2 + c1) * (sigma_x_sq + sigma_y_sq + c2)
    
    return numerator / denominator

def find_medoid_torch(cluster_indices, X):
    if not cluster_indices: return None
    if len(cluster_indices) == 1: return cluster_indices[0]
    
    cluster_data = X[cluster_indices]
    
    # 【最適化】クラスタ内のペアワイズSSIMを一括計算
    # メモリを考慮し、ここもバッチ処理にすることが可能だが、クラスタサイズは
    # 全体より小さいことが多いので、まずはこのままで様子を見る。
    # 大きなクラスタで遅い場合は、ここもバッチ化する。
    ssim_scores_matrix = torch.zeros((len(cluster_indices), len(cluster_indices)), device=device)
    for i in range(len(cluster_indices)):
        ssim_scores_matrix[i, :] = calculate_ssim_pairwise_batch(cluster_data[i].unsqueeze(0), cluster_data).squeeze()

    ssim_sum = torch.sum(ssim_scores_matrix, dim=1)
    medoid_pos_in_cluster = torch.argmax(ssim_sum)
    
    return cluster_indices[medoid_pos_in_cluster]

# --- 5. クラスタ分布分析関数 (変更なし) ---
def analyze_cluster_distribution(clusters, all_labels, all_time_stamps, base_labels, iteration_name):
    logging.info(f"\n--- {iteration_name} クラスタ分布分析 ---")
    for i, cluster_indices in enumerate(clusters):
        num_samples_in_cluster = len(cluster_indices)
        if num_samples_in_cluster == 0: continue
        logging.info(f"\n[クラスタ {i+1}] (データ数: {num_samples_in_cluster})")
        if all_labels:
            cluster_labels = [all_labels[j] for j in cluster_indices]
            base_label_counts = Counter(label for label in cluster_labels if label in base_labels)
            if base_label_counts:
                logging.info("  - ラベル構成 (基本ラベルのみ):")
                sorted_labels = sorted(base_label_counts.items(), key=lambda item: item[1], reverse=True)
                for label, count in sorted_labels:
                    percentage = (count / num_samples_in_cluster) * 100
                    logging.info(f"    - {label:<4}: {count:4d} 件 ({percentage:5.1f}%)")
        if all_time_stamps is not None:
            cluster_time_stamps = np.array(all_time_stamps)[cluster_indices]
            months = pd.to_datetime(cluster_time_stamps).month
            month_counts = Counter(months)
            logging.info("  - 月別分布:")
            for month_val in range(1, 13):
                count = month_counts.get(month_val, 0)
                percentage = (count / num_samples_in_cluster) * 100
                bar = '■' * int(percentage / 4)
                logging.info(f"    - {month_val:2d}月: {count:4d}件 ({percentage:5.1f}%) {bar}")
    logging.info(f"--- {iteration_name} 分析終了 ---\n")

# --- 6. 修正・最適化版：2段階クラスタリングアルゴリズム ---
def two_stage_clustering(X, th_merge, all_labels, all_time_stamps, base_labels, batch_size):
    n_samples = X.shape[0]
    logging.info("2段階クラスタリングを開始します...")
    clusters = [[i] for i in range(n_samples)]
    medoids = list(range(n_samples))
    
    iteration = 1
    while True:
        logging.info(f"\n{'='*10} 反復: {iteration} {'='*10}")
        logging.info(f"現在のクラスタ数: {len(clusters)}")

        # ステップ1: HAC
        logging.info("ステップ1: HAC - 類似クラスタのマージを開始...")
        num_clusters_before_merge = len(clusters)
        medoid_data = X[medoids]
        
        # ▼▼▼▼▼【最適化】SSIM行列計算をバッチ処理化 ▼▼▼▼▼
        ssim_matrix = torch.zeros((num_clusters_before_merge, num_clusters_before_merge), device=device)
        for i in tqdm(range(0, num_clusters_before_merge, batch_size), desc="HAC: SSIM計算中", leave=False):
            end_idx = min(i + batch_size, num_clusters_before_merge)
            x_batch = medoid_data[i:end_idx]
            # バッチと全メドイドデータとのSSIMを一括計算
            ssim_scores_batch = calculate_ssim_pairwise_batch(x_batch, medoid_data)
            ssim_matrix[i:end_idx, :] = ssim_scores_batch
        # ▲▲▲▲▲ 最適化 ▲▲▲▲▲
        
        ssim_matrix.fill_diagonal_(-torch.inf)
        # 対称化は不要（全ペアを計算したため）。ただし、重複を避けるため上三角のみ使う
        merge_candidates = (torch.triu(ssim_matrix) > th_merge).nonzero(as_tuple=False)
        
        if merge_candidates.shape[0] == 0:
            logging.info("マージ可能なクラスタペアが見つかりませんでした。クラスタリングを終了します。")
            break

        # マージ処理 (変更なし)
        unique_candidates = [{'pair': tuple(pair.tolist()), 'ssim': ssim_matrix[pair[0], pair[1]].item()} for pair in merge_candidates]
        unique_candidates.sort(key=lambda x: x['ssim'], reverse=True)
        merged_indices = set()
        new_clusters = []
        for cand in unique_candidates:
            i, j = cand['pair']
            if i not in merged_indices and j not in merged_indices:
                new_clusters.append(clusters[i] + clusters[j])
                merged_indices.add(i); merged_indices.add(j)
        for i in range(num_clusters_before_merge):
            if i not in merged_indices: new_clusters.append(clusters[i])
        clusters = new_clusters
        logging.info(f"マージ後のクラスタ数: {len(clusters)}")
        
        temp_medoids = [find_medoid_torch(c, X) for c in tqdm(clusters, desc="HAC: 一時メドイド計算中", leave=False)]
        
        # ステップ2: k-medoids
        logging.info("ステップ2: k-medoids - クラスタの再構成を開始...")
        current_medoids = temp_medoids
        k_medoids_iter = 0
        while True:
            k_medoids_iter += 1
            medoid_data = X[[m for m in current_medoids if m is not None]]
            
            # ▼▼▼▼▼【最適化】k-medoidsの割り当て処理をベクトル化 ▼▼▼▼▼
            all_ssim_to_medoids = calculate_ssim_pairwise_batch(X, medoid_data)
            # ▲▲▲▲▲ 最適化 ▲▲▲▲▲
            
            assignments = torch.argmax(all_ssim_to_medoids, dim=1)
            new_clusters = [[] for _ in range(len(current_medoids))]
            for i in range(n_samples): new_clusters[assignments[i]].append(i)
            new_medoids = [find_medoid_torch(c, X) for c in new_clusters]
            
            if any(m is None for m in new_medoids):
                logging.warning("k-medoids中に空のクラスタが生成されました。前の状態を維持します。")
                medoids = current_medoids 
                clusters = [[] for _ in range(len(medoids))]
                for i in range(n_samples): clusters[assignments[i]].append(i)
                break
            
            if sorted(new_medoids) == sorted(current_medoids):
                logging.info(f"  k-medoidsが収束しました (反復 {k_medoids_iter} 回)。")
                medoids = new_medoids; clusters = new_clusters
                break
            else:
                current_medoids = new_medoids
        
        analyze_cluster_distribution(clusters, all_labels, all_time_stamps, base_labels, f"反復 {iteration} 完了時点")
        iteration += 1

    return clusters, medoids

# --- 7. 結果可視化関数 (変更なし) ---
def plot_final_clusters(medoids, clusters, X, lat_coords, lon_coords, time_stamps, all_labels, base_labels, save_dir):
    num_clusters = len(medoids)
    logging.info(f"最終結果をプロット中... 合計{num_clusters}クラスタ")
    n_cols = 5
    n_rows = (num_clusters + n_cols - 1) // n_cols if num_clusters > 0 else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5), subplot_kw={"projection": ccrs.PlateCarree()})
    axes = np.atleast_1d(axes).flatten()
    vmax = 3; vmin = -vmax
    cmap = plt.get_cmap('RdBu_r'); norm = Normalize(vmin=vmin, vmax=vmax)
    
    for i in range(num_clusters):
        ax = axes[i]
        if medoids[i] is None:
            ax.set_title(f'Cluster {i+1} (Empty)'); ax.axis("off"); continue
        medoid_pattern_2d = X[medoids[i]].cpu().numpy().reshape(len(lat_coords), len(lon_coords))
        cont = ax.contourf(lon_coords, lat_coords, medoid_pattern_2d, levels=np.linspace(vmin, vmax, 21), cmap=cmap, extend="both", norm=norm)
        ax.contour(lon_coords, lat_coords, medoid_pattern_2d, colors="k", linewidths=0.5, levels=np.arange(vmin, vmax+1, 1))
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="black", linewidth=0.5)
        ax.set_extent([120, 150, 20, 50], crs=ccrs.PlateCarree())
        cluster_indices = clusters[i]
        frequency = len(cluster_indices) / X.shape[0] * 100
        medoid_date = pd.to_datetime(str(time_stamps[medoids[i]])).strftime('%Y-%m-%d')
        dominant_label_str = ""
        if all_labels:
            base_label_counts = Counter(label for label in [all_labels[j] for j in cluster_indices] if label in base_labels)
            if base_label_counts:
                dominant_label = base_label_counts.most_common(1)[0][0]
                dominant_label_str = f"Dom. Label: {dominant_label}"
        ax.set_title(f'Cluster {i+1} (N={len(cluster_indices)}, Freq:{frequency:.1f}%)\n{dominant_label_str}\nMedoid Date: {medoid_date}', fontsize=8)

    for i in range(num_clusters, len(axes)): axes[i].axis("off")
    fig.suptitle(f'Final Synoptic Patterns (Medoids) - TH_merge={TH_MERGE}', fontsize=16)
    fig.tight_layout(rect=[0, 0, 0.9, 0.95])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(cont, cax=cbar_ax, label="Standardized Pressure Anomaly")
    save_path = os.path.join(save_dir, f'final_clusters_th{TH_MERGE}.png')
    plt.savefig(save_path)
    plt.close()
    logging.info(f"最終クラスタの画像を保存: {save_path}")

# --- 8. メイン実行ブロック ---
if __name__ == '__main__':
    logging.info("--- 2段階クラスタリングプログラム開始 ---")
    
    X_data, lat, lon, ts, labels = load_and_prepare_data(DATA_FILE, START_DATE, END_DATE, device)
    
    if X_data is not None:
        start_time = time.time()
        
        final_clusters, final_medoids = two_stage_clustering(X_data, TH_MERGE, labels, ts, BASE_LABELS, CALCULATION_BATCH_SIZE)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"\n{'='*10} クラスタリング完了 {'='*10}")
        logging.info(f"総計算時間: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分)")
        logging.info(f"最終的なクラスタ数: {len(final_clusters)}")
        
        analyze_cluster_distribution(final_clusters, labels, ts, BASE_LABELS, "最終結果")
        
        results = {'medoid_indices': final_medoids, 'clusters': final_clusters, 'th_merge': TH_MERGE, 'start_date': START_DATE, 'end_date': END_DATE, 'lat': lat, 'lon': lon}
        result_file = os.path.join(RESULT_DIR, f'clustering_result_th{TH_MERGE}.pt')
        torch.save(results, result_file)
        logging.info(f"クラスタリング結果を保存しました: {result_file}")

        plot_final_clusters(final_medoids, final_clusters, X_data, lat, lon, ts, labels, BASE_LABELS, RESULT_DIR)

    logging.info("--- プログラム終了 ---")