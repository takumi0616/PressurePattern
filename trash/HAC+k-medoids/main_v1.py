import os
import logging
import torch
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize
from tqdm import tqdm
import time
from collections import Counter

# --- 1. 設定項目 ---

# ファイルパスとディレクトリ
DATA_FILE = './prmsl_era5_all_data_seasonal_large.nc'
RESULT_DIR = './two_stage_clustering_results_v1_045'

# データ期間 (データセットに合わせて調整してください)
START_DATE = '1991-01-01'
END_DATE = '2000-12-31'

# 2段階クラスタリングアルゴリズムのパラメータ
TH_MERGE = 0.45

# 安定化定数
C1 = 1e-8
C2 = 1e-8

# 基本ラベルリスト
BASE_LABELS = [
    '1', '2A', '2B', '2C', '2D', '3A', '3B', '3C', '3D',
    '4A', '4B', '5', '6A', '6B', '6C'
]

# パフォーマンス改善のためのバッチサイズ設定
CALCULATION_BATCH_SIZE = 64

# --- 2. 初期設定 ---
os.makedirs(RESULT_DIR, exist_ok=True)
log_path = os.path.join(RESULT_DIR, 'clustering_analysis.log')
if os.path.exists(log_path): os.remove(log_path)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_path), logging.StreamHandler()])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f"使用デバイス: {device.upper()}")

# --- 3. データ準備関数 (元データも返すように修正) ---
def load_and_prepare_data(filepath, start_date, end_date, device):
    logging.info(f"データファイル '{filepath}' を読み込んでいます...")
    ds = xr.open_dataset(filepath)
    data_period = ds.sel(valid_time=slice(start_date, end_date))
    
    labels = None
    if 'label' in data_period.variables:
        label_values = data_period['label'].values
        labels = [l.decode('utf-8') if isinstance(l, bytes) else str(l) for l in label_values]
        logging.info("ラベルデータを読み込みました。")

    msl_data = data_period['msl']
    n_samples, d_lat, d_lon = msl_data.shape
    
    # 元のデータをnumpy配列として保持 (hPa単位)
    original_data_numpy = msl_data.values / 100.0 # PaからhPaへ変換

    # 正規化処理
    flattened_data = original_data_numpy.reshape(n_samples, d_lat * d_lon)
    mean = np.mean(flattened_data, axis=0, keepdims=True)
    std = np.std(flattened_data, axis=0, keepdims=True)
    std[std == 0] = 1
    normalized_data = (flattened_data - mean) / std

    # 正規化データをPyTorchテンソルに
    X_normalized = torch.from_numpy(normalized_data).to(device, dtype=torch.float32)

    lat_coords = data_period['latitude'].values
    lon_coords = data_period['longitude'].values
    time_stamps = data_period.valid_time.values
    
    logging.info(f"データ期間: {START_DATE} から {END_DATE}")
    logging.info(f"データ形状: {X_normalized.shape[0]}サンプル, {X_normalized.shape[1]}次元 ({d_lat}x{d_lon})")
    
    return X_normalized, original_data_numpy, lat_coords, lon_coords, time_stamps, labels

# --- 4. SSIMおよびクラスタリング関連関数 (最適化版) ---

def calculate_ssim_pairwise_batch(x_batch, y_all, c1=C1, c2=C2):
    B, D = x_batch.shape
    N, D = y_all.shape
    mu_x = torch.mean(x_batch, dim=1).unsqueeze(1)
    mu_y = torch.mean(y_all, dim=1).unsqueeze(0)
    mu_x_prime = (mu_x + mu_y) / 2
    mu_y_prime = mu_x_prime + torch.abs(mu_x - mu_y)
    sigma_x_sq = torch.var(x_batch, dim=1, unbiased=False).unsqueeze(1)
    sigma_y_sq = torch.var(y_all, dim=1, unbiased=False).unsqueeze(0)
    x_expanded = x_batch.unsqueeze(1)
    y_expanded = y_all.unsqueeze(0)
    cov_xy = torch.mean((x_expanded - mu_x.unsqueeze(2)) * (y_expanded - mu_y.unsqueeze(2)), dim=2)
    numerator = (2 * mu_x_prime * mu_y_prime + c1) * (2 * cov_xy + c2)
    denominator = (mu_x_prime**2 + mu_y_prime**2 + c1) * (sigma_x_sq + sigma_y_sq + c2)
    return numerator / denominator

def find_medoid_torch(cluster_indices, X, batch_size):
    if not cluster_indices: return None
    if len(cluster_indices) == 1: return cluster_indices[0]
    
    cluster_data = X[cluster_indices]
    num_in_cluster = len(cluster_indices)
    
    ssim_scores_matrix = torch.zeros((num_in_cluster, num_in_cluster), device=device)
    # クラスタ内でのSSIM総当たり計算もバッチ処理で行う
    for i in range(0, num_in_cluster, batch_size):
        end_idx = min(i + batch_size, num_in_cluster)
        sub_batch = cluster_data[i:end_idx]
        ssim_scores_matrix[i:end_idx, :] = calculate_ssim_pairwise_batch(sub_batch, cluster_data)

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

def calculate_and_log_macro_recall(clusters, all_labels, base_labels, iteration_name):
    """マクロ平均再現率を計算し、途中過程を含めてログに出力する"""
    logging.info(f"\n--- {iteration_name} マクロ平均再現率 (精度) 計算 ---")
    
    # 1. 混同行列の作成 (行: 真のラベル, 列: 予測クラスタ)
    cluster_names = [f'Cluster_{i+1}' for i in range(len(clusters))]
    confusion_matrix = pd.DataFrame(0, index=base_labels, columns=cluster_names)
    
    for i, cluster_indices in enumerate(clusters):
        cluster_name = cluster_names[i]
        cluster_true_labels = [all_labels[j] for j in cluster_indices if all_labels[j] in base_labels]
        label_counts = Counter(cluster_true_labels)
        for label, count in label_counts.items():
            confusion_matrix.loc[label, cluster_name] = count
            
    logging.info("【混同行列 (Confusion Matrix)】")
    logging.info(f"\n{confusion_matrix.to_string()}")

    # 2. 各ラベルの再現率 (Recall) を計算
    logging.info("\n【各ラベルの再現率 (Recall) 計算過程】")
    logging.info("Recall = TP / (TP + FN)  (そのラベルのデータ全体のうち、どれだけを代表クラスタに集められたか)")
    
    recalls = []
    # 各真のラベルに対して、最も多くのデータを含んだクラスタをそのラベルの「代表クラスタ」とする
    representative_clusters = confusion_matrix.idxmax(axis=1)
    
    for label in base_labels:
        total_positives = confusion_matrix.loc[label].sum() # TP + FN (そのラベルの全データ数)
        
        if total_positives == 0:
            logging.info(f" - ラベル '{label}': データなし。Recall = N/A")
            continue
            
        # True Positives (TP)
        pred_cluster_for_label = representative_clusters[label]
        true_positives = confusion_matrix.loc[label, pred_cluster_for_label]
        
        recall = true_positives / total_positives
        recalls.append(recall)
        
        logging.info(f" - ラベル '{label}':")
        logging.info(f"   - 代表クラスタ: {pred_cluster_for_label} (このクラスタが'{label}'の予測を担当)")
        logging.info(f"   - TP (正しく代表クラスタに入った数): {true_positives}")
        logging.info(f"   - TP+FN (ラベル'{label}'の総数): {total_positives}")
        logging.info(f"   - Recall = {true_positives} / {total_positives} = {recall:.4f}")

    # 3. マクロ平均再現率の計算
    if not recalls:
        macro_recall = 0.0
        logging.warning("有効なラベルがなかったため、マクロ平均再現率を計算できませんでした。")
    else:
        macro_recall = np.mean(recalls)

    logging.info("\n【最終スコア】")
    logging.info(f"マクロ平均再現率 (Macro-Average Recall) = {macro_recall:.4f}")
    logging.info(f"--- {iteration_name} 精度計算終了 ---\n")
    return macro_recall

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
        
        ssim_matrix = torch.zeros((num_clusters_before_merge, num_clusters_before_merge), device=device)
        for i in tqdm(range(0, num_clusters_before_merge, batch_size), desc="HAC: SSIM計算中", leave=False):
            end_idx = min(i + batch_size, num_clusters_before_merge)
            x_batch = medoid_data[i:end_idx]
            ssim_scores_batch = calculate_ssim_pairwise_batch(x_batch, medoid_data)
            ssim_matrix[i:end_idx, :] = ssim_scores_batch
        
        ssim_matrix.fill_diagonal_(-torch.inf)
        merge_candidates = (torch.triu(ssim_matrix) > th_merge).nonzero(as_tuple=False)
        
        if merge_candidates.shape[0] == 0:
            logging.info("マージ可能なクラスタペアが見つかりませんでした。クラスタリングを終了します。")
            break

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
        
        temp_medoids = [find_medoid_torch(c, X, batch_size) for c in tqdm(clusters, desc="HAC: 一時メドイド計算中", leave=False)]
        
        # ステップ2: k-medoids
        logging.info("ステップ2: k-medoids - クラスタの再構成を開始...")
        current_medoids = temp_medoids
        k_medoids_iter = 0
        while True:
            k_medoids_iter += 1
            medoid_data = X[[m for m in current_medoids if m is not None]]
            all_ssim_to_medoids = torch.zeros((n_samples, len(current_medoids)), device=device)
            for i in tqdm(range(0, n_samples, batch_size), desc="k-medoids: 割り当て中", leave=False):
                end_idx = min(i + batch_size, n_samples)
                x_batch = X[i:end_idx]
                ssim_scores_batch = calculate_ssim_pairwise_batch(x_batch, medoid_data)
                all_ssim_to_medoids[i:end_idx, :] = ssim_scores_batch
            
            assignments = torch.argmax(all_ssim_to_medoids, dim=1)
            new_clusters = [[] for _ in range(len(current_medoids))]
            for i in range(n_samples): new_clusters[assignments[i]].append(i)
            new_medoids = [find_medoid_torch(c, X, batch_size) for c in new_clusters]
            
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

# --- 7. 新機能：可視化関数 ---
def plot_final_distribution_summary(clusters, all_labels, all_time_stamps, base_labels, save_dir):
    """最終的なクラスタのラベル分布と月別分布をヒートマップで可視化する"""
    logging.info("最終的な分布の可視化画像を作成中...")
    num_clusters = len(clusters)
    cluster_names = [f'Cluster {i+1}' for i in range(num_clusters)]

    # ラベル分布行列
    label_dist_matrix = pd.DataFrame(0, index=cluster_names, columns=base_labels)
    # 月別分布行列
    month_dist_matrix = pd.DataFrame(0, index=cluster_names, columns=range(1, 13))

    for i, cluster_indices in enumerate(clusters):
        cluster_name = cluster_names[i]
        # ラベル
        if all_labels:
            cluster_true_labels = [all_labels[j] for j in cluster_indices if all_labels[j] in base_labels]
            label_counts = Counter(cluster_true_labels)
            for label, count in label_counts.items():
                label_dist_matrix.loc[cluster_name, label] = count
        # 月
        if all_time_stamps is not None:
            cluster_time_stamps = np.array(all_time_stamps)[cluster_indices]
            months = pd.to_datetime(cluster_time_stamps).month
            month_counts = Counter(months)
            for month, count in month_counts.items():
                month_dist_matrix.loc[cluster_name, month] = count
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 14))
    fig.suptitle('Final Cluster Distribution Summary', fontsize=20)

    # ラベル分布ヒートマップ
    sns.heatmap(label_dist_matrix, ax=axes[0], annot=True, fmt='d', cmap='viridis', linewidths=.5)
    axes[0].set_title('Label Distribution per Cluster', fontsize=16)
    axes[0].set_ylabel('Cluster')
    axes[0].set_xlabel('True Label')

    # 月別分布ヒートマップ
    sns.heatmap(month_dist_matrix, ax=axes[1], annot=True, fmt='d', cmap='inferno', linewidths=.5)
    axes[1].set_title('Monthly Distribution per Cluster', fontsize=16)
    axes[1].set_ylabel('Cluster')
    axes[1].set_xlabel('Month')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(save_dir, 'final_distribution_summary.png')
    plt.savefig(save_path)
    plt.close()
    logging.info(f"分布の可視化画像を保存: {save_path}")

def save_daily_maps_for_clusters(clusters, original_data, lat, lon, ts, labels, base_labels, save_dir):
    """各クラスタの日別気圧配置図をフォルダ分けして保存する"""
    logging.info("クラスタ別の日次気圧配置図を保存中...")
    maps_main_dir = os.path.join(save_dir, 'daily_maps')
    os.makedirs(maps_main_dir, exist_ok=True)
    
    # 各クラスタの主要ラベルを特定
    dominant_labels = {}
    for i, cluster_indices in enumerate(clusters):
        if labels:
            base_label_counts = Counter(l for l in [labels[j] for j in cluster_indices] if l in base_labels)
            if base_label_counts:
                dominant_labels[i] = base_label_counts.most_common(1)[0][0]
            else:
                dominant_labels[i] = "Unknown"
        else:
            dominant_labels[i] = ""

    for i, cluster_indices in enumerate(tqdm(clusters, desc="Saving daily maps")):
        dom_label = dominant_labels.get(i, "")
        cluster_dir = os.path.join(maps_main_dir, f'cluster_{i+1:02d}_dom_label_{dom_label}')
        os.makedirs(cluster_dir, exist_ok=True)
        
        for data_idx in cluster_indices:
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            
            pressure_map = original_data[data_idx].reshape(len(lat), len(lon))
            
            # ▼▼▼▼▼ 描画スタイルを final_clusters 形式に統一 ▼▼▼▼▼
            # 1. 発散型カラーマップRdBu_rを使用
            cmap = plt.get_cmap('RdBu_r')

            # 2. 標準大気圧(1013.25hPa)を中心に色を割り当て
            center_pressure = 1013.25
            pressure_span = 35  # 中心から±35hPaの範囲を描画
            vmin, vmax = center_pressure - pressure_span, center_pressure + pressure_span
            norm = Normalize(vmin=vmin, vmax=vmax)

            # 3. 塗りつぶしの等高線レベルを細かく設定し、滑らかな描画に
            fill_levels = np.linspace(vmin, vmax, 21)
            
            # 4. 見やすいように4hPaごとに線を描画
            line_levels = np.arange(960, 1061, 4)
            
            # 塗りつぶし等高線
            cont = ax.contourf(lon, lat, pressure_map, levels=fill_levels, cmap=cmap, norm=norm, extend='both')
            cbar = fig.colorbar(cont, ax=ax, orientation='vertical', pad=0.05, aspect=20)
            cbar.set_label('Sea Level Pressure (hPa)')
            
            # 線状の等高線
            ax.contour(lon, lat, pressure_map, levels=line_levels, colors='k', linewidths=0.5)
            # ▲▲▲▲▲ スタイル統一ここまで ▲▲▲▲▲

            ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="black", linewidth=0.5)
            ax.set_extent([120, 150, 20, 50], crs=ccrs.PlateCarree())
            
            date_str = pd.to_datetime(str(ts[data_idx])).strftime('%Y-%m-%d')
            true_label = labels[data_idx] if labels else "N/A"
            ax.set_title(f"Date: {date_str}\nTrue Label: {true_label}")
            
            plt.tight_layout()
            save_path = os.path.join(cluster_dir, f"{date_str}_label_{true_label.replace('/', '_')}.png")
            plt.savefig(save_path)
            plt.close(fig)

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
    
    X_normalized, X_original, lat, lon, ts, labels = load_and_prepare_data(DATA_FILE, START_DATE, END_DATE, device)
    
    if X_normalized is not None:
        start_time = time.time()
        
        # クラスタリング実行
        final_clusters, final_medoids = two_stage_clustering(X_normalized, TH_MERGE, labels, ts, BASE_LABELS, CALCULATION_BATCH_SIZE)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"\n{'='*10} クラスタリング完了 {'='*10}")
        logging.info(f"総計算時間: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分)")
        logging.info(f"最終的なクラスタ数: {len(final_clusters)}")
        
        # 最終結果の詳細分析と評価
        analyze_cluster_distribution(final_clusters, labels, ts, BASE_LABELS, "最終結果")
        if labels:
            calculate_and_log_macro_recall(final_clusters, labels, BASE_LABELS, "最終結果")
        
        # 最終結果の可視化と保存
        results = {'medoid_indices': final_medoids, 'clusters': final_clusters}
        torch.save(results, os.path.join(RESULT_DIR, f'clustering_result_th{TH_MERGE}.pt'))
        logging.info(f"クラスタリング結果を保存しました。")
        
        # 全ての可視化関数を呼び出し
        if labels and ts is not None:
            plot_final_distribution_summary(final_clusters, labels, ts, BASE_LABELS, RESULT_DIR)
        
        if X_original is not None:
            save_daily_maps_for_clusters(final_clusters, X_original, lat, lon, ts, labels, BASE_LABELS, RESULT_DIR)

    plot_final_clusters(final_medoids, final_clusters, X_normalized, lat, lon, ts, labels, BASE_LABELS, RESULT_DIR)

    logging.info("--- プログラム終了 ---")