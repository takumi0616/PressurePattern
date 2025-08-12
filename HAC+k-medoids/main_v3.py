# -*- coding: utf-8 -*-
# 再現性強化版 main_v3.py

import os
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import random

DATA_FILE = './prmsl_era5_all_data_seasonal_large.nc'
RESULT_DIR = './s1_score_clustering_results'

START_DATE = '1991-01-01'
END_DATE = '2000-12-31'

# S1は小さいほど類似
TH_MERGE = 89
EPSILON = 1e-9

BASE_LABELS = [
    '1', '2A', '2B', '2C', '2D', '3A', '3B', '3C', '3D',
    '4A', '4B', '5', '6A', '6B', '6C'
]

CALCULATION_BATCH_SIZE = 4

# タイブレーク用
TIE_EPS = 1e-12

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
try:
    torch.set_num_threads(1)
except Exception:
    pass

os.makedirs(RESULT_DIR, exist_ok=True)
log_path = os.path.join(RESULT_DIR, 'clustering_analysis.log')
if os.path.exists(log_path):
    os.remove(log_path)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f"使用デバイス: {device.upper()}")

# --- 3. データ準備 ---
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
    
    original_data_numpy = msl_data.values / 100.0
    logging.info("各日のデータから空間平均値を減算し、空間偏差データを生成しています...")
    spatial_mean = np.mean(original_data_numpy, axis=(1, 2), keepdims=True)
    spatial_anomaly_data = original_data_numpy - spatial_mean

    flattened_data = spatial_anomaly_data.reshape(n_samples, d_lat * d_lon)
    mean = np.mean(flattened_data, axis=0, keepdims=True)
    std = np.std(flattened_data, axis=0, keepdims=True)
    std[std == 0] = 1
    normalized_data = (flattened_data - mean) / std

    X_normalized = torch.from_numpy(normalized_data).to(device, dtype=torch.float32)

    lat_coords = data_period['latitude'].values
    lon_coords = data_period['longitude'].values
    time_stamps = data_period.valid_time.values
    
    logging.info(f"データ期間: {START_DATE} から {END_DATE}")
    logging.info(f"データ形状: {X_normalized.shape[0]}サンプル, {X_normalized.shape[1]}次元 ({d_lat}x{d_lon})")
    
    return X_normalized, original_data_numpy, spatial_anomaly_data, lat_coords, lon_coords, d_lat, d_lon, time_stamps, labels

# --- 4. S1スコア ---
def calculate_s1_pairwise_batch(x_batch, y_all, d_lat, d_lon, epsilon=EPSILON):
    B, D = x_batch.shape
    N, _ = y_all.shape
    x_maps = x_batch.view(B, 1, d_lat, d_lon)
    y_maps = y_all.view(1, N, d_lat, d_lon)
    diff_maps = y_maps - x_maps  # (B, N, d_lat, d_lon)

    grad_x_D = (diff_maps[..., :, 1:] - diff_maps[..., :, :-1])[..., :-1, :]
    grad_y_D = (diff_maps[..., 1:, :] - diff_maps[..., :-1, :])[..., :, :-1]
    numerator_term = torch.abs(grad_x_D) + torch.abs(grad_y_D)
    numerator = torch.mean(numerator_term, dim=(-2, -1))  # (B, N)

    grad_x_y = (y_maps[..., :, 1:] - y_maps[..., :, :-1])[..., :-1, :]
    grad_y_y = (y_maps[..., 1:, :] - y_maps[..., :-1, :])[..., :, :-1]
    grad_x_x = (x_maps[..., :, 1:] - x_maps[..., :, :-1])[..., :-1, :]
    grad_y_x = (x_maps[..., 1:, :] - x_maps[..., :-1, :])[..., :, :-1]
    max_grad_x = torch.max(torch.abs(grad_x_y), torch.abs(grad_x_x))
    max_grad_y = torch.max(torch.abs(grad_y_y), torch.abs(grad_y_x))
    denominator_term = max_grad_x + max_grad_y
    denominator = torch.mean(denominator_term, dim=(-2, -1))  # (B, N)

    s1_score = 100 * (numerator / (denominator + epsilon))
    return s1_score  # 低いほど類似

def find_medoid_torch(cluster_indices, X, batch_size, d_lat, d_lon):
    if not cluster_indices:
        return None
    if len(cluster_indices) == 1:
        return cluster_indices[0]
    
    cluster_data = X[cluster_indices]
    num_in_cluster = len(cluster_indices)
    
    s1_scores_matrix = torch.zeros((num_in_cluster, num_in_cluster), device=device)
    for i in range(0, num_in_cluster, batch_size):
        end_idx = min(i + batch_size, num_in_cluster)
        sub_batch = cluster_data[i:end_idx]
        s1_scores_matrix[i:end_idx, :] = calculate_s1_pairwise_batch(sub_batch, cluster_data, d_lat, d_lon)

    s1_sum = torch.sum(s1_scores_matrix, dim=1)
    # タイブレーク: 先頭（小さいインデックス）優先（argmin用）
    tie_bias = TIE_EPS * torch.arange(num_in_cluster, device=device, dtype=s1_sum.dtype)
    s1_sum = s1_sum + tie_bias

    medoid_pos_in_cluster = torch.argmin(s1_sum)
    return cluster_indices[medoid_pos_in_cluster]

# --- 5. 分布/評価（v2と同様） ---
def analyze_cluster_distribution(clusters, all_labels, all_time_stamps, base_labels, iteration_name):
    logging.info(f"\n--- {iteration_name} クラスタ分布分析 ---")
    for i, cluster_indices in enumerate(clusters):
        num_samples_in_cluster = len(cluster_indices)
        if num_samples_in_cluster == 0:
            continue
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

def evaluate_clusters_majority_recall(clusters, all_labels, base_labels, iteration_name):
    logging.info(f"\n--- {iteration_name} 評価 (クラスタ多数決ベースのマクロ平均再現率) ---")
    if not all_labels:
        logging.warning("真のラベルがないため評価をスキップします。")
        return None
    n_samples = len(all_labels)
    num_clusters = len(clusters)
    cluster_names = [f'Cluster_{i+1}' for i in range(num_clusters)]
    confusion_matrix = pd.DataFrame(0, index=base_labels, columns=cluster_names, dtype=int)
    for i, idxs in enumerate(clusters):
        if not idxs:
            continue
        cluster_true_labels = [all_labels[j] for j in idxs if all_labels[j] in base_labels]
        label_counts = Counter(cluster_true_labels)
        for label, count in label_counts.items():
            confusion_matrix.loc[label, cluster_names[i]] = count
    present_labels = [lbl for lbl in base_labels if confusion_matrix.loc[lbl].sum() > 0]
    if len(present_labels) == 0:
        logging.warning("基本ラベルに該当するデータがありません。評価をスキップします。")
        return None
    logging.info("【混同行列 (基本ラベルのみ)】")
    logging.info(f"\n{confusion_matrix.loc[present_labels, :].to_string()}")
    cluster_majority = {}
    logging.info("\n【各クラスタの多数決（代表ラベル）】")
    total_base_count = int(confusion_matrix.values.sum())
    micro_correct_sum = 0
    for k in range(num_clusters):
        col = cluster_names[k]
        col_counts = confusion_matrix[col]
        col_sum = int(col_counts.sum())
        if col_sum == 0:
            cluster_majority[k] = None
            logging.info(f" - {col:<12}: 代表ラベル=None（基本ラベルの出現なし）")
            continue
        top_label = col_counts.idxmax()
        top_count = int(col_counts.max())
        micro_correct_sum += top_count
        share = top_count / col_sum if col_sum > 0 else 0.0
        cluster_majority[k] = top_label
        top3 = col_counts.sort_values(ascending=False)[:3]
        top3_str = ", ".join([f"{lbl}:{int(cnt)}" for lbl, cnt in top3.items()])
        logging.info(f" - {col:<12}: 代表={top_label:<3}  件数={top_count:4d}  シェア={share:5.2f}  | 上位: {top3_str}")
    metrics = {}
    per_label = {}
    logging.info("\n【各ラベルの再現率 (クラスタ多数決ベース)】")
    for lbl in present_labels:
        row_sum = int(confusion_matrix.loc[lbl, :].sum())
        cols_for_lbl = [cluster_names[k] for k in range(num_clusters) if cluster_majority.get(k, None) == lbl]
        num_correct = int(confusion_matrix.loc[lbl, cols_for_lbl].sum()) if cols_for_lbl else 0
        recall = num_correct / row_sum if row_sum > 0 else 0.0
        per_label[lbl] = {'Total': row_sum, 'CorrectInMajorityClusters': num_correct, 'Recall': recall, 'MajorityClusters': cols_for_lbl}
        logging.info(f" - {lbl:<3}: N={row_sum:4d}, Correct(代表クラスタ群内)={num_correct:4d}, Recall={recall:.4f}, 代表クラスタ={cols_for_lbl if cols_for_lbl else 'なし'}")
    macro_recall = float(np.mean([per_label[l]['Recall'] for l in present_labels]))
    metrics['MacroRecall_majority'] = macro_recall
    logging.info("\n【集計】")
    logging.info(f"Macro Recall (多数決ベース) = {macro_recall:.4f}")
    micro_accuracy = micro_correct_sum / total_base_count if total_base_count > 0 else 0.0
    metrics['MicroAccuracy_majority'] = micro_accuracy
    logging.info(f"Micro Accuracy (多数決, base_labels対象) = {micro_accuracy:.4f}")
    sample_to_cluster = [-1] * n_samples
    for ci, idxs in enumerate(clusters):
        for j in idxs:
            sample_to_cluster[j] = ci
    y_true, y_pred = [], []
    for j in range(n_samples):
        true_lbl = all_labels[j]
        if true_lbl not in present_labels:
            continue
        ci = sample_to_cluster[j]
        if ci < 0:
            continue
        pred_lbl = cluster_majority.get(ci, None) or "Unassigned"
        y_true.append(true_lbl)
        y_pred.append(pred_lbl)
    if len(y_true) > 0:
        uniq_true = {l: i for i, l in enumerate(sorted(set(y_true)))}
        uniq_pred = {l: i for i, l in enumerate(sorted(set(y_pred)))}
        y_true_idx = [uniq_true[l] for l in y_true]
        y_pred_idx = [uniq_pred[l] for l in y_pred]
        ari = adjusted_rand_score(y_true_idx, y_pred_idx)
        nmi = normalized_mutual_info_score(y_true_idx, y_pred_idx)
        metrics['ARI_majority'] = float(ari)
        metrics['NMI_majority'] = float(nmi)
        logging.info(f"Adjusted Rand Index (多数決) = {ari:.4f}")
        logging.info(f"Normalized Mutual Info (多数決) = {nmi:.4f}")
    else:
        logging.info("base_labels に該当する評価対象サンプルがないため、ARI/NMI は計算しません。")
    logging.info(f"--- {iteration_name} 評価終了 ---\n")
    return metrics

# --- 6. 2段階クラスタリング（MNNマージ + k-medoids） ---
def two_stage_clustering(X, th_merge, all_labels, all_time_stamps, base_labels, batch_size, d_lat, d_lon):
    n_samples = X.shape[0]
    logging.info("2段階クラスタリングを開始します (類似度指標: S1スコア)...")
    clusters = [[i] for i in range(n_samples)]
    medoids = list(range(n_samples))
    
    iteration = 1
    while True:
        logging.info(f"\n{'='*10} 反復: {iteration} {'='*10}")
        logging.info(f"現在のクラスタ数: {len(clusters)}")

        logging.info("ステップ1: HAC - 相互最近傍(MNN)条件でのマージを開始...")
        num_clusters_before_merge = len(clusters)
        medoid_data = X[medoids]

        s1_matrix = torch.zeros((num_clusters_before_merge, num_clusters_before_merge), device=device)
        for i in tqdm(range(0, num_clusters_before_merge, batch_size), desc="HAC: S1スコア計算中", leave=False):
            end_idx = min(i + batch_size, num_clusters_before_merge)
            x_batch = medoid_data[i:end_idx]
            s1_scores_batch = calculate_s1_pairwise_batch(x_batch, medoid_data, d_lat, d_lon)
            s1_matrix[i:end_idx, :] = s1_scores_batch
        
        s1_matrix.fill_diagonal_(torch.inf)

        finite_mask = torch.isfinite(s1_matrix)
        s1_values_non_diag = s1_matrix[finite_mask]
        if s1_values_non_diag.numel() > 0:
            logging.info(
                f"S1スコア統計 - 最小: {s1_values_non_diag.min().item():.2f}, "
                f"最大: {s1_values_non_diag.max().item():.2f}, "
                f"平均: {s1_values_non_diag.mean().item():.2f}, "
                f"中央値: {s1_values_non_diag.median().item():.2f}"
            )

        upper_tri_mask = torch.triu(torch.ones_like(s1_matrix, dtype=torch.bool), diagonal=1)
        num_pairs_below_threshold = ((s1_matrix < th_merge) & upper_tri_mask).sum().item()
        logging.info(f"しきい値 {th_merge} 未満の(上三角)ペア数: {num_pairs_below_threshold}")

        nn = torch.argmin(s1_matrix, dim=1)  # 最小S1の列index（各行）

        mnn_pairs = []
        for i in range(num_clusters_before_merge):
            j = nn[i].item()
            if j < 0 or j >= num_clusters_before_merge:
                continue
            if nn[j].item() == i and i < j:
                s1_ij = s1_matrix[i, j].item()
                if s1_ij < th_merge:
                    mnn_pairs.append((i, j, s1_ij))

        # 安定ソート: S1昇順、i昇順、j昇順
        mnn_pairs.sort(key=lambda t: (t[2], t[0], t[1]))

        logging.info(f"MNN条件を満たすマージ候補ペア数: {len(mnn_pairs)}")
        if len(mnn_pairs) > 0:
            s1_list = [p[2] for p in mnn_pairs]
            logging.info(
                f"MNNペアS1統計 - 最小: {min(s1_list):.2f}, 最大: {max(s1_list):.2f}, 平均: {np.mean(s1_list):.2f}, 中央値: {np.median(s1_list):.2f}"
            )

        if len(mnn_pairs) == 0:
            logging.info("MNN条件でマージ可能なクラスタペアが見つかりません。クラスタリングを終了します。")
            break

        merged_indices = set()
        new_clusters = []
        for i, j, _ in mnn_pairs:
            if i not in merged_indices and j not in merged_indices:
                new_clusters.append(clusters[i] + clusters[j])
                merged_indices.add(i); merged_indices.add(j)
        for idx in range(num_clusters_before_merge):
            if idx not in merged_indices:
                new_clusters.append(clusters[idx])

        clusters = new_clusters
        logging.info(f"マージ後のクラスタ数: {len(clusters)}")

        temp_medoids = [find_medoid_torch(c, X, batch_size, d_lat, d_lon)
                        for c in tqdm(clusters, desc="HAC: 一時メドイド計算中", leave=False)]

        logging.info("ステップ2: k-medoids - クラスタの再構成を開始...")
        current_medoids = temp_medoids
        k_medoids_iter = 0
        while True:
            k_medoids_iter += 1
            valid_idx_m = [idx for idx, m in enumerate(current_medoids) if m is not None]
            valid_medoids = [current_medoids[idx] for idx in valid_idx_m]
            if len(valid_medoids) == 0:
                logging.warning("有効なメドイドがありません。k-medoidsを終了します。")
                medoids = current_medoids
                break

            medoid_data = X[valid_medoids]
            all_s1_to_medoids = torch.zeros((n_samples, len(valid_medoids)), device=device)
            for i in tqdm(range(0, n_samples, batch_size), desc="k-medoids: 割り当て中", leave=False):
                end_idx = min(i + batch_size, n_samples)
                x_batch = X[i:end_idx]
                s1_scores_batch = calculate_s1_pairwise_batch(x_batch, medoid_data, d_lat, d_lon)
                # タイブレーク: 列（メドイドindex）が小さい方を優先（argmin用）
                tie_vec = TIE_EPS * torch.arange(s1_scores_batch.shape[1], device=device, dtype=s1_scores_batch.dtype).view(1, -1)
                s1_scores_batch = s1_scores_batch + tie_vec
                all_s1_to_medoids[i:end_idx, :] = s1_scores_batch
            
            assignments_valid = torch.argmin(all_s1_to_medoids, dim=1)
            new_clusters = [[] for _ in range(len(current_medoids))]
            for i in range(n_samples):
                col = assignments_valid[i].item()
                target_cluster_idx = valid_idx_m[col]
                new_clusters[target_cluster_idx].append(i)

            new_medoids = [find_medoid_torch(c, X, batch_size, d_lat, d_lon) for c in new_clusters]
            
            if any(m is None for m in new_medoids):
                logging.warning("k-medoids中に空のクラスタが生成されました。前の状態を維持して再割り当てします。")
                medoids = current_medoids
                valid_idx_m2 = [idx for idx, m in enumerate(medoids) if m is not None]
                valid_medoids2 = [medoids[idx] for idx in valid_idx_m2]
                if len(valid_medoids2) == 0:
                    logging.error("再割り当てに使用できる有効メドイドがありません。")
                    clusters = [[]]
                    break
                medoid_data2 = X[valid_medoids2]
                all_s1_to_medoids_recalc = torch.zeros((n_samples, len(valid_medoids2)), device=device)
                for i in tqdm(range(0, n_samples, batch_size), desc="k-medoids: 割り当て再計算中", leave=False):
                    end_idx = min(i + batch_size, n_samples)
                    x_batch = X[i:end_idx]
                    s1_scores_batch_recalc = calculate_s1_pairwise_batch(x_batch, medoid_data2, d_lat, d_lon)
                    tie_vec2 = TIE_EPS * torch.arange(s1_scores_batch_recalc.shape[1], device=device, dtype=s1_scores_batch_recalc.dtype).view(1, -1)
                    s1_scores_batch_recalc = s1_scores_batch_recalc + tie_vec2
                    all_s1_to_medoids_recalc[i:end_idx, :] = s1_scores_batch_recalc
                assignments_valid2 = torch.argmin(all_s1_to_medoids_recalc, dim=1)
                clusters = [[] for _ in range(len(medoids))]
                for i in range(n_samples):
                    col = assignments_valid2[i].item()
                    target_cluster_idx = valid_idx_m2[col]
                    clusters[target_cluster_idx].append(i)
                break
            
            if sorted([m for m in new_medoids if m is not None]) == sorted([m for m in current_medoids if m is not None]):
                logging.info(f"  k-medoidsが収束しました (反復 {k_medoids_iter} 回)。")
                medoids = new_medoids
                clusters = new_clusters
                break
            else:
                current_medoids = new_medoids
        
        iteration += 1

    return clusters, medoids

# --- 7. 可視化（v2と同様） ---
def plot_final_distribution_summary(clusters, all_labels, all_time_stamps, base_labels, save_dir):
    logging.info("最終的な分布の可視化画像を作成中...")
    num_clusters = len(clusters)
    cluster_names = [f'Cluster {i+1}' for i in range(num_clusters)]
    label_dist_matrix = pd.DataFrame(0, index=cluster_names, columns=base_labels)
    month_dist_matrix = pd.DataFrame(0, index=cluster_names, columns=range(1, 13))
    for i, cluster_indices in enumerate(clusters):
        cluster_name = cluster_names[i]
        if all_labels:
            cluster_true_labels = [all_labels[j] for j in cluster_indices if all_labels[j] in base_labels]
            label_counts = Counter(cluster_true_labels)
            for label, count in label_counts.items():
                label_dist_matrix.loc[cluster_name, label] = count
        if all_time_stamps is not None:
            cluster_time_stamps = np.array(all_time_stamps)[cluster_indices]
            months = pd.to_datetime(cluster_time_stamps).month
            month_counts = Counter(months)
            for month, count in month_counts.items():
                month_dist_matrix.loc[cluster_name, month] = count
    fig, axes = plt.subplots(2, 1, figsize=(16, 14))
    fig.suptitle('Final Cluster Distribution Summary', fontsize=20)
    sns.heatmap(label_dist_matrix, ax=axes[0], annot=True, fmt='d', cmap='viridis', linewidths=.5)
    axes[0].set_title('Label Distribution per Cluster', fontsize=16)
    axes[0].set_ylabel('Cluster')
    axes[0].set_xlabel('True Label')
    sns.heatmap(month_dist_matrix, ax=axes[1], annot=True, fmt='d', cmap='inferno', linewidths=.5)
    axes[1].set_title('Monthly Distribution per Cluster', fontsize=16)
    axes[1].set_ylabel('Cluster')
    axes[1].set_xlabel('Month')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(save_dir, 'final_distribution_summary.png')
    plt.savefig(save_path)
    plt.close()
    logging.info(f"分布の可視化画像を保存: {save_path}")

def save_daily_maps_for_clusters(clusters, spatial_anomaly_data, lat, lon, ts, labels, base_labels, save_dir):
    logging.info("クラスタ別の日次気圧配置図（空間偏差）を保存中...")
    maps_main_dir = os.path.join(save_dir, 'daily_anomaly_maps')
    os.makedirs(maps_main_dir, exist_ok=True)
    dominant_labels = {}
    for i, cluster_indices in enumerate(clusters):
        if labels:
            base_label_counts = Counter(l for l in [labels[j] for j in cluster_indices] if l in base_labels)
            dominant_labels[i] = base_label_counts.most_common(1)[0][0] if base_label_counts else "Unknown"
        else:
            dominant_labels[i] = ""
    cmap = plt.get_cmap('RdBu_r')
    pressure_vmin = -40
    pressure_vmax = 40
    pressure_levels = np.linspace(pressure_vmin, pressure_vmax, 21)
    norm = Normalize(vmin=pressure_vmin, vmax=pressure_vmax)
    line_levels = np.arange(pressure_vmin, pressure_vmax + 1, 10)
    for i, cluster_indices in enumerate(tqdm(clusters, desc="Saving daily maps")):
        dom_label = dominant_labels.get(i, "")
        cluster_dir = os.path.join(maps_main_dir, f'cluster_{i+1:02d}_dom_label_{dom_label}')
        os.makedirs(cluster_dir, exist_ok=True)
        for data_idx in cluster_indices:
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            pressure_map = spatial_anomaly_data[data_idx].reshape(len(lat), len(lon))
            cont = ax.contourf(lon, lat, pressure_map, levels=pressure_levels, cmap=cmap, norm=norm, extend='both', transform=ccrs.PlateCarree())
            cbar = fig.colorbar(cont, ax=ax, orientation='vertical', pad=0.05, aspect=20)
            cbar.set_label('Sea Level Pressure Anomaly (hPa)')
            ax.contour(lon, lat, pressure_map, levels=line_levels, colors='k', linewidths=0.5, transform=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="black", linewidth=0.5)
            ax.set_extent([120, 150, 20, 50], crs=ccrs.PlateCarree())
            date_str = pd.to_datetime(str(ts[data_idx])).strftime('%Y-%m-%d')
            true_label = labels[data_idx] if labels else "N/A"
            ax.set_title(f"Date: {date_str}\nTrue Label: {true_label}")
            plt.tight_layout()
            save_path = os.path.join(cluster_dir, f"{date_str}_label_{true_label.replace('/', '_')}.png")
            plt.savefig(save_path)
            plt.close(fig)

def plot_final_clusters(medoids, clusters, spatial_anomaly_data, lat_coords, lon_coords, time_stamps, all_labels, base_labels, save_dir):
    num_clusters = len(medoids)
    logging.info(f"最終結果（空間偏差図）をプロット中... 合計{num_clusters}クラスタ")
    n_cols = 5
    n_rows = (num_clusters + n_cols - 1) // n_cols if num_clusters > 0 else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), subplot_kw={"projection": ccrs.PlateCarree()})
    axes = np.atleast_1d(axes).flatten()
    cmap = plt.get_cmap('RdBu_r')
    pressure_vmin = -40
    pressure_vmax = 40
    pressure_levels = np.linspace(pressure_vmin, pressure_vmax, 21)
    norm = Normalize(vmin=pressure_vmin, vmax=pressure_vmax)
    for i in range(num_clusters):
        ax = axes[i]
        if medoids[i] is None:
            ax.set_title(f'Cluster {i+1} (Empty)'); ax.axis("off"); continue
        medoid_pattern_2d = spatial_anomaly_data[medoids[i]].reshape(len(lat_coords), len(lon_coords))
        cont = ax.contourf(lon_coords, lat_coords, medoid_pattern_2d, levels=pressure_levels, cmap=cmap, extend="both", norm=norm, transform=ccrs.PlateCarree())
        ax.contour(lon_coords, lat_coords, medoid_pattern_2d, colors="k", linewidth=0.5, levels=pressure_levels, transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="black", linewidth=0.5)
        ax.set_extent([120, 150, 20, 50], crs=ccrs.PlateCarree())
        cluster_indices = clusters[i]
        frequency = len(cluster_indices) / spatial_anomaly_data.shape[0] * 100
        medoid_date = pd.to_datetime(str(time_stamps[medoids[i]])).strftime('%Y-%m-%d')
        dominant_label_str = ""
        if all_labels:
            base_label_counts = Counter(label for label in [all_labels[j] for j in cluster_indices] if label in base_labels)
            if base_label_counts:
                dominant_label = base_label_counts.most_common(1)[0][0]
                dominant_label_str = f"Dom. Label: {dominant_label}"
        ax.set_title(f'Cluster {i+1} (N={len(cluster_indices)}, Freq:{frequency:.1f}%)\n{dominant_label_str}\nMedoid Date: {medoid_date}', fontsize=8)
    for i in range(num_clusters, len(axes)):
        axes[i].axis("off")
    fig.suptitle(f'Final Synoptic Patterns (Medoids of Spatial Anomaly) - TH_merge={TH_MERGE}', fontsize=16)
    fig.tight_layout(rect=[0, 0, 0.9, 0.95])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(cont, cax=cbar_ax, label="Sea Level Pressure Anomaly (hPa)")
    save_path = os.path.join(save_dir, f'final_clusters_anomaly_th{TH_MERGE}.png')
    plt.savefig(save_path)
    plt.close()
    logging.info(f"最終クラスタの画像（空間偏差図）を保存: {save_path}")

# --- 8. メイン ---
if __name__ == '__main__':
    logging.info("--- 2段階クラスタリングプログラム開始 (S1スコア使用) ---")
    X_normalized, X_original, X_anomaly, lat, lon, d_lat, d_lon, ts, labels = load_and_prepare_data(DATA_FILE, START_DATE, END_DATE, device)

    logging.info("S1スコアの分布を全データで調査します...")
    X_sample = X_normalized
    s1_matrix_sample = torch.zeros((len(X_sample), len(X_sample)), device=device)
    for i in tqdm(range(0, len(X_sample), CALCULATION_BATCH_SIZE), desc="S1スコア分布調査中 (ALL)"):
        end_idx = min(i + CALCULATION_BATCH_SIZE, len(X_sample))
        x_batch = X_sample[i:end_idx]
        s1_matrix_sample[i:end_idx, :] = calculate_s1_pairwise_batch(x_batch, X_sample, d_lat, d_lon)
    eye_mask = torch.eye(len(X_sample), dtype=torch.bool, device=device)
    s1_scores_flat = s1_matrix_sample[~eye_mask].detach().cpu().numpy()
    plt.figure(figsize=(10, 6))
    plt.hist(s1_scores_flat, bins=50, alpha=0.7, color='blue')
    plt.title('Distribution of S1 Scores (All Data)')
    plt.xlabel('S1 Score (Lower is more similar)')
    plt.ylabel('Frequency')
    plt.grid(True)
    s1_dist_path = os.path.join(RESULT_DIR, 's1_score_distribution.png')
    plt.savefig(s1_dist_path)
    plt.close()
    logging.info(f"S1スコアの分布図(ALL)を保存しました: {s1_dist_path}")
    
    if X_normalized is not None:
        start_time = time.time()
        final_clusters, final_medoids = two_stage_clustering(X_normalized, TH_MERGE, labels, ts, BASE_LABELS, CALCULATION_BATCH_SIZE, d_lat, d_lon)
        elapsed_time = time.time() - start_time
        logging.info(f"\n{'='*10} クラスタリング完了 {'='*10}")
        logging.info(f"総計算時間: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分)")
        logging.info(f"最終的なクラスタ数: {len(final_clusters)}")
        analyze_cluster_distribution(final_clusters, labels, ts, BASE_LABELS, "最終結果")
        if labels:
            _ = evaluate_clusters_majority_recall(final_clusters, labels, BASE_LABELS, "最終結果")
        results = {'medoid_indices': final_medoids, 'clusters': final_clusters}
        torch.save(results, os.path.join(RESULT_DIR, f'clustering_result_th{TH_MERGE}.pt'))
        logging.info("クラスタリング結果を保存しました。")
        if labels and ts is not None:
            plot_final_distribution_summary(final_clusters, labels, ts, BASE_LABELS, RESULT_DIR)
        if X_anomaly is not None:
            save_daily_maps_for_clusters(final_clusters, X_anomaly, lat, lon, ts, labels, BASE_LABELS, RESULT_DIR)
        plot_final_clusters(final_medoids, final_clusters, X_anomaly, lat, lon, ts, labels, BASE_LABELS, RESULT_DIR)

    logging.info("--- プログラム終了 ---")