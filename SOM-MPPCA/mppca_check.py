import os
import logging
import torch
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# 提供されたMPPCAスクリプトをインポート
from mppca_pytorch import initialization_kmeans_torch, mppca_gem_torch

# --- 設定項目 ---
# ファイルパスとディレクトリ
DATA_FILE = './prmsl_era5_all_data_seasonal_small.nc'
RESULT_DIR = './mppca_results'

# データ期間
START_DATE = '1991-01-01'
END_DATE = '2000-12-31'

# MPPCAモデルのパラメータ
P_CLUSTERS = 4        # クラスター数 (ラベル数に合わせて15に設定)
Q_LATENT_DIM = 20         # 潜在変数の次元（2次元に削減）
N_ITERATIONS = 500       # EMアルゴリズムの反復回数
BATCH_SIZE = 512        # バッチサイズ

# 基本ラベルリスト
BASE_LABELS = [
    '1', '2A', '2B', '2C', '2D', '3A', '3B', '3C', '3D',
    '4A', '4B', '5', '6A', '6B', '6C'
]

# --- ロギングと結果ディレクトリの設定 ---
os.makedirs(RESULT_DIR, exist_ok=True)
log_path = os.path.join(RESULT_DIR, 'verification.log')
if os.path.exists(log_path): os.remove(log_path)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_path), logging.StreamHandler()])

# --- データ読み込みと前処理 ---
def load_and_prepare_data(filepath, start_date, end_date):
    """データとラベルを読み込み、前処理を行う"""
    logging.info(f"データファイル '{filepath}' を読み込んでいます...")
    try:
        ds = xr.open_dataset(filepath)
    except FileNotFoundError:
        logging.error(f"データファイルが見つかりません: {filepath}"); return None, None, None, None, None
        
    data_period = ds.sel(valid_time=slice(start_date, end_date))
    
    if data_period.valid_time.size == 0:
        logging.error("指定された期間にデータが見つかりませんでした。"); return None, None, None, None, None
    
    # ラベルデータを読み込む
    if 'label' in data_period.variables:
        # xarrayのデータはバイト列の場合があるので、デコードする
        labels = [l.decode('utf-8') if isinstance(l, bytes) else str(l) for l in data_period['label'].values]
        logging.info("ラベルデータを読み込みました。")
    else:
        logging.warning("'label'変数がデータファイルに見つかりませんでした。ラベル分析はスキップされます。")
        labels = None

    msl_data = data_period['msl']
    if msl_data.isnull().any():
        logging.warning("データに欠損値 (NaN) が見つかりました。データ全体の平均値で補完します。")
        msl_data = msl_data.fillna(msl_data.mean().item())
        
    n_samples, d_lat, d_lon = msl_data.shape[0], msl_data.shape[1], msl_data.shape[2]
    logging.info(f"データ形状: {n_samples}サンプル, {d_lat*d_lon}次元 ({d_lat}x{d_lon})")
    flattened_data = msl_data.values.reshape(n_samples, d_lat * d_lon)
    
    return torch.from_numpy(flattened_data), d_lat, d_lon, data_period.valid_time.values, labels

# --- 計算・可視化・分析関数 ---

def analyze_kmeans_clusters(clusters, labels, time_stamps, p):
    """
    k-meansのクラスタリング結果を、既存ラベルと月別分布の観点から分析し、ログに出力する。
    """
    logging.info("\n--- k-meansクラスタリング結果の総合分析 ---")
    clusters_np = clusters.cpu().numpy()
    
    for i in range(p):
        indices = np.where(clusters_np == i)[0]
        num_samples_in_cluster = len(indices)

        logging.info(f"\n[k-meansクラスタ {i+1}]")
        logging.info(f"  - データポイント数: {num_samples_in_cluster}")

        if num_samples_in_cluster == 0:
            continue

        # --- ラベル構成の分析 (既存の機能) ---
        if labels is not None:
            # (既存のラベル分析コードはここに移動)
            label_counts = {label: 0 for label in BASE_LABELS}; total_label_counts = 0
            cluster_labels = [labels[j] for j in indices]
            for label_str in cluster_labels:
                sub_labels = label_str.split('+') if '+' in label_str else (label_str.split('-') if '-' in label_str else [label_str])
                for sub_label in sub_labels:
                    sub_label = sub_label.strip()
                    if sub_label in label_counts:
                        label_counts[sub_label] += 1; total_label_counts += 1
            if total_label_counts > 0:
                logging.info("  - ラベル構成 (カウント数順):")
                sorted_labels = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)
                for label, count in sorted_labels:
                    if count > 0: logging.info(f"    - {label:<4}: {count:4d} 件 ({(count / total_label_counts) * 100:5.1f}%)")
            else:
                logging.info("  - このクラスタには既知のラベルを持つデータがありませんでした。")
        else:
            logging.info("  - ラベル情報がないため、ラベル構成の分析はスキップします。")
            
        # ★★★【追加機能】月別分布の分析 ★★★
        cluster_time_stamps = time_stamps[indices]
        # pandasを使ってタイムスタンプから月を抽出
        months = pd.to_datetime(cluster_time_stamps).month
        month_counts = {m: 0 for m in range(1, 13)}
        # 各月の出現回数をカウント
        unique_months, counts = np.unique(months, return_counts=True)
        for month_val, count in zip(unique_months, counts):
            month_counts[month_val] = count

        logging.info("  - 月別分布:")
        for month_val in range(1, 13):
            count = month_counts[month_val]
            percentage = (count / num_samples_in_cluster) * 100 if num_samples_in_cluster > 0 else 0
            # ログに見やすいように簡易的なバーを追加
            bar = '■' * int(percentage / 4) 
            logging.info(f"    - {month_val:2d}月: {count:4d}件 ({percentage:5.1f}%) {bar}")

    logging.info("\n--- 総合分析終了 ---")

def calculate_latent_coords(X, R, mu, W, sigma2, device):
    p, d, q = W.shape; N, _ = X.shape
    cluster_assignments = torch.argmax(R, dim=1)
    latent_coords = torch.zeros(N, q, device=device, dtype=X.dtype)
    W_T = W.transpose(-2, -1); I_q = torch.eye(q, device=device, dtype=X.dtype)
    logging.info("主潜在空間の座標を計算中...")
    for i in tqdm(range(p), desc="Calculating main latent coords"):
        mask = (cluster_assignments == i);
        if mask.sum() == 0: continue
        X_c, mu_c, W_c, W_T_c, sigma2_c = X[mask], mu[i], W[i], W_T[i], sigma2[i]
        M_c = sigma2_c * I_q + W_T_c @ W_c; M_inv_c = torch.linalg.inv(M_c)
        deviation = X_c - mu_c; coords = (M_inv_c @ W_T_c @ deviation.T).T
        latent_coords[mask] = coords
    return latent_coords, cluster_assignments

def calculate_all_latent_coords_for_sample(X_sample, mu, W, sigma2, device):
    p, d, q = W.shape
    all_coords = torch.zeros(p, q, device=device, dtype=X_sample.dtype)
    W_T = W.transpose(-2, -1); I_q = torch.eye(q, device=device, dtype=X_sample.dtype)
    for i in range(p):
        mu_c, W_c, W_T_c, sigma2_c = mu[i], W[i], W_T[i], sigma2[i]
        M_c = sigma2_c * I_q + W_T_c @ W_c; M_inv_c = torch.linalg.inv(M_c)
        deviation = X_sample - mu_c
        coords = M_inv_c @ W_T_c @ deviation
        all_coords[i] = coords
    return all_coords

def plot_log_likelihood(log_L, save_path):
    plt.figure(figsize=(10, 6)); plt.plot(log_L.cpu().numpy()); plt.title('Log-Likelihood Convergence')
    plt.xlabel('Iteration'); plt.ylabel('Log-Likelihood'); plt.grid(True); plt.savefig(save_path); plt.close()
    logging.info(f"対数尤度グラフを保存: {save_path}")

def plot_average_patterns(mu, d_lat, d_lon, save_path):
    p, _ = mu.shape; n_cols = 5; n_rows = (p + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3)); axes = axes.flatten()
    mu_np = mu.cpu().numpy().reshape(p, d_lat, d_lon)
    for i in range(p):
        im = axes[i].imshow(mu_np[i], cmap='viridis'); axes[i].set_title(f'Cluster {i+1}'); axes[i].axis('off'); fig.colorbar(im, ax=axes[i])
    for i in range(p, len(axes)): axes[i].axis('off')
    plt.suptitle('Learned Average Pressure Patterns (μ)', fontsize=16); plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(save_path); plt.close()
    logging.info(f"平均パターン画像を保存: {save_path}")

def plot_latent_space(latent_coords, cluster_assignments, p, save_path):
    latent_coords_np = latent_coords.cpu().numpy(); cluster_assignments_np = cluster_assignments.cpu().numpy()
    plt.figure(figsize=(12, 10)); scatter = plt.scatter(latent_coords_np[:, 0], latent_coords_np[:, 1], c=cluster_assignments_np, cmap='tab20', alpha=0.6, s=10)
    plt.title('Latent Space Visualization'); plt.xlabel('Latent Dimension 1'); plt.ylabel('Latent Dimension 2')
    plt.legend(handles=scatter.legend_elements()[0], labels=[f'Cluster {i+1}' for i in range(p)], bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True); plt.tight_layout(rect=[0, 0, 0.85, 1]); plt.savefig(save_path); plt.close()
    logging.info(f"潜在空間プロットを保存: {save_path}")

def plot_reconstructions(X, mu, W, latent_coords, cluster_assignments, d_lat, d_lon, time_stamps, n_samples, save_path):
    fig, axes = plt.subplots(n_samples, 3, figsize=(10, 3 * n_samples)); plt.suptitle('Original vs. Average Pattern vs. Reconstructed', fontsize=16)
    X_np, mu_np = X.cpu().numpy().reshape(-1, d_lat, d_lon), mu.cpu().numpy().reshape(-1, d_lat, d_lon)
    for i in range(n_samples):
        original_img = X_np[i]; assigned_cluster = cluster_assignments[i].item(); latent_vec = latent_coords[i]
        W_c, mu_c = W[assigned_cluster], mu[assigned_cluster]
        reconstructed_img = (W_c @ latent_vec + mu_c).cpu().numpy().reshape(d_lat, d_lon)
        avg_pattern_img = mu_np[assigned_cluster]
        date_str = pd.to_datetime(str(time_stamps[i])).strftime('%Y-%m-%d')
        axes[i, 0].imshow(original_img, cmap='viridis'); axes[i, 0].set_title(f'Original ({date_str})')
        axes[i, 1].imshow(avg_pattern_img, cmap='viridis'); axes[i, 1].set_title(f'Avg Pattern (Cluster {assigned_cluster+1})')
        axes[i, 2].imshow(reconstructed_img, cmap='viridis'); axes[i, 2].set_title('Reconstructed')
        for j in range(3): axes[i,j].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(save_path); plt.close()
    logging.info(f"再構成画像の比較プロットを保存: {save_path}")

# --- メイン処理 ---
if __name__ == '__main__':
    # ★★★【修正箇所】ここから ★★★
    logging.info("--- MPPCA検証プログラム開始 ---")

    # --- 実験条件のログ出力 ---
    logging.info("==================================================")
    logging.info("               実験条件サマリー")
    logging.info("==================================================")
    logging.info(f"入力データファイル: {DATA_FILE}")
    logging.info(f"結果出力ディレクトリ: {RESULT_DIR}")
    logging.info(f"ログファイル: {log_path}")
    logging.info("--------------------------------------------------")
    logging.info(f"データ対象期間: {START_DATE} から {END_DATE}")
    logging.info("--------------------------------------------------")
    logging.info("MPPCAモデル パラメータ:")
    logging.info(f"  - クラスター数 (P_CLUSTERS): {P_CLUSTERS}")
    logging.info(f"  - 潜在変数の次元 (Q_LATENT_DIM): {Q_LATENT_DIM}")
    logging.info(f"  - EMアルゴリズム反復回数 (N_ITERATIONS): {N_ITERATIONS}")
    logging.info(f"  - バッチサイズ (BATCH_SIZE): {BATCH_SIZE}")
    logging.info("--------------------------------------------------")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"使用デバイス: {device.upper()}")
    logging.info("==================================================\n")
    # ★★★【修正箇所】ここまで ★★★

    X_original_dtype, d_lat, d_lon, time_stamps, labels = load_and_prepare_data(DATA_FILE, START_DATE, END_DATE)

    if X_original_dtype is not None:
        X = X_original_dtype.to(device, dtype=torch.float64)
        
        logging.info("k-meansによる初期化を開始..."); 
        pi, mu, W, sigma2, kmeans_clusters = initialization_kmeans_torch(X, P_CLUSTERS, Q_LATENT_DIM, device=device)
        
        if labels is not None:
            analyze_kmeans_clusters(kmeans_clusters, labels, time_stamps, P_CLUSTERS)

        logging.info(f"k-means初期化完了。pi:{pi.shape}, mu:{mu.shape}, W:{W.shape}, sigma2:{sigma2.shape}")
        logging.info("MPPCAモデルの学習 (GEMアルゴリズ) を開始..."); 
        pi, mu, W, sigma2, R, L, _ = mppca_gem_torch(X, pi, mu, W, sigma2, N_ITERATIONS, batch_size=BATCH_SIZE, device=device)
        logging.info(f"学習完了。最終的な対数尤度: {L[-1].item() if not torch.isnan(L[-1]) else 'NaN'}")
        
        model_path = os.path.join(RESULT_DIR, 'mppca_model.pt'); 
        torch.save({'pi': pi, 'mu': mu, 'W': W, 'sigma2': sigma2, 'R': R, 'd_lat': d_lat, 'd_lon': d_lon}, model_path)
        logging.info(f"学習済みモデルを保存: {model_path}")

        logging.info("--- 結果の可視化と分析を開始 ---")
        plot_log_likelihood(L, os.path.join(RESULT_DIR, 'log_likelihood.png'))
        plot_average_patterns(mu, d_lat, d_lon, os.path.join(RESULT_DIR, 'average_patterns.png'))
        latent_coords, cluster_assignments = calculate_latent_coords(X, R, mu, W, sigma2, device)
        plot_latent_space(latent_coords, cluster_assignments, P_CLUSTERS, os.path.join(RESULT_DIR, 'latent_space.png'))

        logging.info("--- 個々のデータに着目した結果の分析 ---")
        N_SAMPLES_TO_ANALYZE = 5; logging.info(f"最初の{N_SAMPLES_TO_ANALYZE}個のサンプルについて、詳細な結果を表示します。")
        for i in range(N_SAMPLES_TO_ANALYZE):
            date_str = pd.to_datetime(str(time_stamps[i])).strftime('%Y-%m-%d'); r_i = R[i]
            assigned_cluster = cluster_assignments[i].item(); max_prob = r_i[assigned_cluster].item()
            latent_coords_i = latent_coords[i].cpu().numpy()
            logging.info(f"\n[サンプル {i+1} ({date_str})]")
            logging.info(f"  - 最も可能性の高いクラスター: {assigned_cluster + 1} (確率: {max_prob:.4f})")
            logging.info(f"  - (主)潜在空間での座標: [{latent_coords_i[0]:.4f}, {latent_coords_i[1]:.4f}]")
            prob_vector_str = ", ".join([f"{p:.4f}" for p in r_i.cpu().numpy()])
            logging.info(f"  - 全クラスターへの所属確率ベクトル [R_i]: [{prob_vector_str}]")
            
            # 全15個の潜在空間での座標を表示
            all_coords_i = calculate_all_latent_coords_for_sample(X[i], mu, W, sigma2, device)
            logging.info(f"  - 全{P_CLUSTERS}個の潜在空間での座標:")
            for c in range(P_CLUSTERS):
                coord_c = all_coords_i[c].cpu().numpy()
                logging.info(f"    - Cluster {c+1:>2} の場合: [{coord_c[0]:9.4f}, {coord_c[1]:9.4f}]")

        plot_reconstructions(X, mu, W, latent_coords, cluster_assignments, d_lat, d_lon, time_stamps, N_SAMPLES_TO_ANALYZE, os.path.join(RESULT_DIR, 'reconstructions.png'))
        logging.info(f"すべての結果は '{RESULT_DIR}' ディレクトリに保存されました。")

    logging.info("--- プログラム終了 ---")