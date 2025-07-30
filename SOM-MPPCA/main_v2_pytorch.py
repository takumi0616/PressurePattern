# main_v2_pytorch.py
# -*- coding: utf-8 -*-
"""
main_v2_pytorch.py (PyTorch GPU対応版)
  
高次元データ構造と不確実性の可視化：ハイブリッドMPPCA-SOMフレームワーク
海面気圧データ（ERA5）を用いた気圧配置パターンの解析
  
[主な改善点]
- MPPCAの計算をPyTorchで実行し、幅広いGPU環境での高速化を実現。
- Python標準の`logging`モジュールを導入し、コンソールとファイルにログを記録。
- 時間のかかるMPPCAとSOMの訓練ループに`tqdm`プログレスバーを適用し、進捗を可視化。
- GPUが利用可能か事前にチェックする機能を追加。
- すべての可視化機能を実装し、一貫したロギングを行う。
"""
import os
import time
import pickle
import warnings
import logging
from collections import Counter, defaultdict

# --- PyTorch ライブラリのインポート ---
import torch

# --- 他ライブラリのインポート ---
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import japanize_matplotlib
from scipy.stats import entropy
from sklearn.metrics import recall_score
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from minisom import MiniSom
# 新しく作成したPyTorch版MPPCAをインポート
from mppca_pytorch import initialization_kmeans_torch, mppca_gem_torch

# --- 1. グローバル設定とパラメータ定義 ---

# 再現性のためのシード
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_SEED)

# 出力ディレクトリ設定
OUTPUT_DIR = "mppca_som_results_pytorch"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ログ設定
LOG_FILE = os.path.join(OUTPUT_DIR, 'execution_pytorch.log')
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# GPU利用可能チェックとデバイス設定
GPU_ENABLED = torch.cuda.is_available()
DEVICE = torch.device("cuda" if GPU_ENABLED else "cpu")
if GPU_ENABLED:
    logging.info(f"GPUが利用可能です。デバイス: {torch.cuda.get_device_name(0)}")
else:
    logging.info("GPUが利用できません。CPUで実行します。")


# データファイルパス
DATA_FILE_PATH = './prmsl_era5_all_data_seasonal_small.nc'

# モデルパラメータ
# MPPCA
P_CLUSTERS = 49 # 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361
Q_LATENT_DIM = 2
N_ITER_MPPCA = 500 # テスト時は短く、本番では長く設定 (例: 100)
MPPCA_BATCH_SIZE = 1024


# MiniSom
MAP_X, MAP_Y = 7, 7
SIGMA_SOM = 3.0
LEARNING_RATE_SOM = 1.0
NEIGHBORHOOD_FUNCTION_SOM = 'gaussian'
TOPOLOGY_SOM = 'rectangular'
ACTIVATION_DISTANCE_SOM = 'euclidean'
N_ITER_SOM = 10000000 # SOMの学習イテレーション

# 基本となる気圧配置パターンのラベル
BASE_LABELS = [
    '1', '2A', '2B', '2C', '2D', '3A', '3B', '3C', '3D',
    '4A', '4B', '5', '6A', '6B', '6C'
]

# --- 2. 補助関数 (変更なし) ---
def format_duration(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

def parse_label(label_str):
    return label_str.replace('+', '-').split('-')

# --- 3. データ処理関数 (変更なし) ---
def load_and_preprocess_data(filepath, start_date='1991-01-01', end_date='2000-12-31'):
    logging.info(f"データファイル '{filepath}' を読み込み中...")
    start_time = time.time()
    try:
        with xr.open_dataset(filepath, engine='netcdf4') as ds:
            ds_labeled = ds.sel(valid_time=slice(start_date, end_date))
            # ★★★ データ読み込み時点でfloat64に統一 ★★★
            msl_data = ds_labeled['msl'].values.astype(np.float64)
            labels = ds_labeled['label'].values.astype(str)
            lat = ds_labeled['latitude'].values
            lon = ds_labeled['longitude'].values
    except FileNotFoundError:
        logging.error(f"データファイルが見つかりません: {filepath}")
        raise
    logging.info(f"データ読み込み完了。対象期間: {start_date} ～ {end_date}")
    n_samples, n_lat, n_lon = msl_data.shape
    logging.info(f"サンプル数: {n_samples}, 緯度点数: {n_lat}, 経度点数: {n_lon}")
    data_reshaped = msl_data.reshape(n_samples, n_lat * n_lon)
    if np.isnan(data_reshaped).any():
        logging.info("欠損値を検出。列ごとの平均値で補完します...")
        col_mean = np.nanmean(data_reshaped, axis=0)
        inds = np.where(np.isnan(data_reshaped))
        data_reshaped[inds] = np.take(col_mean, inds[1])
    logging.info("データを標準化（Z-score normalization）します...")
    data_mean = data_reshaped.mean(axis=0)
    data_std = data_reshaped.std(axis=0)
    data_std[data_std == 0] = 1
    data_normalized = (data_reshaped - data_mean) / data_std
    end_time = time.time()
    logging.info(f"データ前処理完了。所要時間: {format_duration(end_time - start_time)}")
    # 全てfloat64で返すようにする
    return data_normalized.astype(np.float64), labels, lat, lon, data_mean.astype(np.float64), data_std.astype(np.float64)

# --- 4. モデル実行関数 ---

def run_mppca_pytorch(data_np, p, q, niter):
    """MPPCAモデルをPyTorchで訓練し、結果をCPUのNumPy配列で返す"""
    logging.info("MPPCAの訓練を開始します (PyTorch版)...")
    logging.info(f"パラメータ: クラスター数(p)={p}, 潜在次元(q)={q}, 反復回数(niter)={niter}")
    start_time = time.time()
    data_t = torch.from_numpy(data_np.astype(np.float64))
    logging.info(f"K-meansによる初期化を実行中 ({DEVICE})...")
    
    pi_t, mu_t, W_t, sigma2_t, _ = initialization_kmeans_torch(
        data_t, p, q, device=DEVICE
    )
    logging.info("K-means初期化完了。")

    logging.info(f"GEMアルゴリズムによる訓練を実行中 ({DEVICE})...")
    pi_t, mu_t, W_t, sigma2_t, R_t, L_t, _ = mppca_gem_torch(
        X=data_t,
        pi=pi_t, 
        mu=mu_t, 
        W=W_t, 
        sigma2=sigma2_t,
        niter=niter,
        batch_size=MPPCA_BATCH_SIZE,
        device=DEVICE
    )

    end_time = time.time()
    logging.info(f"MPPCA訓練完了。所要時間: {format_duration(end_time - start_time)}")
    
    logging.info("結果をGPUからCPUに転送し、NumPy配列に変換中...")
    L_cpu = L_t.cpu().numpy()
    if L_cpu is not None and len(L_cpu) > 0:
        final_log_likelihood = L_cpu[-1]
        if np.isnan(final_log_likelihood):
            logging.warning("最終的な対数尤度が'nan'です。モデルが正しく収束しなかった可能性があります。")
        else:
            logging.info(f"最終的な対数尤度: {final_log_likelihood:.4f}")

    mppca_results = {
        'pi': pi_t.cpu().numpy(), 'mu': mu_t.cpu().numpy(), 'W': W_t.cpu().numpy(),
        'sigma2': sigma2_t.cpu().numpy(), 'R': R_t.cpu().numpy(), 'L': L_cpu
    }
    
    result_path = os.path.join(OUTPUT_DIR, 'mppca_results_pytorch.pkl')
    with open(result_path, 'wb') as f:
        pickle.dump(mppca_results, f)
    logging.info(f"MPPCAのモデルと結果を '{result_path}' に保存しました。")
    
    return R_t.cpu().numpy(), L_cpu

def run_som(data, map_x, map_y, input_len, sigma, lr, n_iter, seed):
    """MiniSomを訓練し、訓練済みモデルを返す (変更なし)"""
    logging.info("SOMの訓練を開始します...")
    logging.info(f"パラメータ: マップサイズ={map_x}x{map_y}, sigma={sigma}, learning_rate={lr}, 反復回数={n_iter}")
    start_time = time.time()
    
    som = MiniSom(map_x, map_y, input_len,
                  sigma=sigma, learning_rate=lr,
                  neighborhood_function=NEIGHBORHOOD_FUNCTION_SOM,
                  topology=TOPOLOGY_SOM, activation_distance=ACTIVATION_DISTANCE_SOM,
                  random_seed=seed)
    
    if np.isnan(data).any():
        logging.warning("SOMへの入力データにnanが含まれています。SOMの重み初期化と学習に影響する可能性があります。")
        som.random_weights_init(np.nan_to_num(data))
    else:
        som.random_weights_init(data)

    logging.info("SOM訓練中 (進捗表示あり)...")
    q_error_history = som.train_batch(data, n_iter, verbose=True)
    
    end_time = time.time()
    logging.info(f"SOM訓練完了。所要時間: {format_duration(end_time - start_time)}")
    
    model_path = os.path.join(OUTPUT_DIR, 'som_model_pytorch.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(som, f)
    logging.info(f"SOMモデルを '{model_path}' に保存しました。")
    
    return som, q_error_history

# --- 5. 評価・分析関数 (変更なし) ---
def evaluate_classification(som, mppca_posterior, original_labels):
    logging.info("分類性能の評価を開始します (マクロ平均再現率)...")
    start_time = time.time()
    
    if np.isnan(mppca_posterior).any():
        logging.error("MPPCAの事後確率にnanが含まれているため、評価をスキップします。")
        return 0.0, {}

    parsed_true_labels = [parse_label(l) for l in original_labels]
    mlb = MultiLabelBinarizer(classes=BASE_LABELS)
    y_true = mlb.fit_transform(parsed_true_labels)
    win_map_indices = som.win_map(mppca_posterior, return_indices=True)
    node_dominant_label = {}
    for pos, indices in win_map_indices.items():
        if not indices:
            node_dominant_label[pos] = None
            continue
        labels_in_node = [label for i in indices for label in parsed_true_labels[i]]
        if not labels_in_node:
            node_dominant_label[pos] = None
            continue
        most_common = Counter(labels_in_node).most_common(1)
        node_dominant_label[pos] = most_common[0][0]
    winners = [som.winner(x) for x in mppca_posterior]
    predicted_single_labels = [node_dominant_label.get(w) for w in winners]
    y_pred = mlb.transform([[l] if l else [] for l in predicted_single_labels])
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    logging.info(f"評価完了。所要時間: {format_duration(time.time() - start_time)}")
    logging.info(f"マクロ平均再現率: {macro_recall:.4f}")
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    recall_scores_str = "\n".join([f" - {label:<4}: {recall:.4f}" for label, recall in zip(mlb.classes_, per_class_recall)])
    logging.info(f"クラスごとの再現率:\n{recall_scores_str}")
    return macro_recall, node_dominant_label

# --- 6. 可視化関数 ---
def plot_log_likelihood(log_likelihood_history):
    """MPPCAの対数尤度の収束グラフをプロットして保存する"""
    logging.info("対数尤度の収束グラフを作成中...")
    plt.figure(figsize=(10, 6))
    
    # 履歴にnanが含まれている場合、それらを除外してプロット
    valid_indices = ~np.isnan(log_likelihood_history)
    if not np.any(valid_indices):
        logging.warning("有効な対数尤度データがないため、収束グラフは作成されません。")
        plt.close()
        return
        
    iterations = np.arange(len(log_likelihood_history))[valid_indices]
    valid_likelihoods = log_likelihood_history[valid_indices]

    plt.plot(iterations, valid_likelihoods)
    plt.title('MPPCA 対数尤度の収束履歴')
    plt.xlabel('イテレーション (Iteration)')
    plt.ylabel('対数尤度 (Log-Likelihood)')
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'log_likelihood_convergence_pytorch.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.info(f"対数尤度グラフを '{save_path}' に保存しました。")

def plot_quantization_error(q_error_history, log_interval):
    """SOMの量子化誤差の収束グラフをプロットして保存する"""
    logging.info("SOM量子化誤差の収束グラフを作成中...")
    plt.figure(figsize=(10, 6))
    
    iterations = np.arange(len(q_error_history)) * log_interval
    plt.plot(iterations, q_error_history)
    plt.title('SOM 量子化誤差の収束履歴')
    plt.xlabel('イテレーション (Iteration)')
    plt.ylabel('量子化誤差 (Quantization Error)')
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'som_quantization_error_pytorch.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.info(f"SOM量子化誤差グラフを '{save_path}' に保存しました。")

def visualize_results(som, mppca_posterior, original_data, labels, node_dominant_label, lat, lon, data_mean, data_std, log_likelihood_history, q_error_history):
    logging.info("結果の可視化を開始します...")
    start_time = time.time()

    plot_log_likelihood(log_likelihood_history)
    plot_quantization_error(q_error_history, log_interval=1000) # minisom.pyのlog_intervalと値を合わせる
    
    if np.isnan(mppca_posterior).any():
        logging.error("MPPCAの事後確率にnanが含まれているため、可視化をスキップします。")
        return
        
    map_x, map_y = som.get_weights().shape[:2]
    
    plt.figure(figsize=(10, 10)); plt.pcolor(som.distance_map().T, cmap='bone_r'); plt.colorbar(label='ニューロン間の距離'); plt.title('U-Matrix'); plt.grid(True); plt.savefig(os.path.join(OUTPUT_DIR, 'u_matrix_pytorch.png'), dpi=300); plt.close()
    logging.info("U-Matrixを保存しました。")
    
    plt.figure(figsize=(10, 10)); frequencies = som.activation_response(mppca_posterior); plt.pcolor(frequencies.T, cmap='viridis'); plt.colorbar(label='勝者となった回数'); plt.title('ヒットマップ'); plt.grid(True); plt.savefig(os.path.join(OUTPUT_DIR, 'hit_map_pytorch.png'), dpi=300); plt.close()
    logging.info("ヒットマップを保存しました。")
    
    plt.figure(figsize=(12, 10)); label_map = np.full((map_x, map_y), -1, dtype=int); label_names = {label: i for i, label in enumerate(BASE_LABELS)};
    for pos, label in node_dominant_label.items():
        if label: label_map[pos[0], pos[1]] = label_names[label]
    cmap = plt.get_cmap('tab20', len(BASE_LABELS)); plt.pcolor(label_map.T, cmap=cmap, vmin=-0.5, vmax=len(BASE_LABELS)-0.5); cbar = plt.colorbar(ticks=np.arange(len(BASE_LABELS))); cbar.ax.set_yticklabels(BASE_LABELS); plt.title('勝者総取りマップ'); plt.grid(True); plt.savefig(os.path.join(OUTPUT_DIR, 'winner_takes_all_map_pytorch.png'), dpi=300); plt.close()
    logging.info("勝者総取りマップを保存しました。")

    win_map_indices = som.win_map(mppca_posterior, return_indices=True)
    entropy_map = np.full((map_x, map_y), np.nan)
    for pos, indices in win_map_indices.items():
        if indices: entropy_map[pos] = entropy(mppca_posterior[indices].mean(axis=0))
    plt.figure(figsize=(10, 10)); plt.pcolor(entropy_map.T, cmap='magma'); plt.colorbar(label='エントロピー'); plt.title('エントロピーマップ'); plt.grid(True); plt.savefig(os.path.join(OUTPUT_DIR, 'entropy_map_pytorch.png'), dpi=300); plt.close()
    logging.info("エントロピーマップを保存しました。")
    
    valid_nodes = {pos: indices for pos, indices in win_map_indices.items() if indices}
    num_representative_nodes = min(9, len(valid_nodes))
    if num_representative_nodes > 0:
        logging.info("代表的なノードの平均気圧配置図を作成中...")
        sorted_nodes = sorted(valid_nodes.keys(), key=lambda pos: frequencies[pos], reverse=True)
        top_nodes = sorted_nodes[:num_representative_nodes]
        fig, axes = plt.subplots(3, 3, figsize=(15, 15), constrained_layout=True)
        fig.suptitle('ヒット数の多い代表的ノードの平均気圧配置', fontsize=20)
        axes = axes.flatten()
        for i, pos in enumerate(top_nodes):
            ax = axes[i]
            indices = win_map_indices[pos]
            node_data_original_scale = original_data[indices] * data_std + data_mean
            mean_pattern = node_data_original_scale.mean(axis=0).reshape(len(lat), len(lon))
            ax.contourf(lon, lat, mean_pattern / 100, cmap='coolwarm', levels=20)
            ax.contour(lon, lat, mean_pattern / 100, colors='k', linewidths=0.5, levels=15)
            ax.set_title(f"Node {pos} (n={len(indices)})\nDom: {node_dominant_label.get(pos, 'N/A')}")
        for i in range(num_representative_nodes, len(axes)): axes[i].set_visible(False)
        plt.savefig(os.path.join(OUTPUT_DIR, 'representative_node_patterns_pytorch.png'), dpi=300)
        plt.close()
        logging.info("代表的なノードの平均気圧配置図を保存しました。")
    else:
        logging.warning("代表的なノードが見つからず、気圧配置図は作成しませんでした。")
    
    end_time = time.time()
    logging.info(f"可視化完了。所要時間: {format_duration(end_time - start_time)}")


# --- 7. メイン実行ブロック --- 
def main(): 
    """メインの処理フロー""" 
    main_start_time = time.time() 
    logging.info("======= 研究プログラムを開始します (PyTorch版) =======") 

    # 1: データ読み込みと前処理 
    data_normalized, labels, lat, lon, data_mean, data_std = load_and_preprocess_data(DATA_FILE_PATH) 
    
    # 2: MPPCAの実行 (PyTorch) 
    mppca_posterior, log_likelihood_history = run_mppca_pytorch(data_normalized, P_CLUSTERS, Q_LATENT_DIM, N_ITER_MPPCA) 

    # 3: SOMの実行 (CPU, NumPy) 
    som, q_error_history = run_som(mppca_posterior, MAP_X, MAP_Y, P_CLUSTERS, SIGMA_SOM, LEARNING_RATE_SOM, N_ITER_SOM, GLOBAL_SEED) 
    
    # 4: 評価 
    _, node_dominant_label = evaluate_classification(som, mppca_posterior, labels) 
    
    # 5: 可視化 
    # 元データを渡す必要があるので、逆標準化は可視化関数内で行う 
    visualize_results(som, mppca_posterior, data_normalized, labels, node_dominant_label, lat, lon, data_mean, data_std, log_likelihood_history, q_error_history) 
    
    main_end_time = time.time() 
    logging.info("======= すべての処理が完了しました =======") 
    logging.info(f"総実行時間: {format_duration(main_end_time - main_start_time)}")


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning, module='minisom')
    main()