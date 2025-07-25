# -*- coding: utf-8 -*-
"""
main_v1.py (GPU対応・完成版)

高次元データ構造と不確実性の可視化：ハイブリッドMPPCA-SOMフレームワーク
海面気圧データ（ERA5）を用いた気圧配置パターンの解析

[主な改善点]
- MPPCAの計算をGPU (CuPy) で実行し、大幅な高速化を実現。
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

# --- GPU/CPU ライブラリのインポート ---
try:
    import cupy as cp
    from mppca_gpu import initialization_kmeans_gpu, mppca_gem_gpu
    GPU_ENABLED = cp.is_available()
except ImportError:
    GPU_ENABLED = False

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import japanize_matplotlib
from scipy.stats import entropy
from sklearn.metrics import recall_score
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from minisom import MiniSom # minisomは変更なし

# --- 1. グローバル設定とパラメータ定義 ---

# 再現性のためのシード
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
if GPU_ENABLED:
    cp.random.seed(GLOBAL_SEED)

# 出力ディレクトリ設定
OUTPUT_DIR = "mppca_som_results_gpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ログ設定
LOG_FILE = os.path.join(OUTPUT_DIR, 'execution.log')
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# データファイルパス
DATA_FILE_PATH = './prmsl_era5_all_data_seasonal_small.nc'

# モデルパラメータ
# MPPCA
P_CLUSTERS = 100
Q_LATENT_DIM = 2
N_ITER_MPPCA = 10  # テスト時は短く、本番では長く設定 (例: 100)

# MiniSom
MAP_X, MAP_Y = 10, 10
SIGMA_SOM = 3.0
LEARNING_RATE_SOM = 1.0
NEIGHBORHOOD_FUNCTION_SOM = 'gaussian'
TOPOLOGY_SOM = 'rectangular'
ACTIVATION_DISTANCE_SOM = 'euclidean'
N_ITER_SOM = 100000 # SOMの学習イテレーション

# 基本となる気圧配置パターンのラベル
BASE_LABELS = [
    '1', '2A', '2B', '2C', '2D', '3A', '3B', '3C', '3D',
    '4A', '4B', '5', '6A', '6B', '6C'
]

# --- 2. 補助関数 ---

def format_duration(seconds):
    """秒をHH:MM:SS形式に変換する"""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

def parse_label(label_str):
    """ラベル文字列 "A-B" を ['A', 'B'] のリストに変換する"""
    return label_str.replace('+', '-').split('-')

# --- 3. データ処理関数 ---

def load_and_preprocess_data(filepath, start_date='1991-01-01', end_date='2000-12-31'):
    """NetCDFファイルを読み込み、前処理を行う"""
    logging.info(f"データファイル '{filepath}' を読み込み中...")
    start_time = time.time()

    try:
        with xr.open_dataset(filepath, engine='netcdf4') as ds:
            ds_labeled = ds.sel(valid_time=slice(start_date, end_date))
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

    return data_normalized, labels, lat, lon, data_mean, data_std

# --- 4. モデル実行関数 ---

def run_mppca_gpu(data_cpu, p, q, niter):
    """MPPCAモデルをGPUで訓練し、結果をCPUのNumPy配列で返す"""
    logging.info("MPPCAの訓練を開始します (GPU版)...")
    logging.info(f"パラメータ: クラスター数(p)={p}, 潜在次元(q)={q}, 反復回数(niter)={niter}")
    start_time = time.time()

    logging.info("データをCPUからGPUに転送中...")
    data_gpu = cp.asarray(data_cpu, dtype=cp.float64)
    
    logging.info("K-meansによる初期化を実行中 (GPU)...")
    pi_gpu, mu_gpu, W_gpu, sigma2_gpu, _ = initialization_kmeans_gpu(data_gpu, p, q, -1.0)
    logging.info("K-means初期化完了。")

    logging.info("GEMアルゴリズムによる訓練を実行中 (GPU)...")
    pi_gpu, mu_gpu, W_gpu, sigma2_gpu, R_gpu, L_gpu, _ = mppca_gem_gpu(data_gpu, pi_gpu, mu_gpu, W_gpu, sigma2_gpu, niter)

    end_time = time.time()
    logging.info(f"MPPCA訓練完了。所要時間: {format_duration(end_time - start_time)}")

    logging.info("結果をGPUからCPUに転送中...")
    L_cpu = cp.asnumpy(L_gpu)
    if L_cpu is not None and len(L_cpu) > 0:
        logging.info(f"最終的な対数尤度: {L_cpu[-1]:.4f}")

    mppca_results = {
        'pi': cp.asnumpy(pi_gpu), 'mu': cp.asnumpy(mu_gpu), 'W': cp.asnumpy(W_gpu),
        'sigma2': cp.asnumpy(sigma2_gpu), 'R': cp.asnumpy(R_gpu), 'L': L_cpu
    }
    with open(os.path.join(OUTPUT_DIR, 'mppca_results_gpu.pkl'), 'wb') as f:
        pickle.dump(mppca_results, f)
    logging.info(f"MPPCAのモデルと結果を '{os.path.join(OUTPUT_DIR, 'mppca_results_gpu.pkl')}' に保存しました。")

    return cp.asnumpy(R_gpu)

def run_som(data, map_x, map_y, input_len, sigma, lr, n_iter, seed):
    """MiniSomを訓練し、訓練済みモデルを返す"""
    logging.info("SOMの訓練を開始します...")
    logging.info(f"パラメータ: マップサイズ={map_x}x{map_y}, sigma={sigma}, learning_rate={lr}, 反復回数={n_iter}")
    start_time = time.time()

    som = MiniSom(map_x, map_y, input_len,
                  sigma=sigma, learning_rate=lr,
                  neighborhood_function=NEIGHBORHOOD_FUNCTION_SOM,
                  topology=TOPOLOGY_SOM, activation_distance=ACTIVATION_DISTANCE_SOM,
                  random_seed=seed)
    som.random_weights_init(data)

    logging.info("SOM訓練中 (進捗表示あり)...")
    som.train(data, n_iter, verbose=True, random_order=True)

    end_time = time.time()
    logging.info(f"SOM訓練完了。所要時間: {format_duration(end_time - start_time)}")

    with open(os.path.join(OUTPUT_DIR, 'som_model.pkl'), 'wb') as f:
        pickle.dump(som, f)
    logging.info(f"SOMモデルを '{os.path.join(OUTPUT_DIR, 'som_model.pkl')}' に保存しました。")

    return som

# --- 5. 評価・分析関数 ---

def evaluate_classification(som, mppca_posterior, original_labels):
    """マクロ平均再現率で分類性能を評価する"""
    logging.info("分類性能の評価を開始します (マクロ平均再現率)...")
    start_time = time.time()

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
    recall_scores_str = "\n".join([f"  - {label:<4}: {recall:.4f}" for label, recall in zip(mlb.classes_, per_class_recall)])
    logging.info(f"クラスごとの再現率:\n{recall_scores_str}")

    return macro_recall, node_dominant_label

# --- 6. 可視化関数 ---

def visualize_results(som, mppca_posterior, original_data, labels, node_dominant_label, lat, lon, data_mean, data_std):
    """解析結果を多角的に可視化する"""
    logging.info("結果の可視化を開始します...")
    start_time = time.time()

    map_x, map_y = som.get_weights().shape[:2]
    
    # 可視化1: U-Matrix
    plt.figure(figsize=(10, 10))
    plt.pcolor(som.distance_map().T, cmap='bone_r')
    plt.colorbar(label='ニューロン間の距離')
    plt.title('U-Matrix (クラスタ境界の可視化)')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'u_matrix.png'), dpi=300)
    plt.close()
    logging.info("U-Matrixを保存しました。")

    # 可視化2: ヒットマップ
    plt.figure(figsize=(10, 10))
    frequencies = som.activation_response(mppca_posterior)
    plt.pcolor(frequencies.T, cmap='viridis')
    plt.colorbar(label='勝者となった回数')
    plt.title('ヒットマップ (各ニューロンの勝者回数)')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'hit_map.png'), dpi=300)
    plt.close()
    logging.info("ヒットマップを保存しました。")
    
    # 可視化3: 勝者総取りマップ
    plt.figure(figsize=(12, 10))
    label_map = np.full((map_x, map_y), -1, dtype=int)
    label_names = {label: i for i, label in enumerate(BASE_LABELS)}
    for pos, label in node_dominant_label.items():
        if label:
            label_map[pos[0], pos[1]] = label_names[label]

    cmap = plt.get_cmap('tab20', len(BASE_LABELS))
    plt.pcolor(label_map.T, cmap=cmap, vmin=-0.5, vmax=len(BASE_LABELS)-0.5)
    cbar = plt.colorbar(ticks=np.arange(len(BASE_LABELS)))
    cbar.ax.set_yticklabels(BASE_LABELS)
    plt.title('勝者総取りマップ (各ノードの支配的気圧配置パターン)')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'winner_takes_all_map.png'), dpi=300)
    plt.close()
    logging.info("勝者総取りマップを保存しました。")

    # 可視化4: エントロピーマップ
    win_map_indices = som.win_map(mppca_posterior, return_indices=True)
    entropy_map = np.full((map_x, map_y), np.nan)
    for pos, indices in win_map_indices.items():
        if indices:
            avg_posterior_in_node = mppca_posterior[indices].mean(axis=0)
            entropy_map[pos] = entropy(avg_posterior_in_node)
    plt.figure(figsize=(10, 10))
    plt.pcolor(entropy_map.T, cmap='magma')
    plt.colorbar(label='エントロピー')
    plt.title('エントロピーマップ (モデルの不確実性可視化)')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'entropy_map.png'), dpi=300)
    plt.close()
    logging.info("エントロピーマップを保存しました。")

    # 可視化5: 代表ノードの気圧配置図
    logging.info("代表的なノードの平均気圧配置図を作成中...")
    valid_nodes = {pos: indices for pos, indices in win_map_indices.items() if indices}
    num_representative_nodes = min(9, len(valid_nodes))

    if num_representative_nodes > 0:
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
        for i in range(num_representative_nodes, len(axes)):
            axes[i].set_visible(False)
        plt.savefig(os.path.join(OUTPUT_DIR, 'representative_node_patterns.png'), dpi=300)
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
    logging.info("======= 研究プログラムを開始します =======")

    if not GPU_ENABLED:
        logging.error("GPUが利用できません。CuPyが正しくインストールされているか、NVIDIAドライバが最新か確認してください。")
        return

    # 1: データ読み込みと前処理
    data_normalized, labels, lat, lon, data_mean, data_std = load_and_preprocess_data(DATA_FILE_PATH)
    data_reshaped_original = data_normalized * data_std + data_mean

    # 2: MPPCAの実行 (GPU)
    mppca_posterior = run_mppca_gpu(data_normalized, P_CLUSTERS, Q_LATENT_DIM, N_ITER_MPPCA)

    # 3: SOMの実行 (CPU)
    som = run_som(mppca_posterior, MAP_X, MAP_Y, P_CLUSTERS, SIGMA_SOM, LEARNING_RATE_SOM, N_ITER_SOM, GLOBAL_SEED)

    # 4: 評価
    _, node_dominant_label = evaluate_classification(som, mppca_posterior, labels)

    # 5: 可視化
    visualize_results(som, mppca_posterior, data_reshaped_original, labels, node_dominant_label, lat, lon, data_mean, data_std)

    main_end_time = time.time()
    logging.info("======= すべての処理が完了しました =======")
    logging.info(f"総実行時間: {format_duration(main_end_time - main_start_time)}")

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning, module='minisom')
    main()