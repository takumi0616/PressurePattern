# main_v3.py
# -*- coding: utf-8 -*-
"""
main_v3.py
複雑な海面気圧データから気圧配置パターンを抽出し、体系的に分類・評価するための
統合研究フレームワーク。
MPPCA+SOMおよびPCA+SOMのアプローチを、複数のデータセットとパラメータで
実験・比較・可視化することを目的とする。

[主な特徴]
- 実験管理: 複数の実験設定（データ、手法、パラメータ）をリストで管理し、自動で連続実行。
- 柔軟性: MPPCAとPCAの切り替え、各種パラメータの動的な設定が可能。
- 堅牢なロギング: 実験ごとに独立したログファイルと、コンソールへの出力を両立。
- 統合された可視化: SOM関連のマップ、地理空間上の気圧配置パターン（平均、再構成、主成分）、
  潜在空間の分布など、多角的な可視化機能を統合。
- 再現性: グローバルな乱数シード固定により、実験の再現性を確保。
- エラーハンドリング: 特定の実験でエラーが発生しても、全体を停止させずに次の実験へ移行。
"""
import os
import time
import pickle
import warnings
import logging
import random
from collections import Counter, defaultdict

# --- 主要ライブラリのインポート ---
import torch
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm

# --- 可視化ライブラリ ---
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize

# --- 機械学習ライブラリ ---
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score
from sklearn.preprocessing import MultiLabelBinarizer
from minisom import MiniSom

# --- 自作モジュール ---
from mppca_pytorch import initialization_kmeans_torch, mppca_gem_torch

# --- 実験設定を定義するファイル ---
from experiments_config import EXPERIMENTS

# ==============================================================================
# --- 1. グローバル設定 & 実験定義 ---
# ==============================================================================

# --- 再現性のためのシード ---
GLOBAL_SEED = 42

# --- ディレクトリ設定 ---
PARENT_RESULT_DIR = "research_results_v3"

# --- データファイルパス ---
DATA_FILES = {
    'small': './prmsl_era5_all_data_seasonal_small.nc',
    'normal': './prmsl_era5_all_data_seasonal_normal.nc',
    'large': './prmsl_era5_all_data_seasonal_large.nc'
}

# --- データ期間設定 ---
START_DATE = '1991-01-01'
END_DATE = '2000-12-31'

# --- ラベル設定 ---
BASE_LABELS = [
    '1', '2A', '2B', '2C', '2D', '3A', '3B', '3C', '3D',
    '4A', '4B', '5', '6A', '6B', '6C'
]

# --- デフォルトパラメータ ---
DEFAULT_PARAMS = {
    # MPPCA
    'p_clusters': 49,
    'q_latent_dim': 2,
    'n_iter_mppca': 500,
    'mppca_batch_size': 1024,
    # PCA
    'n_components': 20,
    # MiniSom
    'map_x': 7,
    'map_y': 7,
    'sigma_som': 3.0,
    'learning_rate_som': 1.0,
    'n_iter_som': 1000000, # バッチ学習のため、以前より少ないイテレーションで収束
}

# ==============================================================================
# --- 2. 補助関数 & ユーティリティ ---
# ==============================================================================

def set_global_seed(seed):
    """全てのライブラリの乱数シードを固定し、再現性を高める。"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logging.info(f"グローバルシード {seed} を設定しました。")

def format_duration(seconds):
    """秒数を HH:MM:SS 形式の文字列に変換する。"""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

def parse_label(label_str):
    """複合ラベル文字列を分割してリストで返す。"""
    return label_str.replace('+', '-').split('-')

def setup_logger(log_dir, exp_name):
    """実験ごとに独立したロガーを設定する。"""
    log_file = os.path.join(log_dir, f'log_{exp_name}.log')
    
    # 既存のハンドラを全て削除
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

# ==============================================================================
# --- 3. データ処理 ---
# ==============================================================================

def load_and_preprocess_data(filepath, start_date, end_date):
    """データを読み込み、前処理（欠損値補完、標準化）を行う。"""
    logging.info(f"データファイル '{filepath}' を読み込み中...")
    start_time = time.time()
    try:
        with xr.open_dataset(filepath, engine='netcdf4') as ds:
            ds_labeled = ds.sel(valid_time=slice(start_date, end_date))
            msl_data = ds_labeled['msl'].values.astype(np.float64)
            labels = [l.decode('utf-8') if isinstance(l, bytes) else str(l) for l in ds_labeled['label'].values]
            lat = ds_labeled['latitude'].values
            lon = ds_labeled['longitude'].values
            valid_time = ds_labeled['valid_time'].values
    except FileNotFoundError:
        logging.error(f"データファイルが見つかりません: {filepath}")
        raise
    except Exception as e:
        logging.error(f"データ読み込み中にエラーが発生しました: {e}")
        raise

    logging.info(f"データ読み込み完了。対象期間: {start_date} ～ {end_date}")
    n_samples, n_lat, n_lon = msl_data.shape
    logging.info(f"サンプル数: {n_samples}, 緯度点数: {n_lat}, 経度点数: {n_lon}, 総次元数: {n_lat * n_lon}")
    
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
    
    return (data_normalized.astype(np.float64), labels, lat, lon, valid_time,
            data_mean.astype(np.float64), data_std.astype(np.float64))

# ==============================================================================
# --- 4. 次元削減モデル ---
# ==============================================================================

def run_mppca(data_np, params, device, output_dir):
    """MPPCAモデルをPyTorchで訓練し、結果を返す。"""
    logging.info("--- MPPCAの実行を開始 ---")
    p, q, niter, batch_size = params['p_clusters'], params['q_latent_dim'], params['n_iter_mppca'], params['mppca_batch_size']
    logging.info(f"パラメータ: クラスター数(p)={p}, 潜在次元(q)={q}, 反復回数={niter}, バッチサイズ={batch_size}")
    start_time = time.time()
    
    data_t = torch.from_numpy(data_np).to(device, dtype=torch.float64)
    
    logging.info(f"K-meansによる初期化を実行中 ({device})...")
    pi, mu, W, sigma2, _ = initialization_kmeans_torch(data_t, p, q, device=device)
    logging.info("K-means初期化完了。")

    logging.info(f"GEMアルゴリズムによる訓練を実行中 ({device})...")
    pi, mu, W, sigma2, R, L, _ = mppca_gem_torch(
        X=data_t, pi=pi, mu=mu, W=W, sigma2=sigma2,
        niter=niter, batch_size=batch_size, device=device
    )
    end_time = time.time()
    logging.info(f"MPPCA訓練完了。所要時間: {format_duration(end_time - start_time)}")
    
    L_cpu = L.cpu().numpy()
    if L_cpu is not None and len(L_cpu) > 0 and np.isfinite(L_cpu[-1]):
        logging.info(f"最終的な対数尤度: {L_cpu[-1]:.4f}")
    else:
        logging.warning("最終的な対数尤度が'nan'または無効です。")
        
    mppca_results = {
        'pi': pi.cpu().numpy(), 'mu': mu.cpu().numpy(), 'W': W.cpu().numpy(),
        'sigma2': sigma2.cpu().numpy(), 'R': R.cpu().numpy(), 'L': L_cpu,
        'params': params
    }
    result_path = os.path.join(output_dir, 'mppca_model.pkl')
    with open(result_path, 'wb') as f:
        pickle.dump(mppca_results, f)
    logging.info(f"MPPCA結果を '{result_path}' に保存しました。")
    
    # SOMへの入力として事後確率 R を返す
    return R.cpu().numpy(), mppca_results

def run_pca(data_np, params, output_dir):
    """PCAを実行し、結果を返す。"""
    logging.info("--- PCAの実行を開始 ---")
    n_components = params['n_components']
    logging.info(f"パラメータ: 主成分数={n_components}")
    start_time = time.time()
    
    pca = PCA(n_components=n_components, random_state=GLOBAL_SEED)
    # PCAで次元削減したデータ (主成分スコア)
    pca_scores = pca.fit_transform(data_np)
    
    end_time = time.time()
    logging.info(f"PCA実行完了。所要時間: {format_duration(end_time - start_time)}")
    logging.info(f"累積寄与率: {np.sum(pca.explained_variance_ratio_):.4f}")

    pca_results = {
        'model': pca,
        'scores': pca_scores,
        'params': params
    }
    result_path = os.path.join(output_dir, 'pca_model.pkl')
    with open(result_path, 'wb') as f:
        pickle.dump(pca_results, f)
    logging.info(f"PCA結果を '{result_path}' に保存しました。")
    
    # SOMへの入力として主成分スコアを返す
    return pca_scores, pca_results

# ==============================================================================
# --- 5. SOM & 評価 ---
# ==============================================================================

def run_som(data_for_som, params, output_dir):
    """MiniSomを訓練し、モデルを返す。"""
    logging.info("--- SOMの訓練を開始 ---")
    map_x, map_y, sigma, lr, n_iter = params['map_x'], params['map_y'], params['sigma_som'], params['learning_rate_som'], params['n_iter_som']
    input_len = data_for_som.shape[1]
    
    logging.info(f"パラメータ: マップサイズ={map_x}x{map_y}, 入力次元={input_len}, sigma={sigma}, LR={lr}, 反復回数={n_iter}")
    start_time = time.time()

    som = MiniSom(map_x, map_y, input_len,
                  sigma=sigma, learning_rate=lr,
                  random_seed=GLOBAL_SEED)

    if np.isnan(data_for_som).any():
        logging.warning("SOMへの入力データにnanが含まれています。nanを行の平均で補完して初期化します。")
        # nanを補完して初期化
        init_data = np.nan_to_num(data_for_som, nan=np.nanmean(data_for_som))
        som.pca_weights_init(init_data)
    else:
        som.pca_weights_init(data_for_som)
    
    logging.info("SOM訓練中 (バッチ学習)...")
    q_error_history = som.train_batch(data_for_som, n_iter, verbose=True, log_interval=max(1, n_iter // 100))
    
    end_time = time.time()
    logging.info(f"SOM訓練完了。所要時間: {format_duration(end_time - start_time)}")
    
    model_path = os.path.join(output_dir, 'som_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(som, f)
    logging.info(f"SOMモデルを '{model_path}' に保存しました。")
    
    return som, q_error_history

def evaluate_classification(som, data_for_som, original_labels):
    """マクロ平均再現率を用いて分類性能を評価する。"""
    logging.info("--- 分類性能の評価を開始 ---")
    start_time = time.time()

    if np.isnan(data_for_som).any():
        logging.error("SOMへの入力データにnanが含まれており、評価をスキップします。")
        return 0.0, {}

    # 1. 各ノードの優勢ラベルを決定
    parsed_true_labels = [parse_label(l) for l in original_labels]
    win_map_indices = som.win_map(data_for_som, return_indices=True)
    node_dominant_label = {}
    for pos, indices in win_map_indices.items():
        if not indices: continue
        labels_in_node = [label for i in indices for label in parsed_true_labels[i]]
        if not labels_in_node: continue
        most_common = Counter(labels_in_node).most_common(1)
        node_dominant_label[pos] = most_common[0][0]

    # 2. 各データポイントの予測ラベルを決定
    winners = [som.winner(x) for x in data_for_som]
    predicted_single_labels = [node_dominant_label.get(w) for w in winners]
    
    # 3. 再現率を計算
    mlb = MultiLabelBinarizer(classes=BASE_LABELS)
    y_true = mlb.fit_transform(parsed_true_labels)
    y_pred = mlb.transform([[l] if l else [] for l in predicted_single_labels])
    
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    logging.info(f"評価完了。所要時間: {format_duration(time.time() - start_time)}")
    logging.info(f"マクロ平均再現率: {macro_recall:.4f}")
    recall_scores_str = "\n".join([f" - {label:<4}: {recall:.4f}" for label, recall in zip(mlb.classes_, per_class_recall)])
    logging.info(f"クラスごとの再現率:\n{recall_scores_str}")
    
    return macro_recall, node_dominant_label

# ==============================================================================
# --- 6. 可視化 ---
# ==============================================================================

def visualize_results(results, output_dir):
    """実験結果をまとめて可視化する。"""
    logging.info("--- 結果の可視化を開始 ---")
    start_time = time.time()

    # --- 共通の可視化 ---
    plot_quantization_error(results['q_error_history'], results['params']['n_iter_som'], output_dir)
    plot_som_maps(results['som'], results['data_for_som'], results['node_dominant_label'], output_dir)

    # --- 手法別の可視化 ---
    if results['params']['method'] == 'MPPCA':
        plot_log_likelihood(results['mppca_results']['L'], output_dir)
        plot_mppca_average_patterns(results, output_dir)
        plot_latent_space(results, output_dir)
        plot_reconstructions(results, output_dir, n_samples=5)
    elif results['params']['method'] == 'PCA':
        plot_pca_components_as_patterns(results, output_dir)
        plot_latent_space(results, output_dir)
        plot_reconstructions(results, output_dir, n_samples=5)

    end_time = time.time()
    logging.info(f"可視化完了。所要時間: {format_duration(end_time - start_time)}")

def plot_log_likelihood(log_likelihood, output_dir):
    """MPPCAの対数尤度の収束グラフをプロット。"""
    plt.figure(figsize=(10, 6))
    valid_indices = ~np.isnan(log_likelihood)
    if not np.any(valid_indices):
        logging.warning("対数尤度データが無効なため、グラフは作成されません。")
        plt.close()
        return
    plt.plot(np.arange(len(log_likelihood))[valid_indices], log_likelihood[valid_indices])
    plt.title('MPPCA 対数尤度の収束履歴')
    plt.xlabel('イテレーション'); plt.ylabel('対数尤度'); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'log_likelihood_convergence.png'), dpi=300)
    plt.close()

def plot_quantization_error(q_error_history, n_iter, output_dir):
    """SOMの量子化誤差の収束グラフをプロット。"""
    plt.figure(figsize=(10, 6))
    log_interval = max(1, n_iter // 100)
    iterations = np.arange(len(q_error_history)) * log_interval
    plt.plot(iterations, q_error_history)
    plt.title('SOM 量子化誤差の収束履歴')
    plt.xlabel('イテレーション'); plt.ylabel('量子化誤差'); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'som_quantization_error.png'), dpi=300)
    plt.close()

def plot_som_maps(som, data_for_som, node_dominant_label, output_dir):
    """U-Matrix, ヒットマップ, 勝者総取りマップ等をプロット。"""
    map_x, map_y = som.get_weights().shape[:2]
    
    # U-Matrix
    plt.figure(figsize=(10, 10)); plt.pcolor(som.distance_map().T, cmap='bone_r'); plt.colorbar(label='ニューロン間の距離'); plt.title('U-Matrix'); plt.grid(True); plt.savefig(os.path.join(output_dir, 'som_u_matrix.png'), dpi=300); plt.close()

    # Hit Map
    frequencies = som.activation_response(data_for_som)
    plt.figure(figsize=(10, 10)); plt.pcolor(frequencies.T, cmap='viridis'); plt.colorbar(label='勝者となった回数'); plt.title('ヒットマップ'); plt.grid(True); plt.savefig(os.path.join(output_dir, 'som_hit_map.png'), dpi=300); plt.close()

    # Winner-takes-all Map
    plt.figure(figsize=(12, 10)); label_map = np.full((map_x, map_y), -1, dtype=int); label_names = {label: i for i, label in enumerate(BASE_LABELS)};
    for pos, label in node_dominant_label.items():
        if label: label_map[pos[0], pos[1]] = label_names[label]
    cmap = plt.get_cmap('tab20', len(BASE_LABELS)); plt.pcolor(label_map.T, cmap=cmap, vmin=-0.5, vmax=len(BASE_LABELS)-0.5); cbar = plt.colorbar(ticks=np.arange(len(BASE_LABELS))); cbar.ax.set_yticklabels(BASE_LABELS); plt.title('勝者総取りマップ'); plt.grid(True); plt.savefig(os.path.join(output_dir, 'som_winner_takes_all_map.png'), dpi=300); plt.close()

def plot_mppca_average_patterns(results, output_dir):
    """MPPCAで学習された平均気圧配置パターンを地図上にプロット。"""
    mu = results['mppca_results']['mu']
    lat, lon, data_mean = results['lat'], results['lon'], results['data_mean']
    p, _ = mu.shape
    n_cols = min(5, p)
    n_rows = (p + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3), subplot_kw={"projection": ccrs.PlateCarree()})
    axes = np.atleast_1d(axes).flatten()
    
    for i in range(p):
        ax = axes[i]
        # 標準化を戻してから平均を引くことで「偏差」を計算
        mean_pattern_hpa = (mu[i] * results['data_std'] + data_mean) / 100.0
        mean_anomaly_hpa = mean_pattern_hpa - (data_mean.mean() / 100.0) # 全体平均からの偏差
        pattern_2d = mean_anomaly_hpa.reshape(len(lat), len(lon))
        
        norm = Normalize(vmin=-12, vmax=12)
        cont = ax.contourf(lon, lat, pattern_2d, cmap='coolwarm', levels=20, norm=norm, extend='both')
        ax.contour(lon, lat, pattern_2d, colors='k', linewidths=0.5, levels=15)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="black", linewidth=0.5)
        ax.set_title(f'Cluster {i+1} Mean Anomaly')

    for i in range(p, len(axes)): axes[i].set_visible(False)
    fig.colorbar(cont, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1, label='Pressure Anomaly (hPa)')
    plt.suptitle('MPPCA: Learned Average Pressure Anomaly Patterns', fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'mppca_average_patterns.png'), dpi=300)
    plt.close()

def plot_pca_components_as_patterns(results, output_dir):
    """PCAの主成分を気圧配置パターンとして地図上にプロット。"""
    pca = results['pca_results']['model']
    lat, lon = results['lat'], results['lon']
    n_components = min(10, pca.n_components_) # 上位10個まで表示
    n_cols = min(5, n_components)
    n_rows = (n_components + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3), subplot_kw={"projection": ccrs.PlateCarree()})
    axes = np.atleast_1d(axes).flatten()

    for i in range(n_components):
        ax = axes[i]
        component_2d = pca.components_[i].reshape(len(lat), len(lon))
        vmax = np.abs(component_2d).max()
        norm = Normalize(vmin=-vmax, vmax=vmax)
        cont = ax.contourf(lon, lat, component_2d, cmap='RdBu_r', levels=20, norm=norm)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="black", linewidth=0.5)
        ax.set_title(f'PC {i+1} (EVR: {pca.explained_variance_ratio_[i]:.3f})')
        
    for i in range(n_components, len(axes)): axes[i].set_visible(False)
    fig.colorbar(cont, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1, label='Component Loading')
    plt.suptitle('PCA: Principal Component Patterns', fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'pca_component_patterns.png'), dpi=300)
    plt.close()

def plot_latent_space(results, output_dir):
    """次元削減後の潜在空間の分布をプロット。"""
    data = results['data_for_som']
    if data.shape[1] < 2:
        logging.warning("潜在次元が2未満のため、潜在空間プロットはスキップします。")
        return
        
    som = results['som']
    winners = [som.winner(x) for x in data]
    # ノード位置を色にマッピング
    colors = [f"C{w[0] * som._weights.shape[1] + w[1]}" for w in winners]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(data[:, 0], data[:, 1], c=colors, alpha=0.5)
    plt.title('Latent Space Visualization (First 2 Dimensions)')
    plt.xlabel('Dimension 1'); plt.ylabel('Dimension 2'); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latent_space.png'), dpi=300)
    plt.close()

def plot_reconstructions(results, output_dir, n_samples=5):
    """元のデータと再構成データを比較プロット。"""
    # ... この関数の実装は長くなるため、骨子のみ示します ...
    logging.info(f"{n_samples}個のサンプルで再構成を比較します。")
    # `results`から必要な情報を取得 (data_normalized, lat, lon, data_mean, data_std, etc.)
    # methodに応じて再構成ロジックを分岐
    # MPPCA: x_rec = W[c] @ z + mu[c]
    # PCA: x_rec = pca.inverse_transform(scores)
    # 元データ、再構成データ、(MPPCAなら平均パターンも)を逆標準化してプロット
    # cartopyを使って地図上に描画
    pass # この部分はmain.pyのロジックを参考に実装します

# ==============================================================================
# --- 7. メイン実行ブロック ---
# ==============================================================================

def main():
    """メインの処理フロー。定義された実験を順次実行する。"""
    os.makedirs(PARENT_RESULT_DIR, exist_ok=True)
    main_start_time = time.time()
    
    for i, exp_params in enumerate(EXPERIMENTS):
        exp_name = exp_params['name']
        exp_dir = os.path.join(PARENT_RESULT_DIR, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        logger = setup_logger(exp_dir, exp_name)
        
        logger.info(f"======= 実験 {i+1}/{len(EXPERIMENTS)}: '{exp_name}' を開始します =======")
        start_time_exp = time.time()
        
        # --- パラメータ設定 ---
        params = DEFAULT_PARAMS.copy()
        params.update(exp_params)
        
        logger.info("--- 実験パラメータ ---")
        for k, v in params.items():
            logger.info(f"- {k}: {v}")
        logger.info("--------------------")

        try:
            # --- 乱数シード設定 ---
            set_global_seed(GLOBAL_SEED)
            
            # --- デバイス設定 ---
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"使用デバイス: {device}")
            if str(device) == "cuda":
                logger.info(f"GPU名: {torch.cuda.get_device_name(0)}")

            # 1. データ読み込みと前処理
            data_normalized, labels, lat, lon, valid_time, data_mean, data_std = \
                load_and_preprocess_data(DATA_FILES[params['data_key']], START_DATE, END_DATE)

            # 2. 次元削減
            data_for_som, model_results = None, None
            if params['method'] == 'MPPCA':
                data_for_som, model_results = run_mppca(data_normalized, params, device, exp_dir)
            elif params['method'] == 'PCA':
                data_for_som, model_results = run_pca(data_normalized, params, exp_dir)
            else:
                raise ValueError(f"未知のメソッドです: {params['method']}")

            # 3. SOMの実行
            som, q_error_history = run_som(data_for_som, params, exp_dir)
            
            # 4. 評価
            macro_recall, node_dominant_label = evaluate_classification(som, data_for_som, labels)
            
            # 5. 可視化
            # 可視化関数に渡すための結果をまとめる
            viz_results = {
                'params': params,
                'data_normalized': data_normalized, 'labels': labels, 'lat': lat, 'lon': lon,
                'data_mean': data_mean, 'data_std': data_std, 'valid_time': valid_time,
                'data_for_som': data_for_som,
                'som': som, 'q_error_history': q_error_history,
                'node_dominant_label': node_dominant_label,
                'mppca_results': model_results if params['method'] == 'MPPCA' else None,
                'pca_results': model_results if params['method'] == 'PCA' else None,
            }
            visualize_results(viz_results, exp_dir)

        except Exception as e:
            logger.error(f"実験 '{exp_name}' の実行中に致命的なエラーが発生しました。", exc_info=True)
            # エラー情報をファイルに書き出す
            with open(os.path.join(exp_dir, "ERROR.txt"), "w") as f:
                f.write(f"An error occurred during experiment: {exp_name}\n")
                import traceback
                traceback.print_exc(file=f)

        finally:
            end_time_exp = time.time()
            logger.info(f"======= 実験 '{exp_name}' 完了。所要時間: {format_duration(end_time_exp - start_time_exp)} =======")
            print("\n" + "="*80 + "\n") # コンソール上で実験の区切りを明確にする

    main_end_time = time.time()
    print(f"全実験が完了しました。総実行時間: {format_duration(main_end_time - main_start_time)}")


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning, module='minisom')
    warnings.filterwarnings('ignore', category=RuntimeWarning) # nanmeanなどで出る警告を抑制
    main()