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
- ベストモデル選択: SOM学習中に分類性能（マクロ平均再現率）を監視し、最も性能の高いモデルを最終結果として採用。
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
    'n_iter_mppca': 1000,
    'mppca_batch_size': 1024,
    # PCA
    'n_components': 20,
    # MiniSom
    'map_x': 7,
    'map_y': 7,
    'sigma_som': 3.0,
    'learning_rate_som': 1.0,
    'n_iter_som': 100000, # バッチ学習のため、以前より少ないイテレーションで収束
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

def format_coord_vector(vector, precision=4):
    """潜在空間の座標ベクトルを整形して複数行の文字列として返す"""
    return np.array2string(
        vector,
        formatter={'float_kind': lambda x: f"{x:.{precision}f}"},
        separator=', ',
        max_line_width=100
    )
    
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

def calculate_all_latent_coords(X, mu, W, sigma2, device):
    """各データポイントについて、全てのクラスタにおける潜在空間座標を計算し、連結する"""
    p, d, q = W.shape
    N, _ = X.shape
    all_coords_list = []
    
    W_T = W.transpose(-2, -1)
    I_q = torch.eye(q, device=device, dtype=X.dtype)
    
    logging.info(f"全{p}個の潜在空間における座標を計算中...")
    for i in tqdm(range(p), desc="Calculating all latent coords"):
        mu_c, W_c, W_T_c, sigma2_c = mu[i], W[i], W_T[i], sigma2[i]
        M_c = sigma2_c * I_q + W_T_c @ W_c
        M_inv_c = torch.linalg.inv(M_c)
        
        deviation = X - mu_c
        # (N, d) @ (d, q) -> (N, q)
        coords = (deviation @ W_c @ M_inv_c.T)
        all_coords_list.append(coords)
        
    # (N, q)のテンソルがp個入ったリストを、(N, p*q)の単一テンソルに連結
    concatenated_coords = torch.cat(all_coords_list, dim=1)
    logging.info(f"潜在空間座標の連結完了。最終的な次元: {concatenated_coords.shape}")
    
    return concatenated_coords

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
        
    # --- 全ての潜在空間における座標を計算 ---
    all_latent_coords = calculate_all_latent_coords(data_t, mu, W, sigma2, device)

    mppca_results = {
        'pi': pi.cpu().numpy(), 'mu': mu.cpu().numpy(), 'W': W.cpu().numpy(),
        'sigma2': sigma2.cpu().numpy(), 'R': R.cpu().numpy(), 'L': L_cpu,
        'all_latent_coords': all_latent_coords.cpu().numpy(), # ★★★ 新規追加 ★★★
        'params': params
    }
    result_path = os.path.join(output_dir, 'mppca_model.pkl')
    with open(result_path, 'wb') as f:
        pickle.dump(mppca_results, f)
    logging.info(f"MPPCA結果を '{result_path}' に保存しました。")
    
    # SOMへの入力として連結された潜在座標を返す
    return all_latent_coords.cpu().numpy(), mppca_results

def run_pca(data_np, params, output_dir):
    """PCAを実行し、結果を返す。"""
    logging.info("--- PCAの実行を開始 ---")
    n_components = params['n_components']
    logging.info(f"パラメータ: 主成分数={n_components}")
    start_time = time.time()
    
    pca = PCA(n_components=n_components, random_state=GLOBAL_SEED)
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
    
    return pca_scores, pca_results

def log_mppca_sample_details(R_np, all_latent_coords_np, time_stamps, params, n_samples=5):
    """MPPCAの結果について、個々のサンプルの詳細をログに出力する"""
    logging.info("--- 個々のデータに着目したMPPCA結果の分析 ---")
    if n_samples == 0: return
    
    logging.info(f"最初の{min(n_samples, len(R_np))}個のサンプルについて、詳細な結果を表示します。")
    
    p = params['p_clusters']
    q = params['q_latent_dim']
    
    cluster_assignments = np.argmax(R_np, axis=1)

    for i in range(min(n_samples, len(R_np))):
        date_str = pd.to_datetime(str(time_stamps[i])).strftime('%Y-%m-%d')
        r_i = R_np[i]
        assigned_cluster = cluster_assignments[i]
        max_prob = r_i[assigned_cluster]
        
        logging.info(f"\n[サンプル {i+1} ({date_str})]")
        logging.info(f"  - 最も可能性の高いクラスター: {assigned_cluster + 1} (確率: {max_prob:.4f})")
        
        # 主潜在空間の座標を抽出
        main_latent_coords_i = all_latent_coords_np[i, assigned_cluster*q : (assigned_cluster+1)*q]
        latent_coords_str = format_coord_vector(main_latent_coords_i)
        logging.info(f"  - (主)潜在空間での座標 (次元数: {q}):\n {latent_coords_str}")

        prob_vector_str = ", ".join([f"{prob:.4f}" for prob in r_i])
        logging.info(f"  - 全クラスターへの所属確率ベクトル [R_i]: [{prob_vector_str}]")
        
        logging.info(f"  - 全{p}個の潜在空間での座標:")
        for c in range(p):
            # 連結されたベクトルから該当クラスタの座標部分をスライス
            coord_c = all_latent_coords_np[i, c*q : (c+1)*q]
            coord_c_str = format_coord_vector(coord_c)
            logging.info(f"    - Cluster {c+1:>2} の場合:\n {coord_c_str}")

# ==============================================================================
# --- 5. SOM & 評価 ---
# ==============================================================================

def run_som(data_for_som, original_labels, params, output_dir):
    """
    MiniSomを訓練し、最も性能の良いモデルを保存する。
    訓練中にマクロ平均再現率を監視し、グラフ化する。
    """
    logging.info("--- SOMの訓練と評価を開始 ---")
    map_x, map_y, sigma, lr, n_iter = params['map_x'], params['map_y'], params['sigma_som'], params['learning_rate_som'], params['n_iter_som']
    input_len = data_for_som.shape[1]
    
    logging.info(f"パラメータ: マップサイズ={map_x}x{map_y}, 入力次元={input_len}, sigma={sigma}, LR={lr}, 反復回数={n_iter}")
    start_time = time.time()

    som = MiniSom(map_x, map_y, input_len,
                  sigma=sigma, learning_rate=lr,
                  random_seed=GLOBAL_SEED)

    if np.isnan(data_for_som).any():
        logging.warning("SOMへの入力データにnanが含まれています。nanを行の平均で補完して初期化します。")
        init_data = np.nan_to_num(data_for_som, nan=np.nanmean(data_for_som))
        som.pca_weights_init(init_data)
    else:
        som.pca_weights_init(data_for_som)
    
    best_recall = -1.0
    best_som_weights = None
    recall_history = []
    iterations_history = []
    log_interval = max(1, n_iter // 100)

    def evaluation_and_save_callback(som_instance, current_iter):
        nonlocal best_recall, best_som_weights
        macro_recall, _ = evaluate_classification(som_instance, data_for_som, original_labels, verbose=False)
        recall_history.append(macro_recall)
        iterations_history.append(current_iter)
        
        if hasattr(som_instance.tqdm_pbar, 'set_postfix'):
            current_postfix = som_instance.tqdm_pbar.postfix or ""
            q_error_str = current_postfix.split(',')[0]
            som_instance.tqdm_pbar.set_postfix_str(f"{q_error_str}, recall={macro_recall:.4f}")

        if macro_recall > best_recall:
            best_recall = macro_recall
            best_som_weights = som_instance.get_weights().copy()
            logging.info(f"Iter {current_iter}: 新しいベスト再現率を記録: {best_recall:.4f}")

    logging.info("SOM訓練中 (バッチ学習)... ベストモデルを選択します。")
    q_error_history = som.train_batch(data_for_som, n_iter, verbose=True, 
                                      log_interval=log_interval,
                                      evaluation_callback=evaluation_and_save_callback)
    
    if best_som_weights is not None:
        som._weights = best_som_weights
        logging.info(f"最終モデルを、最も再現率の高かったモデル ({best_recall:.4f}) に設定しました。")
    else:
        logging.warning("ベストモデルが見つかりませんでした。学習最終時点のモデルを使用します。")
        
    end_time = time.time()
    logging.info(f"SOM訓練完了。所要時間: {format_duration(end_time - start_time)}")
    
    model_path = os.path.join(output_dir, 'som_model.pkl')
    if hasattr(som, 'tqdm_pbar'): del som.tqdm_pbar
    with open(model_path, 'wb') as f:
        pickle.dump(som, f)
    logging.info(f"SOMベストモデルを '{model_path}' に保存しました。")
    
    recall_data = (iterations_history, recall_history)
    return som, q_error_history, recall_data

def evaluate_classification(som, data_for_som, original_labels, verbose=True):
    """マクロ平均再現率を用いて分類性能を評価する。"""
    if verbose:
        logging.info("--- 分類性能の評価を開始 ---")
        start_time = time.time()

    if np.isnan(data_for_som).any():
        logging.error("SOMへの入力データにnanが含まれており、評価をスキップします。")
        return 0.0, {}

    parsed_true_labels = [parse_label(l) for l in original_labels]
    win_map_indices = som.win_map(data_for_som, return_indices=True)
    node_dominant_label = {}
    for pos, indices in win_map_indices.items():
        if not indices: continue
        labels_in_node = [label for i in indices for label in parsed_true_labels[i]]
        if not labels_in_node: continue
        most_common = Counter(labels_in_node).most_common(1)
        node_dominant_label[pos] = most_common[0][0]

    winners = [som.winner(x) for x in data_for_som]
    predicted_single_labels = [node_dominant_label.get(w) for w in winners]
    
    mlb = MultiLabelBinarizer(classes=BASE_LABELS)
    y_true = mlb.fit_transform(parsed_true_labels)
    y_pred = mlb.transform([[l] if l else [] for l in predicted_single_labels])
    
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    if verbose:
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

    plot_quantization_error(results['q_error_history'], results['params']['n_iter_som'], output_dir)
    plot_som_maps(results['som'], results['data_for_som'], results['node_dominant_label'], output_dir)
    plot_recall_convergence(results['recall_data'], output_dir)
    plot_som_node_average_patterns(results, output_dir)
    plot_som_label_distribution_maps(results, output_dir)
    plot_som_seasonal_distribution(results, output_dir)

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
    ll_finite = log_likelihood[np.isfinite(log_likelihood)]
    if len(ll_finite) < 10:
        logging.warning("有効な対数尤度データが少なすぎるため、外れ値処理をスキップします。")
        plt.plot(log_likelihood)
    else:
        stable_start_idx = int(len(ll_finite) * 0.1)
        ll_stable = ll_finite[stable_start_idx:]
        q1, q3 = np.percentile(ll_stable, 25), np.percentile(ll_stable, 75)
        iqr = q3 - q1
        upper_bound = q3 + 3.0 * iqr 
        ll_plot = np.copy(log_likelihood)
        outlier_indices = np.where(ll_plot > upper_bound)[0]
        ll_plot[outlier_indices] = np.nan
        plt.plot(np.arange(len(ll_plot)), ll_plot, label='対数尤度 (正常範囲)')
        if len(outlier_indices) > 0:
            logging.info(f"{len(outlier_indices)}個の対数尤度の外れ値を検出しました。")
            for i, idx in enumerate(outlier_indices):
                label = '外れ値検出位置' if i == 0 else ""
                plt.axvline(x=idx, color='r', linestyle='--', linewidth=1, label=label)
            plt.legend()

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

def plot_recall_convergence(recall_data, output_dir):
    """SOM学習中のマクロ平均再現率の収束グラフをプロット。"""
    iterations, recall_history = recall_data
    if not iterations or not recall_history:
        logging.warning("再現率の履歴データがないため、グラフは作成されません。")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, recall_history, marker='.', linestyle='-', markersize=4)
    if recall_history:
        best_recall_idx = np.argmax(recall_history)
        plt.scatter(iterations[best_recall_idx], recall_history[best_recall_idx],
                    color='red', s=100, zorder=5, label=f'Best Recall: {recall_history[best_recall_idx]:.4f}')
    
    plt.title('SOM学習過程におけるマクロ平均再現率の変化')
    plt.xlabel('イテレーション'); plt.ylabel('マクロ平均再現率')
    plt.grid(True); plt.legend(); plt.ylim(bottom=0); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'recall_convergence.png'), dpi=300)
    plt.close()

def plot_som_maps(som, data_for_som, node_dominant_label, output_dir):
    """U-Matrix, ヒットマップ, 勝者総取りマップ等をプロット。"""
    map_x, map_y = som.get_weights().shape[:2]
    
    plt.figure(figsize=(10, 10)); plt.pcolor(som.distance_map().T, cmap='bone_r'); plt.colorbar(label='ニューロン間の距離'); plt.title('U-Matrix'); plt.grid(True); plt.savefig(os.path.join(output_dir, 'som_u_matrix.png'), dpi=300); plt.close()
    
    frequencies = som.activation_response(data_for_som)
    plt.figure(figsize=(10, 10)); plt.pcolor(frequencies.T, cmap='viridis'); plt.colorbar(label='勝者となった回数'); plt.title('ヒットマップ'); plt.grid(True); plt.savefig(os.path.join(output_dir, 'som_hit_map.png'), dpi=300); plt.close()

    plt.figure(figsize=(12, 10)); label_map = np.full((map_x, map_y), -1, dtype=int); label_names = {label: i for i, label in enumerate(BASE_LABELS)};
    for pos, label in node_dominant_label.items():
        if label: label_map[pos[0], pos[1]] = label_names.get(label, -1)
    cmap = plt.get_cmap('tab20', len(BASE_LABELS)); plt.pcolor(label_map.T, cmap=cmap, vmin=-0.5, vmax=len(BASE_LABELS)-0.5); cbar = plt.colorbar(ticks=np.arange(len(BASE_LABELS))); cbar.ax.set_yticklabels(BASE_LABELS); plt.title('勝者総取りマップ'); plt.grid(True); plt.savefig(os.path.join(output_dir, 'som_winner_takes_all_map.png'), dpi=300); plt.close()

def plot_mppca_average_patterns(results, output_dir):
    """MPPCAで学習された平均気圧配置パターンを地図上にプロット。"""
    mu = results['mppca_results']['mu']
    lat, lon, data_mean, data_std = results['lat'], results['lon'], results['data_mean'], results['data_std']
    p, _ = mu.shape
    n_cols = min(5, p); n_rows = (p + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3), subplot_kw={"projection": ccrs.PlateCarree()})
    axes = np.atleast_1d(axes).flatten()
    
    for i in range(p):
        ax = axes[i]
        mean_pattern_hpa = (mu[i] * data_std + data_mean) / 100.0
        mean_anomaly_hpa = mean_pattern_hpa - (data_mean.mean() / 100.0)
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
    n_components = min(10, pca.n_components_)
    n_cols = min(5, n_components); n_rows = (n_components + n_cols - 1) // n_cols

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
    som = results['som']
    params = results['params']

    if params['method'] == 'PCA':
        if data.shape[1] < 2:
            logging.warning("潜在次元が2未満のため、潜在空間プロットはスキップします。")
            return
        
        winners = [som.winner(x) for x in data]
        colors = [f"C{w[0] * som._weights.shape[1] + w[1]}" for w in winners]
        
        plt.figure(figsize=(10, 8))
        plt.scatter(data[:, 0], data[:, 1], c=colors, alpha=0.5)
        plt.title('Latent Space Visualization (First 2 PCA Dimensions)')
        plt.xlabel('Principal Component 1'); plt.ylabel('Principal Component 2'); plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'latent_space.png'), dpi=300)
        plt.close()
    
    elif params['method'] == 'MPPCA':
        R = results['mppca_results']['R']
        all_latent_coords = results['mppca_results']['all_latent_coords']
        cluster_assignments = np.argmax(R, axis=1)
        
        p = params['p_clusters']
        q = params['q_latent_dim']
        
        if q < 2:
            logging.warning("潜在次元が2未満のため、MPPCA潜在空間プロットはスキップします。")
            return

        # 主潜在空間の座標を抽出
        main_coords = np.zeros((len(data), q))
        for i in range(len(data)):
            c = cluster_assignments[i]
            main_coords[i] = all_latent_coords[i, c*q:(c+1)*q]
            
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(main_coords[:, 0], main_coords[:, 1], c=cluster_assignments, cmap='tab20', alpha=0.6, s=10)
        plt.title('Latent Space Visualization (Main clusters, first 2 dims)')
        plt.xlabel('Latent Dimension 1'); plt.ylabel('Latent Dimension 2')
        plt.legend(handles=scatter.legend_elements()[0], labels=[f'Cluster {i+1}' for i in range(p)], bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True); plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(os.path.join(output_dir, 'latent_space.png'), dpi=300)
        plt.close()

def plot_reconstructions(results, output_dir, n_samples=5):
    """元のデータと再構成データを比較プロット。"""
    logging.info(f"{n_samples}個のサンプルで再構成を比較します。")
    if n_samples == 0: return

    # ... (この関数の実装は省略します) ...
    pass

def plot_som_node_average_patterns(results, output_dir):
    """SOMの各ノードが代表する平均的な気圧配置パターンをプロット"""
    logging.info("SOMノードごとの平均気圧パターンのプロットを開始...")
    
    som = results['som']
    data_normalized = results['data_normalized']
    data_for_som = results['data_for_som']
    lat, lon = results['lat'], results['lon']
    
    map_x, map_y = som.get_weights().shape[:2]
    win_map_indices = som.win_map(data_for_som, return_indices=True)
    
    mean_patterns = np.full((map_x, map_y, data_normalized.shape[1]), np.nan)
    node_counts = np.zeros((map_x, map_y), dtype=int)
    
    for (i, j), indices_list in win_map_indices.items():
        if indices_list:
            mean_patterns[i, j, :] = data_normalized[indices_list].mean(axis=0)
            node_counts[i, j] = len(indices_list)

    cmap = sns.color_palette("RdBu_r", as_cmap=True)
    norm = Normalize(vmin=-2.5, vmax=2.5)

    fig, axes = plt.subplots(nrows=map_y, ncols=map_x, figsize=(map_x*3, map_y*3), subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.T[::-1, :]
    fig.subplots_adjust(wspace=0.06, hspace=0.06)

    for i in range(map_x):
        for j in range(map_y):
            ax = axes[i, j]
            if not np.isnan(mean_patterns[i, j, :]).all():
                pattern_2d = mean_patterns[i, j, :].reshape(len(lat), len(lon))
                cont = ax.contourf(lon, lat, pattern_2d, levels=21, cmap=cmap, norm=norm, extend='both', transform=ccrs.PlateCarree())
                ax.contour(lon, lat, pattern_2d, levels=21, colors='k', linewidths=0.5, transform=ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='black')
                ax.text(0.05, 0.9, f'({i},{j})\nN={node_counts[i,j]}', transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.6))
            else:
                ax.axis('off')
            ax.set_xticks([]); ax.set_yticks([])

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(cont, cax=cbar_ax, label='Normalized Pressure Anomaly (Std. Dev.)')
    plt.suptitle('SOM Node Average Pressure Patterns', fontsize=16)
    fig.subplots_adjust(top=0.95, right=0.90)
    plt.savefig(os.path.join(output_dir, 'MeanSeaLevelPressurePatterns.png'), dpi=300)
    plt.close(fig)

def plot_som_label_distribution_maps(results, output_dir):
    """SOM上のデータ数とラベル分布のヒートマップを作成"""
    logging.info("SOM上のデータ数とラベル分布のヒートマップ作成を開始...")
    
    som = results['som']; data_for_som = results['data_for_som']; original_labels = results['labels']
    all_possible_labels = sorted(list(set(l for ol in original_labels for l in parse_label(ol))))
    map_x, map_y = som.get_weights().shape[:2]

    win_map_indices = som.win_map(data_for_som, return_indices=True)
    node_total_counts = np.zeros((map_x, map_y), dtype=int)
    node_label_counts = {label: np.zeros((map_x, map_y), dtype=int) for label in all_possible_labels}
    parsed_labels = [parse_label(l) for l in original_labels]

    for (i, j), indices in win_map_indices.items():
        node_total_counts[i, j] = len(indices)
        if not indices: continue
        labels_in_node = [label for idx in indices for label in parsed_labels[idx]]
        label_counts = Counter(labels_in_node)
        for label, count in label_counts.items():
            if label in node_label_counts: node_label_counts[label][i, j] += count
    
    plt.figure(figsize=(10, 8)); plt.title('Data Count on SOM (TrainMap)'); sns.heatmap(node_total_counts.T, annot=True, fmt='d', cmap='viridis'); plt.savefig(os.path.join(output_dir, 'TrainMap.png'), dpi=300); plt.close()

    for label in all_possible_labels:
        plt.figure(figsize=(10, 8)); plt.title(f'Data Count for Label "{label}" (*TypeMap)'); sns.heatmap(node_label_counts[label].T, annot=True, fmt='d', cmap='Reds'); plt.savefig(os.path.join(output_dir, f'{label}TypeMap.png'), dpi=300); plt.close()

    node_proportions = {label: np.nan_to_num(counts / node_total_counts) for label, counts in node_label_counts.items()}
    for label in all_possible_labels:
        plt.figure(figsize=(10, 8)); plt.title(f'Proportion of Label "{label}" (*TypeProportionMap)'); sns.heatmap(node_proportions[label].T, annot=True, fmt='.2f', cmap='coolwarm', vmin=0, vmax=1); plt.savefig(os.path.join(output_dir, f'{label}TypeProportionMap.png'), dpi=300); plt.close()

def plot_som_seasonal_distribution(results, output_dir):
    """SOMノードの季節・月ごとの分布図を作成"""
    logging.info("SOMノードの季節・月ごとの分布図の作成を開始...")
    
    som = results['som']; data_for_som = results['data_for_som']; valid_time = results['valid_time']
    map_x, map_y = som.get_weights().shape[:2]
    
    month_to_season = {m:s for s, months in {'冬': [12, 1, 2], '春': [3, 4, 5], '夏': [6, 7, 8], '秋': [9, 10, 11]}.items() for m in months}
    season_colors = {"春": "#FFE4E1", "夏": "#FFC1C1", "秋": "#F5DEB3", "冬": "#ADD8E6", "": "white"}
    
    win_map_indices = som.win_map(data_for_som, return_indices=True)
    dates = pd.to_datetime(valid_time)
    
    fig, axes = plt.subplots(nrows=map_y, ncols=map_x, figsize=(map_x * 1.5, map_y * 1.5))
    axes = axes.T

    for i in range(map_x):
        for j in range(map_y):
            ax = axes[i, j]; ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False)
            indices = win_map_indices.get((i, j), []); total_dates = len(indices)
            if total_dates == 0: ax.set_facecolor('white'); continue
            node_dates = dates[indices]
            max_season = Counter(month_to_season[d.month] for d in node_dates).most_common(1)[0][0]
            ax.set_facecolor(season_colors.get(max_season, "white"))
            top_months = Counter(d.month for d in node_dates).most_common(3)
            text = f"({i},{j}) N={total_dates}\n{max_season}\n" + "\n".join([f"{pd.to_datetime(f'2000-{m}-01').strftime('%b')}: {c/total_dates:.0%}" for m, c in top_months])
            ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=6, transform=ax.transAxes)

    plt.suptitle('Seasonal & Monthly Distribution on SOM Nodes', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'NodeMonthDistribution.png'), dpi=300)
    plt.close(fig)

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
        
        params = DEFAULT_PARAMS.copy(); params.update(exp_params)
        
        logger.info("--- 実験パラメータ ---")
        for k, v in params.items(): logger.info(f"- {k}: {v}")
        logger.info("--------------------")

        try:
            set_global_seed(GLOBAL_SEED)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"使用デバイス: {device}")
            if str(device) == "cuda": logger.info(f"GPU名: {torch.cuda.get_device_name(0)}")

            data_normalized, labels, lat, lon, valid_time, data_mean, data_std = \
                load_and_preprocess_data(DATA_FILES[params['data_key']], START_DATE, END_DATE)

            data_for_som, model_results = None, None
            if params['method'] == 'MPPCA':
                data_for_som, model_results = run_mppca(data_normalized, params, device, exp_dir)
                log_mppca_sample_details(model_results['R'], model_results['all_latent_coords'], valid_time, params, n_samples=5)
            elif params['method'] == 'PCA':
                data_for_som, model_results = run_pca(data_normalized, params, exp_dir)
            else:
                raise ValueError(f"未知のメソッドです: {params['method']}")

            som, q_error_history, recall_data = run_som(data_for_som, labels, params, exp_dir)
            
            macro_recall, node_dominant_label = evaluate_classification(som, data_for_som, labels)
            
            viz_results = {
                'params': params,
                'data_normalized': data_normalized, 'labels': labels, 'lat': lat, 'lon': lon,
                'data_mean': data_mean, 'data_std': data_std, 'valid_time': valid_time,
                'data_for_som': data_for_som,
                'som': som, 'q_error_history': q_error_history,
                'recall_data': recall_data,
                'node_dominant_label': node_dominant_label,
                'mppca_results': model_results if params['method'] == 'MPPCA' else None,
                'pca_results': model_results if params['method'] == 'PCA' else None,
            }
            visualize_results(viz_results, exp_dir)

        except Exception as e:
            logger.error(f"実験 '{exp_name}' の実行中に致命的なエラーが発生しました。", exc_info=True)
            with open(os.path.join(exp_dir, "ERROR.txt"), "w") as f:
                f.write(f"An error occurred during experiment: {exp_name}\n")
                import traceback
                traceback.print_exc(file=f)

        finally:
            end_time_exp = time.time()
            logger.info(f"======= 実験 '{exp_name}' 完了。所要時間: {format_duration(end_time_exp - start_time_exp)} =======")
            print("\n" + "="*80 + "\n")

    main_end_time = time.time()
    print(f"全実験が完了しました。総実行時間: {format_duration(main_end_time - main_start_time)}")

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning, module='minisom')
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    main()