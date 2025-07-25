# -*- coding: utf-8 -*-
"""
main_v1.py

高次元データ構造と不確実性の可視化：ハイブリッドMPPCA-SOMフレームワーク
海面気圧データ（ERA5）を用いた気圧配置パターンの解析

このスクリプトは以下の処理を実行します。
1. NetCDF形式の気圧データを読み込み、指定された期間（ラベル付きデータ）を抽出・前処理します。
2. 混合確率的主成分分析（MPPCA）を用いて、高次元の気圧配置データを確率的にクラスタリングします。
3. MPPCAから得られる各データポイントの「クラスター所属事後確率ベクトル」を新たな特徴量とします。
4. この事後確率ベクトルを自己組織化マップ（SOM）に入力し、訓練します。これにより、データだけでなく「モデルの不確実性」を2次元マップ上に可視化します。
5. SOMの各ノードがどの気圧配置パターンに対応するかを分析し、マクロ平均再現率を用いて分類性能を評価します。
6. U-Matrix、ヒットマップ、勝者総取りマップ、エントロピーマップなど、多彩な可視化結果を生成し、ファイルに保存します。
"""
import os
import time
import pickle
import warnings
from collections import Counter, defaultdict

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import japanize_matplotlib
from scipy.stats import entropy
from sklearn.metrics import recall_score
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import numba

# 自作モジュールと提供されたモジュールのインポート
from minisom import MiniSom
import mppca

# --- 1. グローバル設定とパラメータ定義 ---

# 再現性のためのグローバルシード
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

# 出力ディレクトリ設定
OUTPUT_DIR = "mppca_som_results_v1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# データファイルパス
DATA_FILE_PATH = './prmsl_era5_all_data_seasonal_small.nc'

# モデルパラメータ
# MPPCA
P_CLUSTERS = 100         # p: クラスタの数
Q_LATENT_DIM = 2         # q: 潜在空間の次元数
N_ITER_MPPCA = 10000    # niter: EMアルゴリズムの反復回数

# MiniSom
MAP_X, MAP_Y = 10, 10
SIGMA_SOM = 3.0
LEARNING_RATE_SOM = 1.0
NEIGHBORHOOD_FUNCTION_SOM = 'gaussian'
TOPOLOGY_SOM = 'rectangular'
ACTIVATION_DISTANCE_SOM = 'euclidean'
N_ITER_SOM = 100000

# 基本となる気圧配置パターンのラベル
BASE_LABELS = [
    '1', '2A', '2B', '2C', '2D', '3A', '3B', '3C', '3D',
    '4A', '4B', '5', '6A', '6B', '6C'
]

# --- 2. 補助関数 ---

def log_message(message):
    """タイムスタンプ付きでログメッセージを出力する関数"""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def format_duration(seconds):
    """秒を分かりやすい時間形式（HH:MM:SS）に変換する関数"""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

def parse_label(label_str):
    """
    ラベル文字列をパースし、基本ラベルのリストを返す。
    例: "4A-6B" -> ['4A', '6B']
        "1" -> ['1']
    """
    return label_str.replace('+', '-').split('-')

# --- 3. データ処理関数 ---

def load_and_preprocess_data(filepath, start_date='1991-01-01', end_date='2000-12-31'):
    """
    NetCDFファイルを読み込み、指定期間のデータを抽出して前処理する。
    """
    log_message(f"データファイル '{filepath}' を読み込み中...")
    start_time = time.time()

    with xr.open_dataset(filepath, engine='netcdf4') as ds:
        # ラベル付き期間のデータをスライス
        ds_labeled = ds.sel(valid_time=slice(start_date, end_date))

        # データをNumpy配列に変換
        msl_data = ds_labeled['msl'].values.astype(np.float64) # <- データ型をfloat64に指定
        labels = ds_labeled['label'].values.astype(str)
        valid_times = ds_labeled['valid_time'].values
        lat = ds_labeled['latitude'].values
        lon = ds_labeled['longitude'].values

    log_message(f"データ読み込み完了。対象期間: {start_date} ～ {end_date}")

    # データ形状の確認
    n_samples, n_lat, n_lon = msl_data.shape
    log_message(f"サンプル数: {n_samples}, 緯度点数: {n_lat}, 経度点数: {n_lon}")

    # データを2次元配列 (サンプル数, 特徴量数) に変形
    data_reshaped = msl_data.reshape(n_samples, n_lat * n_lon)

    # 欠損値を列（グリッド点）ごとの平均値で補完
    if np.isnan(data_reshaped).any():
        log_message("欠損値を検出。列ごとの平均値で補完します...")
        col_mean = np.nanmean(data_reshaped, axis=0)
        inds = np.where(np.isnan(data_reshaped))
        data_reshaped[inds] = np.take(col_mean, inds[1])

    # データを標準化（平均0, 標準偏差1）
    log_message("データを標準化（Z-score normalization）します...")
    data_mean = data_reshaped.mean(axis=0)
    data_std = data_reshaped.std(axis=0)
    # 標準偏差が0の列は0で割るのを避ける
    data_std[data_std == 0] = 1
    data_normalized = (data_reshaped - data_mean) / data_std

    end_time = time.time()
    log_message(f"データ前処理完了。所要時間: {format_duration(end_time - start_time)}")

    return data_normalized, labels, valid_times, lat, lon, data_mean, data_std

# --- 4. モデル実行関数 ---

def run_mppca(data, p, q, niter):
    """
    MPPCAモデルを訓練し、事後確率を返す。
    """
    log_message("MPPCAの訓練を開始します...")
    log_message(f"パラメータ: クラスター数(p)={p}, 潜在次元(q)={q}, 反復回数(niter)={niter}")
    start_time = time.time()

    # K-meansによる初期化
    log_message("K-meansによる初期化を実行中...")
    # Numbaでコンパイルされた関数はnp.random.seedの影響を受けないため、
    # 乱数シードを直接設定する必要はないが、アルゴリズムの性質上、実行ごとに結果は変わりうる。
    pi, mu, W, sigma2, _ = mppca.initialization_kmeans(data, p, q, -1.0)
    log_message("K-means初期化完了。")

    # GEMアルゴリズムによる訓練
    log_message("GEMアルゴリズムによる訓練を実行中... (時間がかかります)")
    # tqdmを使って進捗を表示
    # NOTE: mppca.py内のループにtqdmを統合できないため、ここでは全体の目安として表示
    # 実際の進捗はmppca.pyの実装に依存します。
    # この処理は非常に長くなることが予想されます。
    pi, mu, W, sigma2, R, L, _ = mppca.mppca_gem(data, pi, mu, W, sigma2, niter)

    end_time = time.time()
    log_message(f"MPPCA訓練完了。所要時間: {format_duration(end_time - start_time)}")

    # 最終的な対数尤度を表示
    log_message(f"最終的な対数尤度: {L[-1]}")

    # モデルと結果を保存
    mppca_results = {'pi': pi, 'mu': mu, 'W': W, 'sigma2': sigma2, 'R': R, 'L': L}
    with open(os.path.join(OUTPUT_DIR, 'mppca_results.pkl'), 'wb') as f:
        pickle.dump(mppca_results, f)
    log_message("MPPCAのモデルと結果を 'mppca_results.pkl' に保存しました。")


    return R

def run_som(data, map_x, map_y, input_len, sigma, lr, n_iter, seed):
    """
    MiniSomを訓練し、訓練済みモデルを返す。
    """
    log_message("SOMの訓練を開始します...")
    log_message(f"パラメータ: マップサイズ={map_x}x{map_y}, sigma={sigma}, learning_rate={lr}, 反復回数={n_iter}")
    start_time = time.time()

    som = MiniSom(map_x, map_y, input_len,
                  sigma=sigma,
                  learning_rate=lr,
                  neighborhood_function=NEIGHBORHOOD_FUNCTION_SOM,
                  topology=TOPOLOGY_SOM,
                  activation_distance=ACTIVATION_DISTANCE_SOM,
                  random_seed=seed)

    # PCA初期化の代わりにランダム初期化後、訓練
    som.random_weights_init(data)

    log_message("SOM訓練中... (verbose=Trueで進捗表示)")
    # use_epochs=Falseなので、num_iteration回だけ学習
    som.train(data, n_iter, verbose=True, random_order=True)

    end_time = time.time()
    log_message(f"SOM訓練完了。所要時間: {format_duration(end_time - start_time)}")

    # SOMモデルを保存
    with open(os.path.join(OUTPUT_DIR, 'som_model.pkl'), 'wb') as f:
        pickle.dump(som, f)
    log_message("SOMモデルを 'som_model.pkl' に保存しました。")

    return som

# --- 5. 評価・分析関数 ---

def evaluate_classification(som, mppca_posterior, original_labels):
    """
    マクロ平均再現率を計算して分類性能を評価する。
    """
    log_message("分類性能の評価を開始します (マクロ平均再現率)...")
    start_time = time.time()

    # 1. 真のラベルを多ラベル形式に変換
    parsed_true_labels = [parse_label(l) for l in original_labels]
    mlb = MultiLabelBinarizer(classes=BASE_LABELS)
    y_true = mlb.fit_transform(parsed_true_labels)

    # 2. 各ノードの支配的ラベルを決定
    win_map_indices = som.win_map(mppca_posterior, return_indices=True)
    node_dominant_label = {}
    for pos, indices in win_map_indices.items():
        if not indices:
            node_dominant_label[pos] = None
            continue
        # ノードに属する全ラベルをフラットなリストにする
        labels_in_node = [label for i in indices for label in parsed_true_labels[i]]
        if not labels_in_node:
            node_dominant_label[pos] = None
            continue
        # 最も頻度の高いラベルを支配的ラベルとする
        most_common = Counter(labels_in_node).most_common(1)
        node_dominant_label[pos] = most_common[0][0]

    # 3. 予測ラベルを生成
    winners = [som.winner(x) for x in mppca_posterior]
    predicted_single_labels = [node_dominant_label.get(w) for w in winners]

    # 予測ラベルを多ラベル形式に変換
    y_pred = mlb.transform([[l] if l else [] for l in predicted_single_labels])

    # 4. マクロ平均再現率を計算
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    log_message(f"評価完了。所要時間: {format_duration(time.time() - start_time)}")
    log_message(f"マクロ平均再現率: {macro_recall:.4f}")

    # 各クラスの再現率も表示
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    log_message("クラスごとの再現率:")
    for label, recall in zip(mlb.classes_, per_class_recall):
        print(f"  - {label}: {recall:.4f}")


    return macro_recall, node_dominant_label

# --- 6. 可視化関数 ---

def visualize_results(som, mppca_posterior, original_data, labels, node_dominant_label, lat, lon, data_mean, data_std):
    """
    解析結果を多角的に可視化する。
    """
    log_message("結果の可視化を開始します...")
    start_time = time.time()

    map_x, map_y = som.get_weights().shape[:2]
    parsed_true_labels = [parse_label(l) for l in labels]

    # --- 可視化1: U-Matrix (Unified Distance Matrix) ---
    plt.figure(figsize=(10, 10))
    u_matrix = som.distance_map()
    plt.pcolor(u_matrix.T, cmap='bone_r')
    plt.colorbar(label='ニューロン間の距離')
    plt.title('U-Matrix (クラスタ境界の可視化)')
    plt.xticks(np.arange(map_x + 1))
    plt.yticks(np.arange(map_y + 1))
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, 'u_matrix.png'), dpi=300)
    plt.close()
    log_message("U-Matrixを保存しました。")

    # --- 可視化2: ヒットマップ (Activation Response) ---
    plt.figure(figsize=(10, 10))
    frequencies = som.activation_response(mppca_posterior)
    plt.pcolor(frequencies.T, cmap='viridis')
    plt.colorbar(label='勝者となった回数')
    plt.title('ヒットマップ (各ニューロンの勝者回数)')
    plt.xticks(np.arange(map_x + 1))
    plt.yticks(np.arange(map_y + 1))
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, 'hit_map.png'), dpi=300)
    plt.close()
    log_message("ヒットマップを保存しました。")

    # --- 可視化3: 勝者総取りマップ (ラベルマップ) ---
    plt.figure(figsize=(12, 10))
    label_map = np.zeros((map_x, map_y), dtype=int)
    label_names = {label: i for i, label in enumerate(BASE_LABELS)}
    label_names[None] = -1 # データが割り当てられなかったノード

    for pos, label in node_dominant_label.items():
        label_map[pos[0], pos[1]] = label_names.get(label, -1)

    cmap = plt.get_cmap('tab20', len(BASE_LABELS) + 1)
    plt.pcolor(label_map.T, cmap=cmap, vmin=-1.5, vmax=len(BASE_LABELS)-0.5)

    # カラーバーとラベルの設定
    cbar = plt.colorbar(ticks=np.arange(len(BASE_LABELS)))
    cbar.ax.set_yticklabels(BASE_LABELS)
    plt.title('勝者総取りマップ (各ノードの支配的気圧配置パターン)')
    plt.xticks(np.arange(map_x + 1))
    plt.yticks(np.arange(map_y + 1))
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, 'winner_takes_all_map.png'), dpi=300)
    plt.close()
    log_message("勝者総取りマップを保存しました。")


    # --- 可視化4: エントロピーマップ (不確実性マップ) ---
    win_map_indices = som.win_map(mppca_posterior, return_indices=True)
    entropy_map = np.full((map_x, map_y), np.nan)
    for pos, indices in win_map_indices.items():
        if indices:
            # ノードに属するデータの事後確率の平均を計算
            avg_posterior_in_node = mppca_posterior[indices].mean(axis=0)
            entropy_map[pos] = entropy(avg_posterior_in_node)

    plt.figure(figsize=(10, 10))
    plt.pcolor(entropy_map.T, cmap='magma')
    plt.colorbar(label='エントロピー')
    plt.title('エントロピーマップ (モデルの不確実性可視化)')
    plt.xticks(np.arange(map_x + 1))
    plt.yticks(np.arange(map_y + 1))
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, 'entropy_map.png'), dpi=300)
    plt.close()
    log_message("エントロピーマップを保存しました。")

    # --- 可視化5: 各ノードのラベル分布 (円グラフ) ---
    fig, axes = plt.subplots(map_x, map_y, figsize=(20, 20),
                             subplot_kw={'aspect': 'equal'})
    fig.suptitle('各ノードにおけるラベル分布', fontsize=24)
    win_map_labels = som.labels_map(mppca_posterior, parsed_true_labels)
    cmap_pie = plt.get_cmap('tab20', len(BASE_LABELS))
    label_colors = {label: cmap_pie(i) for i, label in enumerate(BASE_LABELS)}

    for i in range(map_x):
        for j in range(map_y):
            ax = axes[j, i] # subplotは(row, col)なので (j, i)
            ax.set_xticks([])
            ax.set_yticks([])
            pos = (i, j)
            if pos in win_map_labels:
                labels_in_node = win_map_labels[pos]
                # 全ラベルをフラット化
                flat_labels = [l for sublist in labels_in_node.keys() for l in sublist]
                counts = Counter(flat_labels)
                if counts:
                    ax.pie(counts.values(),
                           colors=[label_colors.get(k, 'grey') for k in counts.keys()])
            # ヒットマップの値を背景色に
            ax.set_facecolor(plt.cm.viridis(frequencies[i, j] / frequencies.max()))


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, 'node_label_distribution_pie.png'), dpi=300)
    plt.close()
    log_message("各ノードのラベル分布（円グラフ）を保存しました。")


    # --- 可視化6: 代表的なノードの平均気圧配置図 ---
    log_message("代表的なノードの平均気圧配置図を作成中...")
    num_representative_nodes = min(9, len(win_map_indices)) # 最大9個プロット
    # ヒット数が多いノードを代表として選出
    flat_freq = frequencies.flatten()
    top_indices = np.argsort(flat_freq)[-num_representative_nodes:][::-1]
    top_nodes = [np.unravel_index(i, (map_x, map_y)) for i in top_indices]

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('ヒット数の多い代表的ノードの平均気圧配置', fontsize=20)
    axes = axes.flatten()

    for k, pos in enumerate(top_nodes):
        ax = axes[k]
        indices = win_map_indices.get(pos)
        if not indices:
            ax.set_title(f"Node {pos}\n(データなし)")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # 元のスケールに戻して平均を計算
        node_data_original_scale = original_data[indices] * data_std + data_mean
        mean_pattern = node_data_original_scale.mean(axis=0).reshape(len(lat), len(lon))

        # 等圧線を描画 (Pa -> hPa)
        contour = ax.contourf(lon, lat, mean_pattern / 100, cmap='coolwarm', levels=20)
        ax.contour(lon, lat, mean_pattern / 100, colors='k', linewidths=0.5, levels=15)
        ax.set_title(f"Node {pos} (n={len(indices)})\nDom: {node_dominant_label.get(pos, 'N/A')}")
        ax.set_xlabel('経度')
        ax.set_ylabel('緯度')

    for k in range(num_representative_nodes, len(axes)):
        axes[k].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUTPUT_DIR, 'representative_node_patterns.png'), dpi=300)
    plt.close()
    log_message("代表的なノードの平均気圧配置図を保存しました。")


    end_time = time.time()
    log_message(f"可視化完了。所要時間: {format_duration(end_time - start_time)}")


# --- 7. メイン実行ブロック ---

def main():
    """メインの処理フローを実行する"""
    main_start_time = time.time()
    log_message("研究プログラムを開始します。")

    # --- ステップ1: データ読み込みと前処理 ---
    # `original_data`は可視化のために非正規化状態で保持
    data_normalized, labels, _, lat, lon, data_mean, data_std = load_and_preprocess_data(DATA_FILE_PATH)
    data_reshaped_original = data_normalized * data_std + data_mean


    # --- ステップ2: MPPCAの実行 ---
    # NOTE: niter=100000 は非常に時間がかかります。テスト時は100などに減らすことを推奨します。
    mppca_posterior = run_mppca(data_normalized, P_CLUSTERS, Q_LATENT_DIM, N_ITER_MPPCA)


    # --- ステップ3: SOMの実行 ---
    # MPPCAの事後確率を入力とする
    som_input_data = mppca_posterior
    input_len_som = P_CLUSTERS # SOMの入力長はMPPCAのクラスター数
    som = run_som(som_input_data, MAP_X, MAP_Y, input_len_som, SIGMA_SOM, LEARNING_RATE_SOM, N_ITER_SOM, GLOBAL_SEED)


    # --- ステップ4: 評価 ---
    _, node_dominant_label = evaluate_classification(som, mppca_posterior, labels)


    # --- ステップ5: 可視化 ---
    visualize_results(som, mppca_posterior, data_reshaped_original, labels, node_dominant_label, lat, lon, data_mean, data_std)

    main_end_time = time.time()
    log_message("すべての処理が完了しました。")
    log_message(f"総実行時間: {format_duration(main_end_time - main_start_time)}")

if __name__ == '__main__':
    # 警告を抑制（NumbaやMatplotlibからの警告など）
    warnings.filterwarnings('ignore', category=UserWarning, module='minisom')
    warnings.filterwarnings('ignore', category=numba.NumbaPerformanceWarning)
    main()