# ==============================================================================
# ライブラリのインポート
# ==============================================================================
import sys
from pathlib import Path
import itertools  # グリッドサーチのパラメータ組み合わせ生成
import multiprocessing  # 並列処理
import os  # CPU数の取得やパス操作
import datetime  # タイムスタンプ生成
import random  # 乱数生成
import traceback  # エラー詳細の記録
from functools import partial  # 関数の引数固定
from tqdm import tqdm

import numpy as np
import pandas as pd  # 結果のCSV保存
import xarray as xr  # NetCDFファイルの読み込み
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib  # Matplotlibの日本語表示対応

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from scipy.optimize import linear_sum_assignment

# --- ACSクラスのインポート ---
# このスクリプトと同じディレクトリに acs.py が存在することを前提とします。
try:
    from acs import ACS  # acs.py から ACS クラスをインポート
    print("ACS class (acs.py) imported successfully.")
except ImportError as e:
    print(f"Error: acs.py からACSクラスをインポートできませんでした: {e}")
    print("このスクリプトと同じディレクトリに acs.py ファイルを配置してください。")
    sys.exit(1)

# ==============================================================================
# グローバル設定
# ==============================================================================
# --- 乱数シードの固定 ---
# プログラム全体の再現性を確保するために、乱数シードを固定します。
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

# --- 出力ディレクトリとロギングの設定 ---
# 実行結果を保存するためのディレクトリをタイムスタンプ付きで作成します。
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir_base = Path("./result_prmsl_acs_elliptical_grid")
output_dir = output_dir_base / f"run_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
log_file_path = output_dir / f"grid_search_log_{timestamp}.txt"

class Logger:
    """
    標準出力をコンソールとログファイルの両方へリダイレクトするためのクラス。
    実行中のすべてのプリント文が自動的にファイルに保存されます。
    """
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log_file_handle = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file_handle.write(message)
        self.log_file_handle.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file_handle.flush()

    def close(self):
        if self.log_file_handle and not self.log_file_handle.closed:
            self.log_file_handle.close()

# ==============================================================================
# 並列処理ワーカー関数
# ==============================================================================
def run_acs_trial(param_values_tuple_with_trial_info,
                  fixed_params_dict,
                  X_data, y_data, n_true_cls_worker):
    """
    グリッドサーチの1試行を独立して実行するワーカー関数。
    multiprocessing.Poolによって呼び出されます。
    
    Args:
        param_values_tuple_with_trial_info (tuple): (試行情報, パラメータ名リスト, 試行毎の乱数シード)
        fixed_params_dict (dict): 全試行で共通の固定パラメータ
        X_data (np.ndarray): 学習データ
        y_data (np.ndarray): 真のラベルデータ
        n_true_cls_worker (int): 真のクラスタ数

    Returns:
        dict: 試行結果をまとめた辞書
    """
    # --- パラメータと試行情報の展開 ---
    (trial_count, param_values_tuple), param_names_list, trial_specific_seed = param_values_tuple_with_trial_info
    params_combo = dict(zip(param_names_list, param_values_tuple))

    # --- エポック数をパラメータから取得 ---
    num_epochs_worker = params_combo.pop('num_epochs')

    # --- ACSモデルのパラメータを構築 ---
    current_run_params = {**fixed_params_dict, **params_combo, 'random_state': trial_specific_seed}
    
    # --- ログ出力の開始 ---
    print(f"\n--- [Worker] トライアル {trial_count} 開始 ---")
    print(f"[Worker {trial_count}] パラメータ: {params_combo}")
    print(f"[Worker {trial_count}] エポック数: {num_epochs_worker}")

    # --- 結果格納用の辞書を初期化 ---
    result = {
        'params_combo': {**params_combo, 'num_epochs': num_epochs_worker}, # エポック数も記録
        'ari': -1.0,
        'accuracy_mapped': -1.0,
        'final_clusters': -1,
        'history': [], # ★★★ 改善点: エポック毎の履歴を記録するリストを追加 ★★★
        'error_traceback': None,
        'duration_seconds': 0,
        'acs_random_state_used': trial_specific_seed,
        'trial_count_from_worker': trial_count
    }
    n_samples_worker = X_data.shape[0]
    trial_start_time = datetime.datetime.now()

    try:
        # --- ACSモデルのインスタンス化と学習 ---
        acs_model_trial = ACS(**current_run_params)
        
        # ★★★ 改善点: エポック毎に評価を記録するループに変更 ★★★
        for epoch in range(num_epochs_worker):
            acs_model_trial.fit(X_data, epochs=1) # 1エポックずつ学習
            
            # --- エポック毎の評価 ---
            current_clusters = acs_model_trial.M
            epoch_ari = 0.0
            epoch_acc = 0.0

            if current_clusters > 0:
                preds = acs_model_trial.predict(X_data)
                epoch_ari = adjusted_rand_score(y_data, preds)
                
                contingency = pd.crosstab(preds, y_data)
                row_ind, col_ind = linear_sum_assignment(-contingency.values)
                epoch_acc = contingency.values[row_ind, col_ind].sum() / n_samples_worker
            
            epoch_history = {
                'epoch': epoch + 1,
                'clusters': current_clusters,
                'ari': epoch_ari,
                'accuracy_mapped': epoch_acc
            }
            result['history'].append(epoch_history)

            # ログ出力を少し間引く (例: 10エポック毎と最後のエポック)
            if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs_worker:
                 print(f"[Worker {trial_count}] Epoch {epoch+1}/{num_epochs_worker} - Cls: {current_clusters}, ARI: {epoch_ari:.4f}, Acc: {epoch_acc:.4f}")

        # --- 最終結果を格納 ---
        final_history = result['history'][-1]
        result['ari'] = final_history['ari']
        result['accuracy_mapped'] = final_history['accuracy_mapped']
        result['final_clusters'] = final_history['clusters']

    except Exception:
        # エラー発生時はトレースバックを記録
        result['error_traceback'] = traceback.format_exc()
        print(f"--- [Worker] トライアル {trial_count} でエラー発生 ---")
        print(result['error_traceback'])


    trial_end_time = datetime.datetime.now()
    result['duration_seconds'] = (trial_end_time - trial_start_time).total_seconds()
    
    print(f"--- [Worker] トライアル {trial_count} 終了 | Acc: {result['accuracy_mapped']:.4f}, ARI: {result['ari']:.4f}, Cls: {result['final_clusters']}, Time: {result['duration_seconds']:.2f}s ---")
    return result

# ==============================================================================
# メイン処理
# ==============================================================================
def main_process_logic():
    """
    データ読み込み、前処理、グリッドサーチ、結果評価、プロットまでの一連の処理を統括するメイン関数。
    """
    # --- ロガーの設定: 以降のprint文がファイルにも書き出される ---
    logger_instance = Logger(log_file_path)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = logger_instance
    sys.stderr = logger_instance

    print("=" * 80)
    print("ACSモデル (楕円形活性化) による気圧配置パターンの教師なしクラスタリング")
    print("並列グリッドサーチによるハイパーパラメータ探索")
    print("=" * 80)
    print(f"結果保存先ディレクトリ: {output_dir.resolve()}")
    print(f"ログファイル: {log_file_path.resolve()}")
    print(f"実行開始時刻: {timestamp}")
    print(f"グローバル乱数シード: {GLOBAL_SEED}")

    # --- 1. データの読み込みと前処理 ---
    print("\n--- 1. データ準備 ---")
    data_file = Path("./prmsl_era5_all_data.nc")
    if not data_file.exists():
        print(f"エラー: データファイルが見つかりません: {data_file}")
        sys.exit(1)

    # --- 1a. NetCDFファイルの読み込みと期間・ラベルでのフィルタリング ---
    ds = xr.open_dataset(data_file)
    print(f"✅ データファイル '{data_file}' を読み込みました。")
    
    # 期間(1991-2000)でデータを絞り込み
    ds_period = ds.sel(valid_time=slice('1991-01-01', '2000-12-31'))
    print(f"✅ 期間を 1991-01-01 から 2000-12-31 に絞り込みました。")

    # 評価対象のラベルを定義
    target_labels = ['1', '2A', '2B', '2C', '2D', '3A', '3B', '3C', '3D', '4A', '4B', '5', '6A', '6B', '6C']
    
    # isinメソッドで対象ラベルを持つデータのみを抽出するマスクを作成
    label_mask = ds_period['label'].isin(target_labels)
    ds_filtered = ds_period.where(label_mask, drop=True)
    n_samples = ds_filtered.sizes['valid_time']
    print(f"✅ 評価対象の15種類のラベルでデータをフィルタリングしました。対象サンプル数: {n_samples}")

    # --- 1b. 真のラベル(y)の準備 ---
    # 文字列ラベルを0-14の整数に変換
    y_true_str = ds_filtered['label'].values
    label_encoder = LabelEncoder().fit(target_labels)
    y_true_labels = label_encoder.transform(y_true_str)
    # プロット用に、整数のクラスIDと元の文字列ラベルの対応を保持
    target_names_map = {i: label for i, label in enumerate(label_encoder.classes_)}
    n_true_clusters = len(target_labels)

    # --- 1c. 学習データ(X)の準備 ---
    print("... 学習データ(X)を作成中 ...")
    # 3次元の気圧データ(time, lat, lon)を2次元(time, lat*lon)にフラット化
    msl_data = ds_filtered['msl'].values
    n_lat, n_lon = msl_data.shape[1], msl_data.shape[2]
    msl_flat = msl_data.reshape(n_samples, n_lat * n_lon)
    
    # PCAで20次元に削減
    n_pca_components = 20
    pca_model = PCA(n_components=n_pca_components, random_state=GLOBAL_SEED)
    msl_pca = pca_model.fit_transform(msl_flat)
    print(f"✅ 'msl'データをPCAで{n_pca_components}次元に削減しました。")

    # 時間情報を取得
    sin_time = ds_filtered['sin_time'].values.reshape(-1, 1)
    cos_time = ds_filtered['cos_time'].values.reshape(-1, 1)

    # PCA後の気圧データと時間情報を結合して最終的な特徴量Xを作成
    X_features = np.hstack([msl_pca, sin_time, cos_time])
    n_total_features = X_features.shape[1]
    print(f"✅ PCAデータと時間情報を結合し、{n_total_features}次元の最終特徴量を作成しました。")

    # Min-Maxスケーリングでデータを0-1の範囲に正規化
    scaler = MinMaxScaler()
    X_scaled_data = scaler.fit_transform(X_features)
    print("✅ 全特徴量をMin-Maxスケーリングで正規化しました。")

    # --- 2. グリッドサーチの設定 ---
    print("\n--- 2. グリッドサーチ設定 ---")
    param_grid = {
        'gamma': [1.0],
        'beta': [0.1, 1.0],
        'learning_rate_W': [0.01, 0.1],
        'learning_rate_lambda': [0.1, 1.0],
        'learning_rate_Z': [0.001, 0.01],
        'initial_lambda_vector_val': [0.001, 0.01],
        'initial_lambda_crossterm_val': [-0.01, 0.0, 0.01],
        'initial_Z_val': [0.001, 0.01],
        'initial_Z_new_cluster': [0.001, 0.01],
        'theta_new': [1.0],
        'death_patience_steps': [n_samples // 4, n_samples, n_samples * 2],
        'Z_death_threshold': [0.01],
        'num_epochs': [500] # エポック数も探索対象 500, 1000, 10000
    }

    fixed_params_for_acs = {
        'max_clusters': 30,
        'initial_clusters': 1,
        'n_features': n_total_features,
        'activation_type': 'elliptical',
        'lambda_min_val': 1e-7,
        'bounds_W': (0, 1)
    }

    param_names_grid = list(param_grid.keys())
    value_lists_grid = [param_grid[key] for key in param_names_grid]
    all_param_combinations_values_grid = list(itertools.product(*value_lists_grid))
    num_total_trials_grid = len(all_param_combinations_values_grid)
    print(f"グリッドサーチ試行総数: {num_total_trials_grid}")

    # --- 並列処理のタスクリストを作成 ---
    tasks_for_pool = []
    for i, param_vals_iter in enumerate(all_param_combinations_values_grid):
        trial_specific_seed_val = GLOBAL_SEED + i + 1
        task_item = ((i + 1, param_vals_iter), param_names_grid, trial_specific_seed_val)
        tasks_for_pool.append(task_item)

    # --- 3. 並列グリッドサーチの実行 ---
    print("\n--- 3. 並列グリッドサーチ実行 ---")
    available_cpus = os.cpu_count()
    # CPUリソースの80%を使用
    num_processes_to_use = max(1, int(available_cpus * 0.8)) if available_cpus else 2
    print(f"利用可能CPU数: {available_cpus}, 使用プロセス数: {num_processes_to_use}")

    # partialを使って、ワーカー関数に固定引数を渡す準備
    worker_func_with_fixed_args = partial(run_acs_trial,
                                          fixed_params_dict=fixed_params_for_acs,
                                          X_data=X_scaled_data,
                                          y_data=y_true_labels,
                                          n_true_cls_worker=n_true_clusters)

    start_grid_search_time = datetime.datetime.now()
    all_trial_results_list = []
    
    with multiprocessing.Pool(processes=num_processes_to_use) as pool:
        # pool.imap_unorderedでタスクを非同期に実行し、完了したものから結果を受け取る
        # tqdmで進捗を表示
        for result in tqdm(pool.imap_unordered(worker_func_with_fixed_args, tasks_for_pool), total=num_total_trials_grid, desc="Grid Search Progress"):
            all_trial_results_list.append(result)

    end_grid_search_time = datetime.datetime.now()
    print(f"\n並列グリッドサーチ完了。総所要時間: {end_grid_search_time - start_grid_search_time}")

    # --- 4. 結果の集計と最良モデルの選定 ---
    print("\n--- 4. 結果集計 ---")
    if not all_trial_results_list:
        print("エラー: グリッドサーチから結果が返されませんでした。")
        sys.exit(1)
        
    # 結果をDataFrameに変換してCSVに保存
    if all_trial_results_list:
        df_data = []
        for res in all_trial_results_list:
            row = res['params_combo'].copy()
            row['ari'] = res['ari']
            row['accuracy_mapped'] = res['accuracy_mapped']
            row['final_clusters'] = res['final_clusters']
            row['duration_seconds'] = res['duration_seconds']
            row['error_present'] = bool(res['error_traceback'])
            df_data.append(row)
        
        results_df = pd.DataFrame(df_data)
        results_df = results_df.sort_values(by=['accuracy_mapped', 'ari'], ascending=False)
    
        df_path = output_dir / f"grid_search_all_results_{timestamp}.csv"
        results_df.to_csv(df_path, index=False, encoding='utf-8-sig')
        print(f"全試行結果をCSVファイルに保存しました: {df_path.resolve()}")
    
    # エラーが発生しなかった試行の中から最良モデルを選定
    valid_results = [res for res in all_trial_results_list if not res['error_traceback']]
    if not valid_results:
        print("エラー: 全ての試行でエラーが発生しました。ログを確認してください。")
        sys.exit(1)
        
    # 評価指標(Accuracy > ARI)に基づいて最良のパラメータを探す
    best_result = max(valid_results, key=lambda r: (r['accuracy_mapped'], r['ari']))
    best_params_combo_dict = best_result['params_combo']
    
    print("\n--- 最良パラメータ (グリッドサーチ結果) ---")
    print(f"Accuracy (Mapped): {best_result['accuracy_mapped']:.4f}")
    print(f"ARI: {best_result['ari']:.4f}")
    print(f"最終クラスタ数: {best_result['final_clusters']}")
    print("パラメータ:")
    for key, val in best_params_combo_dict.items():
        print(f"  {key}: {val}")

    # --- 5. 最良モデルでの再学習と最終評価 ---
    print("\n--- 5. 最良モデルでの再学習と最終評価 ---")
    
    params_for_init = best_params_combo_dict.copy()
    epochs_for_refit = params_for_init.pop('num_epochs')

    best_model_full_params_for_refit = {
        **fixed_params_for_acs,
        **params_for_init,
        'random_state': best_result['acs_random_state_used']
    }

    # 可視化用に、学習データ(22次元)をさらに2次元にPCA
    pca_visual = PCA(n_components=2, random_state=GLOBAL_SEED)
    X_pca_visual = pca_visual.fit_transform(X_scaled_data)

    # 最良モデルのインスタンスを作成し、学習
    best_model_instance = ACS(**best_model_full_params_for_refit)
    best_model_instance.fit(X_scaled_data, epochs=epochs_for_refit)
    print("✅ 最良モデルの再学習が完了しました。")

    # 最終的な予測と評価
    final_predicted_labels = best_model_instance.predict(X_scaled_data) if best_model_instance.M > 0 else np.full(n_samples, -1)
    final_ari = adjusted_rand_score(y_true_labels, final_predicted_labels)
    final_clusters = best_model_instance.M

    if final_clusters > 0:
        final_contingency = pd.crosstab(final_predicted_labels, y_true_labels)
        final_row_ind, final_col_ind = linear_sum_assignment(-final_contingency.values)
        final_accuracy = final_contingency.values[final_row_ind, final_col_ind].sum() / n_samples
    else:
        final_accuracy = 0.0

    print("\n--- 最終評価結果 ---")
    print(f"Accuracy (Mapped): {final_accuracy:.4f}")
    print(f"ARI: {final_ari:.4f}")
    print(f"最終クラスタ数: {final_clusters}")
    
    # --- 6. 結果の可視化と保存 ---
    print("\n--- 6. 結果の可視化 ---")
    
    # 6a. 混同行列のプロット
    if final_clusters > 0:
        mapped_pred_labels = np.full_like(y_true_labels, -1)
        pred_to_true_map = {pred_idx: true_idx for pred_idx, true_idx in zip(final_contingency.index[final_row_ind], final_contingency.columns[final_col_ind])}
        for pred_label, true_label_idx in pred_to_true_map.items():
            mapped_pred_labels[final_predicted_labels == pred_label] = true_label_idx
        
        cm = confusion_matrix(y_true_labels, mapped_pred_labels, labels=np.arange(n_true_clusters))
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, annot_kws={"size": 10})
        plt.xlabel("予測ラベル (マッピング後)")
        plt.ylabel("真のラベル")
        plt.title(f"混同行列 (ACS - Elliptical)\nAccuracy: {final_accuracy:.4f}, ARI: {final_ari:.4f}", fontsize=14)
        cm_path = output_dir / f"confusion_matrix_{timestamp}.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 混同行列を保存しました: {cm_path.resolve()}")
    else:
        print("クラスタが形成されなかったため、混同行列はスキップします。")

    # 6b. PCA空間でのクラスタリング結果プロット
    plt.figure(figsize=(16, 7))
    # 左: 真のラベル
    plt.subplot(1, 2, 1)
    scatter_true = sns.scatterplot(x=X_pca_visual[:, 0], y=X_pca_visual[:, 1], hue=[target_names_map[l] for l in y_true_labels], palette='viridis', s=50, alpha=0.7)
    plt.title('真の気圧配置パターン (PCA 2D)')
    plt.xlabel('主成分1')
    plt.ylabel('主成分2')
    scatter_true.legend(title="真のラベル", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 右: ACSによる予測ラベル
    plt.subplot(1, 2, 2)
    if final_clusters > 0:
        mapped_hue = [target_names_map.get(l, 'Unmapped') for l in mapped_pred_labels]
        hue_order = [target_names_map[i] for i in sorted(target_names_map.keys())] + ['Unmapped']
        scatter_pred = sns.scatterplot(x=X_pca_visual[:, 0], y=X_pca_visual[:, 1], hue=mapped_hue, hue_order=hue_order, palette='viridis', s=50, alpha=0.7)
    else:
        # クラスタがない場合はデータ点を灰色でプロット
        scatter_pred = sns.scatterplot(x=X_pca_visual[:, 0], y=X_pca_visual[:, 1], color='gray', s=50, alpha=0.7)

    # クラスタ中心をプロット
    cluster_centers_22d = best_model_instance.get_cluster_centers()
    if cluster_centers_22d.shape[0] > 0:
        # スケーラーを使い逆変換してからPCAで2次元に変換
        cluster_centers_scaled = scaler.inverse_transform(cluster_centers_22d)
        cluster_centers_2d = pca_visual.transform(cluster_centers_scaled)
        plt.scatter(cluster_centers_2d[:, 0], cluster_centers_2d[:, 1], c='red', marker='X', s=200, edgecolor='white', label='クラスタ中心(W)')

    plt.title(f'ACSクラスタリング結果 (PCA 2D)\nAcc: {final_accuracy:.3f}, Cls: {final_clusters}', fontsize=14)
    plt.xlabel('主成分1')
    plt.ylabel('主成分2')
    if final_clusters > 0:
      scatter_pred.legend(title="予測ラベル", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    pca_plot_path = output_dir / f"pca_clustering_plot_{timestamp}.png"
    plt.savefig(pca_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ PCAクラスタリング結果を保存しました: {pca_plot_path.resolve()}")

    # 6c. エネルギー等高線プロット
    if best_model_instance.M > 0 and best_model_instance.fitted_:
        print("... エネルギー関数等高線図を生成中 ...")
        fig_energy, ax_energy = plt.subplots(figsize=(10, 8))
        x_min, x_max = X_pca_visual[:, 0].min() - 0.1, X_pca_visual[:, 0].max() + 0.1
        y_min, y_max = X_pca_visual[:, 1].min() - 0.1, X_pca_visual[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
        
        # グリッドポイントを元の22次元空間に逆変換
        pca_points_flat = np.c_[xx.ravel(), yy.ravel()]
        scaled_points_flat = pca_visual.inverse_transform(pca_points_flat)
        
        # エネルギーを計算
        energy_values_flat = np.array([best_model_instance.calculate_energy_at_point(p) for p in scaled_points_flat])
        E_grid = energy_values_flat.reshape(xx.shape)
        
        # 等高線プロット
        contour = ax_energy.contourf(xx, yy, E_grid, levels=20, cmap='coolwarm', alpha=0.7)
        fig_energy.colorbar(contour, ax=ax_energy, label='エネルギー E')
        ax_energy.contour(xx, yy, E_grid, levels=20, colors='k', alpha=0.5, linewidths=0.5)
        
        # データ点とクラスタ中心を重ねてプロット
        sns.scatterplot(ax=ax_energy, x=X_pca_visual[:, 0], y=X_pca_visual[:, 1], hue=[target_names_map[l] for l in y_true_labels], palette='viridis', alpha=0.5, s=30, zorder=2)
        if 'cluster_centers_2d' in locals() and cluster_centers_2d.shape[0] > 0:
            ax_energy.scatter(cluster_centers_2d[:, 0], cluster_centers_2d[:, 1], c='black', marker='X', s=150, edgecolor='white', label='クラスタ中心 (W)', zorder=3)
        
        ax_energy.set_title(f'エネルギー関数 E (PCA空間)\nAcc: {final_accuracy:.3f}, Cls: {final_clusters}', fontsize=14)
        ax_energy.set_xlabel('主成分1')
        ax_energy.set_ylabel('主成分2')
        ax_energy.legend(title="凡例", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        energy_contour_path = output_dir / f"energy_contour_plot_{timestamp}.png"
        plt.savefig(energy_contour_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ エネルギー等高線図を保存しました: {energy_contour_path.resolve()}")
    
    # ★★★ 改善点: 最良モデルの学習履歴プロットを追加 ★★★
    if best_result and best_result['history']:
        history = best_result['history']
        epochs = [h['epoch'] for h in history]
        history_ari = [h['ari'] for h in history]
        history_acc = [h['accuracy_mapped'] for h in history]
        history_cls = [h['clusters'] for h in history]

        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        # ARI
        color = 'tab:red'
        ax1.set_xlabel('エポック数')
        ax1.set_ylabel('ARI', color=color)
        ax1.plot(epochs, history_ari, 'o-', color=color, label='ARI')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_ylim(-0.05, 1.05)

        # Accuracy
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Accuracy (マッピング後)', color=color)
        ax2.plot(epochs, history_acc, 's--', color=color, label='Accuracy (Mapped)')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(-0.05, 1.05)
        
        # Cluster Count
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 60))
        color = 'tab:green'
        ax3.set_ylabel('クラスタ数', color=color)
        ax3.plot(epochs, history_cls, '^:', color=color, label='クラスタ数')
        ax3.tick_params(axis='y', labelcolor=color)
        ax3.set_ylim(0, fixed_params_for_acs['max_clusters'] + 1)
        
        fig.suptitle(f'最良モデルの学習推移 (グリッドサーチ時)\n最終Acc: {best_result["accuracy_mapped"]:.4f}, ARI: {best_result["ari"]:.4f}, Cls: {best_result["final_clusters"]}')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='best')

        history_plot_path = output_dir / f"best_model_history_{timestamp}.png"
        plt.savefig(history_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 最良モデルの学習推移グラフを保存しました: {history_plot_path.resolve()}")


    print("\n--- 全処理完了 ---")
    if isinstance(sys.stdout, Logger):
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        logger_instance.close()

if __name__ == '__main__':
    # このスクリプトが直接実行された場合にmain処理を呼び出す
    main_process_logic()