# ==============================================================================
# ライブラリのインポート
# ==============================================================================
import sys
from pathlib import Path
import itertools
import multiprocessing
import os
import datetime
import random
import traceback
from functools import partial
from tqdm import tqdm

import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, confusion_matrix, recall_score
from scipy.optimize import linear_sum_assignment

# --- ACSクラスのインポート ---
try:
    # numbaで最適化されたacs.pyをインポートします。
    # 呼び出し側のコードは変更不要です。
    from acs import ACS
    print("ACS class (acs.py) imported successfully.")
except ImportError as e:
    print(f"Error: acs.py からACSクラスをインポートできませんでした: {e}")
    print("このスクリプトと同じディレクトリに acs.py ファイルを配置してください。")
    sys.exit(1)

# ==============================================================================
# グローバル設定
# ==============================================================================
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir_base = Path("./result_prmsl_acs_random_search")
output_dir = output_dir_base / f"run_{timestamp}"
trial_logs_dir = output_dir / "trial_logs"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(trial_logs_dir, exist_ok=True)
log_file_path = output_dir / f"main_log_{timestamp}.txt"

class Logger:
    """標準出力をコンソールとログファイルの両方へリダイレクトするクラス。"""
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
                  pca_data_dict,
                  y_data,
                  sin_time_data,
                  cos_time_data,
                  n_true_cls_worker,
                  trial_log_dir_path,
                  label_class_names):
    """グリッドサーチ/ランダムサーチの1試行を独立して実行するワーカー関数。"""
    (trial_count, params_combo), trial_specific_seed = param_values_tuple_with_trial_info
    
    worker_log_path = trial_log_dir_path / f"trial_{trial_count}.log"
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        with open(worker_log_path, 'w', encoding='utf-8') as log_file:
            sys.stdout = log_file
            sys.stderr = log_file
            
            # --- ワーカー内部で特徴量を動的に構築 ---
            pca_n_components = params_combo.pop('pca_n_components')
            include_time_features = params_combo.pop('include_time_features')
            
            X_pca = pca_data_dict[pca_n_components]
            
            if include_time_features:
                X_features = np.hstack([X_pca, sin_time_data, cos_time_data])
            else:
                X_features = X_pca
            
            # PCA後の特徴量と時間特徴量のスケールを最終的に[0,1]に統一するためにScalerを適用
            scaler = MinMaxScaler()
            X_scaled_data = scaler.fit_transform(X_features)
            n_features_worker = X_scaled_data.shape[1]
            # --- 特徴量構築ここまで ---

            num_epochs_worker = params_combo.pop('num_epochs')
            activation_type_worker = params_combo.pop('activation_type')
            
            if activation_type_worker == 'circular':
                params_combo.pop('initial_lambda_vector_val', None)
                params_combo.pop('initial_lambda_crossterm_val', None)
            else:
                params_combo.pop('initial_lambda_scalar', None)

            current_run_params = {
                **fixed_params_dict,
                'n_features': n_features_worker,
                'activation_type': activation_type_worker,
                **params_combo,
                'random_state': trial_specific_seed
            }
            
            print(f"\n--- [Worker] トライアル {trial_count} 開始 ---")
            print(f"[Worker {trial_count}] PCA Components: {pca_n_components}, Include Time: {include_time_features}")
            print(f"[Worker {trial_count}] Final Feature Dims: {n_features_worker}")
            log_params = {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in params_combo.items()}
            print(f"[Worker {trial_count}] Activation Type: {activation_type_worker}")
            print(f"[Worker {trial_count}] パラメータ: {log_params}")
            print(f"[Worker {trial_count}] エポック数: {num_epochs_worker}")

            result = {
                'params_combo': {
                    **params_combo,
                    'pca_n_components': pca_n_components,
                    'include_time_features': include_time_features,
                    'num_epochs': num_epochs_worker,
                    'activation_type': activation_type_worker
                },
                'ari': -1.0, 'accuracy_mapped': -1.0, 'final_clusters': -1, 'best_epoch': 0,
                'history': [], 'error_traceback': None, 'duration_seconds': 0,
                'acs_random_state_used': trial_specific_seed,
                'trial_count_from_worker': trial_count
            }
            n_samples_worker = X_scaled_data.shape[0]
            trial_start_time = datetime.datetime.now()

            try:
                acs_model_trial = ACS(**current_run_params)
                
                for epoch in range(num_epochs_worker):
                    acs_model_trial.fit(X_scaled_data, epochs=1)
                    current_clusters = acs_model_trial.M
                    epoch_ari, epoch_acc = 0.0, 0.0
                    if current_clusters > 0:
                        preds = acs_model_trial.predict(X_scaled_data)
                        epoch_ari = adjusted_rand_score(y_data, preds)
                        contingency = pd.crosstab(preds, y_data)
                        row_ind, col_ind = linear_sum_assignment(-contingency.values)
                        epoch_acc = contingency.values[row_ind, col_ind].sum() / n_samples_worker
                    result['history'].append({'epoch': epoch + 1, 'clusters': current_clusters, 'ari': epoch_ari, 'accuracy_mapped': epoch_acc})
                    if (epoch + 1) % 100 == 0:
                        print(f"[Worker {trial_count}] Epoch {epoch+1}/{num_epochs_worker} - Cls: {current_clusters}, ARI: {epoch_ari:.4f}, Acc: {epoch_acc:.4f}")

                if result['history']:
                    best_epoch_history = max(result['history'], key=lambda x: x['accuracy_mapped'])
                    result['ari'] = best_epoch_history['ari']
                    result['accuracy_mapped'] = best_epoch_history['accuracy_mapped']
                    result['final_clusters'] = best_epoch_history['clusters']
                    result['best_epoch'] = best_epoch_history['epoch']
                
                if acs_model_trial.M > 0:
                    print("\n--- ラベル別 最終精度 (Recall at final epoch) ---")
                    preds = acs_model_trial.predict(X_scaled_data)
                    contingency = pd.crosstab(preds, y_data)
                    row_ind, col_ind = linear_sum_assignment(-contingency.values)
                    mapped_preds = np.full_like(y_data, -1)
                    pred_idx_to_true_idx_map = {contingency.index[pred_i]: true_i for pred_i, true_i in zip(row_ind, col_ind)}
                    for pred_label_val, true_label_idx in pred_idx_to_true_idx_map.items():
                        mapped_preds[preds == pred_label_val] = true_label_idx
                    recalls = recall_score(y_data, mapped_preds, average=None, labels=np.arange(len(label_class_names)), zero_division=0)
                    for i, class_name in enumerate(label_class_names):
                        print(f"  - {class_name:<4s}: {recalls[i]:.4f}")

            except Exception:
                result['error_traceback'] = traceback.format_exc()
                print(f"--- [Worker] トライアル {trial_count} でエラー発生 ---\n{result['error_traceback']}")

            trial_end_time = datetime.datetime.now()
            result['duration_seconds'] = (trial_end_time - trial_start_time).total_seconds()
            print(f"\n--- [Worker] トライアル {trial_count} 終了 | Best Acc: {result['accuracy_mapped']:.4f} at Epoch {result['best_epoch']}, Time: {result['duration_seconds']:.2f}s ---")
            
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            return result
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr

def sample_random_params(param_dist):
    """ランダムサーチのために、定義されたパラメータ範囲から値をサンプリングする。"""
    params = {}
    for key, value in param_dist.items():
        if isinstance(value, list):
            params[key] = random.choice(value)
        elif isinstance(value, tuple) and len(value) == 2:
            params[key] = round(np.random.uniform(value[0], value[1]), 4)
    return params

# ★★★ 新設: エネルギー等高線プロットを生成する関数 ★★★
def plot_energy_contour_for_epoch(model, epoch, save_path, X_pca_visual, y_true_labels, target_names_map, pca_visual_model):
    """指定されたモデルの状態でエネルギー等高線プロットを生成し保存する。"""
    
    # 2D描画領域のグリッドを作成
    x_min, x_max = X_pca_visual[:, 0].min() - 0.1, X_pca_visual[:, 0].max() + 0.1
    y_min, y_max = X_pca_visual[:, 1].min() - 0.1, X_pca_visual[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 80), np.linspace(y_min, y_max, 80))

    # グリッド上の各点を高次元空間に逆変換
    grid_points_2d = np.c_[xx.ravel(), yy.ravel()]
    try:
        # pca_visual_modelはX_scaled_data_bestで学習済みなので、2D->高次元へ逆変換
        grid_points_high_dim = pca_visual_model.inverse_transform(grid_points_2d)
    except Exception as e:
        print(f"Epoch {epoch}: PCA逆変換中にエラー: {e}")
        return

    # 各点でのエネルギーを計算
    energy_values = np.zeros(grid_points_high_dim.shape[0])
    for i in range(len(grid_points_high_dim)):
        energy_values[i] = model.calculate_energy_at_point(grid_points_high_dim[i, :])
    
    Z_grid = energy_values.reshape(xx.shape)

    # プロット作成
    plt.figure(figsize=(10, 8))
    
    # エネルギー等高線をプロット
    plt.contourf(xx, yy, Z_grid, levels=20, cmap='viridis', alpha=0.5)
    plt.colorbar(label='エネルギー (E)')
    plt.contour(xx, yy, Z_grid, levels=20, colors='white', linewidths=0.5, alpha=0.6)

    # データ点をプロット
    sns.scatterplot(x=X_pca_visual[:, 0], y=X_pca_visual[:, 1], hue=[target_names_map.get(l, 'N/A') for l in y_true_labels], 
                    palette='bright', s=50, alpha=0.9, edgecolor='w', linewidth=0.5)

    # クラスタ中心(W)をプロット
    cluster_centers = model.get_cluster_centers()
    if cluster_centers.shape[0] > 0:
        try:
            # 高次元のクラスタ中心を2Dに変換
            cluster_centers_2d = pca_visual_model.transform(cluster_centers)
            plt.scatter(cluster_centers_2d[:, 0], cluster_centers_2d[:, 1], c='black', marker='o', s=150, edgecolor='white', linewidth=1.5, label='クラスタ中心 (W)')
        except Exception as e:
            print(f"Epoch {epoch}: クラスタ中心のPCA変換中にエラー: {e}")

    plt.title(f'エネルギー等高線とクラスタリング状態 (Epoch: {epoch})', fontsize=16)
    plt.xlabel('主成分1'), plt.ylabel('主成分2')
    plt.legend(title="凡例", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # ファイルに保存
    plot_filepath = save_path / f"energy_contour_epoch_{epoch:04d}.png"
    plt.savefig(plot_filepath, dpi=150, bbox_inches='tight')
    plt.close()

# ==============================================================================
# メイン処理
# ==============================================================================
def main_process_logic():
    """データ読み込み、前処理、ランダムサーチ、結果評価、プロットまでを統括するメイン関数。"""
    logger_instance = Logger(log_file_path)
    sys.stdout, sys.stderr = logger_instance, logger_instance

    print("=" * 80)
    print("ACSモデルによる気圧配置パターンの教師なしクラスタリング")
    print("★★★ 並列ランダムサーチによるハイパーパラメータ探索 ★★★")
    print("=" * 80)
    print(f"結果保存先ディレクトリ: {output_dir.resolve()}")
    print(f"メインログファイル: {log_file_path.resolve()}")
    print(f"各トライアルの詳細ログ: {trial_logs_dir.resolve()}")
    print(f"実行開始時刻: {timestamp}")
    print(f"グローバル乱数シード: {GLOBAL_SEED}")

    # --- 1. データの読み込みと前処理 ---
    print("\n--- 1. データ準備 ---")
    data_file = Path("./prmsl_era5_all_data.nc")
    if not data_file.exists():
        print(f"エラー: データファイルが見つかりません: {data_file}")
        sys.exit(1)
    ds = xr.open_dataset(data_file)
    print(f"✅ データファイル '{data_file}' を読み込みました。")
    ds_period = ds.sel(valid_time=slice('1991-01-01', '2000-12-31'))
    print(f"✅ 期間を 1991-01-01 から 2000-12-31 に絞り込みました。")
    target_labels = ['1', '2A', '2B', '2C', '2D', '3A', '3B', '3C', '3D', '4A', '4B', '5', '6A', '6B', '6C']
    label_mask = ds_period['label'].isin(target_labels)
    ds_filtered = ds_period.where(label_mask, drop=True)
    n_samples = ds_filtered.sizes['valid_time']
    print(f"✅ 評価対象の15種類のラベルでデータをフィルタリングしました。対象サンプル数: {n_samples}")
    
    y_true_str = ds_filtered['label'].values
    label_encoder = LabelEncoder().fit(target_labels)
    y_true_labels = label_encoder.transform(y_true_str)
    target_names_map = {i: label for i, label in enumerate(label_encoder.classes_)}
    n_true_clusters = len(target_labels)
    
    msl_data = ds_filtered['msl'].values
    msl_flat = msl_data.reshape(n_samples, -1)
    sin_time = ds_filtered['sin_time'].values.reshape(-1, 1)
    cos_time = ds_filtered['cos_time'].values.reshape(-1, 1)

    # 前処理の順序を変更 (MinMaxScaler -> PCA)
    print("\n--- 前処理: データを[0,1]にスケーリングしてからPCAを適用します ---")
    scaler_for_pca = MinMaxScaler()
    msl_flat_scaled = scaler_for_pca.fit_transform(msl_flat)
    print("✅ 海面気圧データをスケーリングしました。")
    del msl_flat

    print("\n--- スケーリング済みデータで複数パターンのPCAを事前計算します... ---")
    pca_dims_to_test = [15, 20, 25]
    precalculated_pca_data = {}
    for n_dim in pca_dims_to_test:
        print(f"  ... PCA (n={n_dim}) を計算中 ...")
        pca_model = PCA(n_components=n_dim, random_state=GLOBAL_SEED)
        precalculated_pca_data[n_dim] = pca_model.fit_transform(msl_flat_scaled)
    print("✅ 全パターンのPCA計算が完了しました。")
    del msl_flat_scaled

    # --- 2. ランダムサーチの設定 ---
    print("\n--- 2. ランダムサーチ設定 ---")
    param_dist = {
        'pca_n_components': pca_dims_to_test,
        'include_time_features': [True, False],
        'gamma': (0.01, 3.0),
        'beta': (0.001, 1.0),
        'learning_rate_W': (0.001, 0.1),
        'learning_rate_lambda': (0.001, 0.1),
        'learning_rate_Z': (0.001, 0.1),
        'initial_lambda_scalar': (0.001, 1.0),
        'initial_lambda_vector_val': (0.001, 1.0),
        'initial_lambda_crossterm_val': (-0.5, 0.5),
        'initial_Z_val': (0.01, 1.0),
        'initial_Z_new_cluster': (0.01, 1.0),
        'theta_new': (0.001, 1.0),
        'Z_death_threshold': (0.01, 0.1),
        'death_patience_steps': [n_samples // 10, n_samples // 4, n_samples // 2, n_samples],
        'num_epochs': [1000],
        'activation_type': ['circular', 'elliptical']
    }
    
    N_TRIALS = 1000
    ACCURACY_GOAL = 0.7

    fixed_params_for_acs = {
        'max_clusters': 30, 'initial_clusters': 1,
        'lambda_min_val': 1e-7, 'bounds_W': (0, 1)
    }
    print(f"ランダムサーチ最大試行回数: {N_TRIALS}")
    print(f"早期終了の目標精度 (Accuracy): {ACCURACY_GOAL}")

    # --- 3. 並列ランダムサーチの実行 ---
    print("\n--- 3. 並列ランダムサーチ実行 ---")
    available_cpus = os.cpu_count()
    num_processes_to_use = max(1, int(available_cpus * 0.9)) if available_cpus else 2
    print(f"利用可能CPU数: {available_cpus}, 使用プロセス数: {num_processes_to_use}")

    tasks_for_pool = []
    for i in range(N_TRIALS):
        params_combo = sample_random_params(param_dist)
        trial_specific_seed = GLOBAL_SEED + i + 1
        tasks_for_pool.append(((i + 1, params_combo), trial_specific_seed))

    worker_func_with_fixed_args = partial(run_acs_trial,
                                          fixed_params_dict=fixed_params_for_acs,
                                          pca_data_dict=precalculated_pca_data,
                                          y_data=y_true_labels,
                                          sin_time_data=sin_time,
                                          cos_time_data=cos_time,
                                          n_true_cls_worker=n_true_clusters,
                                          trial_log_dir_path=trial_logs_dir,
                                          label_class_names=label_encoder.classes_)

    start_search_time = datetime.datetime.now()
    all_trial_results_list = []
    best_result = None
    goal_achieved = False

    with multiprocessing.Pool(processes=num_processes_to_use) as pool:
        for result in tqdm(pool.imap_unordered(worker_func_with_fixed_args, tasks_for_pool), total=N_TRIALS, desc="Random Search Progress"):
            all_trial_results_list.append(result)
            if not result['error_traceback']:
                if best_result is None or result['accuracy_mapped'] > best_result['accuracy_mapped']:
                    best_result = result
            if best_result and best_result.get('accuracy_mapped', 0) >= ACCURACY_GOAL:
                print(f"\n✅ 目標精度達成 (Acc >= {ACCURACY_GOAL})！ トライアル {result['trial_count_from_worker']} でサーチを打ち切ります。")
                goal_achieved = True
                pool.terminate()
                break

    end_search_time = datetime.datetime.now()
    print(f"\nランダムサーチ完了。総所要時間: {end_search_time - start_search_time}")

    # --- 4. 結果の集計と最良モデルの選定 ---
    print("\n--- 4. 結果集計 ---")
    if not all_trial_results_list:
        print("エラー: サーチから結果が返されませんでした。")
        sys.exit(1)
        
    if all_trial_results_list:
        df_data = []
        for res in all_trial_results_list:
            row = res['params_combo'].copy()
            row.update({'ari': res['ari'], 'accuracy_mapped': res['accuracy_mapped'], 'final_clusters': res['final_clusters'],
                        'best_epoch': res['best_epoch'],
                        'duration_seconds': res['duration_seconds'], 'error_present': bool(res['error_traceback']),
                        'trial_id': res['trial_count_from_worker']})
            df_data.append(row)
        
        results_df = pd.DataFrame(df_data).sort_values(by=['accuracy_mapped', 'ari'], ascending=False)
        df_path = output_dir / f"random_search_all_results_{timestamp}.csv"
        results_df.to_csv(df_path, index=False, encoding='utf-8-sig')
        print(f"全試行結果をCSVファイルに保存しました: {df_path.resolve()}")

    if best_result is None:
        print("エラー: 全ての試行でエラーが発生、または有効な結果が得られませんでした。")
        sys.exit(1)
        
    print(f"\n--- 最良パラメータ ({'目標達成' if goal_achieved else '探索終了時点'}) ---")
    best_params_combo_dict = best_result['params_combo']
    print(f"Trial ID: {best_result['trial_count_from_worker']}")
    print(f"最高精度 (Mapped): {best_result['accuracy_mapped']:.4f} (at Epoch {best_result['best_epoch']})")
    print(f"最高精度時のARI: {best_result['ari']:.4f}")
    print(f"最高精度時のクラスタ数: {best_result['final_clusters']}")
    print("パラメータ:")
    for key, val in best_params_combo_dict.items():
        print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")

    # --- 5. 最良モデルでの再学習と最終評価 ---
    print("\n--- 5. 最良モデルでの再学習と最終評価 ---")
    best_params_copy = best_result['params_combo'].copy()
    best_pca_n = best_params_copy.pop('pca_n_components')
    best_include_time = best_params_copy.pop('include_time_features')
    
    print(f"最良モデルの特徴量を再構築 (PCA={best_pca_n}, Time={best_include_time})...")
    X_pca_best = precalculated_pca_data[best_pca_n]
    if best_include_time:
        X_features_best = np.hstack([X_pca_best, sin_time, cos_time])
    else:
        X_features_best = X_pca_best
    
    final_scaler = MinMaxScaler()
    X_scaled_data_best = final_scaler.fit_transform(X_features_best)

    params_for_init = best_params_copy
    epochs_for_refit = best_result['best_epoch']
    params_for_init.pop('num_epochs')
    
    best_model_full_params_for_refit = {
        **fixed_params_for_acs, 
        **params_for_init, 
        'n_features': X_scaled_data_best.shape[1],
        'random_state': best_result['acs_random_state_used']
    }
    
    best_model_instance = ACS(**best_model_full_params_for_refit)
    
    # --- 6. 結果の可視化と保存 ---
    print("\n--- 6. 結果の可視化 ---")
    pca_visual = PCA(n_components=2, random_state=GLOBAL_SEED)
    X_pca_visual = pca_visual.fit_transform(X_scaled_data_best)
    
    # ★★★ 改善点: エネルギープロット用のディレクトリを作成 ★★★
    energy_plot_dir = output_dir / f"energy_plots_trial_{best_result['trial_count_from_worker']}"
    os.makedirs(energy_plot_dir, exist_ok=True)
    print(f"✅ エネルギー等高線プロットを {energy_plot_dir.resolve()} に保存します。")

    # ★★★ 改善点: 1エポックずつ再学習と可視化を行うループ ★★★
    print(f"\n--- 最良モデルでの再学習とエネルギー遷移の可視化 (計 {epochs_for_refit} エポック) ---")
    for epoch in tqdm(range(1, epochs_for_refit + 1), desc="再学習とプロット生成"):
        best_model_instance.fit(X_scaled_data_best, epochs=1)
        plot_energy_contour_for_epoch(
            model=best_model_instance,
            epoch=epoch,
            save_path=energy_plot_dir,
            X_pca_visual=X_pca_visual,
            y_true_labels=y_true_labels,
            target_names_map=target_names_map,
            pca_visual_model=pca_visual
        )
    print("\n✅ 再学習とエネルギー遷移プロットの生成が完了しました。")

    # --- 最終状態での評価とプロット ---
    final_predicted_labels = best_model_instance.predict(X_scaled_data_best) if best_model_instance.M > 0 else np.full(n_samples, -1)
    final_ari = adjusted_rand_score(y_true_labels, final_predicted_labels)
    final_clusters = best_model_instance.M
    final_accuracy = 0.0
    if final_clusters > 0:
        final_contingency = pd.crosstab(final_predicted_labels, y_true_labels)
        final_row_ind, final_col_ind = linear_sum_assignment(-final_contingency.values)
        final_accuracy = final_contingency.values[final_row_ind, final_col_ind].sum() / n_samples
    
    print("\n--- 最終評価結果 (After Refit) ---")
    print(f"Accuracy (Mapped): {final_accuracy:.4f}")
    print(f"ARI: {final_ari:.4f}")
    print(f"最終クラスタ数: {final_clusters}")

    # 6a. 混同行列
    if final_clusters > 0:
        mapped_pred_labels = np.full_like(y_true_labels, -1)
        pred_idx_to_true_idx_map = {final_contingency.index[pred_i]: true_i for pred_i, true_i in zip(final_row_ind, final_col_ind)}
        for pred_label_val, true_label_idx in pred_idx_to_true_idx_map.items():
             mapped_pred_labels[final_predicted_labels == pred_label_val] = true_label_idx

        cm = confusion_matrix(y_true_labels, mapped_pred_labels, labels=np.arange(n_true_clusters))
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, annot_kws={"size": 10})
        plt.title(f"混同行列 (ACS - 再学習後)\nAccuracy: {final_accuracy:.4f}, ARI: {final_ari:.4f}", fontsize=14)
        plt.xlabel("予測ラベル (マッピング後)"), plt.ylabel("真のラベル")
        plt.savefig(output_dir / f"confusion_matrix_{timestamp}.png", dpi=300, bbox_inches='tight'), plt.close()
        print(f"✅ 混同行列を保存しました。")
    else:
        print("クラスタが形成されなかったため、混同行列はスキップします。")

    # 6b. PCAクラスタリング結果 (最終状態)
    plt.figure(figsize=(18, 8))
    plt.subplot(1, 2, 1)
    scatter_true = sns.scatterplot(x=X_pca_visual[:, 0], y=X_pca_visual[:, 1], hue=[target_names_map[l] for l in y_true_labels], palette='viridis', s=20, alpha=0.7)
    plt.title('真の気圧配置パターン (PCA 2D)', fontsize=16), plt.xlabel('主成分1'), plt.ylabel('主成分2'), scatter_true.legend(title="真のラベル", bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.subplot(1, 2, 2)
    if final_clusters > 0:
        mapped_hue = [target_names_map.get(l, 'Unmapped') for l in mapped_pred_labels]
        hue_order = [target_names_map[i] for i in sorted(target_names_map.keys())] + ['Unmapped']
        scatter_pred = sns.scatterplot(x=X_pca_visual[:, 0], y=X_pca_visual[:, 1], hue=mapped_hue, hue_order=hue_order, palette='viridis', s=20, alpha=0.7)
        
        cluster_centers = best_model_instance.get_cluster_centers()
        if cluster_centers.shape[0] > 0:
            cluster_centers_2d = pca_visual.transform(cluster_centers)
            plt.scatter(cluster_centers_2d[:, 0], cluster_centers_2d[:, 1], c='red', marker='X', s=250, edgecolor='black', linewidth=1.5, label='クラスタ中心 (W)', zorder=10)
        
        handles, labels = scatter_pred.get_legend_handles_labels()
        center_handle = plt.Line2D([0], [0], marker='X', color='w', label='クラスタ中心 (W)', markerfacecolor='red', markeredgecolor='black', markersize=12)
        handles.insert(0, center_handle)
        labels.insert(0, 'クラスタ中心 (W)')
        scatter_pred.legend(handles=handles, labels=labels, title="予測ラベル", bbox_to_anchor=(1.02, 1), loc='upper left')
    else:
        sns.scatterplot(x=X_pca_visual[:, 0], y=X_pca_visual[:, 1], color='gray', s=20, alpha=0.7)

    plt.title(f'ACSクラスタリング結果 (最終Epoch: {epochs_for_refit})\nAcc: {final_accuracy:.3f}, Cls: {final_clusters}', fontsize=16)
    plt.xlabel('主成分1'), plt.ylabel('主成分2')
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(output_dir / f"pca_clustering_plot_final_{timestamp}.png", dpi=300, bbox_inches='tight'), plt.close()
    print(f"✅ 最終状態のPCAクラスタリング結果を保存しました。")
    
    # 6c. 学習履歴プロット
    if best_result and best_result['history']:
        history = best_result['history']
        epochs = [h['epoch'] for h in history]
        history_acc = [h['accuracy_mapped'] for h in history]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, history_acc, 's-', markersize=4, color='tab:blue', label='Accuracy (Mapped)')
        ax.set_xlabel('エポック数'), ax.set_ylabel('Accuracy (マッピング後)'), ax.set_title('最良モデルの学習推移 (ランダムサーチ時)')
        
        best_epoch_num = best_result.get('best_epoch')
        if best_epoch_num:
            best_acc_val = best_result.get('accuracy_mapped')
            ax.axvline(x=best_epoch_num, color='red', linestyle='--', 
                       label=f'Best Epoch ({best_epoch_num}): Acc={best_acc_val:.3f}')
        
        ax.grid(True, linestyle='--')
        ax.legend()
        fig.tight_layout()
        plt.savefig(output_dir / f"best_model_history_{timestamp}.png", dpi=300, bbox_inches='tight'), plt.close()
        print(f"✅ 最良モデルの学習推移グラフを保存しました。")

    print("\n--- 全処理完了 ---")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    if isinstance(sys.stdout, Logger):
        logger_instance.close()
        sys.stdout, sys.stderr = original_stdout, original_stderr

if __name__ == '__main__':
    original_stdout_main = sys.stdout
    original_stderr_main = sys.stderr
    main_process_logic()