import sys
from pathlib import Path
import itertools # グリッドサーチのために追加
import multiprocessing # 並列処理のために追加
import os # os.cpu_count() やパス操作のために追加

# --- 1. パスの改善: ACSクラスのインポート ---
# acs.py がカレントディレクトリにあることを前提とします。
try:
    from acs import ACS # acs.py から ACS クラスをインポート
    print(f"ACS class (acs.py) imported successfully from current directory.")
except ImportError as e:
    print(f"Error importing ACS class from acs.py in current directory: {e}")
    print("Please ensure acs.py is in the same directory as this script, or adjust Python's import path.")
    sys.exit(1)


# データセット、前処理、評価、プロットに必要なライブラリをインポート
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import japanize_matplotlib # Matplotlibの日本語表示改善
import datetime
import random # ACSモデル内のrandom_state設定のために維持
from collections import Counter
import pandas as pd # 結果のCSV保存のため
import traceback # エラー詳細表示のため
from functools import partial # partial関数で引数を固定するため

# --- グローバルなNumPyとPythonのrandomモジュールのシード値を固定 ---
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

# --- 0. 出力ディレクトリとロギングの設定 ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# --- 2. パスの改善: 出力ディレクトリをカレントディレクトリ配下に ---
output_dir_base = Path("./result_test_acs_elliptical_grid") # カレントディレクトリ基準
output_dir = output_dir_base / f"run_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
log_file_path = output_dir / f"grid_search_log_elliptical_{timestamp}.txt" # ファイル名に elliptical

class Logger(object):
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

# ワーカープロセスのための処理関数 (circular版と同一構造)
def run_acs_trial(param_values_tuple_with_trial_info,
                  fixed_params_dict,
                  X_data, y_data, n_true_cls_worker, num_eps_worker):
    """
    グリッドサーチの1試行を実行するワーカー関数。
    ACSモデルの学習、評価を行い、結果を返す。
    """
    (trial_count, param_values_tuple), param_names_list, trial_specific_seed = param_values_tuple_with_trial_info
    params_combo = dict(zip(param_names_list, param_values_tuple))

    current_random_state_for_acs = trial_specific_seed
    current_run_params = {**fixed_params_dict, **params_combo, 'random_state': current_random_state_for_acs}
    
    result = {
        'params_combo': params_combo,
        'ari': -1.0,
        'accuracy_mapped': -1.0,
        'final_clusters': -1,
        'error_traceback': None,
        'duration_seconds': 0,
        'acs_random_state_used': current_random_state_for_acs,
        'trial_count_from_worker': trial_count
    }
    n_samples_worker = X_data.shape[0]
    trial_start_time = datetime.datetime.now()

    try:
        acs_model_trial = ACS(**current_run_params)
        acs_model_trial.fit(X_data, epochs=num_eps_worker)

        final_n_clusters_trial = acs_model_trial.M
        result['final_clusters'] = final_n_clusters_trial
        predicted_labels_trial = np.full(n_samples_worker, -1, dtype=int)

        if final_n_clusters_trial > 0:
            predicted_labels_trial = acs_model_trial.predict(X_data)

        current_ari_score = -1.0
        current_accuracy_mapped = 0.0
        valid_preds_mask = (predicted_labels_trial != -1)
        n_valid_preds = np.sum(valid_preds_mask)

        if n_valid_preds > 0:
            y_true_subset = y_data[valid_preds_mask].flatten()
            pred_subset = predicted_labels_trial[valid_preds_mask].flatten()
            current_ari_score = adjusted_rand_score(y_true_subset, pred_subset)
            
            pred_unique_labels = np.unique(pred_subset)
            n_predicted_clusters_effective = len(pred_unique_labels)

            if n_predicted_clusters_effective > 0:
                pred_label_map = {label: i for i, label in enumerate(pred_unique_labels)}
                contingency_matrix_trial = np.zeros((n_true_cls_worker, n_predicted_clusters_effective), dtype=np.int64)
                mapped_preds_for_cm = np.array([pred_label_map[l] for l in pred_subset])

                for i_sample in range(n_valid_preds):
                    true_label = y_true_subset[i_sample]
                    pred_label_mapped_idx = mapped_preds_for_cm[i_sample]
                    contingency_matrix_trial[true_label, pred_label_mapped_idx] += 1
                
                if contingency_matrix_trial.shape[0] > 0 and contingency_matrix_trial.shape[1] > 0:
                    cost_matrix = -contingency_matrix_trial
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    mapped_accuracy_count = contingency_matrix_trial[row_ind, col_ind].sum()
                    current_accuracy_mapped = mapped_accuracy_count / n_valid_preds
                else:
                    current_accuracy_mapped = 0.0
        
        result['ari'] = current_ari_score
        result['accuracy_mapped'] = current_accuracy_mapped

    except Exception:
        result['error_traceback'] = traceback.format_exc()

    trial_end_time = datetime.datetime.now()
    result['duration_seconds'] = (trial_end_time - trial_start_time).total_seconds()
    return result


def main_process_logic():
    logger_instance = Logger(log_file_path)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = logger_instance
    sys.stderr = logger_instance

    print(f"ACSモデル (acs.py - 動的クラスタリング - 楕円形活性化) のIrisデータセットに対する並列グリッドサーチと評価を開始します。") # タイトル変更
    print(f"目標: 最終クラスタ数 3, マッピング後Accuracy 0.953以上") # 目標値は元のスクリプトから
    print(f"結果はディレクトリ '{output_dir.resolve()}' に保存されます。")
    print(f"ログファイル: {log_file_path.resolve()}")
    print(f"実行開始時刻: {timestamp}")
    print(f"グローバル乱数シード (メインプロセス): {GLOBAL_SEED}")

    # 1. Irisデータセットの準備
    iris = load_iris()
    X_data_original = iris.data
    y_true_labels = iris.target.astype(int)
    n_samples, n_original_features = X_data_original.shape
    n_true_clusters = len(np.unique(y_true_labels))
    target_names = iris.target_names
    plot_colors = plt.cm.get_cmap('viridis', n_true_clusters)

    print(f"データ準備完了: Irisデータセット - {n_samples} サンプル, {n_original_features} 特徴量, 真のクラスタ数: {n_true_clusters}")

    scaler = MinMaxScaler()
    X_scaled_data = scaler.fit_transform(X_data_original)

    pca_visual = PCA(n_components=2, random_state=GLOBAL_SEED)
    X_pca_visual = pca_visual.fit_transform(X_scaled_data)
    print("PCAによる2次元への削減完了 (可視化用)。ACSモデルへの入力はスケーリング後の元次元データを使用します。")

    # 2. グリッドサーチのためのパラメータグリッドを設定 (楕円形活性化用)
    # 元のスクリプトのグリッドを尊重しつつ、試行数を減らすために一部範囲を狭めるか、値を絞る
    param_grid = {
        'gamma': [0.01, 0.1, 1.0], # 元より絞る
        'beta': [0.01, 0.1, 1.0],  # 元より絞る
        'learning_rate_W': [0.01, 0.1],
        'learning_rate_lambda': [0.01, 0.1],
        'learning_rate_Z': [0.01, 0.1],
        'initial_lambda_vector_val': [0.01, 0.1, 1.0], # 楕円形特有
        'initial_lambda_crossterm_val': [-0.01, 0.0, 0.01, 0.1], # 楕円形特有, 元より絞る
        'initial_Z_val': [0.1, 0.5, 1.0], # 元より絞る
        'initial_Z_new_cluster': [0.01, 0.1, 0.5], # 元より絞る
        'theta_new': [0.01, 0.1, 0.5], # 元より絞る
        'death_patience_steps': [n_samples // 4, n_samples, n_samples * 2], # 元より絞る
        'Z_death_threshold': [0.01, 0.05, 0.1] # 元より絞る
    }

    fixed_params_for_acs = {
        'max_clusters': 10,
        'initial_clusters': 1,
        'n_features': n_original_features,
        'activation_type': 'elliptical', # ★楕円形
        'lambda_min_val': 1e-7,
        'bounds_W': (0, 1),
    }
    num_epochs_per_trial = 200 # グリッドサーチ時のエポック数 (元の300から調整)
    target_accuracy_goal = 0.953
    target_n_clusters_goal = 3

    print(f"使用する活性化関数タイプ: {fixed_params_for_acs['activation_type']}")
    print(f"クラスタ数の上限 (max_clusters): {fixed_params_for_acs['max_clusters']}")
    print(f"初期クラスタ数 (initial_clusters): {fixed_params_for_acs['initial_clusters']}")
    print(f"エポック数 (各試行): {num_epochs_per_trial}")

    param_names_grid = list(param_grid.keys())
    value_lists_grid = [param_grid[key] for key in param_names_grid]
    all_param_combinations_values_grid = list(itertools.product(*value_lists_grid))
    num_total_trials_grid = len(all_param_combinations_values_grid)
    print(f"グリッドサーチ試行総数: {num_total_trials_grid}")

    tasks_for_pool = []
    for i, param_vals_iter in enumerate(all_param_combinations_values_grid):
        trial_specific_seed_val = GLOBAL_SEED + i + 1
        task_item = (
            (i + 1, param_vals_iter),
            param_names_grid,
            trial_specific_seed_val
        )
        tasks_for_pool.append(task_item)

    available_cpus = os.cpu_count()
    if available_cpus is None:
        print("警告: CPU数を取得できませんでした。デフォルト2プロセス使用。")
        num_processes_to_use = 2
    else:
        num_processes_to_use = max(1, int(available_cpus * 0.4))
    print(f"利用CPU数: {available_cpus}, 使用プロセス数(最大): {num_processes_to_use}")

    all_trial_results_list = []
    start_grid_search_time = datetime.datetime.now()
    
    worker_func_with_fixed_args = partial(run_acs_trial,
                                          fixed_params_dict=fixed_params_for_acs,
                                          X_data=X_scaled_data,
                                          y_data=y_true_labels,
                                          n_true_cls_worker=n_true_clusters,
                                          num_eps_worker=num_epochs_per_trial)

    print(f"\n並列グリッドサーチを開始します (プロセス数: {num_processes_to_use})...")
    with multiprocessing.Pool(processes=num_processes_to_use) as pool:
        all_trial_results_list = pool.map(worker_func_with_fixed_args, tasks_for_pool)

    end_grid_search_time = datetime.datetime.now()
    print(f"\n並列グリッドサーチ完了。総所要時間: {end_grid_search_time - start_grid_search_time}")

    best_ari_score = -1.0
    best_accuracy_mapped = -1.0
    best_params_combo_dict = None
    best_model_full_params_for_refit = None
    best_final_n_clusters = 0
    processed_results_for_df = []

    for trial_idx_print, res_dict in enumerate(all_trial_results_list, 1):
        trial_display_id = res_dict.get('trial_count_from_worker', trial_idx_print)
        if res_dict['error_traceback']:
            print(f"\n--- トライアル {trial_display_id}/{num_total_trials_grid} エラー ---")
            print(f"パラメータ: { {k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in res_dict['params_combo'].items()} }")
            print(f"ACS Seed: {res_dict['acs_random_state_used']}")
            print(f"エラー発生(詳細はログ参照): {res_dict['error_traceback'].strip().splitlines()[-1]}")
            print(f"所要時間: {res_dict['duration_seconds']:.2f}s")
        else:
            print(f"\n--- トライアル {trial_display_id}/{num_total_trials_grid} 結果 ---")
            print(f"パラメータ: { {k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in res_dict['params_combo'].items()} }")
            print(f"ACS Seed: {res_dict['acs_random_state_used']}")
            print(f"最終Cls: {res_dict['final_clusters']}, ARI: {res_dict['ari']:.4f}, Acc(map): {res_dict['accuracy_mapped']:.4f}, Time: {res_dict['duration_seconds']:.2f}s")

        df_row = {'trial_id': trial_display_id, **res_dict['params_combo'], 'ari': res_dict['ari'],
                  'accuracy_mapped': res_dict['accuracy_mapped'], 'final_clusters': res_dict['final_clusters'],
                  'duration_seconds': res_dict['duration_seconds'], 'acs_random_state_used': res_dict['acs_random_state_used'],
                  'error_present': bool(res_dict['error_traceback'])}
        processed_results_for_df.append(df_row)

        is_better = False
        if not res_dict['error_traceback']:
            if best_params_combo_dict is None or \
               res_dict['accuracy_mapped'] > best_accuracy_mapped or \
               (res_dict['accuracy_mapped'] == best_accuracy_mapped and res_dict['ari'] > best_ari_score):
                is_better = True
            
            if is_better:
                best_accuracy_mapped = res_dict['accuracy_mapped']
                best_ari_score = res_dict['ari']
                best_params_combo_dict = res_dict['params_combo']
                best_model_full_params_for_refit = {**fixed_params_for_acs, **res_dict['params_combo'], 'random_state': res_dict['acs_random_state_used']}
                best_final_n_clusters = res_dict['final_clusters']
                print(f"*** 新最良スコア: Acc={best_accuracy_mapped:.4f}, ARI={best_ari_score:.4f}, Cls={best_final_n_clusters} ***")

    try:
        results_df = pd.DataFrame(processed_results_for_df)
        results_df = results_df.sort_values(by=['accuracy_mapped', 'ari', 'final_clusters'], ascending=[False, False, True])
        df_path = output_dir / f"grid_search_all_results_elliptical_{timestamp}.csv" # ファイル名
        results_df.to_csv(df_path, index=False, encoding='utf-8-sig')
        print(f"\n全試行結果を {df_path.resolve()} に保存。")
    except Exception as e_df:
        print(f"\nCSV保存中エラー: {e_df}\n{traceback.format_exc()}")
        # (リスト表示は省略)

    print("\n--- グリッドサーチ完了判定 ---")
    goal_achieved = False
    if best_params_combo_dict is not None:
        if best_final_n_clusters == target_n_clusters_goal and best_accuracy_mapped >= target_accuracy_goal:
            print(f"目標達成: Acc {best_accuracy_mapped:.4f} (>= {target_accuracy_goal}), Cls {best_final_n_clusters} (== {target_n_clusters_goal})")
            goal_achieved = True
        else:
             print(f"目標未達 (最良Acc: {best_accuracy_mapped:.4f}, 最良Cls: {best_final_n_clusters}). 最良結果を表示。")
    else:
        print("グリッドサーチで有効な結果が得られませんでした。")
        
    if best_model_full_params_for_refit is None:
        print("最良パラメータが見つからず、再学習スキップ。")
        sys.stdout = original_stdout; sys.stderr = original_stderr; logger_instance.close()
        return

    print(f"\n--- 最良モデル再学習・評価 (Elliptical) ---") # タイトル
    print(f"最良Acc (Grid): {best_accuracy_mapped:.4f}, ARI: {best_ari_score:.4f}, Cls: {best_final_n_clusters}")
    print(f"最良パラメータ (Grid対象): {best_params_combo_dict}")
    print(f"ACS random_state (Grid): {best_model_full_params_for_refit.get('random_state', 'N/A')}")

    print("\n最良パラメータでモデル再学習・評価中...")
    best_model_instance = ACS(**best_model_full_params_for_refit)
    final_epochs_for_refit_history = num_epochs_per_trial
    accuracy_history_refit, ari_history_refit, cluster_count_history_refit = [], [], []
    temp_acs_for_history = ACS(**best_model_full_params_for_refit)

    for epoch in range(final_epochs_for_refit_history):
        temp_acs_for_history.fit(X_scaled_data, epochs=1)
        current_final_n_clusters_hist = temp_acs_for_history.M
        cluster_count_history_refit.append(current_final_n_clusters_hist)
        current_predicted_labels_hist = np.full(n_samples, -1, dtype=int)
        if current_final_n_clusters_hist > 0:
            current_predicted_labels_hist = temp_acs_for_history.predict(X_scaled_data)
        valid_preds_mask_epoch = (current_predicted_labels_hist != -1); n_valid_preds_epoch = np.sum(valid_preds_mask_epoch)
        current_ari_hist, current_accuracy_hist = -1.0, 0.0
        if n_valid_preds_epoch > 0:
            y_true_subset_hist = y_true_labels[valid_preds_mask_epoch].flatten(); pred_subset_hist = current_predicted_labels_hist[valid_preds_mask_epoch].flatten()
            current_ari_hist = adjusted_rand_score(y_true_subset_hist, pred_subset_hist)
            pred_unique_labels_epoch = np.unique(pred_subset_hist); n_predicted_clusters_effective_epoch = len(pred_unique_labels_epoch)
            if n_predicted_clusters_effective_epoch > 0:
                pred_label_map_epoch = {label: i for i, label in enumerate(pred_unique_labels_epoch)}
                contingency_matrix_epoch = np.zeros((n_true_clusters, n_predicted_clusters_effective_epoch), dtype=np.int64)
                mapped_preds_for_cm_epoch = np.array([pred_label_map_epoch[l] for l in pred_subset_hist])
                for i_s in range(n_valid_preds_epoch): contingency_matrix_epoch[y_true_subset_hist[i_s], mapped_preds_for_cm_epoch[i_s]] += 1
                if contingency_matrix_epoch.size > 0: # Check if matrix is not empty
                    cost_matrix_epoch = -contingency_matrix_epoch
                    row_ind_e, col_ind_e = linear_sum_assignment(cost_matrix_epoch)
                    current_accuracy_hist = contingency_matrix_epoch[row_ind_e, col_ind_e].sum() / n_valid_preds_epoch
        ari_history_refit.append(current_ari_hist); accuracy_history_refit.append(current_accuracy_hist)
        if (epoch + 1) % (final_epochs_for_refit_history // 10 if final_epochs_for_refit_history >=10 else 1) == 0 or epoch == final_epochs_for_refit_history -1 :
            print(f"再学習(Hist) Ep {epoch+1}/{final_epochs_for_refit_history} - Cls: {current_final_n_clusters_hist}, ARI: {current_ari_hist:.4f}, Acc: {current_accuracy_hist:.4f}")
    
    best_model_instance.fit(X_scaled_data, epochs=num_epochs_per_trial) # Final model fit
    print("再学習完了。")

    final_predicted_labels_refit = best_model_instance.predict(X_scaled_data) if best_model_instance.M > 0 else np.full(n_samples, -1, dtype=int)
    final_learned_cluster_centers_original_space = best_model_instance.get_cluster_centers()
    final_cluster_centers_pca_visual = pca_visual.transform(final_learned_cluster_centers_original_space) if final_learned_cluster_centers_original_space.shape[0] > 0 else np.empty((0,2))
    final_valid_preds_mask = (final_predicted_labels_refit != -1); n_final_valid_preds = np.sum(final_valid_preds_mask)
    final_ari_refit_val, final_accuracy_refit_val = -1.0, 0.0
    final_cm_refit = np.zeros((n_true_clusters, n_true_clusters), dtype=int)
    mapped_labels_for_plot_cm = np.full_like(final_predicted_labels_refit, -1, dtype=int)
    pred_to_true_map_final = {}

    if n_final_valid_preds > 0:
        y_true_subset_final = y_true_labels[final_valid_preds_mask].flatten(); pred_subset_final = final_predicted_labels_refit[final_valid_preds_mask].flatten()
        final_ari_refit_val = adjusted_rand_score(y_true_subset_final, pred_subset_final)
        final_pred_unique_labels = np.unique(pred_subset_final); final_n_predicted_clusters_effective = len(final_pred_unique_labels)
        if final_n_predicted_clusters_effective > 0:
            final_contingency_matrix = np.zeros((n_true_clusters, final_n_predicted_clusters_effective), dtype=np.int64)
            final_pred_label_map = {label: i for i, label in enumerate(final_pred_unique_labels)}
            final_mapped_preds_for_cm = np.array([final_pred_label_map[l] for l in pred_subset_final])
            for i_s in range(n_final_valid_preds): final_contingency_matrix[y_true_subset_final[i_s], final_mapped_preds_for_cm[i_s]] += 1
            if final_contingency_matrix.size > 0:
                cost_matrix_final_refit = -final_contingency_matrix
                row_ind_f, col_ind_f = linear_sum_assignment(cost_matrix_final_refit)
                for true_idx, pred_mapped_idx in zip(row_ind_f, col_ind_f):
                     original_pred_label = final_pred_unique_labels[pred_mapped_idx]
                     original_pred_indices = np.where(final_predicted_labels_refit == original_pred_label)[0]
                     mapped_labels_for_plot_cm[original_pred_indices] = true_idx
                     pred_to_true_map_final[original_pred_label] = true_idx
                final_accuracy_refit_val = final_contingency_matrix[row_ind_f, col_ind_f].sum() / n_final_valid_preds
                valid_mapped_indices = (mapped_labels_for_plot_cm != -1) & final_valid_preds_mask
                if np.sum(valid_mapped_indices) > 0:
                     final_cm_refit = confusion_matrix(y_true_labels[valid_mapped_indices], mapped_labels_for_plot_cm[valid_mapped_indices], labels=list(range(n_true_clusters)))

    # --- プロット (Elliptical用にタイトルやファイル名を変更) ---
    fig, ax1 = plt.subplots(figsize=(12, 7)); # (略: circular版と同様のプロットコード, タイトルとファイル名変更)
    color = 'tab:red'; ax1.set_xlabel('エポック数'); ax1.set_ylabel('ARI', color=color); ax1.plot(range(1, final_epochs_for_refit_history + 1), ari_history_refit, 'o-', color=color, label='ARI'); ax1.tick_params(axis='y', labelcolor=color); ax1.grid(True, linestyle='--', alpha=0.7); ax1.set_ylim(-1.05, 1.05)
    ax2 = ax1.twinx(); color = 'tab:blue'; ax2.set_ylabel('Accuracy (マッピング後)', color=color); ax2.plot(range(1, final_epochs_for_refit_history + 1), accuracy_history_refit, 's--', color=color, label='Accuracy (Mapped)'); ax2.tick_params(axis='y', labelcolor=color); ax2.set_ylim(0, 1.05)
    ax3 = ax1.twinx(); ax3.spines["right"].set_position(("outward", 60)); color = 'tab:green'; ax3.set_ylabel('クラスタ数', color=color); ax3.plot(range(1, final_epochs_for_refit_history + 1), cluster_count_history_refit, '^:', color=color, label='クラスタ数'); ax3.tick_params(axis='y', labelcolor=color); ax3.set_ylim(0, fixed_params_for_acs['max_clusters'] + 1)
    fig.suptitle(f'最良モデル評価推移 (ACS - Elliptical)\n最終ARI: {final_ari_refit_val:.4f}, Acc: {final_accuracy_refit_val:.4f}, Cls: {best_model_instance.M}', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); lines1, labels1_plot = ax1.get_legend_handles_labels(); lines2, labels2_plot = ax2.get_legend_handles_labels(); lines3, labels3_plot = ax3.get_legend_handles_labels(); ax2.legend(lines1 + lines2 + lines3, labels1_plot + labels2_plot + labels3_plot, loc='center right')
    metrics_plot_path = output_dir / f"metrics_vs_epoch_best_dynamic_elliptical_{timestamp}.png"
    plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f"エポック毎評価プロット保存: {metrics_plot_path.resolve()}");

    fig_cm = plt.figure(figsize=(7, 6)); sns.heatmap(final_cm_refit, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names, annot_kws={"size": 12}); plt.xlabel("予測ラベル"); plt.ylabel("真のラベル")
    plt.title(f"混同行列 (ACS - Elliptical)\nAccuracy: {final_accuracy_refit_val:.4f}, ARI: {final_ari_refit_val:.4f}", fontsize=14)
    cm_path = output_dir / f"confusion_matrix_best_dynamic_elliptical_{timestamp}.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight'); plt.close(fig_cm)
    print(f"混同行列画像保存: {cm_path.resolve()}");

    print(f"\n最終ARI (再学習後): {final_ari_refit_val:.4f}"); print(f"最終Acc (再学習後): {final_accuracy_refit_val:.4f}"); print(f"最終Cls (再学習後): {best_model_instance.M}");

    fig_pca_clusters = plt.figure(figsize=(16, 7)); plt.subplot(1, 2, 1); # (略: circular版と同様のPCAプロットコード, タイトル変更)
    for i_cluster_true in range(n_true_clusters): plt.scatter(X_pca_visual[y_true_labels == i_cluster_true, 0], X_pca_visual[y_true_labels == i_cluster_true, 1], color=plot_colors(i_cluster_true / (n_true_clusters -1 if n_true_clusters > 1 else 1)), label=target_names[i_cluster_true], alpha=0.7, s=50)
    plt.title('Irisデータ (PCA) - 真のクラスタ'); plt.xlabel('PCA特徴量1'); plt.ylabel('PCA特徴量2'); plt.legend(title="真のクラス"); plt.grid(True, linestyle='--', alpha=0.7)
    plt.subplot(1, 2, 2);
    if np.sum(mapped_labels_for_plot_cm != -1) > 0:
        for true_cls_idx in range(n_true_clusters):
            mask = (mapped_labels_for_plot_cm == true_cls_idx)
            if np.sum(mask) > 0: plt.scatter(X_pca_visual[mask, 0], X_pca_visual[mask, 1], color=plot_colors(true_cls_idx / (n_true_clusters -1 if n_true_clusters > 1 else 1)), label=f"Pred as {target_names[true_cls_idx]}", alpha=0.7, s=50)
    unmapped_mask = (mapped_labels_for_plot_cm == -1) & final_valid_preds_mask
    if np.sum(unmapped_mask) > 0: plt.scatter(X_pca_visual[unmapped_mask, 0], X_pca_visual[unmapped_mask, 1], color='gray', label='Unmapped/Other', alpha=0.3, s=30)
    if final_cluster_centers_pca_visual.shape[0] > 0:
        for acs_center_idx in range(final_cluster_centers_pca_visual.shape[0]):
            center_color_val = 'black'; center_label_suffix = f" (ACS Ctr {acs_center_idx})"
            original_acs_label_of_center = acs_center_idx
            if original_acs_label_of_center in pred_to_true_map_final:
                true_class_idx_mapped_to = pred_to_true_map_final[original_acs_label_of_center]
                center_color_val = plot_colors(true_class_idx_mapped_to / (n_true_clusters-1 if n_true_clusters > 1 else 1) )
                center_label_suffix = f" (ACS Ctr {original_acs_label_of_center} -> True {target_names[true_class_idx_mapped_to]})"
            plt.scatter(final_cluster_centers_pca_visual[acs_center_idx, 0], final_cluster_centers_pca_visual[acs_center_idx, 1], c=[center_color_val], marker='X', s=200, edgecolor='white', label=f'Center{center_label_suffix}')
    plt.title(f'ACSクラスタリング (PCA) - (Elliptical)\nARI: {final_ari_refit_val:.3f}, Acc: {final_accuracy_refit_val:.3f}, Cls: {best_model_instance.M}', fontsize=14)
    plt.xlabel('PCA特徴量1'); plt.ylabel('PCA特徴量2'); handles_pca, labels_pca = plt.gca().get_legend_handles_labels(); by_label_pca = dict(zip(labels_pca, handles_pca)); plt.legend(by_label_pca.values(), by_label_pca.keys(), title="予測クラス / 中心", fontsize=8, loc="best"); plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout()
    pca_plot_path = output_dir / f"pca_clustering_best_dynamic_elliptical_{timestamp}.png"
    plt.savefig(pca_plot_path, dpi=300, bbox_inches='tight'); plt.close(fig_pca_clusters)
    print(f"PCAクラスタリングプロット保存: {pca_plot_path.resolve()}");

    if best_model_instance.M > 0 and best_model_instance.fitted_:
        print("\nエネルギー関数等高線図生成中...") # (略: circular版と同様のエネルギープロットコード, タイトル・ファイル名変更)
        fig_energy, ax_energy = plt.subplots(figsize=(10, 8))
        x_min_e, x_max_e = X_pca_visual[:, 0].min() - 0.5, X_pca_visual[:, 0].max() + 0.5; y_min_e, y_max_e = X_pca_visual[:, 1].min() - 0.5, X_pca_visual[:, 1].max() + 0.5
        grid_res = 50; xx_e, yy_e = np.meshgrid(np.linspace(x_min_e, x_max_e, grid_res), np.linspace(y_min_e, y_max_e, grid_res))
        pca_points_flat_e = np.c_[xx_e.ravel(), yy_e.ravel()]; original_space_points_flat_e = pca_visual.inverse_transform(pca_points_flat_e)
        energy_values_flat_e = np.array([best_model_instance.calculate_energy_at_point(p.flatten()) for p in original_space_points_flat_e])
        E_grid_e = energy_values_flat_e.reshape(xx_e.shape); E_grid_e_fin = np.nan_to_num(E_grid_e, nan=np.nanmax(E_grid_e[np.isfinite(E_grid_e)]), posinf=np.nanmax(E_grid_e[np.isfinite(E_grid_e)]))
        num_levels = 15; contour_plot_e = ax_energy.contourf(xx_e, yy_e, E_grid_e_fin, levels=num_levels, cmap='coolwarm', alpha=0.7); plt.colorbar(contour_plot_e, ax=ax_energy, label='Energy E'); ax_energy.contour(xx_e, yy_e, E_grid_e_fin, levels=num_levels, colors='k', alpha=0.5, linewidths=0.5)
        for i_true_cls in range(n_true_clusters): ax_energy.scatter(X_pca_visual[y_true_labels == i_true_cls, 0], X_pca_visual[y_true_labels == i_true_cls, 1], color=plot_colors(i_true_cls / (n_true_clusters -1 if n_true_clusters > 1 else 1)), label=target_names[i_true_cls], alpha=0.5, s=30, zorder=2, edgecolor='k', linewidth=0.5)
        if final_cluster_centers_pca_visual.shape[0] > 0: ax_energy.scatter(final_cluster_centers_pca_visual[:, 0], final_cluster_centers_pca_visual[:, 1], c='black', marker='X', s=150, edgecolor='white', label='学習済みクラスタ中心 (W)', zorder=3, linewidth=1)
        ax_energy.set_title(f'エネルギー関数 E (PCA空間 - ACS - Elliptical)\nAcc: {final_accuracy_refit_val:.3f}, Cls: {best_model_instance.M}', fontsize=14)
        ax_energy.set_xlabel('PCA特徴量1'); ax_energy.set_ylabel('PCA特徴量2'); handles_e, labels_e_plot = ax_energy.get_legend_handles_labels(); by_label_e = dict(zip(labels_e_plot, handles_e)); ax_energy.legend(by_label_e.values(), by_label_e.keys(), fontsize=10, loc='upper right'); ax_energy.grid(True, linestyle=':', alpha=0.5); ax_energy.set_xlim(x_min_e, x_max_e); ax_energy.set_ylim(y_min_e, y_max_e)
        energy_contour_path = output_dir / f"energy_contour_plot_elliptical_{timestamp}.png"
        plt.savefig(energy_contour_path, dpi=300, bbox_inches='tight'); plt.close(fig_energy)
        print(f"エネルギー等高線図保存: {energy_contour_path.resolve()}");
    else: print("有効な学習済みモデルがないため、エネルギー等高線図生成スキップ。")

    print("\n--- 最良モデル学習後パラメータ (Elliptical) ---") # (略: circular版と同様のサマリー保存コード, ファイル名変更)
    params_str_list_final = [];
    for k, v_param in best_model_full_params_for_refit.items(): params_str_list_final.append(f"'{k}': {v_param:.7g}" if isinstance(v_param, float) else f"'{k}': {v_param}")
    formatted_params_refit_final = "{" + ", ".join(params_str_list_final) + "}"
    best_model_summary = f"""
使用パラメータ (固定値含, 再学習時): {formatted_params_refit_final}
学習済クラスタ中心 (W):
{final_learned_cluster_centers_original_space if best_model_instance.M > 0 else "N/A"}
学習済警戒パラメータ (lambda):
{best_model_instance.lambdas if best_model_instance.M > 0 else "N/A"}
学習済深さパラメータ (Z_j):
{best_model_instance.Z.flatten() if best_model_instance.M > 0 else "N/A"}
学習済非活性ステップ (inactive_steps):
{best_model_instance.inactive_steps.flatten() if best_model_instance.M > 0 else "N/A"}
最終形成クラスタ数: {best_model_instance.M}"""
    cluster_assignment_summary = "\n--- クラスタ毎データポイント割当 (再学習後) ---\n"
    if best_model_instance.M > 0 and n_final_valid_preds > 0:
        counts_final = Counter(final_predicted_labels_refit[final_valid_preds_mask]); cluster_assignment_summary += f"総有効予測サンプル数: {n_final_valid_preds}\n"
        for i_acs_cluster_label in range(best_model_instance.M):
            mapped_true_class_name = "N/A"; num_points = counts_final.get(i_acs_cluster_label, 0)
            if i_acs_cluster_label in pred_to_true_map_final: mapped_true_class_name = target_names[pred_to_true_map_final[i_acs_cluster_label]]
            percentage = (num_points / n_samples) * 100 if n_samples > 0 else 0
            cluster_assignment_summary += f"  ACSクラスタ {i_acs_cluster_label} (True Cls as: {mapped_true_class_name}): {num_points} ({percentage:.2f}%)\n"
        unassigned_count_final = counts_final.get(-1, 0)
        if unassigned_count_final > 0: percentage = (unassigned_count_final / n_samples) * 100 if n_samples > 0 else 0; cluster_assignment_summary += f"  未分類 (-1): {unassigned_count_final} ({percentage:.2f}%)\n"
    else: cluster_assignment_summary += "  有効なクラスタまたは予測なし。\n"
    print(best_model_summary); print(cluster_assignment_summary)

    param_log_path_final = output_dir / f"best_model_summary_dynamic_elliptical_{timestamp}.txt" # ファイル名
    with open(param_log_path_final, "w", encoding="utf-8") as f:
        f.write(f"実行開始時刻: {timestamp}\n目標達成: {'達成' if goal_achieved else '未達'}\n\n")
        f.write(f"--- グリッドサーチ最良結果 (Elliptical) ---\nAcc(map): {best_accuracy_mapped:.4f}\nARI: {best_ari_score:.4f}\nCls: {best_final_n_clusters}\nParams: {best_params_combo_dict}\nACS Seed: {best_model_full_params_for_refit.get('random_state', 'N/A')}\n\n")
        f.write(f"--- 再学習後最終評価 (Elliptical) ---\nAcc(map): {final_accuracy_refit_val:.4f}\nARI: {final_ari_refit_val:.4f}\nCls: {best_model_instance.M}\n\n")
        f.write(best_model_summary); f.write(cluster_assignment_summary)
    print(f"最良モデルパラメータ詳細とサマリー保存: {param_log_path_final.resolve()}");

    print(f"\n全処理完了。終了時刻: {datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    print(f"結果はディレクトリ '{output_dir.resolve()}' を確認。")

    if isinstance(sys.stdout, Logger):
        sys.stdout = original_stdout; sys.stderr = original_stderr; logger_instance.close()

if __name__ == '__main__':
    main_process_logic()