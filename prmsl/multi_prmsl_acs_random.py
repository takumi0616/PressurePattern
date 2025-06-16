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
# ★★★ ランダムサーチ用の出力ディレクトリ名に変更 ★★★
output_dir_base = Path("./result_prmsl_acs_random_search") # 汎用的な名前に変更
output_dir = output_dir_base / f"run_{timestamp}"
trial_logs_dir = output_dir / "trial_logs" # ★★★ トライアル別ログ用ディレクトリ ★★★
os.makedirs(output_dir, exist_ok=True)
os.makedirs(trial_logs_dir, exist_ok=True) # ★★★ ディレクトリ作成 ★★★
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
                  X_data, y_data, n_true_cls_worker,
                  trial_log_dir_path, label_class_names): # ★★★ 引数追加 ★★★
    """グリッドサーチ/ランダムサーチの1試行を独立して実行するワーカー関数。"""
    (trial_count, params_combo), trial_specific_seed = param_values_tuple_with_trial_info
    
    # ★★★ ワーカーごとのログファイル設定 ★★★
    worker_log_path = trial_log_dir_path / f"trial_{trial_count}.log"
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        with open(worker_log_path, 'w', encoding='utf-8') as log_file:
            sys.stdout = log_file
            sys.stderr = log_file

            num_epochs_worker = params_combo.pop('num_epochs')
            # activation_typeをparams_comboから取り出す
            activation_type_worker = params_combo.pop('activation_type')
            
            # activation_typeに応じて不要なパラメータを削除
            if activation_type_worker == 'circular':
                params_combo.pop('initial_lambda_vector_val', None)
                params_combo.pop('initial_lambda_crossterm_val', None)
            else: # elliptical
                params_combo.pop('initial_lambda_scalar', None)

            current_run_params = {
                **fixed_params_dict, 
                **params_combo, 
                'activation_type': activation_type_worker, # ワーカーのactivation_typeを設定
                'random_state': trial_specific_seed
            }
            
            # --- ログ出力の開始 ---
            print(f"\n--- [Worker] トライアル {trial_count} 開始 ---")
            log_params = {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in params_combo.items()}
            print(f"[Worker {trial_count}] Activation Type: {activation_type_worker}")
            print(f"[Worker {trial_count}] パラメータ: {log_params}")
            print(f"[Worker {trial_count}] エポック数: {num_epochs_worker}")

            result = {
                'params_combo': {**params_combo, 'num_epochs': num_epochs_worker, 'activation_type': activation_type_worker},
                'ari': -1.0,
                'accuracy_mapped': -1.0,
                'final_clusters': -1,
                'history': [],
                'error_traceback': None,
                'duration_seconds': 0,
                'acs_random_state_used': trial_specific_seed,
                'trial_count_from_worker': trial_count
            }
            n_samples_worker = X_data.shape[0]
            trial_start_time = datetime.datetime.now()

            try:
                acs_model_trial = ACS(**current_run_params)
                
                # --- エポック毎に評価を記録するループ ---
                for epoch in range(num_epochs_worker):
                    acs_model_trial.fit(X_data, epochs=1)
                    
                    current_clusters = acs_model_trial.M
                    epoch_ari, epoch_acc = 0.0, 0.0

                    if current_clusters > 0:
                        preds = acs_model_trial.predict(X_data)
                        epoch_ari = adjusted_rand_score(y_data, preds)
                        contingency = pd.crosstab(preds, y_data)
                        row_ind, col_ind = linear_sum_assignment(-contingency.values)
                        epoch_acc = contingency.values[row_ind, col_ind].sum() / n_samples_worker
                    
                    result['history'].append({'epoch': epoch + 1, 'clusters': current_clusters, 'ari': epoch_ari, 'accuracy_mapped': epoch_acc})
                    
                    print(f"[Worker {trial_count}] Epoch {epoch+1}/{num_epochs_worker} - Cls: {current_clusters}, ARI: {epoch_ari:.4f}, Acc: {epoch_acc:.4f}")

                final_history = result['history'][-1]
                result['ari'], result['accuracy_mapped'], result['final_clusters'] = final_history['ari'], final_history['accuracy_mapped'], final_history['clusters']
                
                # ★★★ ラベルごとの精度を計算・表示 ★★★
                if result['final_clusters'] > 0:
                    print("\n--- ラベル別 最終精度 (Recall) ---")
                    preds = acs_model_trial.predict(X_data)
                    contingency = pd.crosstab(preds, y_data)
                    row_ind, col_ind = linear_sum_assignment(-contingency.values)
                    
                    mapped_preds = np.full_like(y_data, -1)
                    pred_to_true_map = {pred_idx: true_idx for pred_idx, true_idx in zip(row_ind, col_ind)}
                    for pred_val, true_val in pred_to_true_map.items():
                         # contingencyのインデックスから実際の予測ラベル値を取得
                        actual_pred_label = contingency.index[pred_val]
                        mapped_preds[preds == actual_pred_label] = true_val

                    recalls = recall_score(y_data, mapped_preds, average=None, labels=np.arange(len(label_class_names)), zero_division=0)
                    for i, class_name in enumerate(label_class_names):
                        print(f"  - {class_name:<4s}: {recalls[i]:.4f}")

            except Exception:
                result['error_traceback'] = traceback.format_exc()
                print(f"--- [Worker] トライアル {trial_count} でエラー発生 ---\n{result['error_traceback']}")

            trial_end_time = datetime.datetime.now()
            result['duration_seconds'] = (trial_end_time - trial_start_time).total_seconds()
            
            print(f"\n--- [Worker] トライアル {trial_count} 終了 | Acc: {result['accuracy_mapped']:.4f}, ARI: {result['ari']:.4f}, Cls: {result['final_clusters']}, Time: {result['duration_seconds']:.2f}s ---")
            
            # 標準出力を元に戻す
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            
            return result
    finally:
        # エラーが発生しても必ず標準出力を元に戻す
        sys.stdout = original_stdout
        sys.stderr = original_stderr


# ==============================================================================
# ★★★ ランダムサーチのためのパラメータサンプリング関数 ★★★
# ==============================================================================
def sample_random_params(param_dist):
    """
    ランダムサーチのために、定義されたパラメータ範囲から値をサンプリングする。
    """
    params = {}
    for key, value in param_dist.items():
        if isinstance(value, list):
            params[key] = random.choice(value)
        elif isinstance(value, tuple) and len(value) == 2:
            # ★★★ 小数点以下2桁に丸める ★★★
            params[key] = round(np.random.uniform(value[0], value[1]), 2)
    return params

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
    print(f"★★★ 各トライアルの詳細ログ: {trial_logs_dir.resolve()} ★★★") # ★★★ ログ場所の案内 ★★★
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
    print("... 学習データ(X)を作成中 ...")
    msl_data = ds_filtered['msl'].values
    msl_flat = msl_data.reshape(n_samples, -1)
    pca_model = PCA(n_components=20, random_state=GLOBAL_SEED)
    msl_pca = pca_model.fit_transform(msl_flat)
    print(f"✅ 'msl'データをPCAで20次元に削減しました。")
    sin_time = ds_filtered['sin_time'].values.reshape(-1, 1)
    cos_time = ds_filtered['cos_time'].values.reshape(-1, 1)
    X_features = np.hstack([msl_pca, sin_time, cos_time])
    n_total_features = X_features.shape[1]
    print(f"✅ PCAデータと時間情報を結合し、{n_total_features}次元の最終特徴量を作成しました。")
    scaler = MinMaxScaler()
    X_scaled_data = scaler.fit_transform(X_features)
    print("✅ 全特徴量をMin-Maxスケーリングで正規化しました。")

    # --- 2. ランダムサーチの設定 ---
    print("\n--- 2. ランダムサーチ設定 ---")
    param_dist = {
        'gamma': (0.01, 3.0),
        'beta': (0.001, 1.0),
        'learning_rate_W': (0.001, 0.1),
        'learning_rate_lambda': (0.001, 0.1),
        'learning_rate_Z': (0.001, 0.1),
        'initial_lambda_scalar': (0.001, 1.0), # for circular
        'initial_lambda_vector_val': (0.001, 1.0), # for elliptical
        'initial_lambda_crossterm_val': (-0.5, 0.5), # for elliptical
        'initial_Z_val': (0.01, 1.0),
        'initial_Z_new_cluster': (0.01, 1.0),
        'theta_new': (0.001, 1.0),
        'Z_death_threshold': (0.01, 0.1),
        'death_patience_steps': [n_samples // 10, n_samples // 4, n_samples // 2, n_samples, n_samples * 2],
        'num_epochs': [1000],
        'activation_type': ['circular', 'elliptical'] # ★★★ activation_typeを追加 ★★★
    }
    
    N_TRIALS = 10000
    ACCURACY_GOAL = 0.8

    fixed_params_for_acs = {
        'max_clusters': 30, 'initial_clusters': 1, 'n_features': n_total_features,
        'lambda_min_val': 1e-7, 'bounds_W': (0, 1)
        # activation_type はランダムサーチ対象のため削除
    }
    print(f"ランダムサーチ最大試行回数: {N_TRIALS}")
    print(f"早期終了の目標精度 (Accuracy): {ACCURACY_GOAL}")

    # --- 3. 並列ランダムサーチの実行 ---
    print("\n--- 3. 並列ランダムサーチ実行 ---")
    available_cpus = os.cpu_count()
    num_processes_to_use = max(1, int(available_cpus * 0.95)) if available_cpus else 2
    print(f"利用可能CPU数: {available_cpus}, 使用プロセス数: {num_processes_to_use}")

    tasks_for_pool = []
    for i in range(N_TRIALS):
        params_combo = sample_random_params(param_dist)
        trial_specific_seed = GLOBAL_SEED + i + 1
        tasks_for_pool.append(((i + 1, params_combo), trial_specific_seed))

    worker_func_with_fixed_args = partial(run_acs_trial,
                                          fixed_params_dict=fixed_params_for_acs,
                                          X_data=X_scaled_data, y_data=y_true_labels,
                                          n_true_cls_worker=n_true_clusters,
                                          trial_log_dir_path=trial_logs_dir, # ★★★ 引数追加 ★★★
                                          label_class_names=label_encoder.classes_) # ★★★ 引数追加 ★★★

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
            
            if result['accuracy_mapped'] >= ACCURACY_GOAL:
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
                        'duration_seconds': res['duration_seconds'], 'error_present': bool(res['error_traceback'])})
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
    print(f"Accuracy (Mapped): {best_result['accuracy_mapped']:.4f}")
    print(f"ARI: {best_result['ari']:.4f}")
    print(f"最終クラスタ数: {best_result['final_clusters']}")
    print("パラメータ:")
    for key, val in best_params_combo_dict.items():
        print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")
    
    # --- 5. 最良モデルでの再学習と最終評価 ---
    print("\n--- 5. 最良モデルでの再学習と最終評価 ---")
    params_for_init = best_params_combo_dict.copy()
    epochs_for_refit = params_for_init.pop('num_epochs')
    best_model_full_params_for_refit = {**fixed_params_for_acs, **params_for_init, 'random_state': best_result['acs_random_state_used']}
    
    pca_visual = PCA(n_components=2, random_state=GLOBAL_SEED)
    X_pca_visual = pca_visual.fit_transform(X_scaled_data)

    best_model_instance = ACS(**best_model_full_params_for_refit)
    best_model_instance.fit(X_scaled_data, epochs=epochs_for_refit)
    print("✅ 最良モデルの再学習が完了しました。")

    final_predicted_labels = best_model_instance.predict(X_scaled_data) if best_model_instance.M > 0 else np.full(n_samples, -1)
    final_ari = adjusted_rand_score(y_true_labels, final_predicted_labels)
    final_clusters = best_model_instance.M
    final_accuracy = 0.0
    if final_clusters > 0:
        final_contingency = pd.crosstab(final_predicted_labels, y_true_labels)
        final_row_ind, final_col_ind = linear_sum_assignment(-final_contingency.values)
        final_accuracy = final_contingency.values[final_row_ind, final_col_ind].sum() / n_samples
    
    print("\n--- 最終評価結果 ---")
    print(f"Accuracy (Mapped): {final_accuracy:.4f}")
    print(f"ARI: {final_ari:.4f}")
    print(f"最終クラスタ数: {final_clusters}")
    
    # --- 6. 結果の可視化と保存 ---
    print("\n--- 6. 結果の可視化 ---")
    
    # 6a. 混同行列
    if final_clusters > 0:
        mapped_pred_labels = np.full_like(y_true_labels, -1)
        pred_to_true_map = {pred_idx: true_idx for pred_idx, true_idx in zip(final_contingency.index[final_row_ind], label_encoder.classes_[final_col_ind])}
        
        # マッピング辞書を作り直し、ラベルインデックスに変換
        pred_idx_to_true_idx_map = {contingency.index[pred_i]: true_i for pred_i, true_i in zip(final_row_ind, final_col_ind)}
        for pred_label_val, true_label_idx in pred_idx_to_true_idx_map.items():
             mapped_pred_labels[final_predicted_labels == pred_label_val] = true_label_idx

        cm = confusion_matrix(y_true_labels, mapped_pred_labels, labels=np.arange(n_true_clusters))
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, annot_kws={"size": 10})
        plt.title(f"混同行列 (ACS - Random Search)\nAccuracy: {final_accuracy:.4f}, ARI: {final_ari:.4f}", fontsize=14)
        plt.xlabel("予測ラベル (マッピング後)"), plt.ylabel("真のラベル")
        plt.savefig(output_dir / f"confusion_matrix_{timestamp}.png", dpi=300, bbox_inches='tight'), plt.close()
        print(f"✅ 混同行列を保存しました。")
    else:
        print("クラスタが形成されなかったため、混同行列はスキップします。")

    # 6b. PCAクラスタリング結果
    plt.figure(figsize=(16, 7))
    plt.subplot(1, 2, 1)
    scatter_true = sns.scatterplot(x=X_pca_visual[:, 0], y=X_pca_visual[:, 1], hue=[target_names_map[l] for l in y_true_labels], palette='viridis', s=50, alpha=0.7)
    plt.title('真の気圧配置パターン (PCA 2D)'), plt.xlabel('主成分1'), plt.ylabel('主成分2'), scatter_true.legend(title="真のラベル", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.subplot(1, 2, 2)
    if final_clusters > 0:
        mapped_hue = [target_names_map.get(l, 'Unmapped') for l in mapped_pred_labels]
        hue_order = [target_names_map[i] for i in sorted(target_names_map.keys())] + ['Unmapped']
        scatter_pred = sns.scatterplot(x=X_pca_visual[:, 0], y=X_pca_visual[:, 1], hue=mapped_hue, hue_order=hue_order, palette='viridis', s=50, alpha=0.7)
        scatter_pred.legend(title="予測ラベル", bbox_to_anchor=(1.05, 1), loc='upper left')
        cluster_centers_22d = best_model_instance.get_cluster_centers()
        if cluster_centers_22d.shape[0] > 0:
            # スケーリングを逆変換してからPCAにかける
            # 特徴量全体 (PCA+時間) のスケーラーなので、PCA部分だけ取り出して逆変換する必要がある
            # 簡単のため、ここでは22次元の中心を直接PCA変換する（厳密には不正確だが可視化目的）
            if cluster_centers_22d.shape[1] == X_scaled_data.shape[1]:
                 cluster_centers_2d = pca_visual.transform(cluster_centers_22d)
                 plt.scatter(cluster_centers_2d[:, 0], cluster_centers_2d[:, 1], c='red', marker='X', s=200, edgecolor='white', label='クラスタ中心(W)')
    else:
        sns.scatterplot(x=X_pca_visual[:, 0], y=X_pca_visual[:, 1], color='gray', s=50, alpha=0.7)

    plt.title(f'ACSクラスタリング結果 (PCA 2D)\nAcc: {final_accuracy:.3f}, Cls: {final_clusters}', fontsize=14)
    plt.xlabel('主成分1'), plt.ylabel('主成分2'), plt.tight_layout()
    plt.savefig(output_dir / f"pca_clustering_plot_{timestamp}.png", dpi=300, bbox_inches='tight'), plt.close()
    print(f"✅ PCAクラスタリング結果を保存しました。")
    
    # 6c. 学習履歴プロット
    if best_result and best_result['history']:
        history = best_result['history']
        epochs = [h['epoch'] for h in history]
        history_acc = [h['accuracy_mapped'] for h in history]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, history_acc, 's-', color='tab:blue', label='Accuracy (Mapped)')
        ax.set_xlabel('エポック数'), ax.set_ylabel('Accuracy (マッピング後)'), ax.set_title('最良モデルの学習推移 (ランダムサーチ時)')
        ax.grid(True, linestyle='--'), ax.legend(), fig.tight_layout()
        plt.savefig(output_dir / f"best_model_history_{timestamp}.png", dpi=300, bbox_inches='tight'), plt.close()
        print(f"✅ 最良モデルの学習推移グラフを保存しました。")

    print("\n--- 全処理完了 ---")
    # Loggerを閉じる
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    if isinstance(sys.stdout, Logger):
        logger_instance.close()
        sys.stdout, sys.stderr = original_stdout, original_stderr

if __name__ == '__main__':
    # メインロジックを直接呼び出す前に、元のstdout/stderrを保持
    original_stdout_main = sys.stdout
    original_stderr_main = sys.stderr
    main_process_logic()