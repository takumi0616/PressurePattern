import sys
from pathlib import Path
import multiprocessing
import os
import datetime
import random
import traceback
from functools import partial
from tqdm import tqdm
import re
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import japanize_matplotlib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment

try:
    from acs import ACS
    print("ACS class (acs.py) imported successfully.")
except ImportError as e:
    print(f"Error: acs.py からACSクラスをインポートできませんでした: {e}")
    print("このスクリプトと同じディレクトリに acs.py ファイルを配置してください。")
    sys.exit(1)

GLOBAL_SEED = 1
os.environ['PYTHONHASHSEED'] = str(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir_base = Path("./result_prmsl_acs_random_search_v4")
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

def get_sorted_indices(sort_method, valid_times, random_seed=None):
    """
    NumPyのみを使用した高速ソート（決定的な動作を保証）
    """
    n_samples = len(valid_times)
    indices = np.arange(n_samples)
    if sort_method in ['normal_sort', 'change_normal_sort']:
        return indices[np.argsort(valid_times)]
    elif sort_method in ['month_sort', 'change_month_sort']:
        times_dt = valid_times.astype('datetime64[D]')
        times_M = valid_times.astype('datetime64[M]')
        times_Y = valid_times.astype('datetime64[Y]')
        months = ((times_M - times_Y) / np.timedelta64(1, 'M')).astype(int)
        years = times_Y.astype(int) + 1970
        sort_keys = np.lexsort((valid_times, years, months))
        return indices[sort_keys]
    else:
        if random_seed is not None:
            rng = np.random.RandomState(random_seed)
            rng.shuffle(indices)
        else:
            raise ValueError(f"Unknown sort method: {sort_method}")
        return indices

def calculate_all_metrics_multi_label_permissive(preds, y_true_multi, label_encoder):
    """
    【修正版】マクロ平均再現率を計算する。
    1つの正解ラベルが複数のクラスタに分割されることを許容する (多対1マッピング)。
    """
    base_labels = label_encoder.classes_
    n_base_labels = len(base_labels)
    unique_pred_clusters = np.unique(preds[preds != -1])
    n_pred_clusters = len(unique_pred_clusters)

    if n_pred_clusters == 0:
        return {
            'macro_recall': 0.0,
            'n_clusters': 0,
            'pred_map': {},
            'cm': np.zeros((n_base_labels, n_base_labels), dtype=int)
        }

    # 1. コンティンジェンシー行列を作成 (変更なし)
    contingency_np = np.zeros((n_pred_clusters, n_base_labels), dtype=int)
    cluster_to_idx = {cluster: idx for idx, cluster in enumerate(unique_pred_clusters)}
    
    valid_mask = preds != -1
    for i in np.where(valid_mask)[0]:
        cluster_idx = cluster_to_idx[preds[i]]
        for true_idx in y_true_multi[i]:
            if true_idx != -1:
                contingency_np[cluster_idx, true_idx] += 1

    # ================================================================= #
    # 2. 【ここを変更】ハンガリー法から「多対1マッピング」へ
    #    各クラスタ(行)で最も多く含まれる真ラベル(列)をそのクラスタの代表とする
    #    これにより、複数のクラスタが同じ真ラベルにマッピングされることを許容する
    dominant_label_indices = np.argmax(contingency_np, axis=1)
    pred_map = {
        cluster_id: dominant_label_idx 
        for cluster_id, dominant_label_idx in zip(unique_pred_clusters, dominant_label_indices)
    }
    # ================================================================= #

    # 3. マッピング後の混同行列(cm)を作成 (変更なし)
    cm = np.zeros((n_base_labels, n_base_labels), dtype=int)
    mapped_preds = np.full_like(preds, -1, dtype=int)
    for i, p in enumerate(preds):
        if p in pred_map:
            mapped_preds[i] = pred_map[p]

    for i in range(len(preds)):
        pred_label_idx = mapped_preds[i]
        if pred_label_idx == -1:
            continue
        
        for true_label_idx in y_true_multi[i]:
            if true_label_idx != -1:
                cm[pred_label_idx, true_label_idx] += 1
                
    # 4. マクロ平均再現率を計算 (変更なし)
    recalls_per_class = []
    for c in range(n_base_labels):
        total_actual_positives = np.sum(cm[:, c])
        if total_actual_positives > 0:
            tp = cm[c, c]
            recall = tp / total_actual_positives
            recalls_per_class.append(recall)
    
    macro_recall = np.mean(recalls_per_class) if recalls_per_class else 0.0
    
    return {
        'macro_recall': macro_recall,
        'n_clusters': n_pred_clusters,
        'pred_map': pred_map,
        'cm': cm
    }

def run_acs_trial(param_values_tuple_with_trial_info,
                  fixed_params_dict,
                  pca_data_dict,
                  y_data_multi,
                  f1_season_data,
                  f2_season_data,
                  trial_log_dir_path,
                  label_encoder_worker,
                  valid_times_worker):
    """
    ランダムサーチの1試行を独立して実行する。
    指定されたデータ投入順序で学習し、エポックごとにマクロ平均再現率を計算し、全履歴を返す。
    """
    (trial_count, params_combo), trial_specific_seed = param_values_tuple_with_trial_info
    worker_log_path = trial_log_dir_path / f"trial_{trial_count}.log"
    original_stdout, original_stderr = sys.stdout, sys.stderr
    acs_model_trial = None
    
    result = {
        'params_combo': {},
        'history': [],
        'event_log_summary': {},
        'error_traceback': 'Initialization failed',
        'duration_seconds': 0,
        'acs_random_state_used': trial_specific_seed,
        'trial_count_from_worker': trial_count
    }
    
    trial_start_time = datetime.datetime.now()
    
    try:
        with open(worker_log_path, 'w', encoding='utf-8') as log_file:
            sys.stdout = sys.stderr = log_file
            
            data_input_order = params_combo.get('data_input_order')
            pca_n_components = params_combo.get('pca_n_components')
            include_time_features = params_combo.get('include_time_features')
            num_epochs_worker = params_combo.get('num_epochs')
            
            result['params_combo'] = params_combo.copy()
            result['error_traceback'] = None
            
            X_pca = pca_data_dict[pca_n_components]
            X_features = np.hstack([X_pca, f1_season_data, f2_season_data]) if include_time_features else X_pca
            X_scaled_data = MinMaxScaler().fit_transform(X_features).astype(np.float64)
            n_features_worker = X_scaled_data.shape[1]
            
            params_for_acs = params_combo.copy()
            keys_to_remove_for_acs = [
                'data_input_order', 'pca_n_components', 'include_time_features', 'num_epochs'
            ]
            for key in keys_to_remove_for_acs:
                params_for_acs.pop(key, None)
            
            current_run_params = {
                **fixed_params_dict,
                'n_features': n_features_worker,
                'random_state': trial_specific_seed,
                **params_for_acs
            }
            
            print(f"\n--- [Worker] トライアル {trial_count} 開始 ---")
            print(f"[Worker {trial_count}] データ投入順序: {data_input_order}")
            print(f"[Worker {trial_count}] 特徴量: PCA={pca_n_components}, Time={include_time_features}, Total Dim={n_features_worker}")
            print(f"[Worker {trial_count}] パラメータ: { {k: f'{v:.4f}' if isinstance(v, float) else v for k, v in params_combo.items()} }")
            
            np.random.seed(trial_specific_seed)
            random.seed(trial_specific_seed)
            
            acs_model_trial = ACS(**current_run_params)
            initial_indices = get_sorted_indices(data_input_order, valid_times_worker, random_seed=trial_specific_seed)
            
            for epoch in range(1, num_epochs_worker + 1):
                current_indices = initial_indices
                if 'change' in data_input_order and epoch % 2 == 0:
                    current_indices = initial_indices[::-1]
                
                for data_idx in current_indices:
                    U_p = X_scaled_data[data_idx, :]
                    acs_model_trial.partial_fit(U_p, epoch=epoch, data_idx=int(data_idx))
                
                preds = acs_model_trial.predict(X_scaled_data)
                epoch_metrics = calculate_all_metrics_multi_label_permissive(preds, y_data_multi, label_encoder_worker)
                epoch_metrics['epoch'] = epoch
                
                lightweight_metrics = {
                    'epoch': epoch,
                    'macro_recall': epoch_metrics['macro_recall'],
                    'n_clusters': epoch_metrics['n_clusters'],
                }
                
                if epoch == num_epochs_worker:
                    lightweight_metrics['pred_map'] = epoch_metrics['pred_map']
                    lightweight_metrics['cm'] = epoch_metrics['cm']
                
                result['history'].append(lightweight_metrics)
                
                print(f"[Worker {trial_count}] Epoch {epoch}/{num_epochs_worker} - Cls: {epoch_metrics['n_clusters']}, "
                      f"Macro Recall: {epoch_metrics['macro_recall']:.4f}")
                      
    except Exception:
        result['error_traceback'] = traceback.format_exc()
        if 'log_file' in locals() and not log_file.closed:
            print(f"--- [Worker] トライアル {trial_count} で致命的なエラー発生 ---\n{result['error_traceback']}", file=log_file)
    
    finally:
        result['duration_seconds'] = (datetime.datetime.now() - trial_start_time).total_seconds()
        
        if acs_model_trial is not None and acs_model_trial.event_log:
            event_types_count = {}
            for event in acs_model_trial.event_log:
                event_type = event.get('event_type', 'UNKNOWN')
                event_types_count[event_type] = event_types_count.get(event_type, 0) + 1
            
            result['event_log_summary'] = {
                'total_events': len(acs_model_trial.event_log),
                'event_types_count': event_types_count,
                'final_cluster_count': acs_model_trial.M
            }
            
            if len(acs_model_trial.event_log) > 0:
                event_log_path = trial_log_dir_path / f"trial_{trial_count}_events.csv"
                pd.DataFrame(acs_model_trial.event_log).to_csv(event_log_path, index=False)
        
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        print(f"--- [Worker] トライアル {trial_count} 終了 | Time: {result['duration_seconds']:.2f}s | エラー: {'あり' if result['error_traceback'] else 'なし'} ---")
        
        return result
        
def sample_random_params(param_dist, rng=None):
    """ランダムサーチのために、定義されたパラメータ範囲から値をサンプリングする。"""
    if rng is None:
        rng = random.Random()
    
    params = {}
    params['activation_type'] = rng.choice(param_dist['activation_type'])
    
    for key, value in param_dist.items():
        if key == 'activation_type':
            continue
        if params['activation_type'] == 'circular' and key in ['initial_lambda_vector_val', 'initial_lambda_crossterm_val']:
            continue
        if params['activation_type'] == 'elliptical' and key == 'initial_lambda_scalar':
            continue
        if isinstance(value, list):
            params[key] = rng.choice(value)
        elif isinstance(value, tuple) and len(value) == 2:
            if all(isinstance(v, int) for v in value):
                params[key] = rng.randint(value[0], value[1])
            else:
                params[key] = round(rng.uniform(value[0], value[1]), 4)
    return params

def create_grid_search_tasks(param_dist):
    """
    param_distから全組み合わせを生成し、グリッドサーチ用のタスクリストを作成する。
    """
    from itertools import product

    # 各パラメータの選択肢リストを作成
    param_options = {}
    for key, value in param_dist.items():
        if isinstance(value, list):
            param_options[key] = value
        else:
            # タプルなどリストでない値が含まれている場合はグリッドサーチ不可
            raise ValueError(f"グリッドサーチのためには、全てのパラメータがリスト形式である必要があります。'{key}'がリストではありません。")

    # パラメータ名のリストと、値の組み合わせリストを作成
    param_names = list(param_options.keys())
    value_combinations = list(product(*param_options.values()))

    # 全組み合わせの総数を計算
    total_combinations = len(value_combinations)
    print(f"✅ 全通り探索（グリッドサーチ）を実行します。総組み合わせ数: {total_combinations}")

    # プール用のタスクリストを作成
    tasks = []
    for i, combo_values in enumerate(value_combinations):
        params = dict(zip(param_names, combo_values))
        # 各試行にユニークな乱数シードを割り当てる
        trial_seed = GLOBAL_SEED + i + 1
        tasks.append(((i + 1, params), trial_seed))
    
    return tasks, total_combinations

def plot_energy_contour_for_epoch(model, epoch, save_path,
                                  X_scaled_data_for_eval, X_pca_visual, y_true_multi,
                                  label_encoder, pca_visual_model):
    """指定されたモデルの状態でエネルギー等高線プロット等を生成し保存する。"""
    n_base_labels = len(label_encoder.classes_)
    
    if model.M > 0:
        preds = model.predict(X_scaled_data_for_eval)
        metrics = calculate_all_metrics_multi_label_permissive(preds, y_true_multi, label_encoder)
    else:
        metrics = {
            'macro_recall': 0.0, 'n_clusters': 0,
            'cm': np.zeros((n_base_labels, n_base_labels)), 'pred_map': {}
        }

    fig = plt.figure(figsize=(24, 8), constrained_layout=True)
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.8, 1, 1.2])
    ax_contour, ax_info, ax_cm = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    fig.suptitle(f'クラスタリング状態と評価 (Epoch: {epoch})', fontsize=20)
    
    distinguishable_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    ]
    palette_true_labels = distinguishable_colors[:n_base_labels]
    
    # --- 左パネル: エネルギー等高線とクラスタリング結果 ---
    x_min, x_max = X_pca_visual[:, 0].min() - 0.1, X_pca_visual[:, 0].max() + 0.1
    y_min, y_max = X_pca_visual[:, 1].min() - 0.1, X_pca_visual[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 80), np.linspace(y_min, y_max, 80))
    grid_points_2d = np.c_[xx.ravel(), yy.ravel()]
    try:
        grid_points_high_dim = pca_visual_model.inverse_transform(grid_points_2d)
        energy_values = np.array([model.calculate_energy_at_point(p) for p in grid_points_high_dim])
        Z_grid = energy_values.reshape(xx.shape)
        contour = ax_contour.contourf(xx, yy, Z_grid, levels=20, cmap='viridis', alpha=0.5)
        fig.colorbar(contour, ax=ax_contour, label='エネルギー (E)')
    except Exception as e:
        ax_contour.text(0.5, 0.5, f"エネルギー計算/描画エラー:\n{e}", ha='center', va='center')

    y_true_main_labels = [label_encoder.classes_[l[0]] for l in y_true_multi]
    sns.scatterplot(ax=ax_contour, x=X_pca_visual[:, 0], y=X_pca_visual[:, 1],
                    hue=y_true_main_labels, hue_order=label_encoder.classes_,
                    palette=palette_true_labels, s=50, alpha=0.7, edgecolor='w', linewidth=0.5, legend='full')
    ax_contour.legend(title="True Label (Main)", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if model.M > 0 and metrics['pred_map']:
        try:
            all_centers_2d = pca_visual_model.transform(model.get_cluster_centers())
            label_to_color = {label: color for label, color in zip(label_encoder.classes_, palette_true_labels)}
            
            for cluster_id, mapped_label_idx in metrics['pred_map'].items():
                if cluster_id < len(all_centers_2d):
                    center_2d = all_centers_2d[cluster_id]
                    dominant_label_name = label_encoder.classes_[mapped_label_idx]
                    text_color = label_to_color.get(dominant_label_name, 'black')
                    ax_contour.text(center_2d[0], center_2d[1], str(cluster_id),
                                    color=text_color, fontsize=12, fontweight='bold',
                                    ha='center', va='center',
                                    bbox=dict(boxstyle='circle,pad=0.3', fc='white', ec=text_color, alpha=0.8, lw=1.5))
        except Exception as e:
            print(f"Epoch {epoch}: クラスタ中心の描画中にエラー: {e}")

    ax_contour.set_title('クラスタリング結果 (PCA 2D)', fontsize=16)
    ax_contour.set_xlabel('主成分1'); ax_contour.set_ylabel('主成分2')

    # --- 中央パネル: 情報表示 ---
    ax_info.axis('off')
    ax_info.set_title('Learning Status Summary', fontsize=16)
    info_text = (
        f"Epoch: {epoch}\n"
        f"Clusters (M): {metrics['n_clusters']}\n\n"
        f"Macro-averaged Recall: {metrics['macro_recall']:.4f}\n"
    )
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes, fontsize=14,
                 verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.9),
                 family='monospace')

    # --- 右パネル: 混同行列 ---
    sns.heatmap(metrics['cm'], annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_,
                ax=ax_cm, cbar=False)
    ax_cm.set_title('Confusion Matrix (after mapping)', fontsize=16)
    ax_cm.set_xlabel("Predicted Label (mapped)"); ax_cm.set_ylabel("True Label")
    plt.setp(ax_cm.get_xticklabels(), rotation=45, ha="right")
    
    plt.savefig(save_path / f"epoch_{epoch:04d}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_cluster_composition(final_preds, y_true_multi, label_encoder, save_path, trial_id):
    """各クラスタのラベル構成率を示す積み上げ棒グラフを生成・保存する。"""
    base_labels = label_encoder.classes_
    n_base_labels = len(base_labels)
    unique_clusters = sorted([p for p in np.unique(final_preds) if p != -1])
    if not unique_clusters:
        print(f"有効なクラスタが予測されなかったため、構成プロットはスキップします。")
        return

    composition_df = pd.DataFrame(0, index=unique_clusters, columns=base_labels)
    for i in range(len(final_preds)):
        pred_cluster = final_preds[i]
        if pred_cluster == -1: continue
        for true_idx in y_true_multi[i]:
            if true_idx != -1:
                true_label_name = base_labels[true_idx]
                composition_df.loc[pred_cluster, true_label_name] += 1

    cluster_totals = composition_df.sum(axis=1)    
    proportion_df = composition_df.divide(cluster_totals.replace(0, 1), axis=0)
    
    distinguishable_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    ]
    palette = distinguishable_colors[:n_base_labels]
    fig, ax = plt.subplots(figsize=(max(12, len(unique_clusters) * 0.5), 8))
    proportion_df.plot(kind='bar', stacked=True, ax=ax, color=palette, width=0.8,
                       edgecolor='white', linewidth=0.5)
    
    ax.set_title(f'クラスタ構成の割合 (Trial {trial_id}, 基準: Macro Recall)', fontsize=16)
    ax.set_xlabel('クラスタ番号 (Cluster ID)', fontsize=12)
    ax.set_ylabel('ラベルの割合 (Proportion of Labels)', fontsize=12)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title='True Label', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    for i, (cluster_id, total) in enumerate(zip(unique_clusters, cluster_totals)):
        cluster_row = composition_df.loc[cluster_id]
        dominant_label = cluster_row.idxmax() if cluster_row.sum() > 0 else 'None'
        ax.text(i, 1.01, f'n={total}\n({dominant_label})', 
                ha='center', va='bottom', fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    save_filename = save_path / "cluster_composition.png"
    plt.savefig(save_filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ クラスタ構成グラフを保存しました: {save_filename.resolve()}")

def plot_cluster_counts(final_preds, y_true_multi, label_encoder, save_path, trial_id):
    """各クラスタのラベル構成「数」を示す積み上げ棒グラフを生成・保存する。"""
    base_labels = label_encoder.classes_
    n_base_labels = len(base_labels)
    unique_clusters = sorted([p for p in np.unique(final_preds) if p != -1])
    if not unique_clusters:
        print(f"有効なクラスタが予測されなかったため、構成数プロットはスキップします。")
        return

    composition_df = pd.DataFrame(0, index=unique_clusters, columns=base_labels)
    for i in range(len(final_preds)):
        pred_cluster = final_preds[i]
        if pred_cluster == -1: continue
        for true_idx in y_true_multi[i]:
            if true_idx != -1:
                true_label_name = base_labels[true_idx]
                composition_df.loc[pred_cluster, true_label_name] += 1
    
    distinguishable_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    ]
    palette = distinguishable_colors[:n_base_labels]
    fig, ax = plt.subplots(figsize=(max(12, len(unique_clusters) * 0.5), 8))
    composition_df.plot(kind='bar', stacked=True, ax=ax, color=palette, width=0.8,
                        edgecolor='white', linewidth=0.5)

    ax.set_title(f'クラスタ構成のサンプル数 (Trial {trial_id}, 基準: Macro Recall)', fontsize=16)
    ax.set_xlabel('クラスタ番号 (Cluster ID)', fontsize=12)
    ax.set_ylabel('サンプル数 (Number of Samples)', fontsize=12)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title='True Label', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    cluster_totals = composition_df.sum(axis=1)
    for i, (cluster_id, total) in enumerate(zip(unique_clusters, cluster_totals)):
        cluster_row = composition_df.loc[cluster_id]
        dominant_label = cluster_row.idxmax() if cluster_row.sum() > 0 else 'None'
        ax.text(i, total, f'n={total}\n({dominant_label})', 
                ha='center', va='bottom', fontsize=9, color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    save_filename = save_path / "cluster_counts.png"
    plt.savefig(save_filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ クラスタ構成数グラフを保存しました: {save_filename.resolve()}")

def main_process_logic():
    """データ読み込み、前処理、ランダムサーチ、結果評価、プロットまでを統括する。"""
    os.environ['PYTHONHASHSEED'] = str(GLOBAL_SEED)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    np.random.seed(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)
    sys.stdout = sys.stderr = Logger(log_file_path)
    print("=" * 80)
    print("ACSモデルによる気圧配置パターンの教師なしクラスタリング (評価指標: マクロ平均再現率)")
    print("=" * 80)
    print("\n--- 1. データ準備 ---")
    pca_dims_to_test = [20]
    preprocessed_data_cache_file = Path("./preprocessed_prmsl_data_all_labels.pkl")
    if preprocessed_data_cache_file.exists():
        print(f"✅ キャッシュファイルから前処理済みデータを読み込みます...")
        data_cache = pd.read_pickle(preprocessed_data_cache_file)
        precalculated_pca_data, y_true_multi, f1_season, f2_season, label_encoder, n_samples, valid_times, base_labels, all_labels_str = (
        data_cache['precalculated_pca_data'], data_cache['y_true_multi'], data_cache['f1_season'],
        data_cache['f2_season'], data_cache['label_encoder'], data_cache['n_samples'],
        data_cache['valid_times'], data_cache['base_labels'], data_cache['all_labels_str']
        )
    else:
        print(f"✅ キャッシュがないため、データを新規生成します...")
        ds = xr.open_dataset("./prmsl_era5_all_data_seasonal.nc")
        ds_period = ds.sel(valid_time=slice('1991-01-01', '2000-12-31'))
        ds_filtered = ds_period
        base_labels = ['1', '2A', '2B', '2C', '2D', '3A', '3B', '3C', '3D', '4A', '4B', '5', '6A', '6B', '6C']
        label_encoder = LabelEncoder().fit(base_labels)
        all_labels_str = ds_filtered['label'].values
        y_true_multi = []
        for label_str in all_labels_str:
            parts = set(re.split(r'[+-]', label_str))
            valid_indices = [label_encoder.transform([p])[0] for p in parts if p in label_encoder.classes_]
            if not valid_indices:
                valid_indices = [-1]
            y_true_multi.append(tuple(sorted(valid_indices)))

        n_samples = ds_filtered.sizes['valid_time']
        valid_times = ds_filtered['valid_time'].values
        msl_flat = ds_filtered['msl'].values.reshape(n_samples, -1)
        f1_season, f2_season = ds_filtered['f1_season'].values.reshape(-1, 1), ds_filtered['f2_season'].values.reshape(-1, 1)
        msl_flat_scaled = MinMaxScaler().fit_transform(msl_flat)
        precalculated_pca_data = {n: PCA(n_components=n, random_state=GLOBAL_SEED).fit_transform(msl_flat_scaled) for n in pca_dims_to_test}
        data_to_cache = {
            'precalculated_pca_data': precalculated_pca_data, 'y_true_multi': y_true_multi, 'f1_season': f1_season,
            'f2_season': f2_season, 'label_encoder': label_encoder, 'n_samples': n_samples,
            'valid_times': valid_times, 'base_labels': base_labels,'all_labels_str': all_labels_str
        }
        with open(preprocessed_data_cache_file, 'wb') as f: pd.to_pickle(data_to_cache, f)

    n_true_clusters = len(base_labels)
    print(f"✅ データ準備完了。対象サンプル数: {n_samples}, 基本ラベル数: {n_true_clusters}")
    print("\n--- 2. ランダムサーチ設定 ---")
    param_dist = {
        'data_input_order': ['normal_sort'], # ['normal_sort', 'month_sort', 'change_normal_sort', 'change_month_sort']
        'pca_n_components': pca_dims_to_test, 
        'include_time_features': [True], # [True, False]
        'gamma': [1.0, 2.0],   
        'beta': [0.01, 0.1],  
        'learning_rate_W': [0.01, 0.1],
        'learning_rate_lambda': [0.01, 0.1],
        'learning_rate_Z': [0.01, 0.05, 0.1],
        'initial_lambda_scalar': [0.01, 0.1, 1.0], 
        'initial_lambda_vector_val': [0.1], 
        'initial_lambda_crossterm_val': [-0.1, 0.0, 0.1],
        'initial_Z_val': [0.25, 0.5, 0.75], 
        'initial_Z_new_cluster': [0.5, 1.0], 
        'theta_new': [0.01, 0.05, 0.1],  
        'Z_death_threshold': [0.01, 0.05, 0.1], 
        'death_patience_steps': [n_samples // 20], # normal_sort : n_samples // 20, month_sort : n_samples // 2
        'num_epochs': [3000], 
        'activation_type': ['elliptical'] # ['circular', 'elliptical']
    }
    N_TRIALS = 100000
    fixed_params_for_acs = {'max_clusters': 45, 'initial_clusters': 1, 'lambda_min_val': 1e-7, 'bounds_W': (0, 1)}
    print(f"ランダムサーチ最大試行回数: {N_TRIALS}")
    print("\n--- 3. 探索タスクの準備と実行 ---")
    # 全てのパラメータがリスト形式かチェックし、探索方法を決定
    is_grid_search = all(isinstance(v, list) for v in param_dist.values())
    tasks_for_pool = []
    if is_grid_search:
        # --- 全通り探索（グリッドサーチ）の場合 ---
        try:
            tasks_for_pool, N_TRIALS = create_grid_search_tasks(param_dist)
            print(f"試行回数（N_TRIALS）を自動設定しました: {N_TRIALS}")
        except ValueError as e:
            print(f"エラー: {e}")
            sys.exit(1)
    else:
        # --- ランダムサーチの場合 ---
        print(f"⚠️ パラメータに範囲指定（タプル形式）が含まれるため、ランダムサーチを実行します。")
        print(f"ランダムサーチ試行回数: {N_TRIALS}")
        for i in range(N_TRIALS):
            trial_rng = random.Random(GLOBAL_SEED + i + 1)
            params = sample_random_params(param_dist, rng=trial_rng)
            tasks_for_pool.append(((i + 1, params), GLOBAL_SEED + i + 1))

    num_processes_to_use = max(1, int(os.cpu_count() * 0.95)) if os.cpu_count() else 2
    worker_func_with_fixed_args = partial(run_acs_trial, fixed_params_dict=fixed_params_for_acs, pca_data_dict=precalculated_pca_data, y_data_multi=y_true_multi, f1_season_data=f1_season, f2_season_data=f2_season, trial_log_dir_path=trial_logs_dir, label_encoder_worker=label_encoder, valid_times_worker=valid_times)
    start_search_time = datetime.datetime.now()
    all_trial_results = []
    if not tasks_for_pool:
        print("実行するタスクがありません。処理を終了します。")
        sys.exit(0)

    print(f"使用するプロセス数: {num_processes_to_use}")
    with multiprocessing.Pool(processes=num_processes_to_use) as pool:
        desc_text = "Grid Search Progress" if is_grid_search else "Random Search Progress"
        for result in tqdm(pool.imap_unordered(worker_func_with_fixed_args, tasks_for_pool), total=len(tasks_for_pool), desc=desc_text):
            all_trial_results.append(result)

    print(f"\nランダムサーチ完了。総所要時間: {datetime.datetime.now() - start_search_time}")
    print("\n--- 4. 結果集計 (Macro Recallでベストを選出) ---")
    if not all_trial_results: sys.exit("エラー: サーチから結果が返されませんでした。")
    
    processed_results = []
    for res in all_trial_results:
        if res['error_traceback'] or not res['history']:
            continue
        
        history_df = pd.DataFrame(res['history'])
        best_epoch_data = history_df.loc[history_df['macro_recall'].idxmax()]
        
        last_epoch_data = res['history'][-1] if res['history'] else {}
        
        processed_results.append({
            'trial_id': res['trial_count_from_worker'],
            'params': res['params_combo'],
            'random_state': res['acs_random_state_used'],
            'best_epoch_data': best_epoch_data.to_dict(),
            'event_log_summary': res.get('event_log_summary', {}),
            'final_epoch_full_data': last_epoch_data
        })

    if not processed_results: sys.exit("エラー: 有効な結果が得られませんでした。")
    
    summary_df = pd.DataFrame([
        {
            'trial_id': r['trial_id'], **r['params'],
            'best_macro_recall': r['best_epoch_data']['macro_recall'],
            'best_epoch': r['best_epoch_data']['epoch'],
            'clusters_at_best': r['best_epoch_data']['n_clusters']
        } for r in processed_results
    ])
    summary_df.to_csv(output_dir / f"random_search_summary_{timestamp}.csv", index=False)
    print(f"全試行のサマリーをCSVファイルに保存しました: {output_dir.resolve()}")
    # Macro Recallで上位3位までを取得
    top_results = sorted(processed_results, key=lambda x: x['best_epoch_data']['macro_recall'], reverse=True)[:3]
    for rank, best_result_info in enumerate(top_results, 1):
        best_trial_id = best_result_info['trial_id']
        best_epoch_data = best_result_info['best_epoch_data']
        best_epoch = int(best_epoch_data['epoch'])
        best_score = best_epoch_data['macro_recall']
        best_params = best_result_info['params']
        best_random_state = best_result_info['random_state']
        run_output_dir = output_dir / f"{rank}_model_by_recall"
        os.makedirs(run_output_dir, exist_ok=True)
        best_event_log_summary = best_result_info.get('event_log_summary', {})
        if best_event_log_summary:
            trial_event_log_path = trial_logs_dir / f"trial_{best_trial_id}_events.csv"
            if trial_event_log_path.exists():
                import shutil
                dest_path = run_output_dir / "trial_cluster_event_log.csv"
                shutil.copy2(trial_event_log_path, dest_path)
                print(f"✅ 試行時のイベントログをコピーしました: {dest_path.resolve()}")
        
        print("-" * 50)
        print(f"\n--- 第{rank}位モデル (基準: Macro Recall) ---")
        print(f"   Trial ID: {best_trial_id}, Best Epoch: {best_epoch}")
        print(f"🏆 Best Macro Recall: {best_score:.4f}")
        print(f"   - Clusters at Best Epoch: {int(best_epoch_data['n_clusters'])}")
        print("   - パラメータ:")
        for k, v in best_params.items(): 
            print(f"     {k}: {v}")
        
        print(f"\n   再学習とプロット生成を開始します... (出力先: {run_output_dir.resolve()})")
        X_pca = precalculated_pca_data[best_params['pca_n_components']]
        X_features = np.hstack([X_pca, f1_season, f2_season]) if best_params['include_time_features'] else X_pca
        X_scaled_data = MinMaxScaler().fit_transform(X_features).astype(np.float64)
        params_for_refit = {
            **fixed_params_for_acs, 
            **{k:v for k,v in best_params.items() if k not in ['pca_n_components', 'include_time_features', 'num_epochs', 'data_input_order']}, 
            'n_features': X_scaled_data.shape[1], 
            'random_state': best_random_state
        }
        best_model_instance = ACS(**params_for_refit)
        pca_visual = PCA(n_components=2, random_state=GLOBAL_SEED)
        X_pca_visual = pca_visual.fit_transform(X_scaled_data)
        np.random.seed(best_random_state)
        random.seed(best_random_state)
        data_input_order = best_params['data_input_order']
        initial_indices = get_sorted_indices(data_input_order, valid_times, random_seed=best_random_state)
        refit_history = []
        for epoch in tqdm(range(1, best_epoch + 1), desc=f"Refitting for Recall (rank {rank})"):
            current_indices = initial_indices
            if 'change' in data_input_order and epoch % 2 == 0:
                current_indices = initial_indices[::-1]
            for data_idx in current_indices:
                U_p = X_scaled_data[data_idx, :]
                best_model_instance.partial_fit(U_p, epoch=epoch, data_idx=int(data_idx))
            
            preds = best_model_instance.predict(X_scaled_data)
            epoch_metrics = calculate_all_metrics_multi_label_permissive(preds, y_true_multi, label_encoder)
            epoch_metrics['epoch'] = epoch
            refit_history.append(epoch_metrics)
            
            plot_energy_contour_for_epoch(
                model=best_model_instance, epoch=epoch, save_path=run_output_dir,
                X_scaled_data_for_eval=X_scaled_data, X_pca_visual=X_pca_visual,
                y_true_multi=y_true_multi, label_encoder=label_encoder,
                pca_visual_model=pca_visual
            )
        
        final_preds = best_model_instance.predict(X_scaled_data)
        if best_model_instance.event_log:
            pd.DataFrame(best_model_instance.event_log).to_csv(run_output_dir / "refit_cluster_event_log.csv", index=False)
        
        # --- 再現性検証 ---
        print("\n--- 評価指標の一致検証 ---")
        trial_best_metrics = best_epoch_data
        refit_best_metrics = refit_history[best_epoch - 1]
        all_metrics_match = True
        for metric in ['macro_recall', 'n_clusters']:
            trial_value = trial_best_metrics.get(metric)
            refit_value = refit_best_metrics.get(metric)
            if abs(trial_value - refit_value) < 1e-6:
                print(f"   ✅ {metric}: 一致 (値: {trial_value:.6f})")
            else:
                print(f"   ❌ {metric}: 不一致 (Trial: {trial_value:.6f}, Refit: {refit_value:.6f})")
                all_metrics_match = False
        
        if all_metrics_match: print("\n   ✅ [OK] 評価指標が一致しました。")
        else: print("\n   ⚠️ [NG] 評価指標が一致しませんでした。")
        
        # --- プロットと結果保存 ---
        plot_cluster_composition(final_preds, y_true_multi, label_encoder, run_output_dir, best_trial_id)
        plot_cluster_counts(final_preds, y_true_multi, label_encoder, run_output_dir, best_trial_id)
        pd.DataFrame({'valid_time': valid_times, 'true_label_str': all_labels_str, 'predicted_cluster': final_preds}) \
          .to_csv(run_output_dir / "classification_details.csv", index=False)
        print(f"✅ 各データの日付と分類結果をCSVに保存しました。")
        history_df = pd.DataFrame(refit_history)
        fig, ax1 = plt.subplots(figsize=(12, 6))
        fig.suptitle(f'第{rank}位モデルの学習推移 (Trial {best_trial_id}, 基準: Macro Recall)', fontsize=16)
        color = 'tab:green'
        ax1.set_xlabel('エポック数')
        ax1.set_ylabel('Macro Recall', color=color)
        ax1.plot(history_df['epoch'], history_df['macro_recall'], 's-', c=color, label='Macro Recall')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.axvline(x=best_epoch, color='red', linestyle='--', label=f'Best Epoch ({best_epoch})')
        ax1.grid(True, axis='y', linestyle=':')
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Clusters', color=color)
        ax2.plot(history_df['epoch'], history_df['n_clusters'], 'o--', c=color, alpha=0.6, label='Clusters')
        ax2.tick_params(axis='y', labelcolor=color)
        fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(run_output_dir / "learning_history.png", dpi=300)
        plt.close()
        print(f"✅ 第{rank}位モデルの学習推移グラフを保存しました。")

    print("\n--- 全処理完了 ---")

if __name__ == '__main__':
    try:
        main_process_logic()
    except Exception as e:
        print("\n" + "="*30 + " FATAL ERROR " + "="*30)
        traceback.print_exc(file=sys.stdout)
        print("="*73)
    finally:
        if hasattr(sys.stdout, 'close') and callable(sys.stdout.close):
            sys.stdout.close()