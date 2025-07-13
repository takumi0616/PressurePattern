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
from sklearn.metrics import adjusted_rand_score, confusion_matrix, balanced_accuracy_score
from scipy.optimize import linear_sum_assignment

try:
    from acs import ACS
    print("ACS class (acs.py) imported successfully.")
except ImportError as e:
    print(f"Error: acs.py からACSクラスをインポートできませんでした: {e}")
    print("このスクリプトと同じディレクトリに acs.py ファイルを配置してください。")
    sys.exit(1)

GLOBAL_SEED = 17
os.environ['PYTHONHASHSEED'] = str(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir_base = Path("./result_prmsl_acs_random_search_v3")
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
        times_dt = valid_times.astype('datetime64[D]') # NumPyのdatetime64型の操作を直接使用（高速化）
        times_M = valid_times.astype('datetime64[M]') # 月と年を効率的に抽出
        times_Y = valid_times.astype('datetime64[Y]')
        months = ((times_M - times_Y) / np.timedelta64(1, 'M')).astype(int) # 月を0-11の整数として取得
        years = times_Y.astype(int) + 1970  # 年を整数として取得 Unix epoch からの年数
        sort_keys = np.lexsort((valid_times, years, months)) # 複合キーでのソート（月→年→日時の順）lexsortは右から左の順序でソートするので、逆順で指定
        return indices[sort_keys]
    else:
        if random_seed is not None: # ランダムシャッフルの場合は、シードを明示的に設定
            rng = np.random.RandomState(random_seed)
            rng.shuffle(indices)
        else:
            raise ValueError(f"Unknown sort method: {sort_method}") # デフォルトの動作を避け、エラーを発生させる
        return indices

def calculate_composite_score(cluster_report, n_true_clusters, ideal_cluster_range=(1.0, 2.0)):
    """
    クラスタ純度とクラスタ数から複合評価スコアを計算する。
    """
    if cluster_report.empty or 'n_samples' not in cluster_report.columns or cluster_report['n_samples'].sum() == 0:
        return 0.0, 0.0, 0.0

    total_samples = cluster_report['n_samples'].sum()
    weighted_purity = np.sum(cluster_report['purity'] * cluster_report['n_samples']) / total_samples
    n_clusters = len(cluster_report)
    min_ideal_clusters = n_true_clusters * ideal_cluster_range[0]
    max_ideal_clusters = n_true_clusters * ideal_cluster_range[1]
    center_ideal_clusters = (min_ideal_clusters + max_ideal_clusters) / 2.0
    if min_ideal_clusters <= n_clusters <= max_ideal_clusters:
        penalty = 1.0
    else:
        distance_from_center = abs(n_clusters - center_ideal_clusters)
        scale = n_true_clusters
        penalty = np.exp(-distance_from_center / scale)

    final_score = weighted_purity * penalty
    return final_score, weighted_purity, penalty

def calculate_all_metrics_multi_label(preds, y_true_multi, label_encoder):
    """
    最適化版：NumPy配列操作を活用した高速化
    """
    n_samples = len(preds)
    base_labels = label_encoder.classes_
    n_base_labels = len(base_labels)
    unique_pred_clusters = np.unique(preds[preds != -1])# NumPy配列として処理
    n_pred_clusters = len(unique_pred_clusters)
    if n_pred_clusters == 0:
        return {
            'composite_score': 0.0, 'weighted_purity': 0.0, 'accuracy': 0.0,
            'bacc': 0.0, 'ari': 0.0, 'n_clusters': 0,
            'pred_map': {}, 'cm': np.zeros((n_base_labels, n_base_labels), dtype=int),
            'cluster_report': pd.DataFrame()
        }
    
    contingency_np = np.zeros((n_pred_clusters, n_base_labels), dtype=int)# NumPy配列でコンティンジェンシー行列を作成（高速化）
    cluster_to_idx = {cluster: idx for idx, cluster in enumerate(unique_pred_clusters)}
    valid_mask = preds != -1 # ベクトル化された処理
    valid_preds = preds[valid_mask]
    valid_true_multi = [y_true_multi[i] for i in range(n_samples) if valid_mask[i]]
    for i, pred_cluster in enumerate(valid_preds):
        cluster_idx = cluster_to_idx[pred_cluster]
        for true_idx in valid_true_multi[i]:
            if true_idx != -1:
                contingency_np[cluster_idx, true_idx] += 1
    
    row_ind, col_ind = linear_sum_assignment(-contingency_np) # ハンガリアン法による予測クラスタと真ラベルのマッピング
    pred_map = {unique_pred_clusters[pred_i]: true_i for pred_i, true_i in zip(row_ind, col_ind)}
    correct_hits_for_accuracy = 0 # 以降の処理もNumPy配列で効率化
    y_true_for_bacc_cm = []
    y_pred_for_bacc_cm = []
    cluster_report_list = []
    for idx, pred_cluster_id in enumerate(unique_pred_clusters): # クラスタごとの統計を一括計算
        cluster_mask = preds == pred_cluster_id
        n_samples_in_cluster = np.sum(cluster_mask)
        
        if n_samples_in_cluster == 0:
            continue
            
        correct_in_cluster = 0
        dominant_label_name = "Unmapped"
        if pred_cluster_id in pred_map:
            mapped_label_idx = pred_map[pred_cluster_id]
            dominant_label_name = base_labels[mapped_label_idx]
            cluster_indices = np.where(cluster_mask)[0] # ベクトル化された正解判定
            for i in cluster_indices:
                if mapped_label_idx in y_true_multi[i]:
                    correct_in_cluster += 1
        
        purity = correct_in_cluster / n_samples_in_cluster
        cluster_report_list.append({
            'cluster_id': pred_cluster_id,
            'n_samples': n_samples_in_cluster,
            'purity': purity,
            'dominant_label': dominant_label_name
        })
    
    for i in range(n_samples): # BAcc と Confusion Matrix のためのデータ作成（効率化）
        pred_cluster = preds[i]
        if pred_cluster == -1:
            continue
            
        true_label_indices = y_true_multi[i]
        if pred_cluster in pred_map:
            mapped_label_idx = pred_map[pred_cluster]
            y_pred_for_bacc_cm.append(mapped_label_idx)
            
            if mapped_label_idx in true_label_indices:
                correct_hits_for_accuracy += 1
                y_true_for_bacc_cm.append(mapped_label_idx)
            else:
                y_true_for_bacc_cm.append(true_label_indices[0])
    
    accuracy = correct_hits_for_accuracy / n_samples if n_samples > 0 else 0.0 # 各評価指標の最終計算
    bacc = balanced_accuracy_score(y_true_for_bacc_cm, y_pred_for_bacc_cm) if y_true_for_bacc_cm else 0.0
    cm = confusion_matrix(y_true_for_bacc_cm, y_pred_for_bacc_cm, labels=np.arange(n_base_labels)) if y_true_for_bacc_cm else np.zeros((n_base_labels, n_base_labels))
    y_true_representative = [t[0] for t in y_true_multi]
    ari = adjusted_rand_score(y_true_representative, preds)
    cluster_report_df = pd.DataFrame(cluster_report_list)
    composite_score, weighted_purity, _ = calculate_composite_score(cluster_report_df, n_base_labels)
    return {
        'composite_score': composite_score,
        'weighted_purity': weighted_purity,
        'accuracy': accuracy,
        'bacc': bacc,
        'ari': ari,
        'n_clusters': n_pred_clusters,
        'pred_map': pred_map,
        'cm': cm,
        'cluster_report': cluster_report_df
    }

def run_acs_trial(param_values_tuple_with_trial_info,
                  fixed_params_dict,
                  pca_data_dict,
                  y_data_multi,
                  sin_time_data,
                  cos_time_data,
                  n_true_cls_worker,
                  trial_log_dir_path,
                  label_encoder_worker,
                  valid_times_worker):
    """
    ランダムサーチの1試行を独立して実行する。
    指定されたデータ投入順序で学習し、エポックごとに全指標を計算し、全履歴を返す。
    """
    (trial_count, params_combo), trial_specific_seed = param_values_tuple_with_trial_info
    worker_log_path = trial_log_dir_path / f"trial_{trial_count}.log"
    original_stdout, original_stderr = sys.stdout, sys.stderr
    acs_model_trial = None # 変数を関数のトップレベルで初期化
    result = {
        'params_combo': {},
        'history': [],
        'event_log': [],
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
            activation_type_worker = params_combo.get('activation_type')
            result['params_combo'] = params_combo.copy() # resultには元の完全なパラメータを保存
            result['error_traceback'] = None
            X_pca = pca_data_dict[pca_n_components] # 特徴量構築
            X_features = np.hstack([X_pca, sin_time_data, cos_time_data]) if include_time_features else X_pca
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
            np.random.seed(trial_specific_seed) # 乱数状態を明示的に設定（ワーカープロセス内）
            random.seed(trial_specific_seed)
            acs_model_trial = ACS(**current_run_params) # モデルの初期化と学習
            initial_indices = get_sorted_indices(data_input_order, valid_times_worker, random_seed=trial_specific_seed)
            for epoch in range(1, num_epochs_worker + 1):
                current_indices = initial_indices
                if 'change' in data_input_order and epoch % 2 == 0:
                    current_indices = initial_indices[::-1]

                for data_idx in current_indices:
                    U_p = X_scaled_data[data_idx, :]
                    acs_model_trial.partial_fit(U_p, epoch=epoch, data_idx=int(data_idx))

                preds = acs_model_trial.predict(X_scaled_data)
                epoch_metrics = calculate_all_metrics_multi_label(preds, y_data_multi, label_encoder_worker)
                epoch_metrics['epoch'] = epoch
                result['history'].append(epoch_metrics)
                print(f"[Worker {trial_count}] Epoch {epoch}/{num_epochs_worker} - Cls: {epoch_metrics['n_clusters']}, "
                      f"Score: {epoch_metrics['composite_score']:.4f}, BAcc: {epoch_metrics['bacc']:.4f}, Acc: {epoch_metrics['accuracy']:.4f}")

    except Exception: # どの段階でエラーが起きてもトレースバックを記録
        result['error_traceback'] = traceback.format_exc()
        if 'log_file' in locals() and not log_file.closed: # ログファイルがまだ開いている場合は書き込む
             print(f"--- [Worker] トライアル {trial_count} で致命的なエラー発生 ---\n{result['error_traceback']}", file=log_file)

    finally: # 最後に必ず実行される後処理
        result['duration_seconds'] = (datetime.datetime.now() - trial_start_time).total_seconds()
        if acs_model_trial is not None:
            result['event_log'] = acs_model_trial.event_log

        sys.stdout = original_stdout
        sys.stderr = original_stderr
        print(f"--- [Worker] トライアル {trial_count} 終了 | Time: {result['duration_seconds']:.2f}s | エラー: {'あり' if result['error_traceback'] else 'なし'} ---")

        return result
        
def sample_random_params(param_dist, rng=None):
    """ランダムサーチのために、定義されたパラメータ範囲から値をサンプリングする。"""
    if rng is None:
        rng = random.Random()
    
    params = {}
    params['activation_type'] = rng.choice(param_dist['activation_type']) # まず、activation_typeをランダムに選択
    
    for key, value in param_dist.items():
        if key == 'activation_type': # activation_type は既に処理済みなのでスキップ
            continue
        if params['activation_type'] == 'circular' and key in ['initial_lambda_vector_val', 'initial_lambda_crossterm_val']: # activation_typeに応じて不要なパラメータはサンプリングしない
            continue
        if params['activation_type'] == 'elliptical' and key == 'initial_lambda_scalar':
            continue
        if isinstance(value, list): # 既存のサンプリングロジック
            params[key] = rng.choice(value)
        elif isinstance(value, tuple) and len(value) == 2:
            if all(isinstance(v, int) for v in value):
                params[key] = rng.randint(value[0], value[1])
            else:
                params[key] = round(rng.uniform(value[0], value[1]), 4)
    return params

def plot_energy_contour_for_epoch(model, epoch, save_path,
                                  X_scaled_data_for_eval, X_pca_visual, y_true_multi,
                                  label_encoder, pca_visual_model):
    """指定されたモデルの状態でエネルギー等高線プロット等を生成し保存する (複合ラベル対応・改良版)。"""
    n_base_labels = len(label_encoder.classes_)
    current_clusters = model.M

    if current_clusters > 0: # --- 評価指標の計算 (複合ラベル対応) ---
        preds = model.predict(X_scaled_data_for_eval)
        metrics = calculate_all_metrics_multi_label(preds, y_true_multi, label_encoder) # 変更点: 'cluster_report' を含む metrics を受け取る
    else: # クラスタが存在しない場合
        metrics = {
            'composite_score': 0.0, 'weighted_purity': 0.0, 'accuracy': 0.0,
            'bacc': 0.0, 'ari': 0.0, 'n_clusters': 0, 'cm': np.zeros((n_base_labels, n_base_labels)),
            'cluster_report': pd.DataFrame()
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
    if current_clusters > 0 and not metrics['cluster_report'].empty:
        try:
            all_centers_2d = pca_visual_model.transform(model.get_cluster_centers())
            cluster_report_df = metrics['cluster_report']
            label_to_color = {label: color for label, color in zip(label_encoder.classes_, palette_true_labels)}

            for _, row in cluster_report_df.iterrows():
                cluster_id = int(row['cluster_id'])
                dominant_label_name = row['dominant_label']
                
                if cluster_id < len(all_centers_2d):
                    center_2d = all_centers_2d[cluster_id]
                    text_color = label_to_color.get(dominant_label_name, 'black')

                    ax_contour.text(center_2d[0], center_2d[1], str(cluster_id),
                                    color=text_color,
                                    fontsize=12,
                                    fontweight='bold',
                                    ha='center',
                                    va='center',
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
        f"Composite Score: {metrics['composite_score']:.4f}\n"
        f"Weighted Purity: {metrics['weighted_purity']:.4f}\n\n"
        f"Accuracy: {metrics['accuracy']:.4f}\n"
        f"Balanced Accuracy: {metrics['bacc']:.4f}\n"
        f"Adjusted Rand Index: {metrics['ari']:.4f}"
    )
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes, fontsize=12,
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

def plot_cluster_composition(final_preds, y_true_multi, label_encoder, save_path, metric_name, trial_id):
    """
    各クラスタのラベル構成率を示す積み上げ棒グラフを生成・保存する。
    """
    # --- 0. 準備 ---
    base_labels = label_encoder.classes_
    n_base_labels = len(base_labels)
    unique_clusters = sorted([p for p in np.unique(final_preds) if p != -1]) # -1（未分類）を除外し、実際にデータが割り当てられたクラスタのみを対象とする
    if not unique_clusters:
        print(f"[{metric_name.upper()}] 有効なクラスタが予測されなかったため、構成プロットはスキップします。")
        return

    composition_df = pd.DataFrame(0, index=unique_clusters, columns=base_labels) # --- 1. データ集計用のDataFrameを作成 ---
    for i in range(len(final_preds)): # --- 2. 各サンプルの所属クラスタと真ラベルをカウント ---
        pred_cluster = final_preds[i]
        if pred_cluster == -1:
            continue

        true_indices = y_true_multi[i]
        for true_idx in true_indices:
            if true_idx != -1: # ダミーラベル(-1)は無視
                true_label_name = base_labels[true_idx]
                composition_df.loc[pred_cluster, true_label_name] += 1

    cluster_totals = composition_df.sum(axis=1) # --- 3. カウントを割合に変換 ---    
    proportion_df = composition_df.divide(cluster_totals.replace(0, 1), axis=0) # ゼロ除算を避けるため、合計が0のクラスタは1で割る（結果は0のまま）
    # --- 4. プロット ---
    distinguishable_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    ] # 他のプロットと色を合わせる
    palette = distinguishable_colors[:n_base_labels]
    fig, ax = plt.subplots(figsize=(max(12, len(unique_clusters) * 0.5), 8))
    proportion_df.plot(kind='bar', stacked=True, ax=ax, color=palette, width=0.8,
                       edgecolor='white', linewidth=0.5)
    # --- 5. 整形 ---
    ax.set_title(f'クラスタ構成の割合 (Trial {trial_id}, 基準: {metric_name.upper()})', fontsize=16)
    ax.set_xlabel('クラスタ番号 (Cluster ID)', fontsize=12)
    ax.set_ylabel('ラベルの割合 (Proportion of Labels)', fontsize=12)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title='True Label', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    for i, (cluster_id, total) in enumerate(zip(unique_clusters, cluster_totals)): # 各クラスタの総サンプル数と最も多いTrue Labelをバーの上に表示
        cluster_row = composition_df.loc[cluster_id] # 最も多いTrue Labelを見つける
        dominant_label = cluster_row.idxmax() if cluster_row.sum() > 0 else 'None'
        ax.text(i, 1.01, f'n={total}\n({dominant_label})', 
                ha='center', va='bottom', fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)) # n=数とdominant labelを表示

    plt.tight_layout(rect=[0, 0, 0.88, 1]) # 凡例が収まるように調整
    save_filename = save_path / f"cluster_composition_{metric_name}.png" # --- 6. 保存 ---
    plt.savefig(save_filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ クラスタ構成グラフを保存しました: {save_filename.resolve()}")

def plot_cluster_counts(final_preds, y_true_multi, label_encoder, save_path, metric_name, trial_id):
    """
    各クラスタのラベル構成「数」を示す積み上げ棒グラフを生成・保存する。
    """
    # --- 0. 準備 ---
    base_labels = label_encoder.classes_
    n_base_labels = len(base_labels)
    unique_clusters = sorted([p for p in np.unique(final_preds) if p != -1])
    if not unique_clusters:
        print(f"[{metric_name.upper()}] 有効なクラスタが予測されなかったため、構成数プロットはスキップします。")
        return

    # --- 1. データ集計用のDataFrameを作成 (割合グラフと共通) ---
    composition_df = pd.DataFrame(0, index=unique_clusters, columns=base_labels)
    for i in range(len(final_preds)):
        pred_cluster = final_preds[i]
        if pred_cluster == -1:
            continue
        
        true_indices = y_true_multi[i]
        for true_idx in true_indices:
            if true_idx != -1:
                true_label_name = base_labels[true_idx]
                composition_df.loc[pred_cluster, true_label_name] += 1
    
    # --- 2. プロット (💡割合に変換せず、生のカウント数を使用) ---
    distinguishable_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    ]
    palette = distinguishable_colors[:n_base_labels]
    fig, ax = plt.subplots(figsize=(max(12, len(unique_clusters) * 0.5), 8))
    composition_df.plot(kind='bar', stacked=True, ax=ax, color=palette, width=0.8,
                        edgecolor='white', linewidth=0.5) # composition_df (生のカウント数) を直接プロット
    # --- 3. 整形 ---
    ax.set_title(f'クラスタ構成のサンプル数 (Trial {trial_id}, 基準: {metric_name.upper()})', fontsize=16)
    ax.set_xlabel('クラスタ番号 (Cluster ID)', fontsize=12)
    ax.set_ylabel('サンプル数 (Number of Samples)', fontsize=12) # Y軸ラベルを変更
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title='True Label', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    cluster_totals = composition_df.sum(axis=1) # 各クラスタの総サンプル数と最も多いTrue Labelをバーの上に表示
    for i, (cluster_id, total) in enumerate(zip(unique_clusters, cluster_totals)):
        cluster_row = composition_df.loc[cluster_id] # 最も多いTrue Labelを見つける
        dominant_label = cluster_row.idxmax() if cluster_row.sum() > 0 else 'None'
        ax.text(i, total, f'n={total}\n({dominant_label})', 
                ha='center', va='bottom', fontsize=9, color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)) # n=数とdominant labelを表示

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    save_filename = save_path / f"cluster_counts_{metric_name}.png" # --- 4. 保存 ---
    plt.savefig(save_filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ クラスタ構成数グラフを保存しました: {save_filename.resolve()}")

def main_process_logic():
    """データ読み込み、前処理、ランダムサーチ、3基準での結果評価、プロットまでを統括する。"""
    os.environ['PYTHONHASHSEED'] = str(GLOBAL_SEED)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    np.random.seed(GLOBAL_SEED)
    random.seed(GLOBAL_SEED)
    sys.stdout = sys.stderr = Logger(log_file_path)
    print("=" * 80)
    print("ACSモデルによる気圧配置パターンの教師なしクラスタリング (複合ラベル・複数指標対応版)")
    print("=" * 80)
    print("\n--- 1. データ準備 ---")
    pca_dims_to_test = [15, 20, 25]
    preprocessed_data_cache_file = Path("./preprocessed_prmsl_data_all_labels.pkl")
    if preprocessed_data_cache_file.exists():
        print(f"✅ キャッシュファイルから前処理済みデータを読み込みます...")
        data_cache = pd.read_pickle(preprocessed_data_cache_file)
        precalculated_pca_data, y_true_multi, sin_time, cos_time, label_encoder, n_samples, valid_times, base_labels, all_labels_str = (
        data_cache['precalculated_pca_data'], data_cache['y_true_multi'], data_cache['sin_time'],
        data_cache['cos_time'], data_cache['label_encoder'], data_cache['n_samples'],
        data_cache['valid_times'], data_cache['base_labels'], data_cache['all_labels_str']
)
    else:
        print(f"✅ キャッシュがないため、データを新規生成します...")
        ds = xr.open_dataset("./prmsl_era5_all_data.nc")
        ds_period = ds.sel(valid_time=slice('1991-01-01', '2000-12-31'))
        ds_filtered = ds_period
        base_labels = ['1', '2A', '2B', '2C', '2D', '3A', '3B', '3C', '3D', '4A', '4B', '5', '6A', '6B', '6C']
        label_encoder = LabelEncoder().fit(base_labels)
        all_labels_str = ds_filtered['label'].values
        y_true_multi = []
        for label_str in all_labels_str:
            parts = set(re.split(r'[+-]', label_str))# '+' または '-' でラベルを分割し、ユニークなパーツを取得。各パーツを基本ラベルのインデックスに変換（存在しないものは無視）
            valid_indices = [label_encoder.transform([p])[0] for p in parts if p in label_encoder.classes_]
            if not valid_indices:
                valid_indices = [-1] # 万が一、どの基本ラベルにも一致しない場合は、ダミーとして-1を追加
            y_true_multi.append(tuple(sorted(valid_indices)))

        n_samples = ds_filtered.sizes['valid_time']
        valid_times = ds_filtered['valid_time'].values
        msl_flat = ds_filtered['msl'].values.reshape(n_samples, -1)
        sin_time, cos_time = ds_filtered['sin_time'].values.reshape(-1, 1), ds_filtered['cos_time'].values.reshape(-1, 1)
        msl_flat_scaled = MinMaxScaler().fit_transform(msl_flat)
        precalculated_pca_data = {n: PCA(n_components=n, random_state=GLOBAL_SEED).fit_transform(msl_flat_scaled) for n in pca_dims_to_test}
        data_to_cache = {
            'precalculated_pca_data': precalculated_pca_data, 'y_true_multi': y_true_multi, 'sin_time': sin_time,
            'cos_time': cos_time, 'label_encoder': label_encoder, 'n_samples': n_samples,
            'valid_times': valid_times, 'base_labels': base_labels,'all_labels_str': all_labels_str
        }
        with open(preprocessed_data_cache_file, 'wb') as f: pd.to_pickle(data_to_cache, f)

    n_true_clusters = len(base_labels)
    print(f"✅ データ準備完了。対象サンプル数: {n_samples}, 基本ラベル数: {n_true_clusters}")
    print("\n--- 2. ランダムサーチ設定 ---")
    param_dist = {
        'data_input_order': ['normal_sort', 'month_sort', 'change_normal_sort', 'change_month_sort'], # ['normal_sort', 'month_sort', 'change_normal_sort', 'change_month_sort']
        'pca_n_components': pca_dims_to_test, 
        'include_time_features': [True, False], # [True, False]
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
        'death_patience_steps': [n_samples // 32, n_samples // 24, n_samples // 20, n_samples // 16, n_samples // 8, n_samples // 4, n_samples // 2, n_samples], 
        'num_epochs': [1000], 
        'activation_type': ['elliptical'] # ['circular', 'elliptical']
    }
    N_TRIALS = 10000
    fixed_params_for_acs = {'max_clusters': 50, 'initial_clusters': 1, 'lambda_min_val': 1e-7, 'bounds_W': (0, 1)}
    print(f"ランダムサーチ最大試行回数: {N_TRIALS}")
    print("\n--- 3. 並列ランダムサーチ実行 ---")
    num_processes_to_use = max(1, int(os.cpu_count() * 0.9)) if os.cpu_count() else 2
    tasks_for_pool = [] # 各試行用の独立した乱数生成器を作成
    for i in range(N_TRIALS):
        trial_rng = random.Random(GLOBAL_SEED + i + 1)
        params = sample_random_params(param_dist, rng=trial_rng)
        tasks_for_pool.append(((i + 1, params), GLOBAL_SEED + i + 1))
    worker_func_with_fixed_args = partial(run_acs_trial, fixed_params_dict=fixed_params_for_acs, pca_data_dict=precalculated_pca_data, y_data_multi=y_true_multi, sin_time_data=sin_time, cos_time_data=cos_time, n_true_cls_worker=n_true_clusters, trial_log_dir_path=trial_logs_dir, label_encoder_worker=label_encoder, valid_times_worker=valid_times) # 💡 valid_times_workerを追加
    start_search_time = datetime.datetime.now()
    all_trial_results = []
    with multiprocessing.Pool(processes=num_processes_to_use) as pool:
        for result in tqdm(pool.imap_unordered(worker_func_with_fixed_args, tasks_for_pool), total=N_TRIALS, desc="Random Search Progress"):
            all_trial_results.append(result)
    print(f"\nランダムサーチ完了。総所要時間: {datetime.datetime.now() - start_search_time}")
    print("\n--- 4. 結果集計 (3つの評価指標でベストを選出) ---")
    if not all_trial_results: sys.exit("エラー: サーチから結果が返されませんでした。")
    processed_results = []
    for res in all_trial_results:
        if res['error_traceback'] or not res['history']: continue
        history_df = pd.DataFrame(res['history'])
        best_by_composite = history_df.loc[history_df['composite_score'].idxmax()]
        best_by_bacc = history_df.loc[history_df['bacc'].idxmax()]
        best_by_accuracy = history_df.loc[history_df['accuracy'].idxmax()]
        processed_results.append({
            'trial_id': res['trial_count_from_worker'],
            'params': res['params_combo'],
            'random_state': res['acs_random_state_used'],
            'full_history': res['history'],
            'event_log': res['event_log'],
            'best_by_composite_score': best_by_composite.to_dict(),
            'best_by_bacc': best_by_bacc.to_dict(),
            'best_by_accuracy': best_by_accuracy.to_dict(),
        })

    if not processed_results: sys.exit("エラー: 有効な結果が得られませんでした。")
    summary_df = pd.DataFrame([
        {
            'trial_id': r['trial_id'], **r['params'],
            'best_composite_score': r['best_by_composite_score']['composite_score'], 'best_composite_epoch': r['best_by_composite_score']['epoch'],
            'best_bacc': r['best_by_bacc']['bacc'], 'best_bacc_epoch': r['best_by_bacc']['epoch'],
            'best_accuracy': r['best_by_accuracy']['accuracy'], 'best_accuracy_epoch': r['best_by_accuracy']['epoch'],
        } for r in processed_results
    ])
    summary_df.to_csv(output_dir / f"random_search_summary_{timestamp}.csv", index=False)
    print(f"全試行のサマリーをCSVファイルに保存しました: {output_dir.resolve()}")
    top_results_map = {} # 各評価指標で上位3位までを取得
    for metric_name in ['composite_score', 'bacc', 'accuracy']:
        sorted_results = sorted(processed_results, 
                            key=lambda x: x[f'best_by_{metric_name}'][metric_name], 
                            reverse=True)
        top_results_map[metric_name] = sorted_results[:3]  # 上位3つを取得

    for metric_name, top_results in top_results_map.items(): # 上位3位までの結果を処理
        for rank, best_result_info in enumerate(top_results, 1):
            best_trial_id = best_result_info['trial_id']
            best_epoch_data = best_result_info[f'best_by_{metric_name}']
            best_epoch = int(best_epoch_data['epoch'])
            best_score = best_epoch_data[metric_name]
            best_params = best_result_info['params']
            best_random_state = best_result_info['random_state']
            run_output_dir = output_dir / f"{rank}_model_by_{metric_name}" # ディレクトリ名を1_model_by_*, 2_model_by_*, 3_model_by_*の形式に
            os.makedirs(run_output_dir, exist_ok=True)
            best_event_log = best_result_info['event_log'] # 試行時のイベントログを保存
            if best_event_log:
                event_log_df = pd.DataFrame(best_event_log)
                log_save_path = run_output_dir / "trial_cluster_event_log.csv"
                event_log_df.to_csv(log_save_path, index=False)
                print(f"✅ 試行時のイベントログをCSVに保存しました: {log_save_path.resolve()}")
            
            print("-" * 50)
            print(f"\n--- 第{rank}位モデル (基準: {metric_name.upper()}) ---")
            print(f"   Trial ID: {best_trial_id}, Best Epoch: {best_epoch}")
            print(f"🏆 Score ({metric_name}): {best_score:.4f}")
            print(f"   - Composite Score: {best_epoch_data['composite_score']:.4f}")
            print(f"   - Balanced Accuracy: {best_epoch_data['bacc']:.4f}")
            print(f"   - Accuracy: {best_epoch_data['accuracy']:.4f}")
            print(f"   - Final Clusters: {int(best_epoch_data['n_clusters'])}")
            print("   - パラメータ:")
            for k, v in best_params.items(): 
                print(f"     {k}: {v}")
            
            print(f"\n   再学習とプロット生成を開始します... (出力先: {run_output_dir.resolve()})")
            X_pca = precalculated_pca_data[best_params['pca_n_components']] # 特徴量データの準備
            X_features = np.hstack([X_pca, sin_time, cos_time]) if best_params['include_time_features'] else X_pca
            X_scaled_data = MinMaxScaler().fit_transform(X_features).astype(np.float64)
            params_for_refit = {
                **fixed_params_for_acs, 
                **{k:v for k,v in best_params.items() if k not in ['pca_n_components', 'include_time_features', 'num_epochs', 'data_input_order']}, 
                'n_features': X_scaled_data.shape[1], 
                'random_state': best_random_state
            } # ACSモデルのパラメータ設定（'data_input_order' を除外）
            best_model_instance = ACS(**params_for_refit)
            pca_visual = PCA(n_components=2, random_state=GLOBAL_SEED) # 2D可視化用のPCAモデル
            X_pca_visual = pca_visual.fit_transform(X_scaled_data)
            np.random.seed(best_random_state) # 再学習前に乱数状態を完全にリセット
            random.seed(best_random_state)
            data_input_order = best_params['data_input_order'] # データ投入順序を再現するための準備（シードを明示的に指定）
            initial_indices = get_sorted_indices(data_input_order, valid_times, random_seed=best_random_state)
            # partial_fit() を使った再学習・可視化ループ
            refit_history = []  # 再学習時の履歴を記録
            for epoch in tqdm(range(1, best_epoch + 1), desc=f"Refitting for {metric_name} (rank {rank})"):
                current_indices = initial_indices
                if 'change' in data_input_order and epoch % 2 == 0:
                    current_indices = initial_indices[::-1]
                for data_idx in current_indices:
                    U_p = X_scaled_data[data_idx, :]
                    best_model_instance.partial_fit(U_p, epoch=epoch, data_idx=int(data_idx))
                
                # 再学習時の評価指標を計算
                preds = best_model_instance.predict(X_scaled_data)
                epoch_metrics = calculate_all_metrics_multi_label(preds, y_true_multi, label_encoder)
                epoch_metrics['epoch'] = epoch
                refit_history.append(epoch_metrics)
                plot_energy_contour_for_epoch(
                    model=best_model_instance, epoch=epoch, save_path=run_output_dir,
                    X_scaled_data_for_eval=X_scaled_data, X_pca_visual=X_pca_visual,
                    y_true_multi=y_true_multi, label_encoder=label_encoder,
                    pca_visual_model=pca_visual
                )
            
            # --- 最終状態のレポートと学習履歴グラフの保存 ---
            final_preds = best_model_instance.predict(X_scaled_data)
            if best_model_instance.event_log:
                refit_event_log_df = pd.DataFrame(best_model_instance.event_log)
                refit_log_save_path = run_output_dir / "refit_cluster_event_log.csv"
                refit_event_log_df.to_csv(refit_log_save_path, index=False)
                print(f"✅ 再学習時のイベントログをCSVに保存しました: {refit_log_save_path.resolve()}")
            
            trial_log_path = run_output_dir / "trial_cluster_event_log.csv"
            refit_log_path = run_output_dir / "refit_cluster_event_log.csv"
            # 評価指標の一致検証
            print("\n--- 評価指標の一致検証 ---")
            # ランダムサーチ時の最良エポックの指標
            trial_best_metrics = best_epoch_data
            # 再学習時の最良エポックの指標
            refit_best_metrics = refit_history[best_epoch - 1]  # epochは1から始まるため-1
            # 各指標の比較
            metrics_to_compare = ['composite_score', 'weighted_purity', 'accuracy', 'bacc', 'ari', 'n_clusters']
            all_metrics_match = True
            for metric in metrics_to_compare:
                trial_value = trial_best_metrics.get(metric, None)
                refit_value = refit_best_metrics.get(metric, None)
                if trial_value is None or refit_value is None:
                    print(f"   ⚠️ {metric}: 比較不可（データなし）")
                    continue
                
                # 数値の比較（浮動小数点の誤差を考慮）
                if isinstance(trial_value, (int, float)) and isinstance(refit_value, (int, float)):
                    if abs(trial_value - refit_value) < 1e-6:
                        print(f"   ✅ {metric}: 一致 (値: {trial_value:.6f})")
                    else:
                        print(f"   ❌ {metric}: 不一致 (Trial: {trial_value:.6f}, Refit: {refit_value:.6f}, 差: {abs(trial_value - refit_value):.6f})")
                        all_metrics_match = False
                else:
                    if trial_value == refit_value:
                        print(f"   ✅ {metric}: 一致 (値: {trial_value})")
                    else:
                        print(f"   ❌ {metric}: 不一致 (Trial: {trial_value}, Refit: {refit_value})")
                        all_metrics_match = False
            
            if all_metrics_match:
                print("\n   ✅ [OK] すべての評価指標が一致しました。完全な再現性が確認されました。")
            else:
                print("\n   ⚠️ [NG] 一部の評価指標が一致しませんでした。")
            
            print("\n--- イベントログの一致検証 ---")
            if trial_log_path.exists() and refit_log_path.exists():
                try:
                    df_trial = pd.read_csv(trial_log_path)
                    df_refit = pd.read_csv(refit_log_path)
                    are_logs_identical = df_trial.equals(df_refit)
                    if are_logs_identical:
                        print(f"✅ [OK] trial と refit のイベントログは完全に一致しました。再現性が確認されました。")
                    else:
                        print(f"⚠️ [NG] trial と refit のイベントログは一致しませんでした。プログラムの動作に非決定的な要素が含まれている可能性があります。")
                        if len(df_trial) != len(df_refit):
                            print(f"   - 行数が異なります: Trial={len(df_trial)}, Refit={len(df_refit)}")
                        else:
                            try:
                                pd.testing.assert_frame_equal(df_trial, df_refit, check_dtype=True)
                            except AssertionError as e:
                                print(f"   - 内容に差異があります。差分の詳細:\n{e}")
                except Exception as e:
                    print(f"   - ログファイルの比較中にエラーが発生しました: {e}")
            else:
                print("   - 比較対象のログファイルの一方または両方が存在しないため、検証をスキップします。")
            
            print("-" * 32)
            plot_cluster_composition(
                final_preds=final_preds,
                y_true_multi=y_true_multi,
                label_encoder=label_encoder,
                save_path=run_output_dir,
                metric_name=metric_name,
                trial_id=best_trial_id
            )
            plot_cluster_counts(
                final_preds=final_preds,
                y_true_multi=y_true_multi,
                label_encoder=label_encoder,
                save_path=run_output_dir,
                metric_name=metric_name,
                trial_id=best_trial_id
            )
            classification_df = pd.DataFrame({
                'valid_time': valid_times,
                'true_label_str': all_labels_str,
                'predicted_cluster': final_preds
            })
            classification_df.to_csv(run_output_dir / "classification_details.csv", index=False)
            print(f"✅ 各データの日付と分類結果をCSVに保存しました。")
            # 再学習時の履歴から学習推移グラフを生成
            history_df = pd.DataFrame(refit_history)  # 再学習時の履歴を使用
            fig_width = 12 + (best_epoch // 2000) * 4
            fig, axes = plt.subplots(3, 1, figsize=(fig_width, 15), sharex=True)
            fig.suptitle(f'第{rank}位モデルの学習推移 (Trial {best_trial_id}, 基準: {metric_name.upper()}, 再学習時)', fontsize=16)
            # 3つの指標をそれぞれプロット
            for i, (m_name, color) in enumerate([('composite_score', 'green'), ('bacc', 'purple'), ('accuracy', 'orange')]):
                axes[i].plot(history_df['epoch'], history_df[m_name], 's-', c=color, label=m_name)
                axes[i].set_ylabel(m_name, color=color)
                axes[i].tick_params(axis='y', labelcolor=color)
                ax2 = axes[i].twinx()
                ax2.plot(history_df['epoch'], history_df['n_clusters'], 'o--', c='tab:blue', alpha=0.6, label='Clusters')
                ax2.set_ylabel('Clusters', color='tab:blue')
                ax2.tick_params(axis='y', labelcolor='tab:blue')
                axes[i].axvline(x=best_epoch, color='red', linestyle='--', label=f'Best Epoch ({best_epoch})')
                axes[i].grid(True, axis='y', linestyle=':')
            
            axes[-1].set_xlabel('エポック数')
            fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=axes[0].transAxes)
            plt.tight_layout(rect=[0, 0.03, 1, 0.96])
            plt.savefig(run_output_dir / f"learning_history_{metric_name}.png", dpi=300)
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