# ==============================================================================
# ライブラリのインポート
# ==============================================================================
import sys
from pathlib import Path
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
import matplotlib.gridspec as gridspec
import japanize_matplotlib

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, confusion_matrix, recall_score, balanced_accuracy_score
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
GLOBAL_SEED = 17
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
# ★★★★★ 新しい評価指標と分析のためのヘルパー関数 ★★★★★
# ==============================================================================

def analyze_purity_and_splitting(preds, y_true, label_encoder):
    """
    クラスタの純度とラベルの分割状況を分析する。
    Args:
        preds (np.array): モデルによる予測クラスタラベルの配列。
        y_true (np.array): 真の人間ラベルの配列（数値エンコード済み）。
        label_encoder (LabelEncoder): ラベル名を復元するためのエンコーダ。
    Returns:
        tuple: (cluster_report_df, label_report_df)
    """
    # 予測クラスタが存在しない場合は空のレポートを返す
    if len(np.unique(preds)) == 0 or len(preds) == 0:
        return pd.DataFrame(), pd.DataFrame()

    # 1. コンティンジェンシー・マトリックスを作成 (行: 予測クラスタ, 列: 真ラベル)
    contingency_matrix = pd.crosstab(preds, y_true)

    # 2. クラスタ純度レポートの作成
    cluster_total_samples = contingency_matrix.sum(axis=1)
    cluster_max_samples = contingency_matrix.max(axis=1)
    purity = np.divide(cluster_max_samples, cluster_total_samples, 
                       out=np.zeros_like(cluster_max_samples, dtype=float), 
                       where=cluster_total_samples!=0)
    dominant_label_idx = contingency_matrix.idxmax(axis=1)
    dominant_label_name = dominant_label_idx.map(lambda idx: label_encoder.classes_[idx])
    
    cluster_report_df = pd.DataFrame({
        'cluster_id': contingency_matrix.index,
        'n_samples': cluster_total_samples,
        'purity': purity,
        'dominant_label': dominant_label_name
    }).sort_values(by='n_samples', ascending=False).reset_index(drop=True)

    # 3. ラベル分割レポートの作成
    label_report_list = []
    for label_idx, label_name in enumerate(label_encoder.classes_):
        if label_idx in contingency_matrix.columns:
            series = contingency_matrix[label_idx]
            assigned_clusters = series[series > 0]
            label_report_list.append({
                'label_name': label_name,
                'n_total_samples': series.sum(),
                'n_splits': len(assigned_clusters),
                'split_to_clusters': list(assigned_clusters.index)
            })
    label_report_df = pd.DataFrame(label_report_list)

    return cluster_report_df, label_report_df


def calculate_composite_score(cluster_report, n_true_clusters, ideal_cluster_range=(1.0, 2.0)):
    """
    クラスタ純度とクラスタ数から複合評価スコアを計算する。
    Args:
        cluster_report (pd.DataFrame): analyze_purity_and_splittingから得られるクラスタレポート。
        n_true_clusters (int): 真のラベル数 (例: 15)。
        ideal_cluster_range (tuple): 理想的なクラスタ数の、真のラベル数に対する倍率の範囲 (min, max)。
    Returns:
        tuple: (final_score, weighted_purity, penalty)
    """
    if cluster_report.empty:
        return 0.0, 0.0, 0.0
        
    # 1. 加重平均純度の計算
    total_samples = cluster_report['n_samples'].sum()
    weighted_purity = np.sum(cluster_report['purity'] * cluster_report['n_samples']) / total_samples if total_samples > 0 else 0.0

    # 2. クラスタ数ペナルティの計算
    n_clusters = len(cluster_report)
    min_ideal_clusters = n_true_clusters * ideal_cluster_range[0]
    max_ideal_clusters = n_true_clusters * ideal_cluster_range[1]
    
    # 理想範囲の中心
    center_ideal_clusters = (min_ideal_clusters + max_ideal_clusters) / 2.0

    if min_ideal_clusters <= n_clusters <= max_ideal_clusters:
        penalty = 1.0
    else:
        # 範囲外の場合、中心からの距離に応じて指数関数的にペナルティを課す
        distance_from_center = abs(n_clusters - center_ideal_clusters)
        # スケールパラメータを調整してペナルティの効き具合を変える
        scale = n_true_clusters 
        penalty = np.exp(-distance_from_center / scale)

    # 3. 最終スコアの計算
    final_score = weighted_purity * penalty
    
    return final_score, weighted_purity, penalty

# ==============================================================================
# 並列処理ワーカー関数 (改善版)
# ==============================================================================
def run_acs_trial(param_values_tuple_with_trial_info,
                  fixed_params_dict,
                  pca_data_dict,
                  y_data,
                  sin_time_data,
                  cos_time_data,
                  n_true_cls_worker,
                  trial_log_dir_path,
                  label_encoder_worker): # label_class_names から label_encoderオブジェクトに変更
    """
    グリッドサーチ/ランダムサーチの1試行を独立して実行するワーカー関数。
    新しい複合評価スコアに基づいて最良エポックを決定する。
    """
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
            X_features = np.hstack([X_pca, sin_time_data, cos_time_data]) if include_time_features else X_pca
            scaler = MinMaxScaler()
            X_scaled_data = scaler.fit_transform(X_features)
            n_features_worker = X_scaled_data.shape[1]

            num_epochs_worker = params_combo.pop('num_epochs')
            activation_type_worker = params_combo.pop('activation_type')
            if activation_type_worker == 'circular':
                params_combo.pop('initial_lambda_vector_val', None)
                params_combo.pop('initial_lambda_crossterm_val', None)
            else:
                params_combo.pop('initial_lambda_scalar', None)

            current_run_params = {**fixed_params_dict, 'n_features': n_features_worker, 'activation_type': activation_type_worker, **params_combo, 'random_state': trial_specific_seed}
            
            print(f"\n--- [Worker] トライアル {trial_count} 開始 ---")
            print(f"[Worker {trial_count}] 特徴量: PCA={pca_n_components}, Time={include_time_features}, Total Dim={n_features_worker}")
            print(f"[Worker {trial_count}] パラメータ: { {k: f'{v:.4f}' if isinstance(v, float) else v for k, v in params_combo.items()} }")
            print(f"[Worker {trial_count}] エポック数: {num_epochs_worker}")

            # --- 結果格納用辞書の初期化 (新しい指標を追加) ---
            result = {
                'params_combo': {**params_combo, 'pca_n_components': pca_n_components, 'include_time_features': include_time_features, 'num_epochs': num_epochs_worker, 'activation_type': activation_type_worker},
                'final_score': -1.0, 'weighted_purity': -1.0, 'cluster_count_penalty': -1.0,
                'balanced_accuracy_mapped': -1.0, 'ari': -1.0,
                'final_clusters': -1, 'best_epoch': 0, 'history': [], 
                'error_traceback': None, 'duration_seconds': 0,
                'acs_random_state_used': trial_specific_seed,
                'trial_count_from_worker': trial_count
            }
            trial_start_time = datetime.datetime.now()

            try:
                acs_model_trial = ACS(**current_run_params)
                
                for epoch in range(1, num_epochs_worker + 1):
                    acs_model_trial.fit(X_scaled_data, epochs=1)
                    current_clusters = acs_model_trial.M
                    
                    # --- エポックごとの評価 ---
                    epoch_metrics = {'epoch': epoch, 'clusters': current_clusters, 'final_score': 0.0, 'weighted_purity': 0.0, 'bacc': 0.0, 'ari': 0.0}
                    
                    if current_clusters > 0:
                        preds = acs_model_trial.predict(X_scaled_data)
                        
                        # 新しい評価指標の計算
                        cluster_rep, _ = analyze_purity_and_splitting(preds, y_data, label_encoder_worker)
                        final_score, weighted_purity, penalty = calculate_composite_score(cluster_rep, n_true_cls_worker)
                        epoch_metrics['final_score'] = final_score
                        epoch_metrics['weighted_purity'] = weighted_purity
                        
                        # 従来の評価指標も計算
                        epoch_metrics['ari'] = adjusted_rand_score(y_data, preds)
                        contingency = pd.crosstab(preds, y_data)
                        if not contingency.empty:
                            row_ind, col_ind = linear_sum_assignment(-contingency.values)
                            mapped_preds = np.full_like(y_data, -1)
                            pred_map = {contingency.index[pred_i]: true_i for pred_i, true_i in zip(row_ind, col_ind)}
                            for pred_label, true_label_idx in pred_map.items():
                                mapped_preds[preds == pred_label] = true_label_idx
                            epoch_metrics['bacc'] = balanced_accuracy_score(y_data, mapped_preds)

                    result['history'].append(epoch_metrics)
                    print(f"[Worker {trial_count}] Epoch {epoch}/{num_epochs_worker} - Cls: {current_clusters}, Score: {epoch_metrics['final_score']:.4f}, Purity: {epoch_metrics['weighted_purity']:.4f}, BAcc: {epoch_metrics['bacc']:.4f}")

                # --- 最良エポックの決定 (Final Score基準) ---
                if result['history']:
                    best_epoch_history = max(result['history'], key=lambda x: x['final_score'])
                    result.update({
                        'final_score': best_epoch_history['final_score'],
                        'weighted_purity': best_epoch_history['weighted_purity'],
                        'balanced_accuracy_mapped': best_epoch_history['bacc'],
                        'ari': best_epoch_history['ari'],
                        'final_clusters': best_epoch_history['clusters'],
                        'best_epoch': best_epoch_history['epoch']
                    })

            except Exception:
                result['error_traceback'] = traceback.format_exc()
                print(f"--- [Worker] トライアル {trial_count} でエラー発生 ---\n{result['error_traceback']}")

            trial_end_time = datetime.datetime.now()
            result['duration_seconds'] = (trial_end_time - trial_start_time).total_seconds()
            print(f"\n--- [Worker] トライアル {trial_count} 終了 | Best Score: {result['final_score']:.4f} at Epoch {result['best_epoch']}, Time: {result['duration_seconds']:.2f}s ---")
            
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
            # 整数と浮動小数点数の両方に対応
            if all(isinstance(v, int) for v in value):
                params[key] = random.randint(value[0], value[1])
            else:
                params[key] = round(random.uniform(value[0], value[1]), 4)
    return params


def plot_energy_contour_for_epoch(model, epoch, save_path, 
                                  X_scaled_data_for_eval, X_pca_visual, y_true_labels, 
                                  label_encoder, pca_visual_model):
    """指定されたモデルの状態でエネルギー等高線プロット、情報パネル、混同行列を生成し保存する。"""
    n_samples = X_scaled_data_for_eval.shape[0]
    n_true_classes = len(label_encoder.classes_)
    current_clusters = model.M

    # --- 評価指標の計算 ---
    final_score, weighted_purity, bacc, ari = 0.0, 0.0, 0.0, 0.0
    cm = np.zeros((n_true_classes, n_true_classes), dtype=int)
    if current_clusters > 0:
        preds = model.predict(X_scaled_data_for_eval)
        ari = adjusted_rand_score(y_true_labels, preds)
        
        cluster_rep, _ = analyze_purity_and_splitting(preds, y_true_labels, label_encoder)
        final_score, weighted_purity, _ = calculate_composite_score(cluster_rep, n_true_classes)

        contingency = pd.crosstab(preds, y_true_labels)
        if not contingency.empty:
            row_ind, col_ind = linear_sum_assignment(-contingency.values)
            mapped_preds = np.full_like(y_true_labels, -1)
            pred_map = {contingency.index[pred_i]: true_i for pred_i, true_i in zip(row_ind, col_ind)}
            for pred_label, true_label_idx in pred_map.items():
                mapped_preds[preds == pred_label] = true_label_idx
            bacc = balanced_accuracy_score(y_true_labels, mapped_preds)
            cm = confusion_matrix(y_true_labels, mapped_preds, labels=np.arange(n_true_classes))

    # --- プロットの準備 ---
    fig = plt.figure(figsize=(24, 8), constrained_layout=True)
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.8, 1, 1.2])
    ax_contour, ax_info, ax_cm = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    fig.suptitle(f'クラスタリング状態と評価 (Epoch: {epoch})', fontsize=20)

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
        fig.colorbar(contour, ax=ax_contour, label='エネルギー (E)', orientation='vertical')
    except Exception as e:
        ax_contour.text(0.5, 0.5, f"エネルギー計算/描画エラー:\n{e}", transform=ax_contour.transAxes, ha='center', va='center')

    palette = sns.color_palette("bright", n_colors=n_true_classes)
    legend_plot = sns.scatterplot(ax=ax_contour, x=X_pca_visual[:, 0], y=X_pca_visual[:, 1], 
                                  hue=[label_encoder.classes_[l] for l in y_true_labels], 
                                  palette=palette, s=50, alpha=0.9, edgecolor='w', linewidth=0.5)
    legend_plot.legend(title="凡例", bbox_to_anchor=(1.05, 1), loc='upper left')

    if current_clusters > 0 and 'row_ind' in locals() and len(row_ind) > 0:
        cluster_centers = model.get_cluster_centers()
        try:
            cluster_centers_2d = pca_visual_model.transform(cluster_centers)
            pred_map = {pred_i: true_i for pred_i, true_i in zip(row_ind, col_ind)}
            for pred_idx, center_2d in enumerate(cluster_centers_2d):
                true_idx = pred_map.get(pred_idx)
                color = palette[true_idx] if true_idx is not None else 'grey'
                marker = 'o' if true_idx is not None else 'X'
                ax_contour.scatter(center_2d[0], center_2d[1], c=[color], marker=marker, s=300, edgecolor='black', linewidth=2.0, zorder=10)
        except Exception as e:
            print(f"Epoch {epoch}: クラスタ中心の描画中にエラー: {e}")

    ax_contour.set_title('クラスタリング結果 (PCA 2D)', fontsize=16)
    ax_contour.set_xlabel('主成分1'), ax_contour.set_ylabel('主成分2')

    # --- 中央パネル: 情報表示 ---
    ax_info.axis('off')
    ax_info.set_title('学習状況サマリ', fontsize=16)
    info_text = (
        f"Epoch: {epoch}\n"
        f"Clusters (M): {current_clusters}\n\n"
        f"複合評価スコア: {final_score:.4f}\n"
        f"加重平均純度: {weighted_purity:.4f}\n\n"
        f"Balanced Accuracy: {bacc:.4f}\n"
        f"Adjusted Rand Index: {ari:.4f}"
    )
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.9),
                 family='monospace')

    # --- 右パネル: 混同行列 ---
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=ax_cm, cbar=False)
    ax_cm.set_title('混同行列 (マッピング後)', fontsize=16)
    ax_cm.set_xlabel("予測ラベル (マッピング後)"), ax_cm.set_ylabel("真のラベル")
    plt.setp(ax_cm.get_xticklabels(), rotation=45, ha="right")

    plt.savefig(save_path / f"epoch_{epoch:04d}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

# ==============================================================================
# メイン処理 (改善版)
# ==============================================================================
def main_process_logic():
    """データ読み込み、前処理、ランダムサーチ、結果評価、プロットまでを統括するメイン関数。"""
    logger_instance = Logger(log_file_path)
    sys.stdout, sys.stderr = logger_instance, logger_instance

    print("=" * 80)
    print("ACSモデルによる気圧配置パターンの教師なしクラスタリング (評価基準改善版)")
    print("=" * 80)
    
    # --- 1. データ準備 (キャッシュ機能はそのまま活用) ---
    print("\n--- 1. データ準備 ---")
    pca_dims_to_test = [15, 20, 25]
    preprocessed_data_cache_file = Path("./preprocessed_prmsl_data.pkl")
    if preprocessed_data_cache_file.exists():
        print(f"✅ キャッシュファイルから前処理済みデータを読み込みます...")
        with open(preprocessed_data_cache_file, 'rb') as f: data_cache = pd.read_pickle(f)
        precalculated_pca_data = data_cache['precalculated_pca_data']
        y_true_labels, sin_time, cos_time = data_cache['y_true_labels'], data_cache['sin_time'], data_cache['cos_time']
        label_encoder, n_samples = data_cache['label_encoder'], data_cache['n_samples']
        n_true_clusters = len(label_encoder.classes_)
    else:
        print(f"✅ キャッシュがないため、データを新規生成します...")
        ds = xr.open_dataset("./prmsl_era5_all_data.nc")
        ds_period = ds.sel(valid_time=slice('1991-01-01', '2000-12-31'))
        target_labels = ['1', '2A', '2B', '2C', '2D', '3A', '3B', '3C', '3D', '4A', '4B', '5', '6A', '6B', '6C']
        ds_filtered = ds_period.where(ds_period['label'].isin(target_labels), drop=True)
        n_samples = ds_filtered.sizes['valid_time']
        label_encoder = LabelEncoder().fit(target_labels)
        y_true_labels = label_encoder.transform(ds_filtered['label'].values)
        msl_flat = ds_filtered['msl'].values.reshape(n_samples, -1)
        sin_time, cos_time = ds_filtered['sin_time'].values.reshape(-1, 1), ds_filtered['cos_time'].values.reshape(-1, 1)
        msl_flat_scaled = MinMaxScaler().fit_transform(msl_flat)
        precalculated_pca_data = {n: PCA(n_components=n, random_state=GLOBAL_SEED).fit_transform(msl_flat_scaled) for n in pca_dims_to_test}
        data_to_cache = {'precalculated_pca_data': precalculated_pca_data, 'y_true_labels': y_true_labels, 'sin_time': sin_time, 'cos_time': cos_time, 'label_encoder': label_encoder, 'n_samples': n_samples}
        with open(preprocessed_data_cache_file, 'wb') as f: pd.to_pickle(data_to_cache, f)
    print(f"✅ データ準備完了。対象サンプル数: {n_samples}, 真のラベル数: {n_true_clusters}")

    # --- 2. ランダムサーチの設定 ---
    print("\n--- 2. ランダムサーチ設定 ---")
    param_dist = {
        'pca_n_components': pca_dims_to_test, 'include_time_features': [True, False], 'gamma': (0.01, 3.0), 'beta': (0.001, 1.0),
        'learning_rate_W': (0.001, 0.1), 'learning_rate_lambda': (0.001, 0.1), 'learning_rate_Z': (0.001, 0.1),
        'initial_lambda_scalar': (0.001, 1.0), 'initial_lambda_vector_val': (0.001, 1.0), 'initial_lambda_crossterm_val': (-0.5, 0.5),
        'initial_Z_val': (0.01, 1.0), 'initial_Z_new_cluster': (0.01, 1.0), 'theta_new': (0.001, 1.0), 'Z_death_threshold': (0.01, 0.1),
        'death_patience_steps': [n_samples // 4, n_samples // 2, n_samples], 'num_epochs': [10000], 'activation_type': ['circular', 'elliptical']
    }
    N_TRIALS = 500
    SCORE_GOAL = 0.7  # 目標複合スコア
    fixed_params_for_acs = {'max_clusters': 50, 'initial_clusters': 1, 'lambda_min_val': 1e-7, 'bounds_W': (0, 1)}
    print(f"ランダムサーチ最大試行回数: {N_TRIALS}, 早期終了の目標スコア: {SCORE_GOAL}")

    # --- 3. 並列ランダムサーチの実行 ---
    print("\n--- 3. 並列ランダムサーチ実行 ---")
    num_processes_to_use = max(1, int(os.cpu_count() * 0.9)) if os.cpu_count() else 2
    tasks_for_pool = [((i + 1, sample_random_params(param_dist)), GLOBAL_SEED + i + 1) for i in range(N_TRIALS)]
    worker_func_with_fixed_args = partial(run_acs_trial, fixed_params_dict=fixed_params_for_acs, pca_data_dict=precalculated_pca_data, y_data=y_true_labels, sin_time_data=sin_time, cos_time_data=cos_time, n_true_cls_worker=n_true_clusters, trial_log_dir_path=trial_logs_dir, label_encoder_worker=label_encoder)

    start_search_time = datetime.datetime.now()
    all_trial_results = []
    with multiprocessing.Pool(processes=num_processes_to_use) as pool:
        for result in tqdm(pool.imap_unordered(worker_func_with_fixed_args, tasks_for_pool), total=N_TRIALS, desc="Random Search Progress"):
            all_trial_results.append(result)
            # ★★★ 評価基準を Final Score に変更 ★★★
            if not result['error_traceback'] and result.get('final_score', 0) >= SCORE_GOAL:
                print(f"\n✅ 目標スコア達成 (Score >= {SCORE_GOAL})！ トライアル {result['trial_count_from_worker']} でサーチを打ち切ります。")
                pool.terminate()
                break
    print(f"\nランダムサーチ完了。総所要時間: {datetime.datetime.now() - start_search_time}")

    # --- 4. 結果の集計と最良モデルの選定 (Final Score基準) ---
    print("\n--- 4. 結果集計 ---")
    if not all_trial_results: sys.exit("エラー: サーチから結果が返されませんでした。")
    
    df_data = [{**res['params_combo'], **{k:v for k,v in res.items() if k!='params_combo'}} for res in all_trial_results]
    results_df = pd.DataFrame(df_data).sort_values(by=['final_score', 'weighted_purity'], ascending=False)
    results_df.to_csv(output_dir / f"random_search_all_results_{timestamp}.csv", index=False, encoding='utf-8-sig')
    print(f"全試行結果をCSVファイルに保存しました: {output_dir.resolve()}")

    best_result = results_df.iloc[0].to_dict() if not results_df.empty else None
    if best_result is None: sys.exit("エラー: 有効な結果が得られませんでした。")

    print(f"\n--- 最良モデル (Trial ID: {best_result['trial_count_from_worker']}) ---")
    print(f"🏆 最高複合スコア: {best_result['final_score']:.4f} (at Epoch {best_result['best_epoch']})")
    print(f"   - 加重平均純度: {best_result['weighted_purity']:.4f}")
    print(f"   - 最終クラスタ数: {best_result['final_clusters']}")
    print(f"   - (参考) BAcc: {best_result['balanced_accuracy_mapped']:.4f}, ARI: {best_result['ari']:.4f}")
    print("   - パラメータ:")
    for key in param_dist.keys(): print(f"     {key}: {best_result.get(key)}")

    # --- 5. 最良モデルでの再学習 ---
    print("\n--- 5. 最良モデルでの再学習 ---")
    best_params = {k: v for k, v in best_result.items() if k in param_dist.keys()}
    X_pca_best = precalculated_pca_data[best_params['pca_n_components']]
    X_features_best = np.hstack([X_pca_best, sin_time, cos_time]) if best_params['include_time_features'] else X_pca_best
    X_scaled_data_best = MinMaxScaler().fit_transform(X_features_best)
    
    params_for_refit = {**fixed_params_for_acs, **{k: v for k, v in best_params.items() if k not in ['pca_n_components', 'include_time_features', 'num_epochs']}, 'n_features': X_scaled_data_best.shape[1], 'random_state': best_result['acs_random_state_used']}
    best_model_instance = ACS(**params_for_refit)
    
    # --- 6. 結果の可視化と保存 (強化版) ---
    print("\n--- 6. 結果の可視化 ---")
    pca_visual = PCA(n_components=2, random_state=GLOBAL_SEED)
    X_pca_visual = pca_visual.fit_transform(X_scaled_data_best)
    energy_plot_dir = output_dir / f"best_model_plots_trial_{int(best_result['trial_count_from_worker'])}"
    os.makedirs(energy_plot_dir, exist_ok=True)
    print(f"✅ 最良モデルのプロットを {energy_plot_dir.resolve()} に保存します。")
    
    epochs_for_refit = int(best_result['best_epoch'])
    for epoch in tqdm(range(1, epochs_for_refit + 1), desc="再学習とプロット生成"):
        best_model_instance.fit(X_scaled_data_best, epochs=1)
        plot_energy_contour_for_epoch(model=best_model_instance, epoch=epoch, save_path=energy_plot_dir, X_scaled_data_for_eval=X_scaled_data_best, X_pca_visual=X_pca_visual, label_encoder=label_encoder, pca_visual_model=pca_visual)
    
    # --- 6b. 最終状態での詳細レポートと可視化 ---
    print("\n--- 最終評価レポート (After Refit) ---")
    final_preds = best_model_instance.predict(X_scaled_data_best)
    final_cluster_rep, final_label_rep = analyze_purity_and_splitting(final_preds, y_true_labels, label_encoder)
    final_cluster_rep.to_csv(output_dir / "final_cluster_report.csv", index=False)
    final_label_rep.to_csv(output_dir / "final_label_report.csv", index=False)
    print("✅ 最終クラスタ純度・ラベル分割レポートを保存しました。")

    plt.figure(figsize=(12, 6)); sns.histplot(data=final_cluster_rep, x='purity', bins=20, kde=True); plt.title('クラスタ純度の分布 (最良モデル)'); plt.savefig(output_dir / "purity_distribution.png", dpi=300); plt.close()
    plt.figure(figsize=(15, 7)); sns.barplot(data=final_label_rep.sort_values('n_splits', ascending=False), x='label_name', y='n_splits', palette='viridis'); plt.title('人間ラベルごとの分割クラスタ数'); plt.xticks(rotation=45, ha="right"); plt.tight_layout(); plt.savefig(output_dir / "label_splitting_counts.png", dpi=300); plt.close()
    print("✅ 純度分布とラベル分割数のグラフを保存しました。")
    
    # --- 6c. 学習履歴プロット (Final Score基準) ---
    history = best_result.get('history', [])
    if isinstance(history, str): import ast; history = ast.literal_eval(history) # for re-reading from csv
    if history:
        history_df = pd.DataFrame(history)
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(history_df['epoch'], history_df['final_score'], 's-', c='tab:red', label='複合評価スコア')
        ax1.set_xlabel('エポック数'); ax1.set_ylabel('複合評価スコア', color='tab:red'); ax1.tick_params(axis='y', labelcolor='tab:red')
        ax2 = ax1.twinx()
        ax2.plot(history_df['epoch'], history_df['clusters'], 'o--', c='tab:blue', alpha=0.6, label='クラスタ数')
        ax2.set_ylabel('クラスタ数', color='tab:blue'); ax2.tick_params(axis='y', labelcolor='tab:blue')
        ax1.axvline(x=epochs_for_refit, color='green', linestyle='--', label=f'Best Epoch ({epochs_for_refit})')
        ax1.set_title('最良モデルの学習推移 (複合スコア基準)'); fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes);
        plt.savefig(output_dir / "best_model_history.png", dpi=300); plt.close()
        print("✅ 最良モデルの学習推移グラフを保存しました。")

    print("\n--- 全処理完了 ---")
    if isinstance(sys.stdout, Logger): logger_instance.close()

if __name__ == '__main__':
    main_process_logic()