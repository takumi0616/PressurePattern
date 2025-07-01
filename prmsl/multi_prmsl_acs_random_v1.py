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
                'ari': -1.0, 'accuracy_mapped': -1.0, 'balanced_accuracy_mapped': -1.0, # ★追加
                'final_clusters': -1, 'best_epoch': 0, 'history': [], 
                'error_traceback': None, 'duration_seconds': 0,
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
                    epoch_ari, epoch_acc, epoch_bacc = 0.0, 0.0, 0.0
                    
                    if current_clusters > 0:
                        preds = acs_model_trial.predict(X_scaled_data)
                        epoch_ari = adjusted_rand_score(y_data, preds)
                        
                        contingency = pd.crosstab(preds, y_data)
                        row_ind, col_ind = linear_sum_assignment(-contingency.values)
                        epoch_acc = contingency.values[row_ind, col_ind].sum() / n_samples_worker
                        
                        # マッピング後のラベルを計算してバランス化正解率を算出
                        mapped_preds = np.full_like(y_data, -1)
                        pred_idx_to_true_idx_map = {contingency.index[pred_i]: true_i for pred_i, true_i in zip(row_ind, col_ind)}
                        for pred_label_val, true_label_idx in pred_idx_to_true_idx_map.items():
                            mapped_preds[preds == pred_label_val] = true_label_idx
                        epoch_bacc = balanced_accuracy_score(y_data, mapped_preds)

                    # ★改善: 全エポックでログ出力
                    print(f"[Worker {trial_count}] Epoch {epoch+1}/{num_epochs_worker} - Cls: {current_clusters}, ARI: {epoch_ari:.4f}, Acc: {epoch_acc:.4f}, BAcc: {epoch_bacc:.4f}")
                    result['history'].append({
                        'epoch': epoch + 1, 'clusters': current_clusters, 
                        'ari': epoch_ari, 'accuracy_mapped': epoch_acc, 
                        'balanced_accuracy_mapped': epoch_bacc # ★追加
                    })

                if result['history']:
                    # ★改善: バランス化正解率で最良エポックを決定
                    best_epoch_history = max(result['history'], key=lambda x: x['balanced_accuracy_mapped'])
                    result['ari'] = best_epoch_history['ari']
                    result['accuracy_mapped'] = best_epoch_history['accuracy_mapped']
                    result['balanced_accuracy_mapped'] = best_epoch_history['balanced_accuracy_mapped'] # ★追加
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
            print(f"\n--- [Worker] トライアル {trial_count} 終了 | Best BAcc: {result['balanced_accuracy_mapped']:.4f} at Epoch {result['best_epoch']}, Time: {result['duration_seconds']:.2f}s ---")
            
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
            params[key] = round(np.random.uniform(value[0], value[1]), 2)
    return params

# ★★★ レイアウト問題を修正した新しい描画関数 ★★★
def plot_energy_contour_for_epoch(model, epoch, save_path, 
                                  X_scaled_data_for_eval, X_pca_visual, y_true_labels, 
                                  target_names_map, label_encoder_classes, pca_visual_model):
    """
    指定されたモデルの状態でエネルギー等高線プロット、情報パネル、混同行列を生成し保存する。
    """
    n_samples = X_scaled_data_for_eval.shape[0]
    n_true_classes = len(label_encoder_classes)
    current_clusters = model.M

    # --- 評価指標の計算 ---
    acc, bacc, ari, cm = 0.0, 0.0, 0.0, np.zeros((n_true_classes, n_true_classes), dtype=int)
    row_ind, col_ind = [], []
    if current_clusters > 0:
        preds = model.predict(X_scaled_data_for_eval)
        ari = adjusted_rand_score(y_true_labels, preds)
        
        contingency = pd.crosstab(preds, y_true_labels)
        # 予測クラスタ数が0の場合 contingency が空になりうる
        if not contingency.empty:
            row_ind, col_ind = linear_sum_assignment(-contingency.values)
            acc = contingency.values[row_ind, col_ind].sum() / n_samples

            mapped_preds = np.full_like(y_true_labels, -1)
            pred_idx_to_true_idx_map = {contingency.index[pred_i]: true_i for pred_i, true_i in zip(row_ind, col_ind)}
            for pred_label_val, true_label_idx in pred_idx_to_true_idx_map.items():
                mapped_preds[preds == pred_label_val] = true_label_idx
            
            bacc = balanced_accuracy_score(y_true_labels, mapped_preds)
            cm = confusion_matrix(y_true_labels, mapped_preds, labels=np.arange(n_true_classes))
            recalls = recall_score(y_true_labels, mapped_preds, average=None, labels=np.arange(n_true_classes), zero_division=0)


    # --- プロットの準備 (★レイアウト改善: constrained_layout=True を使用) ---
    fig = plt.figure(figsize=(24, 8), constrained_layout=True)
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.8, 1, 1.2])
    
    ax_contour = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])
    ax_cm = fig.add_subplot(gs[0, 2])
    
    fig.suptitle(f'エネルギー等高線とクラスタリング状態 (Epoch: {epoch})', fontsize=20)

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
        ax_contour.contour(xx, yy, Z_grid, levels=20, colors='white', linewidths=0.5, alpha=0.6)
    except Exception as e:
        ax_contour.text(0.5, 0.5, f"エネルギー計算/描画エラー:\n{e}", transform=ax_contour.transAxes, ha='center', va='center')

    palette = sns.color_palette("bright", n_colors=n_true_classes)
    # Seabornプロットの凡例を一度変数に受け取る
    legend_plot = sns.scatterplot(ax=ax_contour, x=X_pca_visual[:, 0], y=X_pca_visual[:, 1], 
                                  hue=[target_names_map.get(l, 'N/A') for l in y_true_labels], 
                                  palette=palette, s=50, alpha=0.9, edgecolor='w', linewidth=0.5)

    # 凡例をプロットの外側に移動させる
    legend_handles, legend_labels = legend_plot.get_legend_handles_labels()
    legend_plot.legend(legend_handles, legend_labels, title="凡例", bbox_to_anchor=(1.05, 1), loc='upper left')

    if current_clusters > 0 and len(row_ind) > 0:
        cluster_centers = model.get_cluster_centers()
        try:
            cluster_centers_2d = pca_visual_model.transform(cluster_centers)
            pred_idx_to_true_idx_map = {pred_i: true_i for pred_i, true_i in zip(row_ind, col_ind)}
            for pred_idx, center_2d in enumerate(cluster_centers_2d):
                true_idx = pred_idx_to_true_idx_map.get(pred_idx)
                if true_idx is not None:
                    color = palette[true_idx]
                    ax_contour.scatter(center_2d[0], center_2d[1], c=[color], marker='o', s=300, edgecolor='black', linewidth=2.0, zorder=10)
                else:
                    ax_contour.scatter(center_2d[0], center_2d[1], c='grey', marker='X', s=250, edgecolor='white', linewidth=1.5, zorder=10)
        except Exception as e:
            print(f"Epoch {epoch}: クラスタ中心のPCA変換/描画中にエラー: {e}")

    ax_contour.set_title(f'クラスタリング結果 (PCA 2D)', fontsize=16)
    ax_contour.set_xlabel('主成分1'), ax_contour.set_ylabel('主成分2')
    ax_contour.set_aspect('equal', adjustable='box')

    # --- 中央パネル: 情報表示 ---
    ax_info.axis('off')
    ax_info.set_title('学習状況サマリ', fontsize=16)

    # クラス別Recallの文字列を生成
    recall_lines = [f"  - {name:<4s}: {recall:.4f}" for name, recall in zip(label_encoder_classes, recalls)]
    recall_text = "--- Class Recall ---\n" + "\n".join(recall_lines)

    # 全ての情報を一つの文字列に結合
    info_text = (
        f"Epoch: {epoch}\n"
        f"Clusters (M): {current_clusters}\n\n"
        f"Balanced Accuracy: {bacc:.4f}\n"
        f"Mapped Accuracy: {acc:.4f}\n"
        f"Adjusted Rand Index: {ari:.4f}\n\n"
        f"{recall_text}"  # ここでRecall情報を結合
    )
    
    # 結合した文字列を一度に描画し、位置とフォントサイズを調整
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes, fontsize=11,
                 verticalalignment='top', horizontalalignment='left',
                 bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.9),
                 family='monospace') # 等幅フォントで見やすくする

    # --- 右パネル: 混同行列 ---
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=label_encoder_classes, yticklabels=label_encoder_classes, 
                annot_kws={"size": 10}, ax=ax_cm, cbar=False)
    ax_cm.set_title('混同行列 (マッピング後)', fontsize=16)
    ax_cm.set_xlabel("予測ラベル (マッピング後)"), ax_cm.set_ylabel("真のラベル")
    plt.setp(ax_cm.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # constrained_layoutを使用するため、tight_layoutは不要
    plot_filepath = save_path / f"energy_contour_epoch_{epoch:04d}.png"
    plt.savefig(plot_filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)

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

    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★★★ 改善点：前処理済みデータのキャッシュ機能 ★★★
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

    # --- 1. データ準備 (キャッシュの確認と読み込み/生成) ---
    pca_dims_to_test = [15, 20, 25]
    preprocessed_data_cache_file = Path("./preprocessed_prmsl_data.pkl")

    if preprocessed_data_cache_file.exists():
        print(f"\n--- 1. データ準備 ---")
        print(f"✅ 発見されたキャッシュファイル '{preprocessed_data_cache_file}' から前処理済みデータを読み込みます...")
        try:
            with open(preprocessed_data_cache_file, 'rb') as f:
                data_cache = pd.read_pickle(f)
            
            # キャッシュからデータを展開
            precalculated_pca_data = data_cache['precalculated_pca_data']
            y_true_labels = data_cache['y_true_labels']
            sin_time = data_cache['sin_time']
            cos_time = data_cache['cos_time']
            label_encoder = data_cache['label_encoder']
            target_names_map = data_cache['target_names_map']
            n_samples = data_cache['n_samples']
            n_true_clusters = len(label_encoder.classes_)
            
            print("✅ データの読み込みが完了しました。")
            print(f"   対象サンプル数: {n_samples}")

        except Exception as e:
            print(f"エラー: キャッシュファイルの読み込みに失敗しました: {e}")
            print("キャッシュを無視して、データの再生成を試みます。")
            # エラーが発生した場合、ファイルが存在しない場合と同じ処理へ
            preprocessed_data_cache_file.unlink() # 壊れたファイルを削除
            # この後、elseブロックの処理が実行されるように見せかけるが、実際は再実行が必要
            # ここではシンプルに終了する
            sys.exit(1)

    else:
        print(f"\n--- 1. データ準備 (キャッシュファイルが見つからないため、新規生成します) ---")
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
        print(f"✅ 学習・評価に使用する全データを、指定の15ラベルを持つもののみにフィルタリングしました。")
        print(f"   最終的な対象サンプル数: {n_samples}")
        
        y_true_str = ds_filtered['label'].values
        label_encoder = LabelEncoder().fit(target_labels)
        y_true_labels = label_encoder.transform(y_true_str)
        target_names_map = {i: label for i, label in enumerate(label_encoder.classes_)}
        n_true_clusters = len(target_labels)
        
        msl_data = ds_filtered['msl'].values
        msl_flat = msl_data.reshape(n_samples, -1)
        sin_time = ds_filtered['sin_time'].values.reshape(-1, 1)
        cos_time = ds_filtered['cos_time'].values.reshape(-1, 1)

        print("\n--- 前処理: データを[0,1]にスケーリングしてからPCAを適用します ---")
        scaler_for_pca = MinMaxScaler()
        msl_flat_scaled = scaler_for_pca.fit_transform(msl_flat)
        print("✅ 海面気圧データをスケーリングしました。")
        del msl_flat

        print("\n--- スケーリング済みデータで複数パターンのPCAを事前計算します... ---")
        precalculated_pca_data = {}
        for n_dim in pca_dims_to_test:
            print(f"  ... PCA (n={n_dim}) を計算中 ...")
            pca_model = PCA(n_components=n_dim, random_state=GLOBAL_SEED)
            precalculated_pca_data[n_dim] = pca_model.fit_transform(msl_flat_scaled)
        print("✅ 全パターンのPCA計算が完了しました。")
        del msl_flat_scaled

        # 生成したデータをキャッシュに保存
        print(f"\n✅ 前処理済みデータをキャッシュファイル '{preprocessed_data_cache_file}' に保存します...")
        data_to_cache = {
            'precalculated_pca_data': precalculated_pca_data,
            'y_true_labels': y_true_labels,
            'sin_time': sin_time,
            'cos_time': cos_time,
            'label_encoder': label_encoder,
            'target_names_map': target_names_map,
            'n_samples': n_samples
        }
        with open(preprocessed_data_cache_file, 'wb') as f:
            pd.to_pickle(data_to_cache, f)
        print("✅ 保存が完了しました。")

    # --- 2. ランダムサーチの設定 ---
    print("\n--- 2. ランダムサーチ設定 ---")
    param_dist = {
        'pca_n_components': pca_dims_to_test, 'include_time_features': [True, False],
        'gamma': (0.01, 3.0), 'beta': (0.001, 1.0),
        'learning_rate_W': (0.001, 0.1), 'learning_rate_lambda': (0.001, 0.1),
        'learning_rate_Z': (0.001, 0.1), 'initial_lambda_scalar': (0.001, 1.0),
        'initial_lambda_vector_val': (0.001, 1.0), 'initial_lambda_crossterm_val': (-0.5, 0.5),
        'initial_Z_val': (0.01, 1.0), 'initial_Z_new_cluster': (0.01, 1.0),
        'theta_new': (0.001, 1.0), 'Z_death_threshold': (0.01, 0.1),
        'death_patience_steps': [n_samples // 10, n_samples // 4, n_samples // 2, n_samples],
        'num_epochs': [1000], 'activation_type': ['circular', 'elliptical']
    }
    
    N_TRIALS = 500
    BACC_GOAL = 0.7 # ★目標をバランス化正解率に変更

    fixed_params_for_acs = {
        'max_clusters': 150, 'initial_clusters': 1,
        'lambda_min_val': 1e-7, 'bounds_W': (0, 1)
    }
    print(f"ランダムサーチ最大試行回数: {N_TRIALS}")
    print(f"早期終了の目標精度 (Balanced Accuracy): {BACC_GOAL}")

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
                                          pca_data_dict=precalculated_pca_data, y_data=y_true_labels,
                                          sin_time_data=sin_time, cos_time_data=cos_time,
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
                # ★改善: バランス化正解率で最良モデルを更新
                if best_result is None or result['balanced_accuracy_mapped'] > best_result['balanced_accuracy_mapped']:
                    best_result = result
            # ★改善: バランス化正解率で早期終了を判断
            if best_result and best_result.get('balanced_accuracy_mapped', 0) >= BACC_GOAL:
                print(f"\n✅ 目標精度達成 (BAcc >= {BACC_GOAL})！ トライアル {result['trial_count_from_worker']} でサーチを打ち切ります。")
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
            row.update({'ari': res['ari'], 
                        'accuracy_mapped': res['accuracy_mapped'],
                        'balanced_accuracy_mapped': res['balanced_accuracy_mapped'], # ★追加
                        'final_clusters': res['final_clusters'], 'best_epoch': res['best_epoch'],
                        'duration_seconds': res['duration_seconds'], 'error_present': bool(res['error_traceback']),
                        'trial_id': res['trial_count_from_worker']})
            df_data.append(row)
        
        # ★改善: バランス化正解率を第一のソートキーにする
        results_df = pd.DataFrame(df_data).sort_values(by=['balanced_accuracy_mapped', 'accuracy_mapped', 'ari'], ascending=False)
        df_path = output_dir / f"random_search_all_results_{timestamp}.csv"
        results_df.to_csv(df_path, index=False, encoding='utf-8-sig')
        print(f"全試行結果をCSVファイルに保存しました: {df_path.resolve()}")

    if best_result is None:
        print("エラー: 全ての試行でエラーが発生、または有効な結果が得られませんでした。")
        sys.exit(1)
        
    print(f"\n--- 最良パラメータ ({'目標達成' if goal_achieved else '探索終了時点'}) ---")
    best_params_combo_dict = best_result['params_combo']
    print(f"Trial ID: {best_result['trial_count_from_worker']}")
    # ★改善: バランス化正解率をメインに表示
    print(f"最高バランス化精度 (Mapped): {best_result['balanced_accuracy_mapped']:.4f} (at Epoch {best_result['best_epoch']})")
    print(f"その時のARI: {best_result['ari']:.4f}")
    print(f"その時のAccuracy: {best_result['accuracy_mapped']:.4f}")
    print(f"その時のクラスタ数: {best_result['final_clusters']}")
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
        **fixed_params_for_acs, **params_for_init, 
        'n_features': X_scaled_data_best.shape[1],
        'random_state': best_result['acs_random_state_used']
    }
    
    best_model_instance = ACS(**best_model_full_params_for_refit)
    
    # --- 6. 結果の可視化と保存 ---
    print("\n--- 6. 結果の可視化 ---")
    pca_visual = PCA(n_components=2, random_state=GLOBAL_SEED)
    X_pca_visual = pca_visual.fit_transform(X_scaled_data_best)
    
    energy_plot_dir = output_dir / f"energy_plots_trial_{best_result['trial_count_from_worker']}"
    os.makedirs(energy_plot_dir, exist_ok=True)
    print(f"✅ エネルギー等高線プロットを {energy_plot_dir.resolve()} に保存します。")

    print(f"\n--- 最良モデルでの再学習とエネルギー遷移の可視化 (計 {epochs_for_refit} エポック) ---")
    for epoch in tqdm(range(1, epochs_for_refit + 1), desc="再学習とプロット生成"):
        best_model_instance.fit(X_scaled_data_best, epochs=1)
        plot_energy_contour_for_epoch(
            model=best_model_instance, epoch=epoch, save_path=energy_plot_dir,
            X_scaled_data_for_eval=X_scaled_data_best, X_pca_visual=X_pca_visual,
            y_true_labels=y_true_labels, target_names_map=target_names_map,
            label_encoder_classes=label_encoder.classes_, pca_visual_model=pca_visual
        )
    print("\n✅ 再学習とエネルギー遷移プロットの生成が完了しました。")

    # --- 最終状態での評価とプロット ---
    final_predicted_labels = best_model_instance.predict(X_scaled_data_best) if best_model_instance.M > 0 else np.full(n_samples, -1)
    final_ari = adjusted_rand_score(y_true_labels, final_predicted_labels)
    final_clusters = best_model_instance.M
    final_accuracy, final_bacc = 0.0, 0.0
    if final_clusters > 0:
        final_contingency = pd.crosstab(final_predicted_labels, y_true_labels)
        final_row_ind, final_col_ind = linear_sum_assignment(-final_contingency.values)
        final_accuracy = final_contingency.values[final_row_ind, final_col_ind].sum() / n_samples
        
        mapped_pred_labels = np.full_like(y_true_labels, -1)
        pred_idx_to_true_idx_map = {final_contingency.index[pred_i]: true_i for pred_i, true_i in zip(final_row_ind, final_col_ind)}
        for pred_label_val, true_label_idx in pred_idx_to_true_idx_map.items():
            mapped_pred_labels[final_predicted_labels == pred_label_val] = true_label_idx
        final_bacc = balanced_accuracy_score(y_true_labels, mapped_pred_labels)
    
    print("\n--- 最終評価結果 (After Refit) ---")
    print(f"Balanced Accuracy (Mapped): {final_bacc:.4f}")
    print(f"Accuracy (Mapped): {final_accuracy:.4f}")
    print(f"ARI: {final_ari:.4f}")
    print(f"最終クラスタ数: {final_clusters}")
    if final_clusters > 0:
        print("\n--- クラス別最終精度 (Recall) ---")
        final_recalls = recall_score(y_true_labels, mapped_pred_labels, average=None, labels=np.arange(n_true_clusters), zero_division=0)
        for i, class_name in enumerate(label_encoder.classes_):
            print(f"  - {class_name:<4s}: {final_recalls[i]:.4f}")

    # 6a. 混同行列
    if final_clusters > 0:
        cm = confusion_matrix(y_true_labels, mapped_pred_labels, labels=np.arange(n_true_clusters))
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, annot_kws={"size": 10})
        plt.title(f"混同行列 (ACS - 再学習後)\nBAcc: {final_bacc:.4f}, Acc: {final_accuracy:.4f}, ARI: {final_ari:.4f}", fontsize=14)
        plt.xlabel("予測ラベル (マッピング後)"), plt.ylabel("真のラベル")
        plt.savefig(output_dir / f"confusion_matrix_{timestamp}.png", dpi=300, bbox_inches='tight'), plt.close()
        print(f"✅ 混同行列を保存しました。")
    else:
        print("クラスタが形成されなかったため、混同行列はスキップします。")
    
    # 6b. 学習履歴プロット
    if best_result and best_result['history']:
        history = best_result['history']
        epochs = [h['epoch'] for h in history]
        history_bacc = [h['balanced_accuracy_mapped'] for h in history]
        history_acc = [h['accuracy_mapped'] for h in history]
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        ax1.plot(epochs, history_bacc, 's-', markersize=4, color='tab:blue', label='Balanced Accuracy (Mapped)')
        ax1.set_xlabel('エポック数')
        ax1.set_ylabel('Balanced Accuracy', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        ax2 = ax1.twinx()
        ax2.plot(epochs, history_acc, 'o-', markersize=4, color='tab:green', alpha=0.6, label='Accuracy (Mapped)')
        ax2.set_ylabel('Accuracy', color='tab:green')
        ax2.tick_params(axis='y', labelcolor='tab:green')

        best_epoch_num = best_result.get('best_epoch')
        if best_epoch_num:
            best_bacc_val = best_result.get('balanced_accuracy_mapped')
            ax1.axvline(x=best_epoch_num, color='red', linestyle='--', 
                       label=f'Best Epoch ({best_epoch_num}) by BAcc: {best_bacc_val:.3f}')
        
        ax1.set_title('最良モデルの学習推移 (ランダムサーチ時)')
        ax1.grid(True, linestyle='--')
        fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
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