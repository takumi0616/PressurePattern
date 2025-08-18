# ACSクラスをインポート (acs.py から)
import sys
from pathlib import Path

# ACSクラスのファイルパスに応じて適宜変更してください
ACS_MODULE_PATH = Path("/home/takumi/Develop/R_Pressure/ACS") # acs.py があるディレクトリ
if str(ACS_MODULE_PATH.resolve()) not in sys.path:
    sys.path.append(str(ACS_MODULE_PATH.resolve()))
try:
    from acs import ACS # acs.py から ACS クラスをインポート
    print(f"ACS class (acs.py) imported successfully from: {ACS_MODULE_PATH / 'acs.py'}")
except ImportError as e:
    print(f"Error importing ACS class from {ACS_MODULE_PATH / 'acs.py'}: {e}")
    print("Please ensure the path is correct and acs.py exists.")
    sys.exit(1)


# データセット、前処理、評価、プロットに必要なライブラリをインポート
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, adjusted_rand_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
# import itertools # ランダムサーチでは直接使わない
from scipy.optimize import linear_sum_assignment
import seaborn as sns
import japanize_matplotlib # Matplotlibの日本語表示改善
import os
import datetime
import random # ランダムサーチのために追加
from collections import Counter # データポイントのカウント用に追加

# --- 追加: グローバルなNumPyのシード値を固定 ---
np.random.seed(42)
random.seed(42) # Pythonのrandomモジュールのシードも固定

# --- 0. 出力ディレクトリとロギングの設定 ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir_base = "./result_test_acs_elliptical_random"
output_dir = f"{output_dir_base}_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
log_file_path = os.path.join(output_dir, f"random_search_log_elliptical_{timestamp}.txt")

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 標準出力と標準エラー出力をファイルにもリダイレクト
sys.stdout = Logger(log_file_path)
sys.stderr = sys.stdout # エラーも同じログファイルへ

print(f"ACSモデル (acs.py - 動的クラスタリング - 楕円形活性化) のIrisデータセットに対するランダムサーチと評価を開始します。")
print(f"目標: 最終クラスタ数 3, マッピング後Accuracy 0.953以上")
print(f"結果はディレクトリ '{output_dir}' に保存されます。")
print(f"ログファイル: {log_file_path}")
print(f"実行開始時刻: {timestamp}")


# 1. Irisデータセットの準備
iris = load_iris()
X_data_original = iris.data
y_true_labels = iris.target
n_samples, n_original_features = X_data_original.shape
n_true_clusters = len(np.unique(y_true_labels))
target_names = iris.target_names
# プロット用のカラーマップを取得
plot_colors = plt.cm.get_cmap('viridis', n_true_clusters)


print(f"データ準備完了: Irisデータセット - {n_samples} サンプル, {n_original_features} 特徴量, 真のクラスタ数: {n_true_clusters}")

scaler = MinMaxScaler()
X_scaled_data = scaler.fit_transform(X_data_original)

pca_visual = PCA(n_components=2, random_state=42)
X_pca_visual = pca_visual.fit_transform(X_scaled_data)
print("PCAによる2次元への削減完了 (可視化用)。ACSモデルへの入力はスケーリング後の元次元データを使用します。")

# 2. ランダムサーチのためのパラメータ範囲を設定 (楕円形活性化用)
param_dist = {
    'gamma': (0.1, 2.0),
    'beta': (0.001, 0.5),
    'learning_rate_W': (0.001, 0.1),
    'learning_rate_lambda': (0.0001, 0.01),
    'learning_rate_Z': (0.001, 0.1),
    'initial_lambda_vector_val': (0.001, 5.0),
    'initial_lambda_crossterm_val': (-0.5, 0.5), # 楕円形ではクロスタームのλKjも考慮
    'initial_Z_val': [0.3, 0.5, 0.7],
    'initial_Z_new_cluster': (0.05, 0.7),
    'theta_new': (0.005, 0.3),
    'death_patience_steps': [n_samples // 10, n_samples // 4, n_samples // 2, n_samples, n_samples * 2],
    'Z_death_threshold': (0.001, 0.2)
}

fixed_params = {
    'max_clusters': 10,
    'initial_clusters': 1,
    'n_features': n_original_features,
    'activation_type': 'elliptical', # ★楕円形に変更
    'lambda_min_val': 1e-7,
    'bounds_W': (0, 1),
    'random_state': None
}
num_epochs = 500 # ランダムサーチ時のエポック数（円形と同じに設定）
max_random_trials = 10000 # 最大試行回数（円形と同じに設定）
target_accuracy = 0.953 # 論文Table2のACS(ellipse) Irisの値 [cite: 171]
target_n_clusters = 3

print(f"使用する活性化関数タイプ: {fixed_params['activation_type']}")
print(f"クラスタ数の上限 (max_clusters): {fixed_params['max_clusters']}")
print(f"初期クラスタ数 (initial_clusters): {fixed_params['initial_clusters']}")
print(f"エポック数: {num_epochs}")
print(f"ランダムサーチ最大試行回数: {max_random_trials}")

best_ari_score = -1.0
best_accuracy_mapped = 0.0
best_params_combination = None
best_model_params_for_refit = None
best_final_n_clusters = 0
goal_achieved = False

all_trial_results = []

# 3. ランダムサーチの実行ループ
for trial_count in range(1, max_random_trials + 1):
    params_combo = {}
    for key, value_spec in param_dist.items():
        if isinstance(value_spec, tuple) and len(value_spec) == 2 and isinstance(value_spec[0], float):
            params_combo[key] = random.uniform(value_spec[0], value_spec[1])
        elif isinstance(value_spec, list):
            params_combo[key] = random.choice(value_spec)
        else:
            raise ValueError(f"Parameter spec for {key} is not valid.")

    current_random_state_for_acs = random.randint(0, 1000000)
    current_run_params = {**fixed_params, **params_combo, 'random_state': current_random_state_for_acs}

    print(f"\n--- トライアル {trial_count}/{max_random_trials} ---")
    print(f"パラメータ: { {k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in params_combo.items()} }")
    trial_start_time = datetime.datetime.now()

    try:
        acs_model_trial = ACS(**current_run_params)
        acs_model_trial.fit(X_scaled_data, epochs=num_epochs)

        final_n_clusters_trial = acs_model_trial.M
        predicted_labels_trial = np.full(n_samples, -1, dtype=int)
        if final_n_clusters_trial > 0:
            predicted_labels_trial = acs_model_trial.predict(X_scaled_data)

        current_ari_score = -1.0
        current_accuracy_mapped = 0.0
        valid_preds_mask = (predicted_labels_trial != -1)
        n_valid_preds = np.sum(valid_preds_mask)

        if n_valid_preds > 0:
            current_ari_score = adjusted_rand_score(y_true_labels[valid_preds_mask], predicted_labels_trial[valid_preds_mask])
            pred_unique_labels = np.unique(predicted_labels_trial[valid_preds_mask])
            n_predicted_clusters_effective = len(pred_unique_labels)

            if n_predicted_clusters_effective > 0:
                contingency_matrix_trial = np.zeros((n_true_clusters, n_predicted_clusters_effective), dtype=np.int64)
                pred_label_map = {label: i for i, label in enumerate(pred_unique_labels)}
                mapped_preds_for_cm = np.array([pred_label_map[l] for l in predicted_labels_trial[valid_preds_mask]])
                for i in range(n_valid_preds):
                    true_label = y_true_labels[valid_preds_mask][i]
                    pred_label_mapped = mapped_preds_for_cm[i]
                    contingency_matrix_trial[true_label, pred_label_mapped] += 1

                cost_matrix = -contingency_matrix_trial
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                mapped_accuracy_count = contingency_matrix_trial[row_ind, col_ind].sum()
                current_accuracy_mapped = mapped_accuracy_count / n_valid_preds

        trial_end_time = datetime.datetime.now()
        trial_duration = trial_end_time - trial_start_time
        print(f"最終クラスタ数: {final_n_clusters_trial}, ARI: {current_ari_score:.4f}, Accuracy(mapped): {current_accuracy_mapped:.4f}, 所要時間: {trial_duration}")

        all_trial_results.append({
            'params': params_combo,
            'ari': current_ari_score,
            'accuracy_mapped': current_accuracy_mapped,
            'final_clusters': final_n_clusters_trial,
            'duration_seconds': trial_duration.total_seconds(),
            'acs_random_state': current_random_state_for_acs
        })

        # 最良スコアの更新ロジック (Accuracy優先、次にARI)
        if current_accuracy_mapped > best_accuracy_mapped:
            best_accuracy_mapped = current_accuracy_mapped
            best_ari_score = current_ari_score
            best_params_combination = params_combo
            best_model_params_for_refit = current_run_params
            best_final_n_clusters = final_n_clusters_trial
            print(f"*** 新しい最良Accuracy発見: {best_accuracy_mapped:.4f} (ARI: {best_ari_score:.4f}, Clusters: {best_final_n_clusters}) ***")
        elif current_accuracy_mapped == best_accuracy_mapped and current_ari_score > best_ari_score:
            best_ari_score = current_ari_score # ARIも更新
            best_params_combination = params_combo
            best_model_params_for_refit = current_run_params
            best_final_n_clusters = final_n_clusters_trial
            print(f"*** Accuracy同点でARI改善: {best_accuracy_mapped:.4f} (ARI: {best_ari_score:.4f}, Clusters: {best_final_n_clusters}) ***")

        # 目標達成の確認
        if final_n_clusters_trial == target_n_clusters and current_accuracy_mapped >= target_accuracy:
            print(f"!!! 目標達成 !!! Trial: {trial_count}")
            print(f"パラメータ: {params_combo}")
            print(f"最終クラスタ数: {final_n_clusters_trial}, Accuracy(mapped): {current_accuracy_mapped:.4f}, ARI: {current_ari_score:.4f}")
            if best_params_combination is None or \
               current_accuracy_mapped > best_accuracy_mapped or \
               (current_accuracy_mapped == best_accuracy_mapped and current_ari_score > best_ari_score) :
                best_accuracy_mapped = current_accuracy_mapped
                best_ari_score = current_ari_score
                best_params_combination = params_combo
                best_model_params_for_refit = current_run_params
                best_final_n_clusters = final_n_clusters_trial
                print(f"*** 目標達成かつ最良モデル更新 ***")
            goal_achieved = True
            break

    except Exception as e:
        import traceback
        trial_end_time = datetime.datetime.now()
        trial_duration = trial_end_time - trial_start_time
        print(f"エラー発生: {e}, 所要時間: {trial_duration}")
        all_trial_results.append({
            'params': params_combo,
            'ari': -1.0,
            'accuracy_mapped': -1.0,
            'final_clusters': -1,
            'error': str(e),
            'duration_seconds': trial_duration.total_seconds(),
            'acs_random_state': current_random_state_for_acs
        })
        traceback.print_exc(file=sys.stderr)

try:
    import pandas as pd
    results_df = pd.DataFrame(all_trial_results)
    results_df = results_df.sort_values(by=['accuracy_mapped', 'ari', 'final_clusters'], ascending=[False, False, True]) # Accuracy優先
    df_path = os.path.join(output_dir, f"random_search_all_results_elliptical_{timestamp}.csv")
    results_df.to_csv(df_path, index=False, encoding='utf-8-sig')
    print(f"\n全試行結果を {df_path} に保存しました。")
except ImportError:
    print("\nPandasがインストールされていないため、全試行結果のCSV保存はスキップされました。")

print("\n--- ランダムサーチ完了 ---")

if goal_achieved:
    print("目標を達成するパラメータが見つかりました。")
elif best_params_combination is None:
    print("有効な結果が得られませんでした。パラメータ範囲やエポック数、ACSクラスの実装を見直してください。")
    sys.exit() # 有効な結果がない場合はここで終了
else:
    print("目標未達のまま最大試行回数に達しました。これまでの最良の結果を表示します。")

if best_params_combination is not None:
    print(f"最良のAccuracy (マッピング後): {best_accuracy_mapped:.4f}")
    print(f"その時のARI: {best_ari_score:.4f}")
    print(f"その時の最終クラスタ数: {best_final_n_clusters}")
    print(f"最良のパラメータの組み合わせ (ランダムサンプリングされた値): {best_params_combination}")
    print(f"その時のACSのrandom_state: {best_model_params_for_refit.get('random_state', 'N/A')}")

    print("\n最良パラメータでモデルを再学習・評価中...")
    best_model_instance = ACS(**best_model_params_for_refit)
    final_epochs_refit = num_epochs

    accuracy_history_refit = []
    ari_history_refit = []
    cluster_count_history_refit = []

    for epoch in range(final_epochs_refit):
        best_model_instance.fit(X_scaled_data, epochs=1)
        current_final_n_clusters = best_model_instance.M
        cluster_count_history_refit.append(current_final_n_clusters)

        current_predicted_labels = np.full(n_samples, -1, dtype=int)
        if current_final_n_clusters > 0:
            current_predicted_labels = best_model_instance.predict(X_scaled_data)

        valid_preds_mask_epoch = (current_predicted_labels != -1)
        n_valid_preds_epoch = np.sum(valid_preds_mask_epoch)
        current_ari = -1.0
        current_accuracy = 0.0

        if n_valid_preds_epoch > 0:
            current_ari = adjusted_rand_score(y_true_labels[valid_preds_mask_epoch], current_predicted_labels[valid_preds_mask_epoch])

            pred_unique_labels_epoch = np.unique(current_predicted_labels[valid_preds_mask_epoch])
            n_predicted_clusters_effective_epoch = len(pred_unique_labels_epoch)
            if n_predicted_clusters_effective_epoch > 0:
                contingency_matrix_epoch = np.zeros((n_true_clusters, n_predicted_clusters_effective_epoch), dtype=np.int64)
                pred_label_map_epoch = {label: i for i, label in enumerate(pred_unique_labels_epoch)}
                mapped_preds_for_cm_epoch = np.array([pred_label_map_epoch[l] for l in current_predicted_labels[valid_preds_mask_epoch]])
                for i in range(n_valid_preds_epoch):
                    contingency_matrix_epoch[y_true_labels[valid_preds_mask_epoch][i], mapped_preds_for_cm_epoch[i]] += 1

                cost_matrix_epoch = -contingency_matrix_epoch
                row_ind_e, col_ind_e = linear_sum_assignment(cost_matrix_epoch)
                current_accuracy = contingency_matrix_epoch[row_ind_e, col_ind_e].sum() / n_valid_preds_epoch

        ari_history_refit.append(current_ari)
        accuracy_history_refit.append(current_accuracy)
        print(f"再学習中 - エポック {epoch+1}/{final_epochs_refit} - クラスタ数: {current_final_n_clusters}, ARI: {current_ari:.4f}, Acc(mapped): {current_accuracy:.4f}")

    print("再学習完了。")

    final_predicted_labels_refit = best_model_instance.predict(X_scaled_data) if best_model_instance.M > 0 else np.full(n_samples, -1, dtype=int)
    final_learned_cluster_centers_original_space = best_model_instance.get_cluster_centers()
    final_cluster_centers_pca_visual = pca_visual.transform(final_learned_cluster_centers_original_space) if final_learned_cluster_centers_original_space.shape[0] > 0 else np.empty((0,2))

    final_valid_preds_mask = (final_predicted_labels_refit != -1)
    n_final_valid_preds = np.sum(final_valid_preds_mask)
    final_ari_refit = -1.0
    final_accuracy_refit = 0.0
    final_cm_refit = np.zeros((n_true_clusters, n_true_clusters))
    mapped_labels_for_plot_cm = np.full_like(final_predicted_labels_refit, -1)
    pred_to_true_map = {}

    if n_final_valid_preds > 0:
        final_ari_refit = adjusted_rand_score(y_true_labels[final_valid_preds_mask], final_predicted_labels_refit[final_valid_preds_mask])

        final_pred_unique_labels = np.unique(final_predicted_labels_refit[final_valid_preds_mask])
        final_n_predicted_clusters_effective = len(final_pred_unique_labels)

        if final_n_predicted_clusters_effective > 0:
            final_contingency_matrix = np.zeros((n_true_clusters, final_n_predicted_clusters_effective), dtype=np.int64)
            final_pred_label_map = {label: i for i, label in enumerate(final_pred_unique_labels)}
            final_mapped_preds_for_cm = np.array([final_pred_label_map[l] for l in final_predicted_labels_refit[final_valid_preds_mask]])
            for i in range(n_final_valid_preds):
                final_contingency_matrix[y_true_labels[final_valid_preds_mask][i], final_mapped_preds_for_cm[i]] += 1

            cost_matrix_final_refit = -final_contingency_matrix
            row_ind_f, col_ind_f = linear_sum_assignment(cost_matrix_final_refit)

            for true_idx, pred_mapped_idx in zip(row_ind_f, col_ind_f):
                 original_pred_label = final_pred_unique_labels[pred_mapped_idx]
                 mapped_labels_for_plot_cm[final_predicted_labels_refit == original_pred_label] = true_idx
                 pred_to_true_map[original_pred_label] = true_idx

            final_accuracy_refit = final_contingency_matrix[row_ind_f, col_ind_f].sum() / n_final_valid_preds

            valid_mapped_indices = (mapped_labels_for_plot_cm != -1) & final_valid_preds_mask
            if np.sum(valid_mapped_indices) > 0:
                 final_cm_refit = confusion_matrix(y_true_labels[valid_mapped_indices], mapped_labels_for_plot_cm[valid_mapped_indices], labels=list(range(n_true_clusters)))

    fig, ax1 = plt.subplots(figsize=(12, 7))
    color = 'tab:red'
    ax1.set_xlabel('エポック数', fontsize=12)
    ax1.set_ylabel('ARI (Adjusted Rand Index)', color=color, fontsize=12)
    ax1.plot(range(1, final_epochs_refit + 1), ari_history_refit, marker='o', linestyle='-', color=color, label='ARI')
    ax1.tick_params(axis='y', labelcolor=color, labelsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(-1.05, 1.05)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy (マッピング後)', color=color, fontsize=12)
    ax2.plot(range(1, final_epochs_refit + 1), accuracy_history_refit, marker='s', linestyle='--', color=color, label='Accuracy (Mapped)')
    ax2.tick_params(axis='y', labelcolor=color, labelsize=10)
    ax2.set_ylim(0, 1.05)

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))
    color = 'tab:green'
    ax3.set_ylabel('クラスタ数', color=color, fontsize=12)
    ax3.plot(range(1, final_epochs_refit + 1), cluster_count_history_refit, marker='^', linestyle=':', color=color, label='クラスタ数')
    ax3.tick_params(axis='y', labelcolor=color, labelsize=10)
    ax3.set_ylim(0, fixed_params['max_clusters'] + 1)

    fig.suptitle(f'最良モデルのエポック毎評価推移 (動的ACS - 楕円形)\n最終ARI: {final_ari_refit:.4f}, 最終Acc: {final_accuracy_refit:.4f}, 最終クラスタ数: {best_model_instance.M}', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='center right')

    metrics_plot_path = os.path.join(output_dir, f"metrics_vs_epoch_best_dynamic_elliptical_{timestamp}.png")
    plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig) # 明示的に閉じる
    print(f"エポック毎評価プロットを {metrics_plot_path} に保存しました。")

    print(f"\n混同行列 (Confusion Matrix) - 最良モデル・再学習後 (最終クラスタ数: {best_model_instance.M}):")
    fig_cm = plt.figure(figsize=(7, 6))
    sns.heatmap(final_cm_refit, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names, annot_kws={"size": 12})
    plt.xlabel("予測ラベル (マッピング後)", fontsize=12)
    plt.ylabel("真のラベル", fontsize=12)
    plt.title(f"混同行列 (最良モデル - 動的ACS - 楕円形)\nAccuracy: {final_accuracy_refit:.4f}, ARI: {final_ari_refit:.4f}", fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    cm_path = os.path.join(output_dir, f"confusion_matrix_best_dynamic_elliptical_{timestamp}.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close(fig_cm) # 明示的に閉じる
    print(f"混同行列の画像を {cm_path} に保存しました。")

    print(f"\n最終ARI (最良モデル・再学習後): {final_ari_refit:.4f}")
    print(f"最終Accuracy (最良モデル・マッピング後・再学習後): {final_accuracy_refit:.4f}")
    print(f"最終形成クラスタ数 (最良モデル・再学習後): {best_model_instance.M}")

    fig_pca_clusters = plt.figure(figsize=(16, 7))
    plt.subplot(1, 2, 1)
    for i_cluster_true in range(n_true_clusters):
        plt.scatter(X_pca_visual[y_true_labels == i_cluster_true, 0],
                    X_pca_visual[y_true_labels == i_cluster_true, 1],
                    color=plot_colors(i_cluster_true / (n_true_clusters-1 if n_true_clusters > 1 else 1)),
                    label=target_names[i_cluster_true], alpha=0.7, s=50)
    plt.title('Irisデータ (PCA) - 真のクラスタ', fontsize=14)
    plt.xlabel('PCA特徴量1 (可視化用)', fontsize=12)
    plt.ylabel('PCA特徴量2 (可視化用)', fontsize=12)
    plt.legend(title="真のクラス", fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    if np.sum(mapped_labels_for_plot_cm != -1) > 0:
        for true_cls_idx in range(n_true_clusters):
            mask = (mapped_labels_for_plot_cm == true_cls_idx)
            if np.sum(mask) > 0:
                 plt.scatter(X_pca_visual[mask, 0], X_pca_visual[mask, 1],
                            color=plot_colors(true_cls_idx / (n_true_clusters-1 if n_true_clusters > 1 else 1)),
                            label=f"Pred as {target_names[true_cls_idx]}", alpha=0.7, s=50)

    unmapped_mask = (mapped_labels_for_plot_cm == -1) & final_valid_preds_mask
    if np.sum(unmapped_mask) > 0:
        plt.scatter(X_pca_visual[unmapped_mask, 0], X_pca_visual[unmapped_mask, 1],
                    color='gray', label='Unmapped/Other', alpha=0.3, s=30)

    if final_cluster_centers_pca_visual.shape[0] > 0:
        for acs_center_idx in range(final_cluster_centers_pca_visual.shape[0]):
            center_color = 'black'
            center_label_suffix = f" (ACS Ctr {acs_center_idx})"
            if acs_center_idx in pred_to_true_map:
                true_class_idx_mapped_to = pred_to_true_map[acs_center_idx]
                center_color = plot_colors(true_class_idx_mapped_to / (n_true_clusters-1 if n_true_clusters > 1 else 1))
                center_label_suffix = f" (ACS Ctr {acs_center_idx} -> True {target_names[true_class_idx_mapped_to]})"
            plt.scatter(final_cluster_centers_pca_visual[acs_center_idx, 0],
                        final_cluster_centers_pca_visual[acs_center_idx, 1],
                        c=[center_color], marker='X', s=200, edgecolor='white',
                        label=f'Center{center_label_suffix}' if trial_count < 5 else None)

    plt.title(f'ACSクラスタリング (PCA) - 最良動的モデル (楕円形)\n最終ARI: {final_ari_refit:.3f}, Acc: {final_accuracy_refit:.3f}, Cls: {best_model_instance.M}', fontsize=14)
    plt.xlabel('PCA特徴量1 (可視化用)', fontsize=12)
    plt.ylabel('PCA特徴量2 (可視化用)', fontsize=12)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title="予測クラス / 中心", fontsize=8, loc="lower right")
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    pca_plot_path = os.path.join(output_dir, f"pca_clustering_best_dynamic_elliptical_{timestamp}.png")
    plt.savefig(pca_plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig_pca_clusters) # 明示的に閉じる
    print(f"PCAクラスタリングプロットを {pca_plot_path} に保存しました。")

    # --- エネルギー等高線図の生成 ---
    if best_model_instance.M > 0 and best_model_instance.fitted_:
        print("\nエネルギー関数の等高線図を生成中...")
        fig_energy, ax_energy = plt.subplots(figsize=(10, 8))

        # PCA空間の範囲でグリッドを生成
        x_min, x_max = X_pca_visual[:, 0].min() - 0.5, X_pca_visual[:, 0].max() + 0.5
        y_min, y_max = X_pca_visual[:, 1].min() - 0.5, X_pca_visual[:, 1].max() + 0.5

        grid_resolution = 50
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_resolution),
                             np.linspace(y_min, y_max, grid_resolution))

        E_grid = np.zeros(xx.shape)

        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                pca_point = np.array([[xx[i, j], yy[i, j]]])
                original_space_point = pca_visual.inverse_transform(pca_point)
                E_grid[i, j] = best_model_instance.calculate_energy_at_point(original_space_point.flatten())

        E_grid = np.nan_to_num(E_grid, nan=np.nanmax(E_grid[np.isfinite(E_grid)]),
                               posinf=np.nanmax(E_grid[np.isfinite(E_grid)]))

        num_contour_levels = 15
        contour_plot = ax_energy.contour(xx, yy, E_grid, levels=num_contour_levels, cmap='coolwarm', alpha=0.8, linewidths=0.7)

        for i_cluster_true in range(n_true_clusters):
            ax_energy.scatter(X_pca_visual[y_true_labels == i_cluster_true, 0],
                              X_pca_visual[y_true_labels == i_cluster_true, 1],
                              color=plot_colors(i_cluster_true / (n_true_clusters -1 if n_true_clusters > 1 else 1)),
                              label=target_names[i_cluster_true], alpha=0.5, s=30, zorder=2)

        if final_cluster_centers_pca_visual.shape[0] > 0:
            ax_energy.scatter(final_cluster_centers_pca_visual[:, 0],
                              final_cluster_centers_pca_visual[:, 1],
                              c='black', marker='o', s=100, edgecolor='white',
                              label='学習済みクラスタ中心 (W)', zorder=3)


        ax_energy.set_title(f'エネルギー関数 E の等高線図 (PCA空間 - 動的ACS - 楕円形)\n最終Acc: {final_accuracy_refit:.3f}, Cls: {best_model_instance.M}', fontsize=14)
        ax_energy.set_xlabel('PCA特徴量1 (可視化用)', fontsize=12)
        ax_energy.set_ylabel('PCA特徴量2 (可視化用)', fontsize=12)
        ax_energy.legend(fontsize=10, loc='upper right')
        ax_energy.grid(True, linestyle=':', alpha=0.5)
        ax_energy.set_xlim(x_min, x_max)
        ax_energy.set_ylim(y_min, y_max)

        energy_contour_path = os.path.join(output_dir, f"energy_contour_plot_elliptical_{timestamp}.png")
        plt.savefig(energy_contour_path, dpi=300, bbox_inches='tight')
        plt.close(fig_energy) # 明示的に閉じる
        print(f"エネルギー関数の等高線図を {energy_contour_path} に保存しました。")
    else:
        print("有効な学習済みモデルがないため、エネルギー等高線図の生成はスキップされました。")


    # --- 最良モデルの学習後パラメータとクラスタ割り当て情報 ---
    print("\n--- 最良モデルの学習後パラメータ ---")
    best_model_summary = f"""
使用されたパラメータ (固定値含む): {best_model_params_for_refit}

学習済みクラスタ中心 (W) (入力空間: {X_scaled_data.shape[1]}次元):
{final_learned_cluster_centers_original_space}

学習済み警戒パラメータ (lambda):
{best_model_instance.lambdas}

学習済み深さパラメータ (Z_j):
{best_model_instance.Z.flatten()}

学習済み非活性ステップ (inactive_steps):
{best_model_instance.inactive_steps.flatten()}

最終形成クラスタ数: {best_model_instance.M}
"""
    cluster_assignment_summary = "\n--- クラスタごとのデータポイント割り当て ---\n"
    if best_model_instance.M > 0 and n_final_valid_preds > 0:
        counts = Counter(final_predicted_labels_refit[final_valid_preds_mask])
        cluster_assignment_summary += f"総有効予測サンプル数: {n_final_valid_preds}\n"
        for i in range(best_model_instance.M):
            mapped_true_class_name = "N/A (Unmapped or Empty)"
            if i in pred_to_true_map:
                mapped_true_class_name = target_names[pred_to_true_map[i]]

            num_points = counts.get(i, 0)
            percentage = (num_points / n_samples) * 100 if n_samples > 0 else 0
            cluster_assignment_summary += f"  ACSクラスタ {i} (True Cls as: {mapped_true_class_name}): {num_points} サンプル ({percentage:.2f}%)\n"

        unassigned_count = counts.get(-1, 0)
        if unassigned_count > 0:
            percentage = (unassigned_count / n_samples) * 100 if n_samples > 0 else 0
            cluster_assignment_summary += f"  未分類 (-1): {unassigned_count} サンプル ({percentage:.2f}%)\n"
    else:
        cluster_assignment_summary += "  有効なクラスタまたは予測がありませんでした。\n"

    print(best_model_summary)
    print(cluster_assignment_summary)

    param_log_path = os.path.join(output_dir, f"best_model_params_dynamic_elliptical_{timestamp}.txt")
    with open(param_log_path, "w", encoding="utf-8") as f:
        f.write(f"実行開始時刻: {timestamp}\n")
        if goal_achieved:
            f.write("目標達成しました。\n")
        else:
            f.write("目標未達でした。\n")
        f.write(f"最良Accuracy (マッピング後): {best_accuracy_mapped:.4f}\n")
        f.write(f"その時のARI: {best_ari_score:.4f}\n")
        f.write(f"その時の最終クラスタ数: {best_final_n_clusters}\n")
        f.write(f"最良のパラメータの組み合わせ (ランダムサンプリングされた値): {best_params_combination}\n")
        f.write(f"その時のACSのrandom_state: {best_model_params_for_refit.get('random_state', 'N/A')}\n\n")
        f.write(best_model_summary)
        f.write(cluster_assignment_summary)
    print(f"最良モデルのパラメータ詳細とクラスタ割り当て情報を {param_log_path} に保存しました。")

print(f"\n全処理完了。実行終了時刻: {datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
print(f"結果はディレクトリ '{output_dir}' を確認してください。")

# Loggerのクローズ処理と標準出力の復元
if isinstance(sys.stdout, Logger):
    log_instance = sys.stdout
    sys.stdout = log_instance.terminal
    log_instance.log.close()
    if sys.stderr is log_instance:
        sys.stderr = sys.__stderr__