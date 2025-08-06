# ==============================================================================
# ライブラリのインポート
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import datetime
import japanize_matplotlib # 日本語化対応
from pathlib import Path

# ==============================================================================
# グローバル設定
# ==============================================================================
# 実行日時をタイムスタンプとして取得
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# 結果CSVファイルが保存されているディレクトリ
# このスクリプトは src/ACS/prmsl/ に配置されることを想定し、
# 統合対象のCSVはその中の all_result_prmsl_acs_random_search_csv にある
# ★修正点: BASE_DIR をスクリプト自身の親ディレクトリに変更
BASE_DIR = Path(__file__).parent 
CSV_DIR = BASE_DIR / "all_result_prmsl_acs_random_search_csv"

# 分析結果を保存するディレクトリ
OUTPUT_DIR = Path("./analysis_results") / f"analysis_{timestamp}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# ヘルパー関数
# ==============================================================================
def load_and_concatenate_csvs(directory_path):
    """
    指定されたディレクトリから 'random_search_all_results_*.csv' ファイルをすべて読み込み、
    一つのDataFrameに結合して返す。
    """
    # glob.globは文字列パスを期待するため、Pathオブジェクトを文字列に変換
    all_files = glob.glob(os.path.join(str(directory_path), 'random_search_all_results_*.csv'))
    
    if not all_files:
        print(f"エラー: 指定されたディレクトリ '{directory_path}' に 'random_search_all_results_*.csv' ファイルが見つかりません。")
        return pd.DataFrame() # 空のDataFrameを返す

    df_list = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            df_list.append(df)
            print(f"  - '{os.path.basename(f)}' を読み込みました。")
        except Exception as e:
            print(f"警告: ファイル '{os.path.basename(f)}' の読み込み中にエラーが発生しました: {e}")
            continue
    
    if not df_list:
        print("エラー: 有効なCSVファイルが一つも読み込めませんでした。")
        return pd.DataFrame()

    combined_df = pd.concat(df_list, ignore_index=True)
    
    # 重複する試行を削除 (同じパラメータセット、同じARI/Accuracyの試行は重複と見なす)
    initial_rows = len(combined_df)
    
    # 除外するカラムが存在するかチェックし、存在するカラムのみをsubsetに含める
    # 'duration_seconds', 'trial_id', 'error_present' が常に存在すると仮定できないため
    cols_to_exclude_from_subset = ['duration_seconds', 'trial_id', 'error_present']
    subset_cols = [col for col in combined_df.columns if col not in cols_to_exclude_from_subset]

    combined_df.drop_duplicates(subset=subset_cols, inplace=True)
    print(f"重複する試行を削除しました。初期データ数: {initial_rows}, 削除後データ数: {len(combined_df)}")
    
    return combined_df

def plot_parameter_distribution_and_accuracy(df, param_name, output_dir, is_categorical=False):
    """
    単一のパラメータについて、その分布とaccuracy_mappedとの関係をプロットする。
    """
    plt.figure(figsize=(12, 6))

    if is_categorical:
        # カテゴリカルデータの場合は箱ひげ図
        # x軸のラベルが長すぎると重なる可能性があるので、調整
        sns.boxplot(x=param_name, y='accuracy_mapped', data=df)
        plt.title(f'{param_name} と Accuracy (Mapped) の関係 (カテゴリカル)', fontsize=16)
        plt.xlabel(param_name, fontsize=12)
        plt.ylabel('Accuracy (Mapped)', fontsize=12)
        if len(df[param_name].unique()) > 5: # カテゴリが多い場合、ラベルを回転
            plt.xticks(rotation=45, ha='right')
    else:
        # 数値データの場合は散布図と回帰直線
        sns.scatterplot(x=param_name, y='accuracy_mapped', data=df, alpha=0.7)
        sns.regplot(x=param_name, y='accuracy_mapped', data=df, scatter=False, color='red', line_kws={"linestyle": "--", "alpha": 0.5})
        plt.title(f'{param_name} と Accuracy (Mapped) の関係 (数値)', fontsize=16)
        plt.xlabel(param_name, fontsize=12)
        plt.ylabel('Accuracy (Mapped)', fontsize=12)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / f'{param_name}_vs_accuracy.png', dpi=300)
    plt.close()
    print(f"  - '{param_name}_vs_accuracy.png' を保存しました。")

def plot_correlation_heatmap(df, output_dir):
    """
    数値パラメータ間の相関ヒートマップを生成する。
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # 精度関連の列と、ランダムサーチ対象ではない列を除外
    exclude_cols = ['ari', 'accuracy_mapped', 'final_clusters', 'best_epoch', 'duration_seconds', 'error_present', 'trial_id']
    analysis_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if not analysis_cols:
        print("警告: 相関ヒートマップを生成するための数値パラメータが見つかりません。")
        return

    # 'accuracy_mapped'がanalysis_colsに含まれていない場合に追加
    if 'accuracy_mapped' not in analysis_cols:
        analysis_cols.append('accuracy_mapped')

    corr_matrix = df[analysis_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('パラメータとAccuracyの相関ヒートマップ', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_correlation_heatmap.png', dpi=300)
    plt.close()
    print(f"  - 'parameter_correlation_heatmap.png' を保存しました。")

def analyze_top_and_bottom_performers(df, output_dir, top_percent=10, bottom_percent=10):
    """
    上位・下位X%のパフォーマンスを持つ試行のパラメータ傾向を分析する。
    """
    total_trials = len(df)
    if total_trials == 0: return

    # Accuracyでソート
    df_sorted = df.sort_values(by='accuracy_mapped', ascending=False).reset_index(drop=True)

    # 上位X%
    num_top = max(1, int(total_trials * top_percent / 100))
    top_performers = df_sorted.head(num_top)
    
    # 下位X%
    num_bottom = max(1, int(total_trials * bottom_percent / 100))
    bottom_performers = df_sorted.tail(num_bottom)

    print(f"\n--- 上位 {top_percent}% の試行 ({num_top}件) のパラメータ傾向 ---")
    # 既存のパラメータリストを参考に、分析対象のパラメータを定義
    # ただし、実際にDataFrameに存在するかはチェックする
    candidate_params_for_summary = [
        'pca_n_components', 'include_time_features', 'gamma', 'beta',
        'learning_rate_W', 'learning_rate_lambda', 'learning_rate_Z',
        'initial_lambda_scalar', 'initial_lambda_vector_val', 'initial_lambda_crossterm_val',
        'initial_Z_val', 'initial_Z_new_cluster', 'theta_new',
        'Z_death_threshold', 'death_patience_steps', 'activation_type'
    ]
    
    for col in candidate_params_for_summary:
        if col in df.columns: # DataFrameに存在するカラムのみ処理
            if df[col].dtype == 'object' or df[col].dtype == 'bool': # カテゴリカル
                print(f"  - {col}:")
                # top_performersが空の場合のエラー回避
                if not top_performers.empty:
                    print(top_performers[col].value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))
                else:
                    print("    (データなし)")
            elif df[col].dtype in ['float64', 'int64']: # 数値
                print(f"  - {col}: 平均={top_performers[col].mean():.4f}, 中央値={top_performers[col].median():.4f}, 範囲=[{top_performers[col].min():.4f}, {top_performers[col].max():.4f}]")
    
    print(f"\n--- 下位 {bottom_percent}% の試行 ({num_bottom}件) のパラメータ傾向 ---")
    for col in candidate_params_for_summary:
        if col in df.columns: # DataFrameに存在するカラムのみ処理
            if df[col].dtype == 'object' or df[col].dtype == 'bool':
                print(f"  - {col}:")
                # bottom_performersが空の場合のエラー回避
                if not bottom_performers.empty:
                    print(bottom_performers[col].value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))
                else:
                    print("    (データなし)")
            elif df[col].dtype in ['float64', 'int64']:
                print(f"  - {col}: 平均={bottom_performers[col].mean():.4f}, 中央値={bottom_performers[col].median():.4f}, 範囲=[{bottom_performers[col].min():.4f}, {bottom_performers[col].max():.4f}]")

    # 例: 特定のパラメータ組み合わせによるヒートマップ (カテゴリカルパラメータの組み合わせ)
    # pca_n_components と include_time_features の組み合わせ
    if 'pca_n_components' in df.columns and 'include_time_features' in df.columns:
        # 型が適切でない場合があるので、型変換を試みる
        # pivot_tableのindexやcolumnsにする前に、型を文字列に統一
        temp_df_pca_time = df.copy() # 元のdfを変更しないようにコピー
        temp_df_pca_time['pca_n_components'] = temp_df_pca_time['pca_n_components'].astype(str)
        temp_df_pca_time['include_time_features'] = temp_df_pca_time['include_time_features'].astype(str)
        
        pivot_table = temp_df_pca_time.pivot_table(values='accuracy_mapped', index='pca_n_components', columns='include_time_features', aggfunc='mean')
        if not pivot_table.empty: # pivot_tableが空でないことを確認
            plt.figure(figsize=(10, 7))
            sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt=".4f")
            plt.title('PCA成分数と時間特徴量による平均Accuracy', fontsize=16)
            plt.xlabel('時間特徴量を含む'), plt.ylabel('PCA成分数')
            plt.tight_layout()
            plt.savefig(output_dir / 'pca_time_features_heatmap.png', dpi=300)
            plt.close()
            print(f"  - 'pca_time_features_heatmap.png' を保存しました。")
        else:
            print("  - 警告: 'pca_time_features_heatmap.png' の生成に必要なデータが不足しています。")

    # activation_type と pca_n_components の組み合わせ
    if 'activation_type' in df.columns and 'pca_n_components' in df.columns:
        temp_df_act_pca = df.copy() # 元のdfを変更しないようにコピー
        temp_df_act_pca['pca_n_components'] = temp_df_act_pca['pca_n_components'].astype(str) # 型変換
        pivot_table = temp_df_act_pca.pivot_table(values='accuracy_mapped', index='pca_n_components', columns='activation_type', aggfunc='mean')
        if not pivot_table.empty: # pivot_tableが空でないことを確認
            plt.figure(figsize=(10, 7))
            sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt=".4f")
            plt.title('活性化タイプとPCA成分数による平均Accuracy', fontsize=16)
            plt.xlabel('活性化タイプ'), plt.ylabel('PCA成分数')
            plt.tight_layout()
            plt.savefig(output_dir / 'activation_pca_heatmap.png', dpi=300)
            plt.close()
            print(f"  - 'activation_pca_heatmap.png' を保存しました。")
        else:
            print("  - 警告: 'activation_pca_heatmap.png' の生成に必要なデータが不足しています。")
        
    # activation_typeごとの数値パラメータ分布
    if 'activation_type' in df.columns:
        numeric_params_to_compare = ['gamma', 'beta', 'learning_rate_W', 'learning_rate_lambda', 'learning_rate_Z']
        for param in numeric_params_to_compare:
            if param in df.columns:
                # activation_type が存在し、かつデータが空でないことを確認
                if not df['activation_type'].empty: 
                    plt.figure(figsize=(8, 5))
                    sns.boxplot(x='activation_type', y=param, data=df)
                    plt.title(f'活性化タイプごとの {param} 分布', fontsize=16)
                    plt.xlabel('活性化タイプ'), plt.ylabel(param)
                    plt.grid(True, linestyle='--', alpha=0.6)
                    plt.tight_layout()
                    plt.savefig(output_dir / f'activation_type_vs_{param}.png', dpi=300)
                    plt.close()
                    print(f"  - 'activation_type_vs_{param}.png' を保存しました。")
                else:
                    print(f"  - 警告: 'activation_type_vs_{param}.png' の生成に必要なデータが不足しています。")

# ==============================================================================
# メイン処理
# ==============================================================================
def main():
    print("=" * 60)
    print("ACSモデル ランダムサーチ結果 統合分析スクリプト")
    print("=" * 60)
    print(f"CSVファイル検索ディレクトリ: {CSV_DIR.resolve()}")
    print(f"分析結果出力ディレクトリ: {OUTPUT_DIR.resolve()}")
    print(f"実行開始時刻: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # --- 1. CSVファイルの読み込みと統合 ---
    print("\n--- 1. 全CSVファイルの読み込みと統合 ---")
    combined_df = load_and_concatenate_csvs(CSV_DIR)

    if combined_df.empty:
        print("\n分析を続行できません。有効なデータがありません。")
        return

    print(f"\n✅ 全てのCSVファイルを統合しました。総試行数 (重複削除後): {len(combined_df)}")

    # --- 2. 基本統計とトップパフォーマンスの表示 ---
    print("\n--- 2. 基本統計とトップパフォーマンス ---")
    print("\n--- 全試行のAccuracy (Mapped) 統計 ---")
    print(combined_df['accuracy_mapped'].describe())

    print("\n--- 最高精度を達成した試行 ---")
    if not combined_df.empty:
        best_trial = combined_df.loc[combined_df['accuracy_mapped'].idxmax()]
        # best_trialを表示する際も、存在するカラムのみを選択する
        cols_to_display_best_trial = [
            'accuracy_mapped', 'ari', 'final_clusters', 'best_epoch', 
            'pca_n_components', 'include_time_features', 'activation_type', 
            'gamma', 'beta', 'learning_rate_W', 'learning_rate_lambda', 
            'learning_rate_Z', 'initial_lambda_scalar', 'initial_lambda_vector_val',
            'initial_lambda_crossterm_val', 'initial_Z_val', 'initial_Z_new_cluster',
            'theta_new', 'Z_death_threshold', 'death_patience_steps'
        ]
        present_cols = [col for col in cols_to_display_best_trial if col in best_trial.index]
        print(best_trial[present_cols])
    else:
        print("  最高精度を特定するデータがありません。")
    
    # 全結果をCSVとして保存
    all_results_output_path = OUTPUT_DIR / "combined_random_search_results.csv"
    combined_df.to_csv(all_results_output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 統合された全結果を '{all_results_output_path.resolve()}' に保存しました。")

    # --- 3. パラメータと精度の相関分析と可視化 ---
    print("\n--- 3. パラメータと精度の関係分析 ---")

    # パラメータ列を特定
    # これらの列はsrc/ACS/prmsl/multi_prmsl_acs_random.pyのparam_distで定義されたもの
    parameters_to_analyze = [
        'pca_n_components', 'include_time_features', 'gamma', 'beta',
        'learning_rate_W', 'learning_rate_lambda', 'learning_rate_Z',
        'initial_lambda_scalar', 'initial_lambda_vector_val', 'initial_lambda_crossterm_val',
        'initial_Z_val', 'initial_Z_new_cluster', 'theta_new',
        'Z_death_threshold', 'death_patience_steps', 'activation_type'
    ]
    
    # 存在しないカラムを除外 (過去の実行ログで存在しない場合があるため)
    parameters_to_analyze = [p for p in parameters_to_analyze if p in combined_df.columns]
    
    # death_patience_steps は値が離散的（リストからの選択）なのでカテゴリカル的に扱うべき
    categorical_params = ['pca_n_components', 'include_time_features', 'activation_type', 'death_patience_steps']
    
    for param in parameters_to_analyze:
        plot_parameter_distribution_and_accuracy(combined_df, param, OUTPUT_DIR, is_categorical=(param in categorical_params))
    
    plot_correlation_heatmap(combined_df, OUTPUT_DIR)

    # --- 4. 高精度を達成するためのパラメータ傾向 / 精度を損なうパラメータ設定の特定 ---
    print("\n--- 4. 高精度/低精度試行のパラメータ傾向分析 ---")
    analyze_top_and_bottom_performers(combined_df, OUTPUT_DIR, top_percent=5, bottom_percent=5) # 上位/下位5%で分析

    print("\n--- 分析完了 ---")
    print(f"全ての分析結果とグラフは '{OUTPUT_DIR.resolve()}' に保存されています。")
    print("=" * 60)

if __name__ == '__main__':
    main()