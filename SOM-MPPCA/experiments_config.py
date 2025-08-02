"""
実験設定を定義するファイル
main_v3.pyから読み込まれて使用される
"""

# --- 実行する実験のリスト ---
# name: 結果を保存するユニークなディレクトリ名
# data_key: 使用するデータ (DATA_FILESのキー)
# method: 'MPPCA' または 'PCA'
# その他: DEFAULT_PARAMSを上書きするパラメータを指定
EXPERIMENTS = [
    # --- smallデータセット, map_size=7x7 ---
    {'name': 'MPPCA_p15_q20_small_7x7', 'data_key': 'small', 'method': 'MPPCA', 'p_clusters': 15, 'q_latent_dim': 20, 'map_x': 7, 'map_y': 7},
    {'name': 'MPPCA_p6_q20_small_7x7',  'data_key': 'small', 'method': 'MPPCA', 'p_clusters': 6,  'q_latent_dim': 20, 'map_x': 7, 'map_y': 7},
    {'name': 'MPPCA_p4_q20_small_7x7',  'data_key': 'small', 'method': 'MPPCA', 'p_clusters': 4,  'q_latent_dim': 20, 'map_x': 7, 'map_y': 7},
    {'name': 'PCA_d20_small_7x7',       'data_key': 'small', 'method': 'PCA',   'n_components': 20, 'map_x': 7, 'map_y': 7},
    {'name': 'PCA_d80_small_7x7',       'data_key': 'small', 'method': 'PCA',   'n_components': 80, 'map_x': 7, 'map_y': 7},
    {'name': 'PCA_d120_small_7x7',      'data_key': 'small', 'method': 'PCA',   'n_components': 120, 'map_x': 7, 'map_y': 7},

    # --- smallデータセット, map_size=8x8 ---
    {'name': 'MPPCA_p15_q20_small_8x8', 'data_key': 'small', 'method': 'MPPCA', 'p_clusters': 15, 'q_latent_dim': 20, 'map_x': 8, 'map_y': 8},
    {'name': 'MPPCA_p6_q20_small_8x8',  'data_key': 'small', 'method': 'MPPCA', 'p_clusters': 6,  'q_latent_dim': 20, 'map_x': 8, 'map_y': 8},
    {'name': 'MPPCA_p4_q20_small_8x8',  'data_key': 'small', 'method': 'MPPCA', 'p_clusters': 4,  'q_latent_dim': 20, 'map_x': 8, 'map_y': 8},
    {'name': 'PCA_d20_small_8x8',       'data_key': 'small', 'method': 'PCA',   'n_components': 20, 'map_x': 8, 'map_y': 8},
    {'name': 'PCA_d80_small_8x8',       'data_key': 'small', 'method': 'PCA',   'n_components': 80, 'map_x': 8, 'map_y': 8},
    {'name': 'PCA_d120_small_8x8',      'data_key': 'small', 'method': 'PCA',   'n_components': 120, 'map_x': 8, 'map_y': 8},

    # --- smallデータセット, map_size=9x9 ---
    {'name': 'MPPCA_p15_q20_small_9x9', 'data_key': 'small', 'method': 'MPPCA', 'p_clusters': 15, 'q_latent_dim': 20, 'map_x': 9, 'map_y': 9},
    {'name': 'MPPCA_p6_q20_small_9x9',  'data_key': 'small', 'method': 'MPPCA', 'p_clusters': 6,  'q_latent_dim': 20, 'map_x': 9, 'map_y': 9},
    {'name': 'MPPCA_p4_q20_small_9x9',  'data_key': 'small', 'method': 'MPPCA', 'p_clusters': 4,  'q_latent_dim': 20, 'map_x': 9, 'map_y': 9},
    {'name': 'PCA_d20_small_9x9',       'data_key': 'small', 'method': 'PCA',   'n_components': 20, 'map_x': 9, 'map_y': 9},
    {'name': 'PCA_d80_small_9x9',       'data_key': 'small', 'method': 'PCA',   'n_components': 80, 'map_x': 9, 'map_y': 9},
    {'name': 'PCA_d120_small_9x9',      'data_key': 'small', 'method': 'PCA',   'n_components': 120, 'map_x': 9, 'map_y': 9},

    # --- smallデータセット, map_size=10x10 ---
    {'name': 'MPPCA_p15_q20_small_10x10', 'data_key': 'small', 'method': 'MPPCA', 'p_clusters': 15, 'q_latent_dim': 20, 'map_x': 10, 'map_y': 10},
    {'name': 'MPPCA_p6_q20_small_10x10',  'data_key': 'small', 'method': 'MPPCA', 'p_clusters': 6,  'q_latent_dim': 20, 'map_x': 10, 'map_y': 10},
    {'name': 'MPPCA_p4_q20_small_10x10',  'data_key': 'small', 'method': 'MPPCA', 'p_clusters': 4,  'q_latent_dim': 20, 'map_x': 10, 'map_y': 10},
    {'name': 'PCA_d20_small_10x10',       'data_key': 'small', 'method': 'PCA',   'n_components': 20, 'map_x': 10, 'map_y': 10},
    {'name': 'PCA_d80_small_10x10',       'data_key': 'small', 'method': 'PCA',   'n_components': 80, 'map_x': 10, 'map_y': 10},
    {'name': 'PCA_d120_small_10x10',      'data_key': 'small', 'method': 'PCA',   'n_components': 120, 'map_x': 10, 'map_y': 10},

    # --- normalデータセット, map_size=7x7 ---
    {'name': 'MPPCA_p15_q20_normal_7x7', 'data_key': 'normal', 'method': 'MPPCA', 'p_clusters': 15, 'q_latent_dim': 20, 'map_x': 7, 'map_y': 7},
    {'name': 'MPPCA_p6_q20_normal_7x7',  'data_key': 'normal', 'method': 'MPPCA', 'p_clusters': 6,  'q_latent_dim': 20, 'map_x': 7, 'map_y': 7},
    {'name': 'MPPCA_p4_q20_normal_7x7',  'data_key': 'normal', 'method': 'MPPCA', 'p_clusters': 4,  'q_latent_dim': 20, 'map_x': 7, 'map_y': 7},
    {'name': 'PCA_d20_normal_7x7',       'data_key': 'normal', 'method': 'PCA',   'n_components': 20, 'map_x': 7, 'map_y': 7},
    {'name': 'PCA_d80_normal_7x7',       'data_key': 'normal', 'method': 'PCA',   'n_components': 80, 'map_x': 7, 'map_y': 7},
    {'name': 'PCA_d120_normal_7x7',      'data_key': 'normal', 'method': 'PCA',   'n_components': 120, 'map_x': 7, 'map_y': 7},

    # --- normalデータセット, map_size=8x8 ---
    {'name': 'MPPCA_p15_q20_normal_8x8', 'data_key': 'normal', 'method': 'MPPCA', 'p_clusters': 15, 'q_latent_dim': 20, 'map_x': 8, 'map_y': 8},
    {'name': 'MPPCA_p6_q20_normal_8x8',  'data_key': 'normal', 'method': 'MPPCA', 'p_clusters': 6,  'q_latent_dim': 20, 'map_x': 8, 'map_y': 8},
    {'name': 'MPPCA_p4_q20_normal_8x8',  'data_key': 'normal', 'method': 'MPPCA', 'p_clusters': 4,  'q_latent_dim': 20, 'map_x': 8, 'map_y': 8},
    {'name': 'PCA_d20_normal_8x8',       'data_key': 'normal', 'method': 'PCA',   'n_components': 20, 'map_x': 8, 'map_y': 8},
    {'name': 'PCA_d80_normal_8x8',       'data_key': 'normal', 'method': 'PCA',   'n_components': 80, 'map_x': 8, 'map_y': 8},
    {'name': 'PCA_d120_normal_8x8',      'data_key': 'normal', 'method': 'PCA',   'n_components': 120, 'map_x': 8, 'map_y': 8},

    # --- normalデータセット, map_size=9x9 ---
    {'name': 'MPPCA_p15_q20_normal_9x9', 'data_key': 'normal', 'method': 'MPPCA', 'p_clusters': 15, 'q_latent_dim': 20, 'map_x': 9, 'map_y': 9},
    {'name': 'MPPCA_p6_q20_normal_9x9',  'data_key': 'normal', 'method': 'MPPCA', 'p_clusters': 6,  'q_latent_dim': 20, 'map_x': 9, 'map_y': 9},
    {'name': 'MPPCA_p4_q20_normal_9x9',  'data_key': 'normal', 'method': 'MPPCA', 'p_clusters': 4,  'q_latent_dim': 20, 'map_x': 9, 'map_y': 9},
    {'name': 'PCA_d20_normal_9x9',       'data_key': 'normal', 'method': 'PCA',   'n_components': 20, 'map_x': 9, 'map_y': 9},
    {'name': 'PCA_d80_normal_9x9',       'data_key': 'normal', 'method': 'PCA',   'n_components': 80, 'map_x': 9, 'map_y': 9},
    {'name': 'PCA_d120_normal_9x9',      'data_key': 'normal', 'method': 'PCA',   'n_components': 120, 'map_x': 9, 'map_y': 9},

    # --- normalデータセット, map_size=10x10 ---
    {'name': 'MPPCA_p15_q20_normal_10x10', 'data_key': 'normal', 'method': 'MPPCA', 'p_clusters': 15, 'q_latent_dim': 20, 'map_x': 10, 'map_y': 10},
    {'name': 'MPPCA_p6_q20_normal_10x10',  'data_key': 'normal', 'method': 'MPPCA', 'p_clusters': 6,  'q_latent_dim': 20, 'map_x': 10, 'map_y': 10},
    {'name': 'MPPCA_p4_q20_normal_10x10',  'data_key': 'normal', 'method': 'MPPCA', 'p_clusters': 4,  'q_latent_dim': 20, 'map_x': 10, 'map_y': 10},
    {'name': 'PCA_d20_normal_10x10',       'data_key': 'normal', 'method': 'PCA',   'n_components': 20, 'map_x': 10, 'map_y': 10},
    {'name': 'PCA_d80_normal_10x10',       'data_key': 'normal', 'method': 'PCA',   'n_components': 80, 'map_x': 10, 'map_y': 10},
    {'name': 'PCA_d120_normal_10x10',      'data_key': 'normal', 'method': 'PCA',   'n_components': 120, 'map_x': 10, 'map_y': 10},

    # --- largeデータセット, map_size=7x7 ---
    {'name': 'MPPCA_p15_q20_large_7x7', 'data_key': 'large', 'method': 'MPPCA', 'p_clusters': 15, 'q_latent_dim': 20, 'map_x': 7, 'map_y': 7},
    {'name': 'MPPCA_p6_q20_large_7x7',  'data_key': 'large', 'method': 'MPPCA', 'p_clusters': 6,  'q_latent_dim': 20, 'map_x': 7, 'map_y': 7},
    {'name': 'MPPCA_p4_q20_large_7x7',  'data_key': 'large', 'method': 'MPPCA', 'p_clusters': 4,  'q_latent_dim': 20, 'map_x': 7, 'map_y': 7},
    {'name': 'PCA_d20_large_7x7',       'data_key': 'large', 'method': 'PCA',   'n_components': 20, 'map_x': 7, 'map_y': 7},
    {'name': 'PCA_d80_large_7x7',       'data_key': 'large', 'method': 'PCA',   'n_components': 80, 'map_x': 7, 'map_y': 7},
    {'name': 'PCA_d120_large_7x7',      'data_key': 'large', 'method': 'PCA',   'n_components': 120, 'map_x': 7, 'map_y': 7},

    # --- largeデータセット, map_size=8x8 ---
    {'name': 'MPPCA_p15_q20_large_8x8', 'data_key': 'large', 'method': 'MPPCA', 'p_clusters': 15, 'q_latent_dim': 20, 'map_x': 8, 'map_y': 8},
    {'name': 'MPPCA_p6_q20_large_8x8',  'data_key': 'large', 'method': 'MPPCA', 'p_clusters': 6,  'q_latent_dim': 20, 'map_x': 8, 'map_y': 8},
    {'name': 'MPPCA_p4_q20_large_8x8',  'data_key': 'large', 'method': 'MPPCA', 'p_clusters': 4,  'q_latent_dim': 20, 'map_x': 8, 'map_y': 8},
    {'name': 'PCA_d20_large_8x8',       'data_key': 'large', 'method': 'PCA',   'n_components': 20, 'map_x': 8, 'map_y': 8},
    {'name': 'PCA_d80_large_8x8',       'data_key': 'large', 'method': 'PCA',   'n_components': 80, 'map_x': 8, 'map_y': 8},
    {'name': 'PCA_d120_large_8x8',      'data_key': 'large', 'method': 'PCA',   'n_components': 120, 'map_x': 8, 'map_y': 8},

    # --- largeデータセット, map_size=9x9 ---
    {'name': 'MPPCA_p15_q20_large_9x9', 'data_key': 'large', 'method': 'MPPCA', 'p_clusters': 15, 'q_latent_dim': 20, 'map_x': 9, 'map_y': 9},
    {'name': 'MPPCA_p6_q20_large_9x9',  'data_key': 'large', 'method': 'MPPCA', 'p_clusters': 6,  'q_latent_dim': 20, 'map_x': 9, 'map_y': 9},
    {'name': 'MPPCA_p4_q20_large_9x9',  'data_key': 'large', 'method': 'MPPCA', 'p_clusters': 4,  'q_latent_dim': 20, 'map_x': 9, 'map_y': 9},
    {'name': 'PCA_d20_large_9x9',       'data_key': 'large', 'method': 'PCA',   'n_components': 20, 'map_x': 9, 'map_y': 9},
    {'name': 'PCA_d80_large_9x9',       'data_key': 'large', 'method': 'PCA',   'n_components': 80, 'map_x': 9, 'map_y': 9},
    {'name': 'PCA_d120_large_9x9',      'data_key': 'large', 'method': 'PCA',   'n_components': 120, 'map_x': 9, 'map_y': 9},

    # --- largeデータセット, map_size=10x10 ---
    {'name': 'MPPCA_p15_q20_large_10x10', 'data_key': 'large', 'method': 'MPPCA', 'p_clusters': 15, 'q_latent_dim': 20, 'map_x': 10, 'map_y': 10},
    {'name': 'MPPCA_p6_q20_large_10x10',  'data_key': 'large', 'method': 'MPPCA', 'p_clusters': 6,  'q_latent_dim': 20, 'map_x': 10, 'map_y': 10},
    {'name': 'MPPCA_p4_q20_large_10x10',  'data_key': 'large', 'method': 'MPPCA', 'p_clusters': 4,  'q_latent_dim': 20, 'map_x': 10, 'map_y': 10},
    {'name': 'PCA_d20_large_10x10',       'data_key': 'large', 'method': 'PCA',   'n_components': 20, 'map_x': 10, 'map_y': 10},
    {'name': 'PCA_d80_large_10x10',       'data_key': 'large', 'method': 'PCA',   'n_components': 80, 'map_x': 10, 'map_y': 10},
    {'name': 'PCA_d120_large_10x10',      'data_key': 'large', 'method': 'PCA',   'n_components': 120, 'map_x': 10, 'map_y': 10},
]