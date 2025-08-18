import os
import json
import numpy as np
import xarray as xr
import pandas as pd
import torch

import matplotlib
matplotlib.use('Agg')  # サーバ上でも保存できるように
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
HAS_CARTOPY = True
import seaborn as sns
HAS_SEABORN = True
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from s1_minisom import MiniSom  # 同ディレクトリのPyTorch実装

# 入力
DATA_FILE = './prmsl_era5_all_data_seasonal_large.nc'
START_DATE = '1991-01-01'
END_DATE   = '2000-12-31'
BASE_LABELS = [
    '1', '2A', '2B', '2C', '2D', '3A', '3B', '3C', '3D',
    '4A', '4B', '5', '6A', '6B', '6C'
]

# SOM設定（デフォルトをGPU・ミニバッチ想定に）
SOM_X, SOM_Y = 10, 10
NUM_ITER = 300             # 以前の10,000は重過ぎるため、まずは300程度から
BATCH_SIZE = 1024          # RTX 4060 Ti 16GBなら 512～2048 で調整
NODES_CHUNK = 16           # S1距離でノード方向を分割計算（VRAM節約）
LOG_INTERVAL = 10          # ログ間隔（反復数）
EVAL_SAMPLE_LIMIT = 2048   # 量子化誤差評価に使うサンプルの上限（高速化用）

OUTPUT_DIR = './outputs_s1_som'
ONLY_BASE_DIR = os.path.join(OUTPUT_DIR, 'only_base_label')
MULTI_LABEL_DIR = os.path.join(OUTPUT_DIR, 'multi_label')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ONLY_BASE_DIR, exist_ok=True)
os.makedirs(MULTI_LABEL_DIR, exist_ok=True)

# =========================
# データ読み込み
# =========================
def load_data(data_file, start_date, end_date):
    ds = xr.open_dataset(data_file, decode_times=True)

    # 時間スライス
    if 'valid_time' in ds:
        ds = ds.sel(valid_time=slice(start_date, end_date))
        time_coord = 'valid_time'
    elif 'time' in ds:
        ds = ds.sel(time=slice(start_date, end_date))
        time_coord = 'time'
    else:
        raise ValueError('No time coordinate named "valid_time" or "time" found.')

    if 'msl' not in ds:
        raise ValueError('Variable "msl" not found in dataset.')
    msl = ds['msl']

    # 次元名の標準化
    lat_name = 'latitude'
    lon_name = 'longitude'
    if lat_name not in msl.dims or lon_name not in msl.dims:
        for dn in msl.dims:
            if 'lat' in dn.lower(): lat_name = dn
            if 'lon' in dn.lower(): lon_name = dn

    # データ型揃え（float32で省メモリ）
    msl = msl.astype('float32')
    # 時間×緯度×経度の順序に
    msl = msl.transpose(time_coord, lat_name, lon_name)
    ntime = msl.sizes[time_coord]
    nlat = msl.sizes[lat_name]
    nlon = msl.sizes[lon_name]

    # 値の取得 (N, lat, lon) -> (N, D)
    arr = msl.values  # (N, nlat, nlon), NaNあり得る
    arr2 = arr.reshape(ntime, nlat*nlon)

    # NaN行を除外（SOM学習にはNaNを含まないサンプルのみ使う）
    valid_mask = ~np.isnan(arr2).any(axis=1)
    data = arr2[valid_mask]
    times = msl[time_coord].values[valid_mask]

    # 緯度・経度の配列
    lat = ds[lat_name].values
    lon = ds[lon_name].values

    # ラベル（存在する場合）
    labels = None
    if 'label' in ds.variables:
        raw = ds['label'].values  # (N,) string dtype
        raw = raw[valid_mask]
        labels = [l.decode('utf-8') if isinstance(l, (bytes, bytearray)) else str(l) for l in raw]

    return data, (nlat, nlon), times, lat, lon, labels

# =========================
# 可視化：SOM各ノード平均異常(hPa)
# =========================
def plot_som_node_average_patterns(
    data_flat: np.ndarray,
    winners_xy: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    som_shape: tuple,
    save_path: str,
    title: str = 'SOM Node Average Pressure Anomaly Patterns (hPa)'
):
    """
    各ノードに割り当てられたサンプルの平均「海面更正気圧の異常値（hPa）」をマップ状に描く。
    - 全期間の平均場を基準に異常を算出
    - サブプロットをSOMの (x,y) 配置通りに並べる
    - cartopy があれば地図投影、無ければimshowで代替
    """
    H, W = len(lat), len(lon)
    X2 = data_flat.reshape(-1, H, W)  # (N, H, W)
    clim = np.nanmean(X2, axis=0)     # 全期間平均 (H, W)
    anom_hpa = (X2 - clim)[...,] / 100.0  # hPa

    map_x, map_y = som_shape
    mean_patterns = np.full((map_x, map_y, H, W), np.nan, dtype=np.float32)
    counts = np.zeros((map_x, map_y), dtype=int)

    # ノードごとに平均
    for ix in range(map_x):
        for iy in range(map_y):
            mask = (winners_xy[:, 0] == ix) & (winners_xy[:, 1] == iy)
            idxs = np.where(mask)[0]
            counts[ix, iy] = len(idxs)
            if len(idxs) > 0:
                mean_patterns[ix, iy] = np.nanmean(anom_hpa[idxs], axis=0)

    # カラースケール
    absmax = np.nanmax(np.abs(mean_patterns))
    if not np.isfinite(absmax) or absmax == 0:
        absmax = 1.0
    vmin, vmax = -absmax, absmax
    levels = np.linspace(vmin, vmax, 21)

    # 図
    nrows, ncols = som_shape[1], som_shape[0]
    figsize = (ncols * 2.6, nrows * 2.6)

    if HAS_CARTOPY:
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
                                 subplot_kw={'projection': ccrs.PlateCarree()})
    else:
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = np.atleast_2d(axes)
    axes = axes.T[::-1, :]  # 左下が(0,0)に見えるように

    last_cf = None
    for ix in range(map_x):
        for iy in range(map_y):
            ax = axes[ix, iy]
            mp = mean_patterns[ix, iy]
            if np.isnan(mp).all():
                ax.set_axis_off()
                continue

            if HAS_CARTOPY:
                cf = ax.contourf(lon, lat, mp, levels=levels, cmap='RdBu_r', extend='both', transform=ccrs.PlateCarree())
                ax.contour(lon, lat, mp, levels=levels, colors='k', linewidths=0.3, transform=ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='black', linewidth=0.5)
                ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
            else:
                extent = [lon.min(), lon.max(), lat.min(), lat.max()]
                cf = ax.imshow(mp[::-1, :], extent=extent, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
                try:
                    ax.contour(lon, lat, mp, levels=levels, colors='k', linewidths=0.3)
                except Exception:
                    pass
                ax.set_xlim(lon.min(), lon.max()); ax.set_ylim(lat.min(), lat.max())

            ax.text(0.02, 0.96, f'({ix},{iy}) N={counts[ix,iy]}',
                    transform=ax.transAxes, fontsize=7, va='top',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            ax.set_xticks([]); ax.set_yticks([])
            last_cf = cf

    if last_cf is not None:
        fig.subplots_adjust(right=0.88, top=0.94)
        cax = fig.add_axes([0.90, 0.12, 0.02, 0.72])
        fig.colorbar(last_cf, cax=cax, label='Pressure Anomaly (hPa)')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.88, 0.94])
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

# =========================
# ラベル整形と複合展開
# =========================
def normalize_label_str(lbl):
    if lbl is None:
        return None
    s = str(lbl).strip()
    s = s.replace('＋', '+').replace('－', '-').replace('−', '-')
    s = s.replace('　', '').replace(' ', '')
    return s

def extract_base_labels_from_label(label_str, base_labels, mode: str):
    """
    mode: 'only_base' or 'multi_label'
    戻り値: そのサンプルが持つ基本ラベルのリスト（0,1,2個）
    """
    s = normalize_label_str(label_str)
    if not s:
        return []
    if mode == 'only_base':
        return [s] if s in base_labels else []
    # multi_label
    import re
    m = re.split(r'(\+|\-)', s)
    if len(m) == 3:
        left, sep, right = m[0], m[1], m[2]
        out = []
        if left in base_labels:
            out.append(left)
        if right in base_labels and right != left:
            out.append(right)
        return out
    else:
        return [s] if s in base_labels else []

# =========================
# 評価：各ノードをクラスタとして扱う
# =========================
def build_clusters_from_winners(winners_xy: np.ndarray, num_nodes: int, som_shape: tuple):
    """
    winners_xy: (N,2) 各サンプルのBMU座標
    戻り値: clusters(list of list[indices]), node_names(list[str])
    """
    clusters = [[] for _ in range(num_nodes)]
    node_names = []
    # ノード名は "Node_(ix,iy)" とする
    for ix in range(som_shape[0]):
        for iy in range(som_shape[1]):
            node_names.append(f'Node_({ix},{iy})')
    # サンプル割当
    for i, (ix, iy) in enumerate(winners_xy):
        k = ix * som_shape[1] + iy
        clusters[k].append(i)
    return clusters, node_names

def build_confusion_matrix_multimode(clusters, labels, base_labels, mode, node_names):
    """
    混同行列（行: base_labels, 列: ノード）
    各サンプルは持つラベル数（0,1,2）に応じて該当ラベルにカウントを加算（複合は2つに加算）。
    """
    num_nodes = len(clusters)
    df = pd.DataFrame(0, index=base_labels, columns=node_names, dtype=int)
    if labels is None:
        return df
    for node_idx, idxs in enumerate(clusters):
        if not idxs:
            continue
        c = {}
        # サンプルを（複合であれば）2ラベルに展開してカウント
        from collections import Counter
        cnt = Counter()
        for j in idxs:
            for lbl in extract_base_labels_from_label(labels[j], base_labels, mode):
                cnt[lbl] += 1
        for lbl, count in cnt.items():
            df.loc[lbl, node_names[node_idx]] = count
    return df

def evaluate_by_majority(clusters, labels, base_labels, mode, node_names, out_dir):
    """
    多数決ベースの評価（only_base / multi_label）：
      - 混同行列（保存）
      - 各ノードの多数決ラベル
      - ラベルごとの再現率、Macro Recall、Micro Accuracy
      - ARI, NMI（複合ラベルは展開）
    戻り値: metrics(dict), majority_per_node(dict)
    """
    os.makedirs(out_dir, exist_ok=True)
    metrics = {}
    if labels is None:
        print("評価スキップ：ラベルがデータセットに存在しません。")
        return metrics, {}

    # 混同行列
    conf = build_confusion_matrix_multimode(clusters, labels, base_labels, mode, node_names)
    # presence: そのラベルがデータに出現したか
    present = [lbl for lbl in base_labels if conf.loc[lbl, :].sum() > 0]
    conf_csv = os.path.join(out_dir, f'confusion_matrix_{mode}.csv')
    conf.to_csv(conf_csv, encoding='utf-8-sig')

    # 各ノードの多数決ラベル
    majority_per_node = {}
    micro_correct_sum = 0
    total_assignments = int(conf.values.sum())  # 複合分も含む総カウント
    for node_idx in range(len(node_names)):
        col_series = conf.iloc[:, node_idx]
        col_sum = int(col_series.sum())
        if col_sum == 0:
            majority_per_node[node_idx] = None
            continue
        top_label = col_series.idxmax()
        top_count = int(col_series.max())
        majority_per_node[node_idx] = top_label
        micro_correct_sum += top_count

    # 各ラベルの再現率（代表ノード群に入った割合）
    per_label_rows = []
    for lbl in present:
        row = conf.loc[lbl, :]
        row_sum = int(row.sum())
        node_indices_for_lbl = [i for i, maj in majority_per_node.items() if maj == lbl]
        correct = int(row.iloc[node_indices_for_lbl].sum()) if node_indices_for_lbl else 0
        recall = correct / row_sum if row_sum > 0 else 0.0
        per_label_rows.append([lbl, row_sum, correct, recall, str([node_names[i] for i in node_indices_for_lbl])])

    per_label_df = pd.DataFrame(per_label_rows, columns=['Label', 'Total', 'CorrectInMajorityNodes', 'Recall', 'MajorityNodes'])
    per_label_csv = os.path.join(out_dir, f'per_label_recall_{mode}.csv')
    per_label_df.to_csv(per_label_csv, index=False, encoding='utf-8-sig')

    # Macro / Micro
    macro_recall = float(np.mean(per_label_df['Recall'])) if len(per_label_df) > 0 else 0.0
    micro_accuracy = micro_correct_sum / total_assignments if total_assignments > 0 else 0.0
    metrics['MacroRecall_majority'] = macro_recall
    metrics['MicroAccuracy_majority'] = micro_accuracy

    # ARI / NMI（複合展開）
    n_samples = len(labels)
    # サンプル -> ノードindex
    sample_to_node = [-1] * n_samples
    for node_idx, idxs in enumerate(clusters):
        for j in idxs:
            sample_to_node[j] = node_idx

    y_true, y_pred = [], []
    for j in range(n_samples):
        if sample_to_node[j] < 0:
            continue
        node_idx = sample_to_node[j]
        pred_lbl = majority_per_node.get(node_idx, None) or "Unassigned"
        true_lbls = extract_base_labels_from_label(labels[j], base_labels, mode)
        if not true_lbls:
            continue
        for t in true_lbls:
            y_true.append(t)
            y_pred.append(pred_lbl)

    if len(y_true) > 0:
        uniq_true = {l: i for i, l in enumerate(sorted(set(y_true)))}
        uniq_pred = {l: i for i, l in enumerate(sorted(set(y_pred)))}
        y_true_idx = [uniq_true[l] for l in y_true]
        y_pred_idx = [uniq_pred[l] for l in y_pred]
        ari = adjusted_rand_score(y_true_idx, y_pred_idx)
        nmi = normalized_mutual_info_score(y_true_idx, y_pred_idx)
        metrics['ARI_majority'] = float(ari)
        metrics['NMI_majority'] = float(nmi)
    else:
        metrics['ARI_majority'] = None
        metrics['NMI_majority'] = None

    # メトリクス保存（JSON）
    metrics_path = os.path.join(out_dir, f'metrics_summary_{mode}.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 混同行列のヒートマップ（任意）
    if HAS_SEABORN:
        plt.figure(figsize=(min(22, 1 + 0.4 * len(node_names)), 8))
        sns.heatmap(conf.loc[present, :], annot=False, cmap='viridis')
        plt.title(f'Confusion Matrix (labels x SOM nodes) - {mode}')
        plt.ylabel('Base Labels'); plt.xlabel('SOM Nodes')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'confusion_matrix_{mode}.png'), dpi=250)
        plt.close()

        # per-label recall bar
        plt.figure(figsize=(10, 6))
        plt.bar(per_label_df['Label'], per_label_df['Recall'])
        plt.ylim(0, 1); plt.grid(True, axis='y', alpha=0.3)
        plt.title(f'Per Label Recall (majority, {mode})')
        plt.xlabel('Base Label'); plt.ylabel('Recall')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'per_label_recall_{mode}.png'), dpi=250)
        plt.close()

    # 画面出力
    print(f"\n=== Evaluation ({mode}) ===")
    print(f"Macro Recall (majority): {macro_recall:.4f}")
    print(f"Micro Accuracy (majority): {micro_accuracy:.4f}")
    print(f"ARI (majority): {metrics['ARI_majority']}")
    print(f"NMI (majority): {metrics['NMI_majority']}")
    print(f"Saved confusion matrix -> {conf_csv}")
    print(f"Saved per-label recall -> {per_label_csv}")
    print(f"Saved metrics summary -> {metrics_path}")

    return metrics, majority_per_node

# =========================
# メイン
# =========================
def main():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data, field_shape, times, lat, lon, labels = load_data(DATA_FILE, START_DATE, END_DATE)
    n_samples, input_len = data.shape
    print(f'Loaded data: samples={n_samples}, field_shape={field_shape}, input_len={input_len}')
    print(f'Labels available: {"Yes" if labels is not None else "No"}')

    # SOM初期化（S1距離、s1_field_shapeに2次元形状）
    som = MiniSom(SOM_X, SOM_Y, input_len,
                  sigma=2.5, learning_rate=0.5,
                  neighborhood_function='gaussian',
                  topology='rectangular',
                  activation_distance='s1',
                  random_seed=42,
                  sigma_decay='asymptotic_decay',
                  s1_field_shape=field_shape,
                  device='cuda' if torch.cuda.is_available() else 'cpu',
                  dtype=torch.float32,
                  nodes_chunk=NODES_CHUNK)

    # 初期重みをデータからランダムに初期化
    som.random_weights_init(data)

    # 学習（ミニバッチ）
    qhist = som.train_batch(
        data,
        num_iteration=NUM_ITER,
        batch_size=BATCH_SIZE,
        verbose=True,
        log_interval=LOG_INTERVAL,
        update_per_iteration=False,
        shuffle=True
    )

    # 学習後の平均S1（量子化誤差）
    qe = som.quantization_error(data, sample_limit=EVAL_SAMPLE_LIMIT, batch_size=max(32, BATCH_SIZE))
    print(f'Final mean S1 (quantization_error): {qe:.4f}')

    # 全サンプルのBMUを推定（GPUでチャンク処理）
    winners_xy = som.predict(data, batch_size=max(64, BATCH_SIZE))
    winners_xy = winners_xy.astype(int)  # (N, 2) [x, y]

    # 割当リスト（CSV）
    # BMUをラベルにマップ（行優先でBASE_LABELSを割り当て）
    if SOM_X * SOM_Y != len(BASE_LABELS):
        print('Warning: BASE_LABELS size does not match SOM grid; will cut or pad as needed.')
    labels_extended = BASE_LABELS[:SOM_X*SOM_Y] + [''] * max(0, SOM_X*SOM_Y - len(BASE_LABELS))
    label_grid = np.array(labels_extended).reshape(SOM_X, SOM_Y)
    assigned_labels = [label_grid[xy[0], xy[1]] for xy in winners_xy]

    assign_csv = os.path.join(OUTPUT_DIR, 'som_s1_assignments.csv')
    df = pd.DataFrame({
        'time': times,
        'bmu_x': winners_xy[:, 0],
        'bmu_y': winners_xy[:, 1],
        'som_label': assigned_labels,
        'true_label': labels if labels is not None else ['']*len(winners_xy)
    })
    df.to_csv(assign_csv, index=False, encoding='utf-8-sig')
    print(f'Assignments saved: {assign_csv}')

    # 可視化：SOMノード平均パターン（異常hPa）を地図に配置して保存
    out_plot = os.path.join(OUTPUT_DIR, 'som_node_average_patterns.png')
    plot_som_node_average_patterns(
        data_flat=data,
        winners_xy=winners_xy,
        lat=lat,
        lon=lon,
        som_shape=(SOM_X, SOM_Y),
        save_path=out_plot,
        title='SOM Node Average Sea-Level Pressure Anomaly (hPa)'
    )
    print(f'Node average pattern map saved: {out_plot}')

    # 評価（ラベルがある場合のみ）
    if labels is not None:
        # 各ノードをクラスタと見なす
        num_nodes = SOM_X * SOM_Y
        clusters, node_names = build_clusters_from_winners(winners_xy, num_nodes, (SOM_X, SOM_Y))

        # only_base 評価
        metrics_base, majority_base = evaluate_by_majority(
            clusters, labels, BASE_LABELS, mode='only_base', node_names=node_names, out_dir=ONLY_BASE_DIR
        )
        # multi_label 評価
        metrics_multi, majority_multi = evaluate_by_majority(
            clusters, labels, BASE_LABELS, mode='multi_label', node_names=node_names, out_dir=MULTI_LABEL_DIR
        )

        # 結果の概要JSON
        summary = {
            'SOM_shape': [SOM_X, SOM_Y],
            'Final_mean_S1': qe,
            'metrics_only_base': metrics_base,
            'metrics_multi_label': metrics_multi
        }
        with open(os.path.join(OUTPUT_DIR, 'evaluation_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Evaluation summary saved: {os.path.join(OUTPUT_DIR, 'evaluation_summary.json')}")
    else:
        print("Labels not found in dataset; skipped evaluation.")

if __name__ == '__main__':
    main()