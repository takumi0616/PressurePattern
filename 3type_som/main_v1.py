import os
import sys
import json
import numpy as np
import xarray as xr
import pandas as pd
import torch
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # サーバ上でも保存できるように
import matplotlib.pyplot as plt

# cartopy / seaborn は無い環境もあるため保護
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False

from minisom import MiniSom  # PyTorch実装（S1/SSIM/Euclidean対応）

# ====== 入力 ======
DATA_FILE = './prmsl_era5_all_data_seasonal_large.nc'
TRAIN_START = '1991-01-01'
TRAIN_END   = '1999-12-31'
TEST_START  = '2000-01-01'
TEST_END    = '2000-12-31'

# 基本ラベル（15）＋ 複合ラベル（16番）
BASE_LABELS = [
    '1', '2A', '2B', '2C', '2D',
    '3A', '3B', '3C', '3D',
    '4A', '4B',
    '5',
    '6A', '6B', '6C'
]
COMPOSITE_LABEL = 'COMPOSITE'  # 16番目（代表ラベル不可）

# ====== SOM設定 ======
SOM_X, SOM_Y = 10, 10
NUM_ITER = 300
BATCH_SIZE = 512
NODES_CHUNK = 16    # ノード方向チャンク（S1/SSIMのVRAM節約）
LOG_INTERVAL = 10
EVAL_SAMPLE_LIMIT = 2048

OUTPUT_ROOT = './outputs_compare_som'
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# =====================================================
# ユーティリティ：ログライタ
# =====================================================
class Logger:
    def __init__(self, path):
        self.path = path
        self.f = open(path, 'w', encoding='utf-8')
    def write(self, s):
        sys.stdout.write(s)
        self.f.write(s)
        self.f.flush()
    def close(self):
        self.f.close()

# =====================================================
# データ読み込み ＆ 前処理（領域平均差し引き）
# =====================================================
def load_data(data_file, start_date, end_date):
    ds = xr.open_dataset(data_file, decode_times=True)
    if 'valid_time' in ds:
        ds = ds.sel(valid_time=slice(start_date, end_date))
        time_coord = 'valid_time'
    elif 'time' in ds:
        ds = ds.sel(time=slice(start_date, end_date))
        time_coord = 'time'
    else:
        raise ValueError('No time coordinate named "valid_time" or "time".')

    if 'msl' not in ds:
        raise ValueError('Variable "msl" not found.')
    msl = ds['msl'].astype('float32')

    # 次元名の標準化
    lat_name = 'latitude'
    lon_name = 'longitude'
    for dn in msl.dims:
        if 'lat' in dn.lower(): lat_name = dn
        if 'lon' in dn.lower(): lon_name = dn

    msl = msl.transpose(time_coord, lat_name, lon_name)
    ntime = msl.sizes[time_coord]
    nlat  = msl.sizes[lat_name]
    nlon  = msl.sizes[lon_name]

    arr = msl.values  # (N, H, W)
    arr2 = arr.reshape(ntime, nlat*nlon)

    # NaN行除外
    valid_mask = ~np.isnan(arr2).any(axis=1)
    data = arr2[valid_mask]
    times = msl[time_coord].values[valid_mask]
    lat = ds[lat_name].values
    lon = ds[lon_name].values

    # ラベル
    labels = None
    if 'label' in ds.variables:
        raw = ds['label'].values
        raw = raw[valid_mask]
        labels = [l.decode('utf-8') if isinstance(l, (bytes, bytearray)) else str(l) for l in raw]

    # 領域平均（各サンプルの空間平均）を引く
    mean_per_sample = np.nanmean(data, axis=1, keepdims=True)
    data = data - mean_per_sample

    return data, (nlat, nlon), times, lat, lon, labels

def split_train_test(times, data, labels):
    """1991–1999を学習、2000をテストに分割"""
    t_pd = pd.to_datetime(times)
    train_mask = (t_pd >= pd.to_datetime(TRAIN_START)) & (t_pd <= pd.to_datetime(TRAIN_END))
    test_mask  = (t_pd >= pd.to_datetime(TEST_START)) & (t_pd <= pd.to_datetime(TEST_END))

    data_train = data[train_mask]
    data_test  = data[test_mask]
    times_train = t_pd[train_mask].values
    times_test  = t_pd[test_mask].values

    labels_train = None
    labels_test  = None
    if labels is not None:
        ls = np.array(labels, dtype=object)
        labels_train = ls[train_mask].tolist()
        labels_test  = ls[test_mask].tolist()

    return data_train, times_train, labels_train, data_test, times_test, labels_test

# =====================================================
# ラベル処理
# =====================================================
def normalize_label(l):
    if l is None:
        return None
    s = str(l).strip()
    s = s.replace('＋','+').replace('－','-').replace('−','-').replace('　','').replace(' ','')
    return s

def map_to_base_or_composite(label_str):
    """基本ラベルならそのまま、複合はCOMPOSITE、その他はNone"""
    s = normalize_label(label_str)
    if not s:
        return None
    if s in BASE_LABELS:
        return s
    # 複合判定（+/-が含まれる）
    if ('+' in s) or ('-' in s):
        return COMPOSITE_LABEL
    # 一致しないものは複合扱いにしても良いが、明確に不明はNoneにしておく
    return COMPOSITE_LABEL

# =====================================================
# SOM結果から評価・可視化
# =====================================================
def winners_to_clusters(winners_xy, som_shape):
    clusters = [[] for _ in range(som_shape[0]*som_shape[1])]
    for i,(ix,iy) in enumerate(winners_xy):
        k = ix*som_shape[1] + iy
        clusters[k].append(i)
    return clusters

def choose_representative_labels_train(clusters, labels_mapped, base_labels, forbid_label):
    """
    学習データ上の代表ラベル（最多）をノード毎に選ぶ
      - forbid_label（COMPOSITE）は代表不可
      - forbid_labelが最多の場合は代表None（ラベルなしクラスタ）
    """
    rep = {}
    for k, idxs in enumerate(clusters):
        if len(idxs)==0:
            rep[k] = None
            continue
        # 出現カウント
        from collections import Counter
        c = Counter()
        for j in idxs:
            lab = labels_mapped[j]
            if lab is not None:
                c[lab] += 1
        if len(c)==0:
            rep[k] = None
            continue
        # forbid_labelを代表にできない
        # base_labelsから最多を選ぶ。base側で何もなければNone。
        max_lbl = None
        max_cnt = -1
        for bl in base_labels:
            if c[bl] > max_cnt:
                max_cnt = c[bl]
                max_lbl = bl
        if max_cnt > 0:
            rep[k] = max_lbl
        else:
            rep[k] = None
    return rep

def macro_recall_train(clusters, labels_mapped, rep_labels, base_labels):
    """学習データのマクロ平均再現率（指定通り）"""
    # 各baseラベルの総数
    from collections import Counter
    label_indices = {}
    for bl in base_labels:
        label_indices[bl] = []
    for i,lab in enumerate(labels_mapped):
        if lab in base_labels:
            label_indices[lab].append(i)

    recalls = []
    for bl in base_labels:
        total = len(label_indices[bl])
        if total == 0:
            continue
        # そのサンプルの属するノードの代表が bl かどうか
        correct = 0
        for j in label_indices[bl]:
            # 探す：jの属するノードk
            # まず全クラスタを逆引きするコストを避けるために、後で呼び出し元でsample_to_nodeを渡すのが望ましいが、
            # ここでは規模が小さい想定のため素直に線形探索
            found_k = None
            for k,idxs in enumerate(clusters):
                if j in idxs:
                    found_k = k
                    break
            if found_k is None:
                continue
            if rep_labels.get(found_k, None) == bl:
                correct += 1
        recalls.append(correct/total)
    macro = float(np.mean(recalls)) if len(recalls)>0 else 0.0
    return macro, recalls

def accuracy_test(winners_xy_test, labels_test_mapped, rep_labels, base_labels, som_shape):
    """テスト2000での正解率（基本ラベルのみ）"""
    # ノード代表ラベルによる予測
    correct = 0
    total   = 0
    per_label_total = {bl:0 for bl in base_labels}
    per_label_correct = {bl:0 for bl in base_labels}

    for i,(ix,iy) in enumerate(winners_xy_test):
        true_lab = labels_test_mapped[i]
        if true_lab not in base_labels:
            continue  # 複合は評価対象外
        node_k = ix*som_shape[1] + iy
        pred_lab = rep_labels.get(node_k, None)
        per_label_total[true_lab] += 1
        total += 1
        if (pred_lab is not None) and (pred_lab == true_lab):
            correct += 1
            per_label_correct[true_lab] += 1

    acc = correct/max(1,total)
    per_label_acc = {bl:(per_label_correct[bl]/per_label_total[bl] if per_label_total[bl]>0 else None)
                     for bl in base_labels}
    return acc, total, per_label_acc

# ======== 可視化 =========
def plot_som_node_average_patterns(data_flat, winners_xy, lat, lon, som_shape, save_path, title):
    """各ノードに割り当てられたサンプルの平均「SLP異常（hPa）」のタイル図"""
    H, W = len(lat), len(lon)
    X2 = data_flat.reshape(-1, H, W)
    clim = np.nanmean(X2, axis=0)
    anom_hpa = (X2 - clim) / 100.0

    map_x, map_y = som_shape
    mean_patterns = np.full((map_x, map_y, H, W), np.nan, dtype=np.float32)
    counts = np.zeros((map_x, map_y), dtype=int)
    for ix in range(map_x):
        for iy in range(map_y):
            mask = (winners_xy[:,0]==ix) & (winners_xy[:,1]==iy)
            idxs = np.where(mask)[0]
            counts[ix,iy] = len(idxs)
            if len(idxs)>0:
                mean_patterns[ix,iy] = np.nanmean(anom_hpa[idxs], axis=0)

    absmax = np.nanmax(np.abs(mean_patterns))
    if not np.isfinite(absmax) or absmax==0:
        absmax = 1.0
    vmin,vmax = -absmax, absmax
    levels = np.linspace(vmin, vmax, 21)

    nrows, ncols = som_shape[1], som_shape[0]
    figsize=(ncols*2.6, nrows*2.6)
    if HAS_CARTOPY:
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
                                 subplot_kw={'projection': ccrs.PlateCarree()})
    else:
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = np.atleast_2d(axes)
    axes = axes.T[::-1,:]

    last_cf=None
    for ix in range(map_x):
        for iy in range(map_y):
            ax = axes[ix,iy]
            mp = mean_patterns[ix,iy]
            if np.isnan(mp).all():
                ax.set_axis_off(); continue
            if HAS_CARTOPY:
                cf = ax.contourf(lon, lat, mp, levels=levels, cmap='RdBu_r', extend='both', transform=ccrs.PlateCarree())
                ax.contour(lon, lat, mp, levels=levels, colors='k', linewidths=0.3, transform=ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='black', linewidth=0.4)
                ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], ccrs.PlateCarree())
            else:
                extent=[lon.min(), lon.max(), lat.min(), lat.max()]
                cf = ax.imshow(mp[::-1,:], extent=extent, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
                try:
                    ax.contour(lon, lat, mp, levels=levels, colors='k', linewidths=0.3)
                except Exception:
                    pass
            ax.text(0.02,0.96,f'({ix},{iy}) N={counts[ix,iy]}', transform=ax.transAxes,
                    fontsize=7, va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            ax.set_xticks([]); ax.set_yticks([])
            last_cf=cf
    if last_cf is not None:
        fig.subplots_adjust(right=0.88, top=0.94)
        cax = fig.add_axes([0.90, 0.12, 0.02, 0.72])
        fig.colorbar(last_cf, cax=cax, label='SLP Anomaly (hPa)')
    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0,0,0.88,0.94])
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def save_each_node_mean_image(data_flat, winners_xy, lat, lon, som_shape, out_dir, prefix):
    """各ノードの平均SLP異常の図を個別に保存"""
    os.makedirs(out_dir, exist_ok=True)
    H, W = len(lat), len(lon)
    X2 = data_flat.reshape(-1, H, W)
    clim = np.nanmean(X2, axis=0)
    anom_hpa = (X2 - clim)/100.0
    map_x, map_y = som_shape
    for ix in range(map_x):
        for iy in range(map_y):
            mask = (winners_xy[:,0]==ix) & (winners_xy[:,1]==iy)
            idxs = np.where(mask)[0]
            mean_img = None
            if len(idxs)>0:
                mean_img = np.nanmean(anom_hpa[idxs], axis=0)
            else:
                mean_img = np.full((H,W), np.nan, dtype=np.float32)
            # 画像保存
            fig = plt.figure(figsize=(4,3))
            if HAS_CARTOPY:
                ax = plt.axes(projection=ccrs.PlateCarree())
                cf = ax.contourf(lon, lat, mean_img, 21, cmap='RdBu_r', transform=ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.4)
                ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], ccrs.PlateCarree())
            else:
                ax = plt.gca()
                im = ax.imshow(mean_img[::-1,:], cmap='RdBu_r',
                               vmin=np.nanmin(mean_img), vmax=np.nanmax(mean_img),
                               aspect='auto', extent=[lon.min(), lon.max(), lat.min(), lat.max()])
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f'({ix},{iy}) N={len(idxs)}')
            ax.set_xticks([]); ax.set_yticks([])
            fpath = os.path.join(out_dir, f'{prefix}_node_{ix}_{iy}.png')
            plt.tight_layout()
            plt.savefig(fpath, dpi=180)
            plt.close(fig)

def plot_label_distributions(winners_xy, labels_mapped, label_list, som_shape, save_dir, title_prefix):
    """各ラベルごとのノード分布（カウント）をヒートマップで保存"""
    os.makedirs(save_dir, exist_ok=True)
    node_counts = {lbl: np.zeros((som_shape[0], som_shape[1]), dtype=int) for lbl in label_list}
    for i,(ix,iy) in enumerate(winners_xy):
        lab = labels_mapped[i]
        if lab in node_counts:
            node_counts[lab][ix,iy] += 1
    # まとめ図（ラベルごとサブプロット）
    cols = 5
    rows = int(np.ceil(len(label_list)/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.6, rows*2.6))
    axes = np.atleast_2d(axes)
    for idx,lbl in enumerate(label_list):
        r = idx//cols; c=idx%cols
        ax = axes[r,c]
        im = ax.imshow(node_counts[lbl].T[::-1,:], cmap='viridis')
        ax.set_title(lbl); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for k in range(len(label_list), rows*cols):
        r = k//cols; c=k%cols
        axes[r,c].axis('off')
    plt.suptitle(f'{title_prefix} Label Distributions on SOM nodes', fontsize=14)
    plt.tight_layout(rect=[0,0,1,0.95])
    fpath = os.path.join(save_dir, f'{title_prefix}_label_distributions.png')
    plt.savefig(fpath, dpi=250)
    plt.close(fig)

# =====================================================
# 学習・評価パイプライン（1方式分）
# =====================================================
def run_one_method(method_name, activation_distance, data_train, labels_train, times_train,
                   data_test, labels_test, times_test, field_shape, lat, lon, out_dir):
    """
    method_name: 'euclidean' | 'ssim' | 's1'
    activation_distance: 同上
    """
    os.makedirs(out_dir, exist_ok=True)
    log = Logger(os.path.join(out_dir, f'{method_name}_results.log'))
    log.write(f'=== {method_name} SOM ===\n')
    log.write(f'Device CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}\n')
    log.write(f'SOM size: {SOM_X} x {SOM_Y}, iter={NUM_ITER}, batch={BATCH_SIZE}, nodes_chunk={NODES_CHUNK}\n')
    log.write(f'Train samples: {data_train.shape[0]}, Test samples: {data_test.shape[0]}\n')

    # ---- SOMの構築 ----
    som = MiniSom(
        x=SOM_X, y=SOM_Y, input_len=data_train.shape[1],
        sigma=2.5, learning_rate=0.5,
        neighborhood_function='gaussian',
        topology='rectangular',
        activation_distance=activation_distance,             # 'euclidean'/'ssim'/'s1'
        random_seed=42,
        sigma_decay='asymptotic_decay',
        s1_field_shape=field_shape,                          # SSIM/S1で必須
        device='cuda' if torch.cuda.is_available() else 'cpu',
        dtype=torch.float32,
        nodes_chunk=NODES_CHUNK
    )
    som.random_weights_init(data_train)

    # ---- 学習 ----
    qhist = som.train_batch(
        data_train, num_iteration=NUM_ITER,
        batch_size=BATCH_SIZE, verbose=True, log_interval=LOG_INTERVAL,
        update_per_iteration=False, shuffle=True
    )
    qe_train = som.quantization_error(data_train, sample_limit=EVAL_SAMPLE_LIMIT, batch_size=max(32,BATCH_SIZE))
    log.write(f'Final quantization_error (mean distance) on train: {qe_train:.6f}\n')

    # ---- BMU推定 ----
    winners_train = som.predict(data_train, batch_size=max(64,BATCH_SIZE))
    winners_test  = som.predict(data_test,  batch_size=max(64,BATCH_SIZE))

    # 割当CSV
    assign_csv_train = os.path.join(out_dir, f'{method_name}_assign_train.csv')
    pd.DataFrame({
        'time': times_train,
        'bmu_x': winners_train[:,0], 'bmu_y': winners_train[:,1],
        'label_raw': labels_train if labels_train is not None else ['']*len(winners_train)
    }).to_csv(assign_csv_train, index=False, encoding='utf-8-sig')
    log.write(f'Assigned BMU (train) -> {assign_csv_train}\n')

    assign_csv_test = os.path.join(out_dir, f'{method_name}_assign_test.csv')
    pd.DataFrame({
        'time': times_test,
        'bmu_x': winners_test[:,0], 'bmu_y': winners_test[:,1],
        'label_raw': labels_test if labels_test is not None else ['']*len(winners_test)
    }).to_csv(assign_csv_test, index=False, encoding='utf-8-sig')
    log.write(f'Assigned BMU (test)  -> {assign_csv_test}\n')

    # ---- 可視化：SOMノード平均パターン（異常）大図 ----
    bigmap_train = os.path.join(out_dir, f'{method_name}_som_node_avg_train.png')
    plot_som_node_average_patterns(data_train, winners_train, lat, lon, (SOM_X,SOM_Y),
                                   save_path=bigmap_train,
                                   title=f'{method_name.upper()} SOM Node Avg SLP Anomaly (Train)')
    log.write(f'Node average patterns (train) -> {bigmap_train}\n')

    bigmap_test = os.path.join(out_dir, f'{method_name}_som_node_avg_test.png')
    plot_som_node_average_patterns(data_test, winners_test, lat, lon, (SOM_X,SOM_Y),
                                   save_path=bigmap_test,
                                   title=f'{method_name.upper()} SOM Node Avg SLP Anomaly (Test)')
    log.write(f'Node average patterns (test)  -> {bigmap_test}\n')

    # ---- 各ノードの平均画像を個別保存 ----
    pernode_dir_train = os.path.join(out_dir, f'{method_name}_pernode_train')
    save_each_node_mean_image(data_train, winners_train, lat, lon, (SOM_X,SOM_Y),
                              out_dir=pernode_dir_train, prefix='train')
    log.write(f'Per-node mean images (train) -> {pernode_dir_train}\n')

    pernode_dir_test = os.path.join(out_dir, f'{method_name}_pernode_test')
    save_each_node_mean_image(data_test, winners_test, lat, lon, (SOM_X,SOM_Y),
                              out_dir=pernode_dir_test, prefix='test')
    log.write(f'Per-node mean images (test)  -> {pernode_dir_test}\n')

    # ---- ラベルがある場合のみ評価 ----
    if labels_train is not None and labels_test is not None:
        # ラベルを base or composite に正規化
        labels_train_mapped = [map_to_base_or_composite(l) for l in labels_train]
        labels_test_mapped  = [map_to_base_or_composite(l) for l in labels_test]

        # 学習データで評価（マクロ平均再現率のみ）
        clusters_train = winners_to_clusters(winners_train, (SOM_X,SOM_Y))
        rep_labels = choose_representative_labels_train(
            clusters=clusters_train, labels_mapped=labels_train_mapped,
            base_labels=BASE_LABELS, forbid_label=COMPOSITE_LABEL
        )
        macro, per_label_recalls = macro_recall_train(
            clusters_train, labels_train_mapped, rep_labels, BASE_LABELS
        )
        log.write('\n[Train 1991–1999] Macro Recall (base only): {:.4f}\n'.format(macro))
        # 各ラベル再現率
        for bl,rc in zip(BASE_LABELS, per_label_recalls):
            log.write(f'  Recall[{bl}]: {rc:.4f}\n')
        # 代表ラベル統計
        cnt_rep = {}
        for k,v in rep_labels.items():
            cnt_rep[v] = cnt_rep.get(v,0)+1
        log.write(f'  Representative labels per node (count): {cnt_rep}\n')

        # テスト精度（基本ラベルのみ）
        acc, total, per_label_acc = accuracy_test(
            winners_test, labels_test_mapped, rep_labels, BASE_LABELS, (SOM_X,SOM_Y)
        )
        log.write('\n[Test 2000] Accuracy (base only): {:.4f}  [N={}]\n'.format(acc, total))
        for bl in BASE_LABELS:
            v = per_label_acc[bl]
            if v is None:
                log.write(f'  Acc[{bl}]: None (no sample)\n')
            else:
                log.write(f'  Acc[{bl}]: {v:.4f}\n')

        # ラベル分布の可視化（学習・テスト）
        # 学習
        dist_dir_train = os.path.join(out_dir, f'{method_name}_label_dist_train')
        plot_label_distributions(winners_train, labels_train_mapped, BASE_LABELS+[COMPOSITE_LABEL],
                                 (SOM_X,SOM_Y), dist_dir_train, title_prefix='Train')
        log.write(f'Label-distribution heatmaps (train) -> {dist_dir_train}\n')
        # テスト
        dist_dir_test = os.path.join(out_dir, f'{method_name}_label_dist_test')
        plot_label_distributions(winners_test, labels_test_mapped, BASE_LABELS+[COMPOSITE_LABEL],
                                 (SOM_X,SOM_Y), dist_dir_test, title_prefix='Test')
        log.write(f'Label-distribution heatmaps (test)  -> {dist_dir_test}\n')

    else:
        log.write('Labels not found; skip evaluation.\n')

    log.write('\n=== Done ===\n')
    log.close()

# =====================================================
# メイン
# =====================================================
def main():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1991–2000 をロード（領域平均引き済み）
    data_all, field_shape, times_all, lat, lon, labels_all = load_data(DATA_FILE, TRAIN_START, TEST_END)
    print(f'Loaded: N={data_all.shape[0]}, field_shape={field_shape}, input_len={data_all.shape[1]}')
    print(f'Labels available: {"Yes" if labels_all is not None else "No"}')

    # 学習/テスト分割
    data_train, times_train, labels_train, data_test, times_test, labels_test = \
        split_train_test(times_all, data_all, labels_all)
    print(f'Train: {data_train.shape}, Test: {data_test.shape}')

    # 方式ごとに実行
    methods = [
        ('euclidean', 'euclidean'),
        ('ssim',      'ssim'),
        ('s1',        's1')
    ]
    for mname, adist in methods:
        out_dir = os.path.join(OUTPUT_ROOT, f'{mname}_som')
        run_one_method(
            method_name=mname, activation_distance=adist,
            data_train=data_train, labels_train=labels_train, times_train=times_train,
            data_test=data_test, labels_test=labels_test, times_test=times_test,
            field_shape=field_shape, lat=lat, lon=lon, out_dir=out_dir
        )

if __name__ == '__main__':
    main()