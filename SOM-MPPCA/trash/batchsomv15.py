import gc  # ガベージコレクションのために追加
import os
import random
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime, timedelta

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import japanize_matplotlib  # 日本語フォント対応
import matplotlib.pyplot as plt
import numpy as np
import psutil  # メモリ使用量を計測するために追加
import seaborn as sns
import xarray as xr
from matplotlib.colors import Normalize
from minisom import MiniSom
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data.py import (
    data_label_dict,
    rain_dates_list,
    summer_dates_list,
    test_rain_dates_list,
    test_summer_dates_list,
    test_typhoon_dates_list,
    test_winter_dates_list,
    typhoon_dates_list,
    winter_dates_list,
)


# メモリ使用量を表示する関数を定義
def print_memory_usage(message=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"{message} メモリ使用量: {mem_info.rss / 1024 ** 2:.2f} MB")


# 乱数シードの固定
random.seed(0)
np.random.seed(0)

# データのパス設定
data_dir = "/home/devel/work_C/work_ytakano/GSMGPV/nc/prmsl/"

# 解析期間の設定
start_date = datetime.strptime("2016-03-01", "%Y-%m-%d")
end_date = datetime.strptime("2024-10-31", "%Y-%m-%d")

# 使用する緯度・経度の範囲を設定（日本付近）
lat_range = slice(15, 55)
lon_range = slice(115, 155)

# データファイルの一覧を取得（サブディレクトリ内も含める）
file_paths = []
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".nc"):
            file_paths.append(os.path.join(root, file))

# ファイルパスをソート
file_paths = sorted(file_paths)

# ファイル名から日付を取得し、辞書を作成
file_date_dict = {}
for file_path in file_paths:
    file_name = os.path.basename(file_path)
    date_str = file_name.replace("prmsl_", "").replace(".nc", "")
    try:
        file_date = datetime.strptime(date_str, "%Y%m%d").date()
        file_date_dict[file_date] = file_path
    except ValueError:
        print(f"ファイル名から日付を抽出できませんでした: {file_name}")

# 共通の緯度・経度グリッドを定義
common_lat = np.arange(15, 55.5, 0.5)
common_lon = np.arange(115, 155.5, 0.5)

# データを格納するリスト
data_list = []
date_list = []

# 期待されるデータの形状を初期化
expected_shape = None

# データの読み込み
current_date = start_date
while current_date <= end_date:
    current_date_date = current_date.date()
    if current_date_date in file_date_dict:
        nc_file = file_date_dict[current_date_date]
        ds = xr.open_dataset(nc_file)

        # 座標名を標準化
        if "lat" in ds.coords and "lon" in ds.coords:
            ds = ds.rename({"lat": "latitude", "lon": "longitude"})

        # 海面更正気圧データを取得し、指定の緯度経度範囲でスライス
        slp = (
            (ds["PRMSL_meansealevel"] / 100)
            .isel(time=0)
            .sel(latitude=lat_range, longitude=lon_range)
        )

        # slp データの存在確認
        if slp.size == 0:
            print(
                f"警告: slp データが空です。日付 {current_date_date} のデータをスキップします。"
            )
            current_date += timedelta(days=1)
            continue

        # データを共通グリッドに再補間
        slp_interp = slp.interp(latitude=common_lat, longitude=common_lon)
        slp_shape = slp_interp.values.shape

        # 期待されるデータ形状を設定
        if expected_shape is None:
            expected_shape = slp_shape
        else:
            if slp_shape != expected_shape:
                print(
                    f"日付 {current_date_date}: データ形状が {slp_shape} で、期待される形状 {expected_shape} と一致しません。"
                )
                current_date += timedelta(days=1)
                continue

        # 領域平均を引く（再補間後のデータで）
        slp_interp = slp_interp - slp_interp.mean()

        # データを1次元配列にフラット化してリストに追加
        data_list.append(slp_interp.values.flatten())
        date_list.append(current_date_date)

        # print(f"データを読み込みました: 日付 {current_date_date}")
    else:
        print(f"データが存在しない日付: {current_date_date}")
    current_date += timedelta(days=1)

# リストをNumPy配列に変換
data_array = np.array(data_list)
print("data_array の形状:", data_array.shape)

# データの正規化（平均0、標準偏差1）
scaler = StandardScaler()
if data_array.size == 0:
    print("エラー: data_array が空です。データの読み込みに問題があります。")
else:
    data_array_norm = scaler.fit_transform(data_array)

    # 主成分分析による次元削減
    pca_components = 20
    pca = PCA(n_components=pca_components, svd_solver="full", random_state=0)
    data_array_pca = pca.fit_transform(data_array_norm)
    print(f"データの形状（PCA後）: {data_array_pca.shape}")

# 開始日と終了日を date 型に変換
start_date_date = start_date.date()
end_date_date = end_date.date()

# 冬型の気圧配置の日付を抽出し、解析期間内にフィルタリング
dates_with_label_A = [date for date, label in data_label_dict.items() if label == "A"]
dates_with_label_A_str = [date.strftime("%Y-%m-%d") for date in dates_with_label_A]
all_winter_dates_list = list(
    OrderedDict.fromkeys(winter_dates_list + dates_with_label_A_str)
)
filtered_winter_dates_list = [
    date_str
    for date_str in all_winter_dates_list
    if start_date_date
    <= datetime.strptime(date_str, "%Y-%m-%d").date()
    <= end_date_date
]
print(f"冬型の気圧配置の日付の総数: {len(filtered_winter_dates_list)} 日")
winter_dates_set = set(
    datetime.strptime(date_str, "%Y-%m-%d").date()
    for date_str in filtered_winter_dates_list
)
is_winter_type = [date in winter_dates_set for date in date_list]

# 夏型の気圧配置の日付を抽出し、解析期間内にフィルタリング
dates_with_label_L = [date for date, label in data_label_dict.items() if label == "L"]
dates_with_label_L_str = [date.strftime("%Y-%m-%d") for date in dates_with_label_L]
all_summer_dates_list = list(
    OrderedDict.fromkeys(summer_dates_list + dates_with_label_L_str)
)
filtered_summer_dates_list = [
    date_str
    for date_str in all_summer_dates_list
    if start_date_date
    <= datetime.strptime(date_str, "%Y-%m-%d").date()
    <= end_date_date
]
print(f"夏型の気圧配置の日付の総数: {len(filtered_summer_dates_list)} 日")
summer_dates_set = set(
    datetime.strptime(date_str, "%Y-%m-%d").date()
    for date_str in filtered_summer_dates_list
)
is_summer_type = [date in summer_dates_set for date in date_list]

# 梅雨型の気圧配置の日付を抽出し、解析期間内にフィルタリング
dates_with_label_JK = [
    date for date, label in data_label_dict.items() if label in ("J", "K")
]
dates_with_label_JK_str = [date.strftime("%Y-%m-%d") for date in dates_with_label_JK]
all_rain_dates_list = list(
    OrderedDict.fromkeys(rain_dates_list + dates_with_label_JK_str)
)
filtered_rain_dates_list = [
    date_str
    for date_str in all_rain_dates_list
    if start_date_date
    <= datetime.strptime(date_str, "%Y-%m-%d").date()
    <= end_date_date
]
print(f"梅雨型の気圧配置の日付の総数: {len(filtered_rain_dates_list)} 日")
rain_dates_set = set(
    datetime.strptime(date_str, "%Y-%m-%d").date()
    for date_str in filtered_rain_dates_list
)
is_rain_type = [date in rain_dates_set for date in date_list]

# 台風型の気圧配置の日付を抽出し、解析期間内にフィルタリング
dates_with_label_MNO = [
    date for date, label in data_label_dict.items() if label in ("M", "N", "O")
]
dates_with_label_MNO_str = [date.strftime("%Y-%m-%d") for date in dates_with_label_MNO]
all_typhoon_dates_list = list(
    OrderedDict.fromkeys(typhoon_dates_list + dates_with_label_MNO_str)
)
filtered_typhoon_dates_list = [
    date_str
    for date_str in all_typhoon_dates_list
    if start_date_date
    <= datetime.strptime(date_str, "%Y-%m-%d").date()
    <= end_date_date
]
print(f"台風型の気圧配置の日付の総数: {len(filtered_typhoon_dates_list)} 日")
typhoon_dates_set = set(
    datetime.strptime(date_str, "%Y-%m-%d").date()
    for date_str in filtered_typhoon_dates_list
)
is_typhoon_type = [date in typhoon_dates_set for date in date_list]

print_memory_usage("気圧配置のフィルタリング後")

# SOMのサイズ (行数, 列数)
som_rows = 10
som_cols = 10

# 近傍関数の範囲
sigma = 3.0

# 学習率
learning_rate = 1.0

# イテレーション数
num_iterations = 100000

# MiniSomのインスタンスを作成
som = MiniSom(
    som_rows,
    som_cols,
    data_array_pca.shape[1],
    sigma=sigma,
    learning_rate=learning_rate,
    neighborhood_function="gaussian",
    random_seed=0,
)

# 初期化
som.random_weights_init(data_array_pca)

# 学習（バッチ学習）
som.train_batch(data_array_pca, num_iterations, verbose=False)

# 各データポイントの対応するノードを取得
mapped_nodes = [som.winner(x) for x in data_array_pca]

# ノードごとにデータの数をカウントする配列を作成
node_total_counts = np.zeros((som_rows, som_cols), dtype=int)

# 各ノードごとのデータインデックスのリストを作成
node_data_indices = defaultdict(list)

# ノードごとの各タイプのデータ数をカウントする配列を作成
node_winter_counts = np.zeros((som_rows, som_cols), dtype=int)
node_summer_counts = np.zeros((som_rows, som_cols), dtype=int)
node_rain_counts = np.zeros((som_rows, som_cols), dtype=int)
node_typhoon_counts = np.zeros((som_rows, som_cols), dtype=int)

for idx, node in enumerate(mapped_nodes):
    i, j = node
    node_total_counts[i, j] += 1
    node_data_indices[(i, j)].append(idx)
    if is_winter_type[idx]:
        node_winter_counts[i, j] += 1
    if is_summer_type[idx]:
        node_summer_counts[i, j] += 1
    if is_rain_type[idx]:
        node_rain_counts[i, j] += 1
    if is_typhoon_type[idx]:
        node_typhoon_counts[i, j] += 1

# 不要な変数を削除
del mapped_nodes, is_winter_type, is_summer_type, is_rain_type, is_typhoon_type
gc.collect()
print_memory_usage("不要な変数削除後")

# ノードごとの各タイプのデータの割合を計算
node_winter_proportions = np.zeros((som_rows, som_cols))
node_summer_proportions = np.zeros((som_rows, som_cols))
node_rain_proportions = np.zeros((som_rows, som_cols))
node_typhoon_proportions = np.zeros((som_rows, som_cols))

for i in range(som_rows):
    for j in range(som_cols):
        total = node_total_counts[i, j]
        if total > 0:
            node_winter_proportions[i, j] = node_winter_counts[i, j] / total
            node_summer_proportions[i, j] = node_summer_counts[i, j] / total
            node_rain_proportions[i, j] = node_rain_counts[i, j] / total
            node_typhoon_proportions[i, j] = node_typhoon_counts[i, j] / total
        else:
            node_winter_proportions[i, j] = np.nan
            node_summer_proportions[i, j] = np.nan
            node_rain_proportions[i, j] = np.nan
            node_typhoon_proportions[i, j] = np.nan

# 結果を保存するディレクトリを作成
result_dir = f"/home/devel/work_C/work_ytakano/GSMGPV/som_{som_rows}x{som_cols}_sigma{sigma}_lr{learning_rate}_iter{num_iterations}"
os.makedirs(result_dir, exist_ok=True)

# 以下、図の作成と保存
# U-Matrixの作成と保存
u_matrix = som.distance_map()
plt.figure(figsize=(7, 7))
plt.title("U-Matrix")
plt.imshow(u_matrix, origin="lower", extent=(0, som_cols, 0, som_rows), cmap="coolwarm")
plt.colorbar(label="Distance")
plt.xlabel("Column Index (j)")
plt.ylabel("Row Index (i)")
umatrix_filename = "UMatrix.png"
plt.savefig(os.path.join(result_dir, umatrix_filename))
plt.close()

# 学習データのヒートマップを作成と保存
plt.figure(figsize=(8, 6))
plt.title("Data Mapping on SOM (Training Data)")
im = plt.imshow(
    node_total_counts,
    origin="lower",
    extent=(0, som_cols, 0, som_rows),
    cmap="Blues",
    aspect="auto",
)
plt.colorbar(label="Number of Data Points")
plt.xlabel("Column Index (j)")
plt.ylabel("Row Index (i)")

# 各セルにデータ数を表示
for i in range(som_rows):
    for j in range(som_cols):
        count = node_total_counts[i, j]
        plt.text(j + 0.5, i + 0.5, str(count), ha="center", va="center", color="black")
trainmap_filename = "TrainMap.png"
plt.savefig(os.path.join(result_dir, trainmap_filename))
plt.close()


# 各タイプごとにヒートマップを作成と保存
def plot_type_counts_and_proportions(
    type_name, node_type_counts, node_type_proportions
):
    # Counts heatmap
    plt.figure(figsize=(8, 6))
    plt.title(f"{type_name} Type Data Mapping on SOM")
    im = plt.imshow(
        node_type_counts,
        origin="lower",
        extent=(0, som_cols, 0, som_rows),
        cmap="Reds",
        aspect="auto",
    )
    plt.colorbar(label=f"Number of {type_name} Type Data Points")
    plt.xlabel("Column Index (j)")
    plt.ylabel("Row Index (i)")

    # 各セルにデータ数を表示
    for i in range(som_rows):
        for j in range(som_cols):
            count = node_type_counts[i, j]
            plt.text(
                j + 0.5, i + 0.5, str(count), ha="center", va="center", color="black"
            )
    counts_filename = f"{type_name}TypeMap.png"
    plt.savefig(os.path.join(result_dir, counts_filename))
    plt.close()

    # Proportions heatmap
    plt.figure(figsize=(8, 6))
    plt.title(f"Proportion of {type_name} Type Data on SOM Nodes")
    im = plt.imshow(
        node_type_proportions,
        origin="lower",
        extent=(0, som_cols, 0, som_rows),
        cmap="coolwarm",
        aspect="auto",
        vmin=0,
        vmax=1,
    )
    plt.colorbar(label=f"Proportion of {type_name} Type Data")
    plt.xlabel("Column Index (j)")
    plt.ylabel("Row Index (i)")

    # 各セルに割合を表示
    for i in range(som_rows):
        for j in range(som_cols):
            proportion = node_type_proportions[i, j]
            if not np.isnan(proportion):
                plt.text(
                    j + 0.5,
                    i + 0.5,
                    f"{proportion:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                )
    prop_filename = f"{type_name}TypeProportionMap.png"
    plt.savefig(os.path.join(result_dir, prop_filename))
    plt.close()


# 各タイプのヒートマップを作成
plot_type_counts_and_proportions("Winter", node_winter_counts, node_winter_proportions)
plot_type_counts_and_proportions("Summer", node_summer_counts, node_summer_proportions)
plot_type_counts_and_proportions("Rain", node_rain_counts, node_rain_proportions)
plot_type_counts_and_proportions(
    "Typhoon", node_typhoon_counts, node_typhoon_proportions
)

# 各ノードの平均パターンを計算
mean_patterns = np.full((som_rows, som_cols, data_array.shape[1]), np.nan)
for (i, j), indices_list in node_data_indices.items():
    indices = indices_list
    if len(indices) > 0:
        mean_pattern = data_array[indices].mean(axis=0)
        mean_patterns[i, j, :] = mean_pattern

# プロットの準備
cmap = sns.color_palette("RdBu_r", as_cmap=True)

# カラーバーの範囲を統一
pressure_vmin = -40
pressure_vmax = 40
pressure_levels = np.linspace(pressure_vmin, pressure_vmax, 21)
pressure_norm = Normalize(vmin=pressure_vmin, vmax=pressure_vmax)

# 図とサブプロットを作成
fig, axes = plt.subplots(
    nrows=som_rows,
    ncols=som_cols,
    figsize=(som_cols * 4, som_rows * 4),
    subplot_kw={"projection": ccrs.PlateCarree()},
)

# 行インデックスを反転して左下を(0,0)に設定
axes = axes[::-1, :]

# サブプロット間の間隔を調整
fig.subplots_adjust(wspace=0.06, hspace=0.06)

# 各ノードの平均気圧パターンをプロット
for i in range(som_rows):
    for j in range(som_cols):
        ax = axes[i, j]
        mean_pattern = mean_patterns[i, j, :]
        if not np.isnan(mean_pattern).all():
            # 平均パターンを元の形状にリシェイプ
            mean_pattern_2d = mean_pattern.reshape(len(common_lat), len(common_lon))
            # 塗りつぶし等高線プロット
            cont = ax.contourf(
                common_lon,
                common_lat,
                mean_pattern_2d,
                levels=pressure_levels,
                cmap=cmap,
                extend="both",
                norm=pressure_norm,
                transform=ccrs.PlateCarree(),
            )
            # 等高線（線のみ、黒色）
            ax.contour(
                common_lon,
                common_lat,
                mean_pattern_2d,
                levels=pressure_levels,
                colors="k",
                linewidths=0.5,
                transform=ccrs.PlateCarree(),
            )
            # 海岸線をプロット
            ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="black")
            # 軸設定
            ax.set_extent([115, 155, 15, 55], crs=ccrs.PlateCarree())
            ax.set_xticks([])
            ax.set_yticks([])
            # ノードに含まれるデータ数を表示
            count = node_total_counts[i, j]
            ax.text(
                0.05,
                0.9,
                f"({i},{j}) {count}",
                transform=ax.transAxes,
                fontsize=16,
                color="black",
                ha="left",
                va="top",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
            )
        else:
            # データがない場合は軸をオフに
            ax.axis("off")

# カラーバーを追加（位置とサイズを調整して余白を削除）
cbar_ax = fig.add_axes([0.92, 0.02, 0.015, 0.96])  # [left, bottom, width, height]
sm = plt.cm.ScalarMappable(cmap=cmap, norm=pressure_norm)
sm.set_array([])
fig.colorbar(sm, cax=cbar_ax, label="Sea Level Pressure (hPa)")

# 図の余白を調整（サブプロット間の間隔と全体の余白を削減）
fig.subplots_adjust(
    wspace=0.06, hspace=0.06, left=0.02, right=0.90, bottom=0.02, top=0.98
)

mean_pattern_filename = "MeanSeaLevelPressurePatterns.png"
plt.savefig(
    os.path.join(result_dir, mean_pattern_filename), dpi=300, bbox_inches="tight"
)  # bbox_inches='tight'を追加
plt.close()


# 不要な変数を削除
del fig, axes, mean_patterns
gc.collect()
print_memory_usage("図関連の変数削除後")

# 月から季節へのマッピングを定義
month_to_season = {
    1: "冬",
    2: "冬",
    3: "春",
    4: "春",
    5: "春",
    6: "夏",
    7: "夏",
    8: "夏",
    9: "秋",
    10: "秋",
    11: "秋",
    12: "冬",
}
seasons = ["春", "夏", "秋", "冬"]

# 季節から色へのマッピングを定義（薄い色を使用）
season_colors = {
    "春": "#FFE4E1",  # Misty Rose（薄いピンク）
    "夏": "#FFB6C1",  # Light Pink（薄い赤）
    "秋": "#F5DEB3",  # Wheat（薄い茶色）
    "冬": "#ADD8E6",  # Light Blue（薄い青）
    "": "white",  # データがない場合は白
}

# 各ノードに対応する日付のリストを作成
node_date_list = {}

# ノードごとの月ごとのカウントと割合を計算
node_month_counts = np.zeros((som_rows, som_cols, 12), dtype=int)
node_month_proportions = np.zeros((som_rows, som_cols, 12), dtype=float)
node_season_counts = {}  # 季節ごとのカウントを保存する辞書

# 各ノードの割り当てられた日付をmdファイルに保存
dates_output_dir = os.path.join(result_dir, "node_dates")
os.makedirs(dates_output_dir, exist_ok=True)

for (i, j), indices_list in node_data_indices.items():
    # インデックスから対応する日付を取得
    dates = [date_list[idx] for idx in indices_list]
    node_date_list[(i, j)] = dates

    # 月ごとのカウントを計算
    months = [date.month for date in dates]
    total_dates = len(dates)
    month_counts = np.zeros(12, dtype=int)
    for month in months:
        month_counts[month - 1] += 1  # 月を0ベースのインデックスに変換

    node_month_counts[i, j, :] = month_counts

    # 月ごとの割合を計算
    if total_dates > 0:
        month_proportions = month_counts / total_dates
    else:
        month_proportions = np.zeros(12)

    node_month_proportions[i, j, :] = month_proportions

    # 季節ごとのカウントを計算
    season_counts = {season: 0 for season in seasons}
    for date in dates:
        month = date.month
        season = month_to_season[month]
        season_counts[season] += 1

    # 一番多い季節を取得
    if total_dates > 0:
        max_season = max(season_counts.items(), key=lambda x: x[1])[0]
    else:
        max_season = ""

    # ノードの季節情報を保存
    node_season_counts[(i, j)] = {"counts": season_counts, "max_season": max_season}

    # 結果をファイルに保存（mdファイルとして）
    node_filename = f"node_{i}_{j}_dates.md"
    node_file_path = os.path.join(dates_output_dir, node_filename)
    with open(node_file_path, "w", encoding="utf-8") as f:
        f.write(f"# ノード ({i},{j}) に割り当てられた日付:\\n")
        for date in dates:
            f.write(f"- {date}\\n")
        f.write("\\n")
        f.write("## 各月のデータ数と割合:\\n")
        for month in range(1, 13):
            count = month_counts[month - 1]
            proportion = month_proportions[month - 1]
            f.write(f"- {month}月: {count}日 ({proportion:.2%})\\n")
        f.write("\\n")
        f.write("## 各季節のデータ数:\\n")
        for season in seasons:
            count = season_counts[season]
            f.write(f"- {season}: {count}日\\n")
        f.write(f"\\n### 最も多い季節: {max_season}\\n")

# 図とサブプロットを作成
figure_scale = 1.2  # 図のサイズを調整
fig, axes = plt.subplots(
    nrows=som_rows,
    ncols=som_cols,
    figsize=(som_cols * figure_scale, som_rows * figure_scale),
)

# サブプロット間の間隔を調整
fig.subplots_adjust(wspace=0.5, hspace=0.5)

# 行インデックスを反転して左下を(0,0)に設定
axes = axes[::-1, :]

# フォントサイズを調整
fontsize = 5

# 各ノードの上位3つの月と最も多い季節を表示し、背景色を設定
for i in range(som_rows):
    for j in range(som_cols):
        ax = axes[i, j]
        # 軸目盛りと枠線を非表示
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        total_dates = node_total_counts[i, j]
        if total_dates == 0:
            # データがない場合は背景を白に設定
            ax.set_facecolor("white")
            continue

        # 月ごとのデータ数と割合を取得
        month_counts = node_month_counts[i, j, :]
        month_proportions = node_month_proportions[i, j, :]

        # 上位3つの月を取得
        top_indices = np.argsort(month_counts)[::-1][:3]
        top_months = top_indices + 1  # 月は1から12
        top_counts = month_counts[top_indices]
        top_proportions = month_proportions[top_indices]

        # 一番多い季節を取得
        max_season = node_season_counts[(i, j)]["max_season"]

        # 季節に応じて背景色を設定
        season_color = season_colors.get(max_season, "white")
        ax.set_facecolor(season_color)

        # 各行のテキストを準備
        text_lines = [f"({i},{j}) N={total_dates} {max_season}"]
        for k in range(len(top_months)):
            month = top_months[k]
            count = top_counts[k]
            proportion = top_proportions[k]
            text_lines.append(f"{month}月: {count}日 ({proportion:.0%})")

        # 各行を個別にプロット（y座標をずらす）
        y_positions = np.linspace(0.7, 0.3, len(text_lines))
        for text, y in zip(text_lines, y_positions):
            ax.text(
                0.5,
                y,
                text,
                ha="center",
                va="center",
                fontsize=fontsize,
                transform=ax.transAxes,
                color="black",
            )

# 全体のタイトルを設定
plt.suptitle("各ノードにおける上位3つの月の分布と最も多い季節", fontsize=12)

# 図を保存
month_distribution_filename = "NodeMonthDistribution.png"
plt.savefig(
    os.path.join(result_dir, month_distribution_filename), dpi=300, bbox_inches="tight"
)  # 解像度を300dpiに設定
plt.close()

# 必要なライブラリのインポート
import gc  # ガベージコレクションのために追加
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import psutil  # メモリ使用量の取得のために追加
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap

process = psutil.Process()

print(f"プログラム開始時のメモリ使用量: {process.memory_info().rss / 1024 ** 2:.2f} MB")

# 天気コードと色のマッピングを定義（変更後）
weather_mapping = {
    0: ("undefined", "white"),  # 未定義の地点（0）は白色
    1: ("sunny", "orange"),
    2: ("cloudy", "grey"),
    3: ("rain/snow", "lightblue"),  # 3、4、5をまとめてrain/snowにする
}
weather_codes = list(weather_mapping.keys())  # [0, 1, 2, 3]

# 天気データのディレクトリ（適宜変更してください）
weather_data_dir = "/home/devel/work_C/work_ytakano/GSMGPV/nc/weather/"

# 緯度・経度の範囲を定義（日本付近）
lat_range = slice(23, 47)
lon_range = slice(122, 146)

print(f"初期設定完了時のメモリ使用量: {process.memory_info().rss / 1024 ** 2:.2f} MB")

# 天気データから共通の緯度・経度グリッドを取得
# 全体のdate_listから最初の日付を使用
sample_date = date_list[0]  # 利用可能な日付の最初のものを使用
sample_date_str = sample_date.strftime("%Y%m%d")
sample_file_path = os.path.join(
    weather_data_dir, f"{sample_date.strftime('%Y%m')}", f"wm_{sample_date_str}.nc"
)

print(f"サンプルファイル {sample_file_path} を読み込みます")

# 天気データを読み込み
ds_sample = xr.open_dataset(sample_file_path)

print(
    f"サンプルファイル読み込み後のメモリ使用量: {process.memory_info().rss / 1024 ** 2:.2f} MB"
)

# 座標名を標準化
if "lat" in ds_sample.coords and "lon" in ds_sample.coords:
    ds_sample = ds_sample.rename({"lat": "latitude", "lon": "longitude"})

# 緯度・経度を取得し、指定の範囲でフィルタリング
all_latitudes = ds_sample["latitude"].values
all_longitudes = ds_sample["longitude"].values

# 緯度・経度を指定の範囲でフィルタリング
common_lat = all_latitudes[
    (all_latitudes >= lat_range.start) & (all_latitudes <= lat_range.stop)
]
common_lon = all_longitudes[
    (all_longitudes >= lon_range.start) & (all_longitudes <= lon_range.stop)
]

n_lat = len(common_lat)
n_lon = len(common_lon)

print(f"共通緯度数: {n_lat}, 共通経度数: {n_lon}")
print(
    f"緯度・経度グリッド取得後のメモリ使用量: {process.memory_info().rss / 1024 ** 2:.2f} MB"
)

# 最後に使用したデータセットを閉じて削除
ds_sample.close()
del ds_sample
gc.collect()
print("サンプルデータセットを削除しました")
print(
    f"サンプルデータセット削除後のメモリ使用量: {process.memory_info().rss / 1024 ** 2:.2f} MB"
)

# 各ノードのfiltered_weather_codesを格納する配列を初期化
filtered_weather_codes = np.zeros((som_rows, som_cols, n_lat, n_lon), dtype=int)
print(
    f"filtered_weather_codes 配列を初期化しました。サイズ: {filtered_weather_codes.nbytes / 1024 ** 2:.2f} MB"
)

for i in range(som_rows):
    for j in range(som_cols):
        node = (i, j)
        dates = node_date_list.get(node, [])
        if len(dates) == 0:
            continue  # データがない場合は次へ

        print(f"ノード ({i},{j}) の処理を開始します。日付数: {len(dates)}")
        wm_list = []
        for date in dates:
            # 日付からファイルパスを構築
            date_str = date.strftime("%Y%m%d")
            file_path = os.path.join(
                weather_data_dir, f"{date.strftime('%Y%m')}", f"wm_{date_str}.nc"
            )

            if not os.path.exists(file_path):
                print(f"警告: 天気データファイルが見つかりません: {file_path}")
                continue

            # 天気データを読み込み
            ds = xr.open_dataset(file_path)

            # 座標名を標準化
            if "lat" in ds.coords and "lon" in ds.coords:
                ds = ds.rename({"lat": "latitude", "lon": "longitude"})

            # 緯度・経度を昇順にソート
            if ds["latitude"].values[0] > ds["latitude"].values[-1]:
                ds = ds.sortby("latitude")
            if ds["longitude"].values[0] > ds["longitude"].values[-1]:
                ds = ds.sortby("longitude")

            # 指定の緯度経度範囲でデータをサブセット
            ds_subset = ds.sel(latitude=lat_range, longitude=lon_range)

            # 天気コードを取得
            wm = ds_subset["wm"]

            # 天気コードのリマッピング（3,4,5を3に統合）
            wm_values = wm.values.astype(int)  # 形状: (n_lat, n_lon)
            wm_values[wm_values == 4] = 3
            wm_values[wm_values == 5] = 3

            # データをリストに追加
            wm_list.append(wm_values)

            # 使い終わったデータセットを閉じて削除
            ds.close()
            del ds, ds_subset, wm, wm_values
            gc.collect()

        if len(wm_list) == 0:
            print(f"ノード ({i},{j}) に利用可能な天気データがありません。")
            continue

        print(f"ノード ({i},{j}) のデータをスタックします。")
        # データをスタックして3次元配列に
        wm_stack = np.stack(wm_list, axis=0)  # 形状: (日数, n_lat, n_lon)
        print(
            f"wm_stack の形状: {wm_stack.shape}, サイズ: {wm_stack.nbytes / 1024 ** 2:.2f} MB"
        )
        print(
            f"wm_stack 作成後のメモリ使用量: {process.memory_info().rss / 1024 ** 2:.2f} MB"
        )

        # 各グリッドポイントで最頻値とその割合を計算（ベクトル化）
        print(f"ノード ({i},{j}) の最頻天気コードとその割合を計算します。")

        # データをリシェイプ
        n_dates = wm_stack.shape[0]
        wm_stack_reshaped = wm_stack.reshape(
            n_dates, -1
        )  # 形状: (n_dates, n_lat*n_lon)

        # 天気コードの出現回数を計算
        counts = np.zeros(
            (len(weather_codes), wm_stack_reshaped.shape[1]), dtype=int
        )  # (4, n_lat*n_lon)

        for idx, code in enumerate(weather_codes):
            counts[idx, :] = np.sum(wm_stack_reshaped == code, axis=0)

        # 最頻天気コードとその出現回数を取得
        most_common_indices = np.argmax(counts, axis=0)  # 形状: (n_lat*n_lon,)
        max_counts = counts[
            most_common_indices, np.arange(wm_stack_reshaped.shape[1])
        ]  # 形状: (n_lat*n_lon,)
        total_counts = np.sum(counts, axis=0)  # 形状: (n_lat*n_lon,)

        # 最頻天気コードを取得
        most_common_codes = np.array(weather_codes)[
            most_common_indices
        ]  # 形状: (n_lat*n_lon,)

        # 比率を計算
        proportions = max_counts / total_counts

        # 条件を適用して天気コードをフィルタリング
        filtered_codes_flat = np.where(
            (proportions >= 0.5) & (most_common_codes != 0),
            most_common_codes,
            0,  # それ以外は0（白）を設定
        )

        # 元の形状に戻す
        filtered_codes_node = filtered_codes_flat.reshape(n_lat, n_lon)

        # このノードの結果を保存
        filtered_weather_codes[i, j, :, :] = filtered_codes_node

        # 使用し終わった変数を削除してメモリを解放
        del (
            wm_list,
            wm_stack,
            wm_stack_reshaped,
            counts,
            most_common_indices,
            max_counts,
            total_counts,
            most_common_codes,
            proportions,
            filtered_codes_flat,
            filtered_codes_node,
        )
        gc.collect()
        print(
            f"ノード ({i},{j}) の処理が完了しました。メモリ使用量: {process.memory_info().rss / 1024 ** 2:.2f} MB"
        )

print("全ノードの処理が完了しました。")
print(
    f"filtered_weather_codes の最終サイズ: {filtered_weather_codes.nbytes / 1024 ** 2:.2f} MB"
)

# プロットの準備
codes = list(weather_mapping.keys())  # [0,1,2,3]
colors = [weather_mapping[code][1] for code in codes]

cmap = ListedColormap(colors)
norm = BoundaryNorm(np.arange(-0.5, len(codes) - 0.5 + 1), cmap.N)  # 境界を正しく設定

# 結果をプロットします
print("結果をプロットします...")
fig, axes = plt.subplots(
    nrows=som_rows,
    ncols=som_cols,
    figsize=(som_cols * 4, som_rows * 4),
    subplot_kw={"projection": ccrs.PlateCarree()},
)

# 各ノードの天気パターンをまとめて nc ファイルに保存
print("全てのノードの天気パターンを一つの nc ファイルに保存します...")

# xarray Dataset を作成
ds = xr.Dataset(
    {
        "weather_code": (
            ("node_row", "node_col", "latitude", "longitude"),
            filtered_weather_codes,
        )
    },
    coords={
        "node_row": np.arange(som_rows),
        "node_col": np.arange(som_cols),
        "latitude": common_lat,
        "longitude": common_lon,
    },
)

# エンコーディングを設定
encoding = {
    "weather_code": {"dtype": "int8", "zlib": True, "complevel": 5, "contiguous": False}
}

# nc ファイルに保存
output_filename = os.path.join(result_dir, "all_nodes_weather_patterns.nc")
ds.to_netcdf(output_filename, encoding=encoding)

print("全てのノードの天気パターンの nc ファイルの保存が完了しました。")


# 行インデックスを反転して左下を(0,0)に設定
axes = axes[::-1, :]

# サブプロット間の間隔を調整
fig.subplots_adjust(wspace=0.06, hspace=0.06)

# 各ノードの天気パターンをプロット
for i in range(som_rows):
    for j in range(som_cols):
        ax = axes[i, j]
        wm_mode = filtered_weather_codes[i, j, :, :]

        if np.any(wm_mode > 0):  # データが存在するか確認（0以外の値があるか）
            # 天気パターンをプロット
            im = ax.pcolormesh(
                common_lon,
                common_lat,
                wm_mode,
                cmap=cmap,
                norm=norm,
                shading="auto",
                transform=ccrs.PlateCarree(),
            )

            # 海岸線と国境を追加
            ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="black")
            ax.add_feature(cfeature.BORDERS.with_scale("10m"), linestyle=":")

            # 範囲を設定
            ax.set_extent(
                [lon_range.start, lon_range.stop, lat_range.start, lat_range.stop],
                crs=ccrs.PlateCarree(),
            )

            # 軸目盛りを非表示
            ax.set_xticks([])
            ax.set_yticks([])

            # ノード情報をテキストで表示
            count = node_total_counts[i, j]
            ax.text(
                0.05,
                0.9,
                f"({i},{j}) {count}",
                transform=ax.transAxes,
                fontsize=16,
                color="black",
                ha="left",
                va="top",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
            )
        else:
            # データがない場合は軸をオフに
            ax.axis("off")

# サブプロット間の間隔を調整（上と右の余白を削除）
fig.subplots_adjust(
    wspace=0.06, hspace=0.06, left=0.02, right=0.98, bottom=0.02, top=0.98
)

# カスタム凡例を作成（0以外の天気コードのみ）
legend_handles = [
    mpatches.Patch(color=weather_mapping[code][1], label=weather_mapping[code][0])
    for code in codes
    if code != 0
]

# 凡例を右下に配置
fig.legend(
    handles=legend_handles, loc="lower right", bbox_to_anchor=(0.99, 0.01), fontsize=16
)

# 図を保存
weather_pattern_filename = "MostCommonWeatherPatterns.png"
plt.savefig(
    os.path.join(result_dir, weather_pattern_filename), dpi=300, bbox_inches="tight"
)  # 解像度を300dpiに設定
plt.close()
print("図の作成と保存が完了しました。")

# 使用し終わった変数を削除
del fig, axes, im, filtered_weather_codes
gc.collect()
print(f"最終的なメモリ使用量: {process.memory_info().rss / 1024 ** 2:.2f} MB")

# 冬型ノードのリストを定義
winter_nodes = [
    (6, 0),
    (7, 0),
    (8, 0),
    (9, 0),
    (5, 1),
    (6, 1),
    (7, 1),
    (8, 1),
    (5, 2),
    (6, 2),
    (7, 2),
    (6, 3),
    (7, 3),
    (6, 4),
    (5, 3),
]

# 指定されたノードに割り当てられた日付をすべて結合
winter_dates = []
for node in winter_nodes:
    dates = node_date_list.get(node, [])
    winter_dates.extend(dates)

# 日付の重複を除去
winter_dates = list(set(winter_dates))

# 天気データを格納するリストを初期化
wm_list = []

print(f"冬型ノードに割り当てられた日付の数: {len(winter_dates)}")

# 各日付に対して天気データを読み込み
for date in winter_dates:
    # 日付からファイルパスを構築
    date_str = date.strftime("%Y%m%d")
    file_path = os.path.join(
        weather_data_dir, f"{date.strftime('%Y%m')}", f"wm_{date_str}.nc"
    )

    if not os.path.exists(file_path):
        print(f"警告: 天気データファイルが見つかりません: {file_path}")
        continue

    # 天気データを読み込み
    ds = xr.open_dataset(file_path)

    # 座標名を標準化
    if "lat" in ds.coords and "lon" in ds.coords:
        ds = ds.rename({"lat": "latitude", "lon": "longitude"})

    # 緯度・経度を昇順にソート
    if ds["latitude"].values[0] > ds["latitude"].values[-1]:
        ds = ds.sortby("latitude")
    if ds["longitude"].values[0] > ds["longitude"].values[-1]:
        ds = ds.sortby("longitude")

    # 指定の緯度経度範囲でデータをサブセット
    ds_subset = ds.sel(latitude=lat_range, longitude=lon_range)

    # 天気コードを取得
    wm = ds_subset["wm"]

    # 天気コードのリマッピング（3,4,5を3に統合）
    wm_values = wm.values.astype(int)  # 形状: (n_lat, n_lon)
    wm_values[wm_values == 4] = 3
    wm_values[wm_values == 5] = 3

    # データをリストに追加
    wm_list.append(wm_values)

    # 使い終わったデータセットを閉じて削除
    ds.close()
    del ds, ds_subset, wm, wm_values
    gc.collect()

# データが存在するか確認
if len(wm_list) == 0:
    print("冬型ノードに割り当てられた日付の天気データがありません。")
else:
    print(f"冬型ノードの日付のデータをスタックします。")
    # データをスタックして3次元配列に
    wm_stack = np.stack(wm_list, axis=0)  # 形状: (日数, n_lat, n_lon)
    print(f"wm_stack の形状: {wm_stack.shape}")

    # 各グリッドポイントで最頻値とその割合を計算（ベクトル化）
    print("最頻天気コードとその割合を計算します。")

    # データをリシェイプ
    n_dates = wm_stack.shape[0]
    wm_stack_reshaped = wm_stack.reshape(n_dates, -1)  # 形状: (n_dates, n_lat*n_lon)

    # 天気コードの出現回数を計算
    counts = np.zeros(
        (len(weather_codes), wm_stack_reshaped.shape[1]), dtype=int
    )  # (コード数, n_lat*n_lon)

    for idx, code in enumerate(weather_codes):
        counts[idx, :] = np.sum(wm_stack_reshaped == code, axis=0)

    # 最頻天気コードとその出現回数を取得
    most_common_indices = np.argmax(counts, axis=0)  # 形状: (n_lat*n_lon,)
    max_counts = counts[
        most_common_indices, np.arange(wm_stack_reshaped.shape[1])
    ]  # 形状: (n_lat*n_lon,)
    total_counts = np.sum(counts, axis=0)  # 形状: (n_lat*n_lon,)

    # 最頻天気コードを取得
    most_common_codes = np.array(weather_codes)[
        most_common_indices
    ]  # 形状: (n_lat*n_lon,)

    # 比率を計算
    proportions = max_counts / total_counts

    # 条件を適用して天気コードをフィルタリング
    filtered_codes_flat = np.where(
        (proportions >= 0.5) & (most_common_codes != 0),
        most_common_codes,
        0,  # それ以外は0（白）を設定
    )

    # 元の形状に戻す
    filtered_codes_winter = filtered_codes_flat.reshape(n_lat, n_lon)

    # プロットの準備
    codes = list(weather_mapping.keys())  # [0,1,2,3]
    colors = [weather_mapping[code][1] for code in codes]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(
        np.arange(-0.5, len(codes) - 0.5 + 1), cmap.N
    )  # 境界を正しく設定

    # 結果をプロットします
    # 修正: plt.figure() ではなく plt.subplots() を使用し、軸を取得
    fig, ax = plt.subplots(
        figsize=(8, 6), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    # 天気パターンをプロット
    im = ax.pcolormesh(
        common_lon,
        common_lat,
        filtered_codes_winter,
        cmap=cmap,
        norm=norm,
        shading="auto",
        transform=ccrs.PlateCarree(),
    )

    # 海岸線と国境を追加
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="black")
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linestyle=":")

    # 範囲を設定
    ax.set_extent(
        [lon_range.start, lon_range.stop, lat_range.start, lat_range.stop],
        crs=ccrs.PlateCarree(),
    )

    # カスタム凡例を作成（0以外の天気コードのみ）
    legend_handles = [
        mpatches.Patch(color=weather_mapping[code][1], label=weather_mapping[code][0])
        for code in codes
        if code != 0
    ]

    # 凡例を図に追加
    ax.legend(handles=legend_handles, loc="upper right", fontsize=12)

    # タイトルを設定
    plt.title(
        "冬型ノードに割り当てられた日付の最も一般的な天気パターン（過半数の場合）",
        fontsize=16,
    )

    # 結果を保存するディレクトリを確認
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 図を保存
    winter_weather_pattern_filename = "WinterNodesWeatherPattern.png"
    plt.savefig(
        os.path.join(result_dir, winter_weather_pattern_filename),
        dpi=300,
        bbox_inches="tight",
    )  # 解像度を300dpiに設定
    plt.close()
    print("冬型ノードの日付の天気パターンの図を作成し、保存しました。")

    # 冬型ノードの天気パターンを nc ファイルに保存
    print("冬型ノードの天気パターンを nc ファイルに保存します...")
    ds = xr.Dataset(
        {"weather_code": (("latitude", "longitude"), filtered_codes_winter)},
        coords={"latitude": common_lat, "longitude": common_lon},
    )
    # エンコーディングを設定
    encoding = {
        "weather_code": {
            "dtype": "int8",
            "zlib": True,
            "complevel": 5,
            "contiguous": False,
        }
    }
    output_filename = os.path.join(result_dir, "winter_weather_pattern.nc")
    ds.to_netcdf(output_filename, encoding=encoding)
    print("冬型ノードの天気パターンの nc ファイルの保存が完了しました。")

    # 使用し終わった変数を削除してメモリを解放
    del (
        fig,
        ax,
        im,
        filtered_codes_winter,
        wm_stack,
        wm_list,
        wm_stack_reshaped,
        counts,
        most_common_indices,
        max_counts,
        total_counts,
        most_common_codes,
        proportions,
        filtered_codes_flat,
    )
    gc.collect()
    print(
        f"冬型ノードの処理完了後のメモリ使用量: {process.memory_info().rss / 1024 ** 2:.2f} MB"
    )

# 夏型ノードのリストを定義
summer_nodes = [
    (8, 6),
    (9, 6),
    (8, 7),
    (9, 7),
    (8, 8),
    (9, 8),
    (9, 9),
    (7, 7),
    (9, 5),
    (9, 4),
    (9, 3),
]

# 指定されたノードに割り当てられた日付をすべて結合
summer_dates = []
for node in summer_nodes:
    dates = node_date_list.get(node, [])
    summer_dates.extend(dates)

# 日付の重複を除去
summer_dates = list(set(summer_dates))

# 天気データを格納するリストを初期化
wm_list = []

print(f"夏型ノードに割り当てられた日付の数: {len(summer_dates)}")

# 各日付に対して天気データを読み込み
for date in summer_dates:
    # 日付からファイルパスを構築
    date_str = date.strftime("%Y%m%d")
    file_path = os.path.join(
        weather_data_dir, f"{date.strftime('%Y%m')}", f"wm_{date_str}.nc"
    )

    if not os.path.exists(file_path):
        print(f"警告: 天気データファイルが見つかりません: {file_path}")
        continue

    # 天気データを読み込み
    ds = xr.open_dataset(file_path)

    # 座標名を標準化
    if "lat" in ds.coords and "lon" in ds.coords:
        ds = ds.rename({"lat": "latitude", "lon": "longitude"})

    # 緯度・経度を昇順にソート
    if ds["latitude"].values[0] > ds["latitude"].values[-1]:
        ds = ds.sortby("latitude")
    if ds["longitude"].values[0] > ds["longitude"].values[-1]:
        ds = ds.sortby("longitude")

    # 指定の緯度経度範囲でデータをサブセット
    ds_subset = ds.sel(latitude=lat_range, longitude=lon_range)

    # 天気コードを取得
    wm = ds_subset["wm"]

    # 天気コードのリマッピング（3,4,5を3に統合）
    wm_values = wm.values.astype(int)  # 形状: (n_lat, n_lon)
    wm_values[wm_values == 4] = 3
    wm_values[wm_values == 5] = 3

    # データをリストに追加
    wm_list.append(wm_values)

    # 使い終わったデータセットを閉じて削除
    ds.close()
    del ds, ds_subset, wm, wm_values
    gc.collect()

# データが存在するか確認
if len(wm_list) == 0:
    print("夏型ノードに割り当てられた日付の天気データがありません。")
    # データがない場合、全て0の配列を作成
    filtered_codes_summer = np.zeros((n_lat, n_lon), dtype=int)
else:
    print(f"夏型ノードの日付のデータをスタックします。")
    # データをスタックして3次元配列に
    wm_stack = np.stack(wm_list, axis=0)  # 形状: (日数, n_lat, n_lon)
    print(f"wm_stack の形状: {wm_stack.shape}")

    # 各グリッドポイントで最頻値とその割合を計算（ベクトル化）
    print("最頻天気コードとその割合を計算します。")

    # データをリシェイプ
    n_dates = wm_stack.shape[0]
    wm_stack_reshaped = wm_stack.reshape(n_dates, -1)  # 形状: (n_dates, n_lat*n_lon)

    # 天気コードの出現回数を計算
    counts = np.zeros(
        (len(weather_codes), wm_stack_reshaped.shape[1]), dtype=int
    )  # (コード数, n_lat*n_lon)

    for idx, code in enumerate(weather_codes):
        counts[idx, :] = np.sum(wm_stack_reshaped == code, axis=0)

    # 最頻天気コードとその出現回数を取得
    most_common_indices = np.argmax(counts, axis=0)  # 形状: (n_lat*n_lon,)
    max_counts = counts[
        most_common_indices, np.arange(wm_stack_reshaped.shape[1])
    ]  # 形状: (n_lat*n_lon,)
    total_counts = np.sum(counts, axis=0)  # 形状: (n_lat*n_lon,)

    # 最頻天気コードを取得
    most_common_codes = np.array(weather_codes)[
        most_common_indices
    ]  # 形状: (n_lat*n_lon,)

    # 比率を計算
    proportions = max_counts / total_counts

    # 条件を適用して天気コードをフィルタリング
    filtered_codes_flat = np.where(
        (proportions >= 0.5) & (most_common_codes != 0),
        most_common_codes,
        0,  # それ以外は0（白）を設定
    )

    # 元の形状に戻す
    filtered_codes_summer = filtered_codes_flat.reshape(n_lat, n_lon)

    # 使用し終わった変数を削除してメモリを解放
    del (
        wm_stack,
        wm_stack_reshaped,
        counts,
        most_common_indices,
        max_counts,
        total_counts,
        most_common_codes,
        proportions,
        filtered_codes_flat,
    )
    gc.collect()

# 夏型の天気パターンをプロット
# プロットの準備
codes = list(weather_mapping.keys())  # [0,1,2,3]
colors = [weather_mapping[code][1] for code in codes]

cmap = ListedColormap(colors)
norm = BoundaryNorm(np.arange(-0.5, len(codes) - 0.5 + 1), cmap.N)

# 結果をプロットします
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": ccrs.PlateCarree()})

# 天気パターンをプロット
im = ax.pcolormesh(
    common_lon,
    common_lat,
    filtered_codes_summer,
    cmap=cmap,
    norm=norm,
    shading="auto",
    transform=ccrs.PlateCarree(),
)

# 海岸線と国境を追加
ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="black")
ax.add_feature(cfeature.BORDERS.with_scale("10m"), linestyle=":")

# 範囲を設定
ax.set_extent(
    [lon_range.start, lon_range.stop, lat_range.start, lat_range.stop],
    crs=ccrs.PlateCarree(),
)

# カスタム凡例を作成（0以外の天気コードのみ）
legend_handles = [
    mpatches.Patch(color=weather_mapping[code][1], label=weather_mapping[code][0])
    for code in codes
    if code != 0
]

# 凡例を図に追加
ax.legend(handles=legend_handles, loc="upper right", fontsize=12)

# タイトルを設定
plt.title(
    "夏型ノードに割り当てられた日付の最も一般的な天気パターン（過半数の場合）",
    fontsize=16,
)

# 結果を保存するディレクトリを確認
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 図を保存
summer_weather_pattern_filename = "SummerNodesWeatherPattern.png"
plt.savefig(
    os.path.join(result_dir, summer_weather_pattern_filename),
    dpi=300,
    bbox_inches="tight",
)
plt.close()
print("夏型ノードの日付の天気パターンの図を作成し、保存しました。")

# 夏型ノードの天気パターンを nc ファイルに保存
print("夏型ノードの天気パターンを nc ファイルに保存します...")
ds = xr.Dataset(
    {"weather_code": (("latitude", "longitude"), filtered_codes_summer)},
    coords={"latitude": common_lat, "longitude": common_lon},
)
# エンコーディングを設定
encoding = {
    "weather_code": {"dtype": "int8", "zlib": True, "complevel": 5, "contiguous": False}
}
output_filename = os.path.join(result_dir, "summer_weather_pattern.nc")
ds.to_netcdf(output_filename, encoding=encoding)
print("夏型ノードの天気パターンの nc ファイルの保存が完了しました。")

# 使用し終わった変数を削除してメモリを解放
del fig, ax, im, filtered_codes_summer, wm_list, ds
gc.collect()
print(
    f"夏型ノードの処理完了後のメモリ使用量: {process.memory_info().rss / 1024 ** 2:.2f} MB"
)

# 梅雨型ノードのリストを定義
rain_nodes = [
    (6, 7),
    (5, 8),
    (6, 8),
    (7, 8),
    (6, 9),
    (7, 9),
    (8, 9),
    (5, 7),
    (5, 9),
    (4, 7),
    (4, 8),
    (4, 9),
    (3, 7),
    (3, 8),
    (3, 9),
]

# 指定されたノードに割り当てられた日付をすべて結合
rain_dates = []
for node in rain_nodes:
    dates = node_date_list.get(node, [])
    rain_dates.extend(dates)

# 日付の重複を除去
rain_dates = list(set(rain_dates))

# 天気データを格納するリストを初期化
wm_list = []

print(f"梅雨型ノードに割り当てられた日付の数: {len(rain_dates)}")

# 各日付に対して天気データを読み込み
for date in rain_dates:
    # 日付からファイルパスを構築
    date_str = date.strftime("%Y%m%d")
    file_path = os.path.join(
        weather_data_dir, f"{date.strftime('%Y%m')}", f"wm_{date_str}.nc"
    )

    if not os.path.exists(file_path):
        print(f"警告: 天気データファイルが見つかりません: {file_path}")
        continue

    # 天気データを読み込み
    ds = xr.open_dataset(file_path)

    # 座標名を標準化
    if "lat" in ds.coords and "lon" in ds.coords:
        ds = ds.rename({"lat": "latitude", "lon": "longitude"})

    # 緯度・経度を昇順にソート
    if ds["latitude"].values[0] > ds["latitude"].values[-1]:
        ds = ds.sortby("latitude")
    if ds["longitude"].values[0] > ds["longitude"].values[-1]:
        ds = ds.sortby("longitude")

    # 指定の緯度経度範囲でデータをサブセット
    ds_subset = ds.sel(latitude=lat_range, longitude=lon_range)

    # 天気コードを取得
    wm = ds_subset["wm"]

    # 天気コードのリマッピング（3,4,5を3に統合）
    wm_values = wm.values.astype(int)  # 形状: (n_lat, n_lon)
    wm_values[wm_values == 4] = 3
    wm_values[wm_values == 5] = 3

    # データをリストに追加
    wm_list.append(wm_values)

    # 使い終わったデータセットを閉じて削除
    ds.close()
    del ds, ds_subset, wm, wm_values
    gc.collect()

# データが存在するか確認
if len(wm_list) == 0:
    print("梅雨型ノードに割り当てられた日付の天気データがありません。")
    # データがない場合、全て0の配列を作成
    filtered_codes_rain = np.zeros((n_lat, n_lon), dtype=int)
else:
    print(f"梅雨型ノードの日付のデータをスタックします。")
    # データをスタックして3次元配列に
    wm_stack = np.stack(wm_list, axis=0)  # 形状: (日数, n_lat, n_lon)
    print(f"wm_stack の形状: {wm_stack.shape}")

    # 各グリッドポイントで最頻値とその割合を計算（ベクトル化）
    print("最頻天気コードとその割合を計算します。")

    # データをリシェイプ
    n_dates = wm_stack.shape[0]
    wm_stack_reshaped = wm_stack.reshape(n_dates, -1)  # 形状: (n_dates, n_lat*n_lon)

    # 天気コードの出現回数を計算
    counts = np.zeros(
        (len(weather_codes), wm_stack_reshaped.shape[1]), dtype=int
    )  # (コード数, n_lat*n_lon)

    for idx, code in enumerate(weather_codes):
        counts[idx, :] = np.sum(wm_stack_reshaped == code, axis=0)

    # 最頻天気コードとその出現回数を取得
    most_common_indices = np.argmax(counts, axis=0)  # 形状: (n_lat*n_lon,)
    max_counts = counts[
        most_common_indices, np.arange(wm_stack_reshaped.shape[1])
    ]  # 形状: (n_lat*n_lon,)
    total_counts = np.sum(counts, axis=0)  # 形状: (n_lat*n_lon,)

    # 最頻天気コードを取得
    most_common_codes = np.array(weather_codes)[
        most_common_indices
    ]  # 形状: (n_lat*n_lon,)

    # 比率を計算
    proportions = max_counts / total_counts

    # 条件を適用して天気コードをフィルタリング
    filtered_codes_flat = np.where(
        (proportions >= 0.5) & (most_common_codes != 0),
        most_common_codes,
        0,  # それ以外は0（白）を設定
    )

    # 元の形状に戻す
    filtered_codes_rain = filtered_codes_flat.reshape(n_lat, n_lon)

    # 使用し終わった変数を削除してメモリを解放
    del (
        wm_stack,
        wm_stack_reshaped,
        counts,
        most_common_indices,
        max_counts,
        total_counts,
        most_common_codes,
        proportions,
        filtered_codes_flat,
    )
    gc.collect()

# 梅雨型の天気パターンをプロット
# プロットの準備
codes = list(weather_mapping.keys())  # [0,1,2,3]
colors = [weather_mapping[code][1] for code in codes]

cmap = ListedColormap(colors)
norm = BoundaryNorm(np.arange(-0.5, len(codes) - 0.5 + 1), cmap.N)

# 結果をプロットします
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": ccrs.PlateCarree()})

# 天気パターンをプロット
im = ax.pcolormesh(
    common_lon,
    common_lat,
    filtered_codes_rain,
    cmap=cmap,
    norm=norm,
    shading="auto",
    transform=ccrs.PlateCarree(),
)

# 海岸線と国境を追加
ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="black")
ax.add_feature(cfeature.BORDERS.with_scale("10m"), linestyle=":")

# 範囲を設定
ax.set_extent(
    [lon_range.start, lon_range.stop, lat_range.start, lat_range.stop],
    crs=ccrs.PlateCarree(),
)

# カスタム凡例を作成（0以外の天気コードのみ）
legend_handles = [
    mpatches.Patch(color=weather_mapping[code][1], label=weather_mapping[code][0])
    for code in codes
    if code != 0
]

# 凡例を図に追加
ax.legend(handles=legend_handles, loc="upper right", fontsize=12)

# タイトルを設定
plt.title(
    "梅雨型ノードに割り当てられた日付の最も一般的な天気パターン（過半数の場合）",
    fontsize=16,
)

# 結果を保存するディレクトリを確認
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 図を保存
rain_weather_pattern_filename = "RainNodesWeatherPattern.png"
plt.savefig(
    os.path.join(result_dir, rain_weather_pattern_filename),
    dpi=300,
    bbox_inches="tight",
)
plt.close()
print("梅雨型ノードの日付の天気パターンの図を作成し、保存しました。")

# 梅雨型ノードの天気パターンを nc ファイルに保存
print("梅雨型ノードの天気パターンを nc ファイルに保存します...")
ds = xr.Dataset(
    {"weather_code": (("latitude", "longitude"), filtered_codes_rain)},
    coords={"latitude": common_lat, "longitude": common_lon},
)
# エンコーディングを設定
encoding = {
    "weather_code": {"dtype": "int8", "zlib": True, "complevel": 5, "contiguous": False}
}
output_filename = os.path.join(result_dir, "rain_weather_pattern.nc")
ds.to_netcdf(output_filename, encoding=encoding)
print("梅雨型ノードの天気パターンの nc ファイルの保存が完了しました。")

# 使用し終わった変数を削除してメモリを解放
del fig, ax, im, filtered_codes_rain, wm_list, ds
gc.collect()
print(
    f"梅雨型ノードの処理完了後のメモリ使用量: {process.memory_info().rss / 1024 ** 2:.2f} MB"
)

# 台風型ノードのリストを定義
typhoon_nodes = [
    (0, 6),
    (0, 7),
    (1, 7),
    (0, 8),
    (1, 8),
    (2, 8),
    (1, 9),
    (2, 9),
    (0, 9),
    (1, 6),
    (0, 4),
    (0, 5),
]

# 指定されたノードに割り当てられた日付をすべて結合
typhoon_dates = []
for node in typhoon_nodes:
    dates = node_date_list.get(node, [])
    typhoon_dates.extend(dates)

# 日付の重複を除去
typhoon_dates = list(set(typhoon_dates))

# 天気データを格納するリストを初期化
wm_list = []

print(f"台風型ノードに割り当てられた日付の数: {len(typhoon_dates)}")

# 各日付に対して天気データを読み込み
for date in typhoon_dates:
    # 日付からファイルパスを構築
    date_str = date.strftime("%Y%m%d")
    file_path = os.path.join(
        weather_data_dir, f"{date.strftime('%Y%m')}", f"wm_{date_str}.nc"
    )

    if not os.path.exists(file_path):
        print(f"警告: 天気データファイルが見つかりません: {file_path}")
        continue

    # 天気データを読み込み
    ds = xr.open_dataset(file_path)

    # 座標名を標準化
    if "lat" in ds.coords and "lon" in ds.coords:
        ds = ds.rename({"lat": "latitude", "lon": "longitude"})

    # 緯度・経度を昇順にソート
    if ds["latitude"].values[0] > ds["latitude"].values[-1]:
        ds = ds.sortby("latitude")
    if ds["longitude"].values[0] > ds["longitude"].values[-1]:
        ds = ds.sortby("longitude")

    # 指定の緯度経度範囲でデータをサブセット
    ds_subset = ds.sel(latitude=lat_range, longitude=lon_range)

    # 天気コードを取得
    wm = ds_subset["wm"]

    # 天気コードのリマッピング（3,4,5を3に統合）
    wm_values = wm.values.astype(int)  # 形状: (n_lat, n_lon)
    wm_values[wm_values == 4] = 3
    wm_values[wm_values == 5] = 3

    # データをリストに追加
    wm_list.append(wm_values)

    # 使い終わったデータセットを閉じて削除
    ds.close()
    del ds, ds_subset, wm, wm_values
    gc.collect()

# データが存在するか確認
if len(wm_list) == 0:
    print("台風型ノードに割り当てられた日付の天気データがありません。")
    # データがない場合、全て0の配列を作成
    filtered_codes_typhoon = np.zeros((n_lat, n_lon), dtype=int)
else:
    print(f"台風型ノードの日付のデータをスタックします。")
    # データをスタックして3次元配列に
    wm_stack = np.stack(wm_list, axis=0)  # 形状: (日数, n_lat, n_lon)
    print(f"wm_stack の形状: {wm_stack.shape}")

    # 各グリッドポイントで最頻値とその割合を計算（ベクトル化）
    print("最頻天気コードとその割合を計算します。")

    # データをリシェイプ
    n_dates = wm_stack.shape[0]
    wm_stack_reshaped = wm_stack.reshape(n_dates, -1)  # 形状: (n_dates, n_lat*n_lon)

    # 天気コードの出現回数を計算
    counts = np.zeros(
        (len(weather_codes), wm_stack_reshaped.shape[1]), dtype=int
    )  # (コード数, n_lat*n_lon)

    for idx, code in enumerate(weather_codes):
        counts[idx, :] = np.sum(wm_stack_reshaped == code, axis=0)

    # 最頻天気コードとその出現回数を取得
    most_common_indices = np.argmax(counts, axis=0)  # 形状: (n_lat*n_lon,)
    max_counts = counts[
        most_common_indices, np.arange(wm_stack_reshaped.shape[1])
    ]  # 形状: (n_lat*n_lon,)
    total_counts = np.sum(counts, axis=0)  # 形状: (n_lat*n_lon,)

    # 最頻天気コードを取得
    most_common_codes = np.array(weather_codes)[
        most_common_indices
    ]  # 形状: (n_lat*n_lon,)

    # 比率を計算
    proportions = max_counts / total_counts

    # 条件を適用して天気コードをフィルタリング
    filtered_codes_flat = np.where(
        (proportions >= 0.5) & (most_common_codes != 0),
        most_common_codes,
        0,  # それ以外は0（白）を設定
    )

    # 元の形状に戻す
    filtered_codes_typhoon = filtered_codes_flat.reshape(n_lat, n_lon)

    # 使用し終わった変数を削除してメモリを解放
    del (
        wm_stack,
        wm_stack_reshaped,
        counts,
        most_common_indices,
        max_counts,
        total_counts,
        most_common_codes,
        proportions,
        filtered_codes_flat,
    )
    gc.collect()

# 台風型の天気パターンをプロット
# プロットの準備
codes = list(weather_mapping.keys())  # [0,1,2,3]
colors = [weather_mapping[code][1] for code in codes]

cmap = ListedColormap(colors)
norm = BoundaryNorm(np.arange(-0.5, len(codes) - 0.5 + 1), cmap.N)

# 結果をプロットします
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": ccrs.PlateCarree()})

# 天気パターンをプロット
im = ax.pcolormesh(
    common_lon,
    common_lat,
    filtered_codes_typhoon,
    cmap=cmap,
    norm=norm,
    shading="auto",
    transform=ccrs.PlateCarree(),
)

# 海岸線と国境を追加
ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="black")
ax.add_feature(cfeature.BORDERS.with_scale("10m"), linestyle=":")

# 範囲を設定
ax.set_extent(
    [lon_range.start, lon_range.stop, lat_range.start, lat_range.stop],
    crs=ccrs.PlateCarree(),
)

# カスタム凡例を作成（0以外の天気コードのみ）
legend_handles = [
    mpatches.Patch(color=weather_mapping[code][1], label=weather_mapping[code][0])
    for code in codes
    if code != 0
]

# 凡例を図に追加
ax.legend(handles=legend_handles, loc="upper right", fontsize=12)

# タイトルを設定
plt.title(
    "台風型ノードに割り当てられた日付の最も一般的な天気パターン（過半数の場合）",
    fontsize=16,
)

# 結果を保存するディレクトリを確認
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 図を保存
typhoon_weather_pattern_filename = "TyphoonNodesWeatherPattern.png"
plt.savefig(
    os.path.join(result_dir, typhoon_weather_pattern_filename),
    dpi=300,
    bbox_inches="tight",
)
plt.close()
print("台風型ノードの日付の天気パターンの図を作成し、保存しました。")

# 台風型ノードの天気パターンを nc ファイルに保存
print("台風型ノードの天気パターンを nc ファイルに保存します...")
ds = xr.Dataset(
    {"weather_code": (("latitude", "longitude"), filtered_codes_typhoon)},
    coords={"latitude": common_lat, "longitude": common_lon},
)
# エンコーディングを設定
encoding = {
    "weather_code": {"dtype": "int8", "zlib": True, "complevel": 5, "contiguous": False}
}
output_filename = os.path.join(result_dir, "typhoon_weather_pattern.nc")
ds.to_netcdf(output_filename, encoding=encoding)
print("台風型ノードの天気パターンの nc ファイルの保存が完了しました。")

# 使用し終わった変数を削除してメモリを解放
del fig, ax, im, filtered_codes_typhoon, wm_list, ds
gc.collect()
print(
    f"台風型ノードの処理完了後のメモリ使用量: {process.memory_info().rss / 1024 ** 2:.2f} MB"
)

# テストデータの処理と評価

# テストデータの日付リストをインポートしています

# ノードごとのラベルを定義します（先ほどの続き）
winter_nodes = [
    (6, 0),
    (7, 0),
    (8, 0),
    (9, 0),
    (5, 1),
    (6, 1),
    (7, 1),
    (8, 1),
    (5, 2),
    (6, 2),
    (7, 2),
    (6, 3),
    (7, 3),
    (6, 4),
    (5, 3),
]
summer_nodes = [
    (8, 6),
    (9, 6),
    (8, 7),
    (9, 7),
    (8, 8),
    (9, 8),
    (9, 9),
    (7, 7),
    (9, 5),
    (9, 4),
    (9, 3),
]
rain_nodes = [
    (6, 7),
    (5, 8),
    (6, 8),
    (7, 8),
    (6, 9),
    (7, 9),
    (8, 9),
    (5, 7),
    (5, 9),
    (4, 7),
    (4, 8),
    (4, 9),
    (3, 7),
    (3, 8),
    (3, 9),
]
typhoon_nodes = [
    (0, 6),
    (0, 7),
    (1, 7),
    (0, 8),
    (1, 8),
    (2, 8),
    (1, 9),
    (2, 9),
    (0, 9),
    (1, 6),
    (0, 4),
    (0, 5),
]

# テストデータの日付と対応するラベルを準備します
test_dates_with_labels = []
for date_str in test_winter_dates_list:
    date = datetime.strptime(date_str, "%Y-%m-%d").date()
    test_dates_with_labels.append((date, "winter"))
for date_str in test_summer_dates_list:
    date = datetime.strptime(date_str, "%Y-%m-%d").date()
    test_dates_with_labels.append((date, "summer"))
for date_str in test_rain_dates_list:
    date = datetime.strptime(date_str, "%Y-%m-%d").date()
    test_dates_with_labels.append((date, "rain"))
for date_str in test_typhoon_dates_list:
    date = datetime.strptime(date_str, "%Y-%m-%d").date()
    test_dates_with_labels.append((date, "typhoon"))

# テストデータの日付を表示
print("テストデータの日付とラベル:")
for date, label in test_dates_with_labels:
    print(f"{date}: {label}")

# データのパス設定（以前と同じ）
data_dir = "/home/devel/work_C/work_ytakano/GSMGPV/nc/prmsl/"

# 使用する緯度・経度の範囲を設定（日本付近）（以前と同じ）
lat_range = slice(15, 55)
lon_range = slice(115, 155)

# データファイルの一覧を取得（サブディレクトリ内も含める）（以前と同じ）
file_paths = []
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith(".nc"):
            file_paths.append(os.path.join(root, file))

# ファイルパスをソート
file_paths = sorted(file_paths)

# ファイル名から日付を取得し、辞書を作成（以前と同じ）
file_date_dict = {}
for file_path in file_paths:
    file_name = os.path.basename(file_path)
    date_str = file_name.replace("prmsl_", "").replace(".nc", "")
    try:
        file_date = datetime.strptime(date_str, "%Y%m%d").date()
        file_date_dict[file_date] = file_path
    except ValueError:
        print(f"ファイル名から日付を抽出できませんでした: {file_name}")

# 共通の緯度・経度グリッドを定義（以前と同じ）
common_lat = np.arange(15, 55.5, 0.5)
common_lon = np.arange(115, 155.5, 0.5)

# テストデータを格納するリスト
test_data_list = []
test_date_list = []
test_labels = []  # 対応するラベルを格納

# データの読み込みと前処理
for date, label in test_dates_with_labels:
    if date in file_date_dict:
        nc_file = file_date_dict[date]
        ds = xr.open_dataset(nc_file)

        # 座標名を標準化
        if "lat" in ds.coords and "lon" in ds.coords:
            ds = ds.rename({"lat": "latitude", "lon": "longitude"})

        # 海面更正気圧データを取得し、指定の緯度経度範囲でスライス
        slp = (
            (ds["PRMSL_meansealevel"] / 100)
            .isel(time=0)
            .sel(latitude=lat_range, longitude=lon_range)
        )

        # slp データの存在確認
        if slp.size == 0:
            print(f"警告: slp データが空です。日付 {date} のデータをスキップします。")
            continue

        # データを共通グリッドに再補間
        slp_interp = slp.interp(latitude=common_lat, longitude=common_lon)
        slp_shape = slp_interp.values.shape

        # 期待されるデータ形状をチェック
        if expected_shape is not None and slp_shape != expected_shape:
            print(
                f"日付 {date}: データ形状が {slp_shape} で、期待される形状 {expected_shape} と一致しません。"
            )
            continue

        # 領域平均を引く
        slp_interp = slp_interp - slp_interp.mean()

        # データを1次元配列にフラット化してリストに追加
        test_data_list.append(slp_interp.values.flatten())
        test_date_list.append(date)
        test_labels.append(label)

        # データセットを閉じて削除
        ds.close()
        del ds, slp, slp_interp
        gc.collect()
    else:
        print(f"データが存在しない日付: {date}")

# リストをNumPy配列に変換
test_data_array = np.array(test_data_list)
print("test_data_array の形状:", test_data_array.shape)
print("テストデータの日付数:", len(test_date_list))

# データの正規化（訓練データと同じスケーラーを使用）
if test_data_array.size == 0:
    print("エラー: test_data_array が空です。データの読み込みに問題があります。")
else:
    test_data_array_norm = scaler.transform(test_data_array)

    # 主成分分析による次元削減（訓練データと同じPCAを使用）
    test_data_array_pca = pca.transform(test_data_array_norm)
    print(f"テストデータの形状（PCA後）: {test_data_array_pca.shape}")

# テストデータをSOMにマッピング
test_mapped_nodes = [som.winner(x) for x in test_data_array_pca]

# テストデータの分類結果を保持するリスト
test_results = []

# 正解数のカウント
correct_counts = {"winter": 0, "summer": 0, "rain": 0, "typhoon": 0}
total_counts = {"winter": 0, "summer": 0, "rain": 0, "typhoon": 0}

# ノードラベルの逆引き辞書を作成
node_label_dict = {}
for node in winter_nodes:
    node_label_dict[node] = "winter"
for node in summer_nodes:
    node_label_dict[node] = "summer"
for node in rain_nodes:
    node_label_dict[node] = "rain"
for node in typhoon_nodes:
    node_label_dict[node] = "typhoon"

# テストデータごとに結果を判定
for idx, node in enumerate(test_mapped_nodes):
    date = test_date_list[idx]
    true_label = test_labels[idx]
    predicted_label = node_label_dict.get(
        node, "other"
    )  # ラベルが定義されていないノードは 'other' とする

    # 正解をカウント
    total_counts[true_label] += 1
    if true_label == predicted_label:
        correct_counts[true_label] += 1

    test_results.append(
        {
            "date": date,
            "true_label": true_label,
            "node": node,
            "predicted_label": predicted_label,
        }
    )

# 結果を表示
print("\\\\nテストデータの分類結果:")
for result in test_results:
    print(
        f"日付: {result['date']} | 真のラベル: {result['true_label']} | "
        f"マッピングされたノード: {result['node']} | 予測ラベル: {result['predicted_label']}"
    )

# ラベルごとの正解率を計算
print("\\\\nラベルごとの正解数と正解率:")
for label in ["winter", "summer", "rain", "typhoon"]:
    correct = correct_counts[label]
    total = total_counts[label]
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"{label}: 正解数 {correct} / {total} | 正解率: {accuracy:.2f}%")

# 総合正解率
total_correct = sum(correct_counts.values())
total_total = sum(total_counts.values())
total_accuracy = total_correct / total_total * 100 if total_total > 0 else 0
print(
    f"\\\\n総合正解率: {total_correct} / {total_total} | 正解率: {total_accuracy:.2f}%"
)

# ヒートマップを作成するために、ノードごとのテストデータ数を集計
test_node_counts = np.zeros((som_rows, som_cols), dtype=int)
test_node_correct_counts = np.zeros((som_rows, som_cols), dtype=int)

for result in test_results:
    node = result["node"]
    i, j = node
    test_node_counts[i, j] += 1
    if result["true_label"] == result["predicted_label"]:
        test_node_correct_counts[i, j] += 1

# 正解率ヒートマップを計算
test_node_accuracies = np.zeros((som_rows, som_cols), dtype=float)
with np.errstate(divide="ignore", invalid="ignore"):
    test_node_accuracies = np.where(
        test_node_counts > 0, test_node_correct_counts / test_node_counts, np.nan
    )

# 結果を保存するディレクトリを確認
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# ヒートマップを作成して保存
import matplotlib.colors as colors

# テストデータの数のヒートマップ
plt.figure(figsize=(8, 6))
plt.title("Test Data Mapping on SOM")
im = plt.imshow(
    test_node_counts,
    origin="lower",
    extent=(0, som_cols, 0, som_rows),
    cmap="Blues",
    aspect="auto",
)
plt.colorbar(label="Number of Test Data Points")
plt.xlabel("Column Index (j)")
plt.ylabel("Row Index (i)")

# 各セルにデータ数を表示
for i in range(som_rows):
    for j in range(som_cols):
        count = test_node_counts[i, j]
        if count > 0:
            plt.text(
                j + 0.5, i + 0.5, str(count), ha="center", va="center", color="black"
            )
test_map_filename = "TestDataMap.png"
plt.savefig(os.path.join(result_dir, test_map_filename))
plt.close()

# テストデータの正解率のヒートマップ
plt.figure(figsize=(8, 6))
plt.title("Test Data Accuracy on SOM Nodes")
im = plt.imshow(
    test_node_accuracies,
    origin="lower",
    extent=(0, som_cols, 0, som_rows),
    cmap="coolwarm",
    aspect="auto",
    vmin=0,
    vmax=1,
)
plt.colorbar(label="Accuracy")
plt.xlabel("Column Index (j)")
plt.ylabel("Row Index (i)")

# 各セルに正解率を表示
for i in range(som_rows):
    for j in range(som_cols):
        if not np.isnan(test_node_accuracies[i, j]):
            accuracy = test_node_accuracies[i, j]
            plt.text(
                j + 0.5,
                i + 0.5,
                f"{accuracy:.0%}",
                ha="center",
                va="center",
                color="black",
            )
accuracy_map_filename = "TestAccuracyMap.png"
plt.savefig(os.path.join(result_dir, accuracy_map_filename))
plt.close()

# ラベルごとのヒートマップを作成
label_list = ["winter", "summer", "rain", "typhoon"]
label_node_counts = {
    label: np.zeros((som_rows, som_cols), dtype=int) for label in label_list
}
label_colors = {"winter": "blue", "summer": "red", "rain": "green", "typhoon": "purple"}

for result in test_results:
    node = result["node"]
    i, j = node
    label = result["true_label"]
    label_node_counts[label][i, j] += 1

# 各ラベルのヒートマップを作成
for label in label_list:
    plt.figure(figsize=(8, 6))
    plt.title(f"Test Data Mapping on SOM ({label.capitalize()})")
    im = plt.imshow(
        label_node_counts[label],
        origin="lower",
        extent=(0, som_cols, 0, som_rows),
        cmap="Blues",
        aspect="auto",
    )
    plt.colorbar(label=f"Number of {label.capitalize()} Test Data Points")
    plt.xlabel("Column Index (j)")
    plt.ylabel("Row Index (i)")

    # 各セルにデータ数を表示
    for i in range(som_rows):
        for j in range(som_cols):
            count = label_node_counts[label][i, j]
            if count > 0:
                plt.text(
                    j + 0.5,
                    i + 0.5,
                    str(count),
                    ha="center",
                    va="center",
                    color="black",
                )
    label_map_filename = f"TestDataMap_{label.capitalize()}.png"
    plt.savefig(os.path.join(result_dir, label_map_filename))
    plt.close()

# 混同行列の作成と可視化
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 真のラベルと予測ラベルのリストを作成
true_labels_list = [result["true_label"] for result in test_results]
predicted_labels_list = [
    result["predicted_label"]
    if result["predicted_label"] in ["winter", "summer", "rain", "typhoon"]
    else "other"
    for result in test_results
]

# ラベルの順序を定義
labels = ["winter", "summer", "rain", "typhoon", "other"]
cm = confusion_matrix(true_labels_list, predicted_labels_list, labels=labels)

# 混同行列を視覚化して保存
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
)
plt.xlabel("予測ラベル")
plt.ylabel("真のラベル")
plt.title("混同行列")
confusion_matrix_filename = "ConfusionMatrix.png"
plt.savefig(os.path.join(result_dir, confusion_matrix_filename))
plt.close()

print("\\\\nテストデータの分類と評価が完了しました。結果の図は保存されています。")
