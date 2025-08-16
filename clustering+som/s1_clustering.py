# s1_clustering.py
# -*- coding: utf-8 -*-
import os
import logging
import unicodedata
import re
from typing import List, Tuple, Callable, Optional, Dict
from collections import Counter

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

logger = logging.getLogger(__name__)

EPSILON = 1e-9
TIE_EPS = 1e-12


@torch.no_grad()
def calculate_s1_pairwise_batch(x_batch: torch.Tensor,
                                y_all: torch.Tensor,
                                d_lat: int,
                                d_lon: int,
                                epsilon: float = EPSILON) -> torch.Tensor:
    """
    S1スコア（小さいほど類似）をバッチで計算
    x_batch: (B, D) torch tensor
    y_all  : (N, D) torch tensor
    戻り値: (B, N) のS1スコア
    備考:
      - 本実装では「差分場の勾配」に基づく近似式（元実装）を保持
      - SOM側のS1は '差分の勾配' ではなく '勾配の差分' を用いる版だが、
        ここは閾値TH_MERGEとの互換を保つため現状の式を維持
    """
    B, D = x_batch.shape
    N, _ = y_all.shape
    x_maps = x_batch.view(B, 1, d_lat, d_lon)
    y_maps = y_all.view(1, N, d_lat, d_lon)
    diff_maps = y_maps - x_maps  # (B, N, H, W)

    # 分子: 差分場の勾配
    grad_x_D = (diff_maps[..., :, 1:] - diff_maps[..., :, :-1])[..., :-1, :]
    grad_y_D = (diff_maps[..., 1:, :] - diff_maps[..., :-1, :])[..., :, :-1]
    numerator_term = torch.abs(grad_x_D) + torch.abs(grad_y_D)
    numerator = torch.mean(numerator_term, dim=(-2, -1))  # (B, N)

    # 分母: 各場の勾配の最大値の和
    grad_x_y = (y_maps[..., :, 1:] - y_maps[..., :, :-1])[..., :-1, :]
    grad_y_y = (y_maps[..., 1:, :] - y_maps[..., :-1, :])[..., :, :-1]
    grad_x_x = (x_maps[..., :, 1:] - x_maps[..., :, :-1])[..., :-1, :]
    grad_y_x = (x_maps[..., 1:, :] - x_maps[..., :-1, :])[..., :, :-1]
    max_grad_x = torch.max(torch.abs(grad_x_y), torch.abs(grad_x_x))
    max_grad_y = torch.max(torch.abs(grad_y_y), torch.abs(grad_y_x))
    denominator_term = max_grad_x + max_grad_y
    denominator = torch.mean(denominator_term, dim=(-2, -1))  # (B, N)

    s1_score = 100.0 * (numerator / (denominator + epsilon))
    return s1_score


def _normalize_to_base_candidate(label_str: Optional[str]) -> Optional[str]:
    """
    与えられたラベル文字列を正規化し、英数字のみ残す。
    例:
      '２Ａ' -> '2A'
      '2 A' -> '2A'
      '2A＋' -> '2A'
      '3B-移行' -> '3B'
      '2A/3B' -> '2A3B' （複合はこの後でNone判定に回す）
    """
    if label_str is None:
        return None
    s = str(label_str)
    # Unicode正規化（全角→半角、合成文字等の正規化）
    s = unicodedata.normalize('NFKC', s)
    s = s.upper().strip()
    # 記号類の正規化
    s = s.replace('＋', '+').replace('－', '-').replace('−', '-')
    # 英数字以外は除去（日本語・記号・空白など）
    s = re.sub(r'[^0-9A-Z]', '', s)
    return s if s != '' else None


def basic_label_or_none(label_str: Optional[str], base_labels: List[str]) -> Optional[str]:
    """
    ラベルが基本ラベル集合に含まれる場合のみ返す。複合/移行型はNone。
    正規化を強化し、'2 A', '２Ａ', '2A+', '3B-移行' などを '2A', '3B' として扱う。
    '2A/3B' のような複合は None にする。
    """
    cand = _normalize_to_base_candidate(label_str)
    if cand is None:
        return None
    # 候補中に基本ラベルが1つだけ含まれていれば採用、それ以外（0または2つ以上）はNone
    hits = [bl for bl in base_labels if bl in cand]
    # よくある単一表記（完全一致）を優先
    if cand in base_labels:
        return cand
    # '2A+' → '2A' のように末尾記号や語尾が削られて一致する場合を許可
    for bl in base_labels:
        if cand == bl:
            return bl
        # 先頭が完全一致かつ残りが短い記号のみとみなせる場合に許容
        if cand.startswith(bl):
            # bl を除いた残りに英数字が含まれていれば複合と判断して除外
            rest = cand[len(bl):]
            if re.search(r'[0-9A-Z]', rest) is None:
                return bl
    # 残りは複合（例: '2A3B'）等とみなして除外
    return None


def analyze_cluster_distribution(clusters: List[List[int]],
                                 all_labels: List[Optional[str]],
                                 all_time_stamps: Optional[np.ndarray],
                                 base_labels: List[str],
                                 title: str = "クラスタ分布分析") -> None:
    logger.info(f"\n--- {title} ---")
    for i, idxs in enumerate(clusters):
        n = len(idxs)
        if n == 0:
            continue
        logger.info(f"\n[クラスタ {i+1}] (N={n})")
        # ラベル分布（基本ラベルのみ）
        c = Counter()
        for j in idxs:
            lbl = basic_label_or_none(all_labels[j], base_labels)
            if lbl is not None:
                c[lbl] += 1
        if c:
            logger.info("  - ラベル構成（基本ラベルのみ）:")
            for lbl, cnt in sorted(c.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"    {lbl:<3}: {cnt:4d} ({cnt/n*100:5.1f}%)")
        if all_time_stamps is not None:
            months = pd.to_datetime(all_time_stamps[idxs]).month
            mon_c = Counter(months)
            logger.info("  - 月別分布:")
            for m in range(1, 13):
                cnt = mon_c.get(m, 0)
                logger.info(f"    {m:2d}月: {cnt:4d} ({cnt/n*100:5.1f}%)")
    logger.info(f"--- {title} 終了 ---\n")


def build_confusion_matrix_only_base(clusters: List[List[int]],
                                     all_labels: List[Optional[str]],
                                     base_labels: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    num_clusters = len(clusters)
    cluster_names = [f'Cluster_{i+1}' for i in range(num_clusters)]
    cm = pd.DataFrame(0, index=base_labels, columns=cluster_names, dtype=int)
    for i, idxs in enumerate(clusters):
        col = cluster_names[i]
        cnt = Counter()
        for j in idxs:
            lbl = basic_label_or_none(all_labels[j], base_labels)
            if lbl is not None:
                cnt[lbl] += 1
        for lbl, k in cnt.items():
            cm.loc[lbl, col] = k
    return cm, cluster_names


def evaluate_clusters_only_base(clusters: List[List[int]],
                                all_labels: List[Optional[str]],
                                base_labels: List[str],
                                title: str = "評価（基本ラベルのみ）",
                                medoids: Optional[List[Optional[int]]] = None) -> Optional[Dict[str, float]]:
    """
    - 各クラスタの代表（多数決）ラベルを base_labels のみで決定（複数クラスタが同一ラベルも可）
    - マクロ平均再現率を「各基本ラベル l に対して、代表ラベルが l であるクラスタ群に入った l の件数 / l 全体の件数」の平均で算出
    - マイクロ精度は「各クラスタの多数派件数の総和 / 基本ラベル総件数」
    - ARI/NMI は、基本ラベルを持ち、かつクラスタの代表ラベルが None でないサンプルのみで算出
    - 代表ラベルとメドイドの真ラベルの一致状況もログ出力（medoids 指定時）
    """
    logger.info(f"\n--- {title} ---")
    if not all_labels:
        logger.warning("ラベル無しのため評価をスキップします。")
        return None

    cm, cluster_names = build_confusion_matrix_only_base(clusters, all_labels, base_labels)
    present_labels = [l for l in base_labels if cm.loc[l].sum() > 0]
    if len(present_labels) == 0:
        logger.warning("基本ラベルに該当するサンプルがありません。評価をスキップします。")
        return None

    logger.info("【混同行列（基本ラベルのみ）】")
    logger.info(f"\n{cm.loc[present_labels, :].to_string()}")

    # 各クラスタの多数決（代表ラベル）を決定
    cluster_majority: Dict[int, Optional[str]] = {}
    logger.info("\n【各クラスタの多数決（代表ラベル）】")
    total_count = int(cm.values.sum())
    micro_correct_sum = 0
    for k in range(len(cluster_names)):
        col = cluster_names[k]
        col_counts = cm[col]
        col_sum = int(col_counts.sum())
        if col_sum == 0:
            cluster_majority[k] = None
            logger.info(f" - {col:<12}: 代表ラベル=None（基本ラベル出現なし）")
            continue
        # 同数タイは index順で先勝（再現性のため）
        top_label = col_counts.idxmax()
        top_count = int(col_counts.max())
        micro_correct_sum += top_count
        share = top_count / col_sum if col_sum > 0 else 0.0
        top3 = col_counts.sort_values(ascending=False)[:3]
        top3_str = ", ".join([f"{lbl}:{int(cnt)}" for lbl, cnt in top3.items()])
        logger.info(f" - {col:<12}: 代表={top_label:<3} 件数={top_count:4d} シェア={share:5.2f} | 上位: {top3_str}")
        cluster_majority[k] = top_label

    # メドイドと代表ラベルの一致状況
    medoid_match_rate = np.nan
    if medoids is not None and len(medoids) == len(clusters):
        logger.info("\n【メドイドと代表ラベルの一致状況】")
        matches = 0
        valid = 0
        for k, (col, med_idx) in enumerate(zip(cluster_names, medoids)):
            rep = cluster_majority.get(k, None)
            if med_idx is None:
                logger.info(f" - {col:<12}: 代表={rep}, メドイド=None")
                continue
            med_true = basic_label_or_none(all_labels[med_idx], base_labels)
            if rep is None or med_true is None:
                logger.info(f" - {col:<12}: 代表={rep}, メドイド真ラベル={med_true}, 判定=スキップ")
                continue
            valid += 1
            ok = (rep == med_true)
            matches += int(ok)
            logger.info(f" - {col:<12}: 代表={rep}, メドイド真ラベル={med_true}, 一致={ok}")
        medoid_match_rate = (matches / valid) if valid > 0 else np.nan
        logger.info(f"メドイド-代表ラベル一致率: {medoid_match_rate:.4f}")

    logger.info("\n【各ラベルの再現率（代表クラスタ群ベース）】")
    per_label = {}
    for lbl in present_labels:
        row_sum = int(cm.loc[lbl, :].sum())
        cols_for_lbl = [cluster_names[k] for k in range(len(cluster_names)) if cluster_majority.get(k, None) == lbl]
        correct = int(cm.loc[lbl, cols_for_lbl].sum()) if cols_for_lbl else 0
        recall = correct / row_sum if row_sum > 0 else 0.0
        per_label[lbl] = {'N': row_sum, 'Correct': correct, 'Recall': recall}
        logger.info(f" - {lbl:<3}: N={row_sum:4d} Correct={correct:4d} Recall={recall:.4f} 代表={cols_for_lbl if cols_for_lbl else 'なし'}")

    macro_recall = float(np.mean([per_label[l]['Recall'] for l in present_labels]))
    micro_accuracy = micro_correct_sum / total_count if total_count > 0 else 0.0
    logger.info("\n【集計】")
    logger.info(f"Macro Recall (基本ラベル) = {macro_recall:.4f}")
    logger.info(f"Micro Accuracy (基本ラベル) = {micro_accuracy:.4f}")

    # ARI/NMI 計算（基本ラベルを持ち、かつクラスタ代表ラベルが有効なサンプルのみ）
    n_samples = len(all_labels)
    sample_to_cluster = [-1] * n_samples
    for ci, idxs in enumerate(clusters):
        for j in idxs:
            sample_to_cluster[j] = ci

    y_true, y_pred = [], []
    for j in range(n_samples):
        lbl = basic_label_or_none(all_labels[j], base_labels)
        if lbl is None:
            continue
        ci = sample_to_cluster[j]
        if ci < 0:
            continue
        rep = cluster_majority.get(ci, None)
        if rep is None:
            continue  # Unassignedは評価から除外
        y_true.append(lbl)
        y_pred.append(rep)

    metrics: Dict[str, float] = {
        'MacroRecall_majority': macro_recall,
        'MicroAccuracy_majority': micro_accuracy,
        'MedoidMajorityMatchRate': float(medoid_match_rate) if not np.isnan(medoid_match_rate) else np.nan
    }
    if len(y_true) > 1:
        uniq_true = {l: i for i, l in enumerate(sorted(set(y_true)))}
        uniq_pred = {l: i for i, l in enumerate(sorted(set(y_pred)))}
        y_true_idx = [uniq_true[l] for l in y_true]
        y_pred_idx = [uniq_pred[l] for l in y_pred]
        ari = adjusted_rand_score(y_true_idx, y_pred_idx)
        nmi = normalized_mutual_info_score(y_true_idx, y_pred_idx)
        metrics['ARI_majority'] = float(ari)
        metrics['NMI_majority'] = float(nmi)
        logger.info(f"Adjusted Rand Index (基本ラベル) = {ari:.4f}")
        logger.info(f"Normalized Mutual Info (基本ラベル) = {nmi:.4f}")
    else:
        logger.info("評価対象サンプルが少ないため ARI/NMI を計算しません。")

    logger.info(f"--- {title} 終了 ---\n")
    return metrics


@torch.no_grad()
def find_medoid_torch(cluster_indices: List[int],
                      X: torch.Tensor,
                      row_batch_size: int,
                      col_chunk_size: int,
                      d_lat: int,
                      d_lon: int) -> Optional[int]:
    """
    クラスタ内メドイドを、列チャンク逐次合計で求める（GPUメモリ節約版）
    """
    if not cluster_indices:
        return None
    if len(cluster_indices) == 1:
        return cluster_indices[0]

    device = X.device
    k = len(cluster_indices)
    cluster_data = X[cluster_indices]  # (k, D)

    s1_sums_cpu = torch.zeros(k, dtype=torch.float64, device='cpu')  # 精度余裕
    for i0 in range(0, k, row_batch_size):
        i1 = min(i0 + row_batch_size, k)
        xb = cluster_data[i0:i1]  # (B, D)
        # 列チャンクの最小値ではなく"総和"が必要なので、逐次合計
        partial_sum = torch.zeros((i1 - i0,), dtype=torch.float64, device=device)
        for j0 in range(0, k, col_chunk_size):
            j1 = min(j0 + col_chunk_size, k)
            yb = cluster_data[j0:j1]  # (Nc, D)
            s1_chunk = calculate_s1_pairwise_batch(xb, yb, d_lat, d_lon)  # (B, Nc)
            # 自分自身の位置は除外（infに）
            if (j0 < i1) and (j1 > i0):
                o_start = max(i0, j0)
                o_end = min(i1, j1)
                t = o_end - o_start
                if t > 0:
                    ar = torch.arange(t, device=device)
                    s1_chunk[o_start - i0 + ar, o_start - j0 + ar] = float('inf')
            # 合計（infは無視）
            finite_mask = torch.isfinite(s1_chunk)
            s1_chunk = torch.where(finite_mask, s1_chunk, torch.zeros_like(s1_chunk))
            partial_sum += s1_chunk.sum(dim=1, dtype=torch.float64)
        s1_sums_cpu[i0:i1] = partial_sum.cpu()

    # タイブレーク（インデックス小優先）
    tie_bias = TIE_EPS * torch.arange(k, dtype=torch.float64)
    s1_sums_cpu = s1_sums_cpu + tie_bias
    medoid_pos = int(torch.argmin(s1_sums_cpu).item())
    return cluster_indices[medoid_pos]


def two_stage_clustering(X: torch.Tensor,
                         th_merge: float,
                         all_labels: List[Optional[str]],
                         all_time_stamps: Optional[np.ndarray],
                         base_labels: List[str],
                         row_batch_size: int,
                         col_chunk_size: int,
                         d_lat: int,
                         d_lon: int,
                         eval_callback: Optional[Callable[[int, List[List[int]], List[Optional[int]]], None]] = None
                         ) -> Tuple[List[List[int]], List[Optional[int]]]:
    """
    2段階クラスタリング
      1) HAC: 相互最近傍(MNN) & S1 < th_merge を満たすペアをマージ
         -> 距離行列を持たず、行×列チャンクで各行の最小相手のみ追跡（メモリ節約）
      2) k-medoids: メドイド中心の再割当てでクラスタを再構成（割当も列チャンクでBMU探索）

    戻り値: (clusters, medoids)
    """
    n_samples = X.shape[0]
    logger.info("二段階クラスタリング開始 (S1, 入力=空間偏差[hPa])...")
    clusters: List[List[int]] = [[i] for i in range(n_samples)]
    medoids: List[Optional[int]] = list(range(n_samples))

    iteration = 1
    while True:
        logger.info(f"\n========== 反復 {iteration} ==========")
        logger.info(f"現在のクラスタ数: {len(clusters)}")

        logger.info("ステップ1: HAC - MNNマージ開始（列チャンクで近傍探索）...")
        # 有効メドイドのみ抽出
        index_map = [i for i, m in enumerate(medoids) if m is not None]  # cluster index -> medoid row index in following array
        medoid_ids = [m for m in medoids if m is not None]               # 元データindex
        C = len(medoid_ids)
        if C == 0:
            logger.info("有効メドイドがありません。終了します。")
            break

        medoid_data = X[medoid_ids]  # (C, D)
        # 各行（クラス）について最小相手と距離を求める（列チャンクで走査）
        nn_idx = np.full(C, -1, dtype=np.int64)
        nn_val = np.full(C, np.inf, dtype=np.float64)

        for i0 in tqdm(range(0, C, row_batch_size), desc="HAC: 最近傍探索", leave=False):
            i1 = min(i0 + row_batch_size, C)
            xb = medoid_data[i0:i1]  # (B, D)
            best_val = torch.full((i1 - i0,), float('inf'), device=X.device, dtype=torch.float64)
            best_j = torch.full((i1 - i0,), -1, device=X.device, dtype=torch.long)
            for j0 in range(0, C, col_chunk_size):
                j1 = min(j0 + col_chunk_size, C)
                yb = medoid_data[j0:j1]  # (Nc, D)
                s1_chunk = calculate_s1_pairwise_batch(xb, yb, d_lat, d_lon).to(torch.float64)  # (B, Nc)
                # 自己距離除外（対角）
                if (j0 < i1) and (j1 > i0):
                    o_start = max(i0, j0)
                    o_end = min(i1, j1)
                    t = o_end - o_start
                    if t > 0:
                        ar = torch.arange(t, device=X.device)
                        s1_chunk[o_start - i0 + ar, o_start - j0 + ar] = float('inf')
                # タイブレーク: 小さい列インデックス優先（グローバル列番号でバイアス）
                col_idx = torch.arange(j0, j1, device=X.device, dtype=torch.float64)
                s1_chunk = s1_chunk + (TIE_EPS * col_idx.view(1, -1))
                # チャンク内の最小
                minvals, minpos = torch.min(s1_chunk, dim=1)
                # これまでの最小と比較
                better = minvals < best_val
                best_val = torch.where(better, minvals, best_val)
                best_j = torch.where(better, col_idx[minpos].to(torch.long), best_j)
            nn_idx[i0:i1] = best_j.cpu().numpy()
            nn_val[i0:i1] = best_val.cpu().numpy()

        # 参考統計
        finite_vals = nn_val[np.isfinite(nn_val)]
        if finite_vals.size > 0:
            logger.info(f"NN距離統計 - 最小: {finite_vals.min():.2f}, 最大: {finite_vals.max():.2f}, 平均: {finite_vals.mean():.2f}, 中央: {np.median(finite_vals):.2f}")
        count_below = int((finite_vals < th_merge).sum())
        logger.info(f"NN距離がしきい値 {th_merge} 未満の行数: {count_below}")

        # 相互最近傍ペア抽出（medoid配列上のインデックス）
        mnn_pairs = []
        for i_med in range(C):
            j_med = nn_idx[i_med]
            if j_med < 0 or j_med >= C:
                continue
            if nn_idx[j_med] == i_med and i_med < j_med:
                d_ij = nn_val[i_med]
                if d_ij < th_merge:
                    i_cluster = index_map[i_med]
                    j_cluster = index_map[j_med]
                    mnn_pairs.append((i_cluster, j_cluster, float(d_ij)))
        logger.info(f"MNN条件を満たすマージ候補ペア数: {len(mnn_pairs)}")
        if len(mnn_pairs) == 0:
            logger.info("MNN条件でマージ可能なペア無し。クラスタリング終了。")
            if eval_callback is not None:
                try:
                    eval_callback(iteration, clusters, medoids)
                except Exception as e:
                    logger.warning(f"評価コールバックで例外: {e}")
            break

        # S1昇順で安定ソート
        mnn_pairs.sort(key=lambda t: (t[2], t[0], t[1]))

        # マージ
        num_clusters_before = len(clusters)
        merged_indices = set()
        new_clusters = []
        for i_c, j_c, _ in mnn_pairs:
            if i_c not in merged_indices and j_c not in merged_indices:
                new_clusters.append(clusters[i_c] + clusters[j_c])
                merged_indices.add(i_c)
                merged_indices.add(j_c)
        for idx in range(num_clusters_before):
            if idx not in merged_indices:
                new_clusters.append(clusters[idx])
        clusters = new_clusters
        logger.info(f"マージ後クラスタ数: {len(clusters)}")

        # 一時メドイド（クラスごとに安全に算出）
        temp_medoids: List[Optional[int]] = []
        for c in tqdm(clusters, desc="HAC: 一時メドイド", leave=False):
            m = find_medoid_torch(c, X, row_batch_size, col_chunk_size, d_lat, d_lon)
            temp_medoids.append(m)

        # ステップ2: k-medoids - 割当（列チャンクBMU）
        logger.info("ステップ2: k-medoids - 再構成開始（列チャンクBMU）...")
        current_medoids = temp_medoids
        n_total = X.shape[0]
        while True:
            valid_idx_m = [idx for idx, m in enumerate(current_medoids) if m is not None]
            valid_medoids = [current_medoids[idx] for idx in valid_idx_m]
            if len(valid_medoids) == 0:
                logger.warning("有効メドイドなし。k-medoids終了。")
                medoids = current_medoids
                break

            # 新しいクラスタの器
            new_clusters = [[] for _ in range(len(current_medoids))]

            # 行バッチで全サンプルを処理し、列チャンクでBMU更新
            medoid_data = X[valid_medoids]
            for i0 in tqdm(range(0, n_total, row_batch_size), desc="k-medoids: 割当", leave=False):
                i1 = min(i0 + row_batch_size, n_total)
                xb = X[i0:i1]
                best_val = torch.full((i1 - i0,), float('inf'), device=X.device, dtype=torch.float64)
                best_col = torch.full((i1 - i0,), -1, device=X.device, dtype=torch.long)
                for j0 in range(0, len(valid_medoids), col_chunk_size):
                    j1 = min(j0 + col_chunk_size, len(valid_medoids))
                    yb = medoid_data[j0:j1]
                    s1_chunk = calculate_s1_pairwise_batch(xb, yb, d_lat, d_lon).to(torch.float64)
                    col_idx = torch.arange(j0, j1, device=X.device, dtype=torch.float64)
                    s1_chunk = s1_chunk + (TIE_EPS * col_idx.view(1, -1))
                    minvals, minpos = torch.min(s1_chunk, dim=1)
                    better = minvals < best_val
                    best_val = torch.where(better, minvals, best_val)
                    best_col = torch.where(better, col_idx[minpos].to(torch.long), best_col)

                # この行バッチの割当を反映
                for r in range(i1 - i0):
                    col = int(best_col[r].item())
                    target_cluster_idx = valid_idx_m[col]
                    new_clusters[target_cluster_idx].append(i0 + r)

            # 新メドイド算出
            new_medoids = [find_medoid_torch(c, X, row_batch_size, col_chunk_size, d_lat, d_lon) for c in new_clusters]

            if sorted([m for m in new_medoids if m is not None]) == sorted([m for m in current_medoids if m is not None]):
                logger.info("  k-medoids収束。")
                medoids = new_medoids
                clusters = new_clusters
                break
            else:
                current_medoids = new_medoids

        if eval_callback is not None:
            try:
                eval_callback(iteration, clusters, medoids)
            except Exception as e:
                logger.warning(f"評価コールバックで例外: {e}")
        iteration += 1

    return clusters, medoids


def plot_s1_distribution_histogram(X: torch.Tensor,
                                   d_lat: int,
                                   d_lon: int,
                                   save_path: str,
                                   row_batch_size: int = 4,
                                   col_chunk_size: int = 64,
                                   max_samples: Optional[int] = 1200) -> None:
    """
    S1スコアの分布を可視化。上三角のみ、行×列チャンクで収集（メモリ節約）
    """
    device = X.device
    n = X.shape[0]
    if (max_samples is not None) and (n > max_samples):
        idx = np.random.choice(n, max_samples, replace=False)
        Xs = X[idx]
        m = Xs.shape[0]
        idx_global_rows = np.arange(m)
    else:
        Xs = X
        m = n
        idx_global_rows = np.arange(m)

    values = []
    for i0 in tqdm(range(0, m, row_batch_size), desc="S1分布用計算", leave=False):
        i1 = min(i0 + row_batch_size, m)
        xb = Xs[i0:i1]
        for j0 in range(0, m, col_chunk_size):
            j1 = min(j0 + col_chunk_size, m)
            yb = Xs[j0:j1]
            s1_chunk = calculate_s1_pairwise_batch(xb, yb, d_lat, d_lon)  # (B, Nc)
            # 上三角のみ採用（全組のうち i < j ）
            for r in range(i1 - i0):
                gi = i0 + r
                gj = np.arange(j0, j1)
                mask = gj > gi
                if mask.any():
                    vals = s1_chunk[r, mask].detach().cpu().numpy().ravel()
                    values.append(vals)

    if len(values) == 0:
        flat = np.array([], dtype=np.float32)
    else:
        flat = np.concatenate(values, axis=0)

    plt.figure(figsize=(10, 6))
    plt.hist(flat, bins=50, color='steelblue', alpha=0.8)
    plt.title('Distribution of S1 Scores (Lower = More Similar)')
    plt.xlabel('S1 Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_final_distribution_summary(clusters: List[List[int]],
                                    all_labels: List[Optional[str]],
                                    all_time_stamps: np.ndarray,
                                    base_labels: List[str],
                                    save_path: str) -> None:
    num_clusters = len(clusters)
    cluster_names = [f'Cluster {i+1}' for i in range(num_clusters)]
    label_dist_matrix = pd.DataFrame(0, index=cluster_names, columns=base_labels)
    month_dist_matrix = pd.DataFrame(0, index=cluster_names, columns=range(1, 13))
    for i, idxs in enumerate(clusters):
        name = cluster_names[i]
        c = Counter()
        for j in idxs:
            lbl = basic_label_or_none(all_labels[j], base_labels)
            if lbl is not None:
                c[lbl] += 1
        for lbl, k in c.items():
            label_dist_matrix.loc[name, lbl] = k
        months = pd.to_datetime(all_time_stamps[idxs]).month if all_time_stamps is not None else []
        mon_c = Counter(months)
        for month, k in mon_c.items():
            month_dist_matrix.loc[name, month] = k

    fig, axes = plt.subplots(2, 1, figsize=(16, 14))
    fig.suptitle('Final Cluster Distribution Summary (Basic Labels Only)', fontsize=20)
    sns.heatmap(label_dist_matrix, ax=axes[0], annot=True, fmt='d', cmap='viridis', linewidths=.5)
    axes[0].set_title('Label Distribution per Cluster', fontsize=16)
    axes[0].set_ylabel('Cluster')
    axes[0].set_xlabel('True Base Label')
    sns.heatmap(month_dist_matrix, ax=axes[1], annot=True, fmt='d', cmap='inferno', linewidths=.5)
    axes[1].set_title('Monthly Distribution per Cluster', fontsize=16)
    axes[1].set_ylabel('Cluster')
    axes[1].set_xlabel('Month')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    plt.close()


def plot_final_clusters_medoids(medoids: List[Optional[int]],
                                clusters: List[List[int]],
                                spatial_anomaly_data: np.ndarray,
                                lat_coords: np.ndarray,
                                lon_coords: np.ndarray,
                                time_stamps: np.ndarray,
                                all_labels: List[Optional[str]],
                                base_labels: List[str],
                                save_path: str) -> None:
    num_clusters = len(medoids)
    logger.info(f"メドイドの空間偏差図をプロット: {num_clusters}クラス")
    n_cols = 5
    n_rows = (num_clusters + n_cols - 1) // n_cols if num_clusters > 0 else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), subplot_kw={"projection": ccrs.PlateCarree()})
    axes = np.atleast_1d(axes).flatten()

    cmap = plt.get_cmap('RdBu_r')
    vmin, vmax = -40, 40
    levels = np.linspace(vmin, vmax, 21)
    norm = Normalize(vmin=vmin, vmax=vmax)

    last_cont = None
    for i in range(num_clusters):
        ax = axes[i]
        if medoids[i] is None:
            ax.set_title(f'Cluster {i+1} (Empty)')
            ax.axis("off")
            continue
        medoid_pattern_2d = spatial_anomaly_data[medoids[i]].reshape(len(lat_coords), len(lon_coords))
        cont = ax.contourf(lon_coords, lat_coords, medoid_pattern_2d, levels=levels, cmap=cmap, extend="both", norm=norm, transform=ccrs.PlateCarree())
        last_cont = cont
        ax.contour(lon_coords, lat_coords, medoid_pattern_2d, colors="k", linewidth=0.5, levels=levels, transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="black", linewidth=0.5)
        ax.set_extent([120, 150, 20, 50], crs=ccrs.PlateCarree())

        cluster_indices = clusters[i]
        frequency = len(cluster_indices) / spatial_anomaly_data.shape[0] * 100.0
        medoid_date = pd.to_datetime(str(time_stamps[medoids[i]])).strftime('%Y-%m-%d')
        cnt = Counter()
        for j in cluster_indices:
            lbl = basic_label_or_none(all_labels[j], base_labels)
            if lbl is not None:
                cnt[lbl] += 1
        rep = cnt.most_common(1)[0][0] if cnt else "Unknown"  # 代表（多数決）
        med_true = basic_label_or_none(all_labels[medoids[i]], base_labels) if all_labels else None
        match_str = ""
        if med_true is not None and rep is not None:
            match_str = f" | MedoidMatch={rep == med_true}"
        elif med_true is not None:
            match_str = f" | MedoidLbl={med_true}"
        ax.set_title(f'Cluster {i+1} (N={len(cluster_indices)}, Freq:{frequency:.1f}%)\nRep:{rep}  Medoid:{medoid_date} Lbl:{med_true}{match_str}', fontsize=8)

    for i in range(num_clusters, len(axes)):
        axes[i].axis("off")

    fig.tight_layout(rect=[0, 0, 0.9, 0.95])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    if last_cont is not None:
        fig.colorbar(last_cont, cax=cbar_ax, label="Sea Level Pressure Anomaly (hPa)")
    fig.suptitle('Final Synoptic Patterns (Medoids of Spatial Anomaly)', fontsize=16)
    plt.savefig(save_path)
    plt.close()


def save_daily_maps_per_cluster(clusters: List[List[int]],
                                spatial_anomaly_data: np.ndarray,
                                lat: np.ndarray,
                                lon: np.ndarray,
                                time_stamps: np.ndarray,
                                labels: List[Optional[str]],
                                base_labels: List[str],
                                out_dir: str,
                                per_cluster_limit: Optional[int] = None) -> None:
    logger.info("クラスタ別の日次気圧配置図（空間偏差）を保存...")
    os.makedirs(out_dir, exist_ok=True)

    cmap = plt.get_cmap('RdBu_r')
    vmin, vmax = -40, 40
    levels = np.linspace(vmin, vmax, 21)
    norm = Normalize(vmin=vmin, vmax=vmax)
    line_levels = np.arange(vmin, vmax + 1, 10)

    for i, idxs in enumerate(tqdm(clusters, desc="Saving daily maps")):
        cnt = Counter()
        for j in idxs:
            lbl = basic_label_or_none(labels[j], base_labels)
            if lbl is not None:
                cnt[lbl] += 1
        rep = cnt.most_common(1)[0][0] if cnt else "Unknown"
        cluster_dir = os.path.join(out_dir, f'cluster_{i+1:02d}_rep_{rep}')
        os.makedirs(cluster_dir, exist_ok=True)

        saved = 0
        for data_idx in idxs:
            if per_cluster_limit is not None and saved >= per_cluster_limit:
                break
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            pressure_map = spatial_anomaly_data[data_idx].reshape(len(lat), len(lon))
            cont = ax.contourf(lon, lat, pressure_map, levels=levels, cmap=cmap, norm=norm, extend='both', transform=ccrs.PlateCarree())
            ax.contour(lon, lat, pressure_map, levels=line_levels, colors='k', linewidths=0.5, transform=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="black", linewidth=0.5)
            ax.set_extent([120, 150, 20, 50], crs=ccrs.PlateCarree())
            cbar = fig.colorbar(cont, ax=ax, orientation='vertical', pad=0.05, aspect=20)
            cbar.set_label('Sea Level Pressure Anomaly (hPa)')

            date_str = pd.to_datetime(str(time_stamps[data_idx])).strftime('%Y-%m-%d')
            true_label = labels[data_idx] if labels else "N/A"
            ax.set_title(f"Date: {date_str}\nTrue Label: {true_label}")
            plt.tight_layout()
            save_path = os.path.join(cluster_dir, f"{date_str}_label_{str(true_label).replace('/', '_')}.png")
            plt.savefig(save_path)
            plt.close(fig)
            saved += 1


def plot_iteration_metrics(history: Dict[str, List[float]], save_path: str) -> None:
    iters = history.get('iteration', [])
    metrics_names = [k for k in history.keys() if k != 'iteration']
    n = len(metrics_names)
    n_cols = 2
    n_rows = (n + n_cols - 1) // n_cols if n > 0 else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    for idx, mname in enumerate(metrics_names):
        ax = axes[idx]
        ax.plot(iters, history.get(mname, []), marker='o')
        ax.set_title(mname)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(mname)
        ax.grid(True)
    for i in range(n, len(axes)):
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_metrics_history_to_csv(history: Dict[str, List[float]], out_csv: str) -> None:
    df = pd.DataFrame(history)
    df.to_csv(out_csv, index=False)


def summarize_cluster_info(clusters: List[List[int]],
                           medoids: List[Optional[int]],
                           labels: List[Optional[str]],
                           base_labels: List[str],
                           time_stamps: np.ndarray) -> pd.DataFrame:
    """
    クラスタごとに代表（メドイド）、頻度、代表（多数決）ラベル、
    メドイド真ラベル、一致有無などを集計
    """
    rows = []
    total = sum(len(c) for c in clusters)
    for i, idxs in enumerate(clusters):
        med = medoids[i]
        cnt = len(idxs)
        share = cnt / max(1, total)
        # 代表（多数決）ラベル
        c = Counter()
        for j in idxs:
            lbl = basic_label_or_none(labels[j], base_labels)
            if lbl is not None:
                c[lbl] += 1
        rep = c.most_common(1)[0][0] if c else None
        rep_count = c[rep] if rep is not None else 0
        rep_share_in_cluster = rep_count / max(1, cnt)

        med_date = pd.to_datetime(str(time_stamps[med])).strftime('%Y-%m-%d') if med is not None else None
        med_lbl = basic_label_or_none(labels[med], base_labels) if (med is not None and labels) else None
        match = (rep == med_lbl) if (rep is not None and med_lbl is not None) else None

        rows.append({
            'Cluster': i + 1,
            'Size': cnt,
            'Share': share,
            'MajorityLabel': rep,
            'MajorityShare': rep_share_in_cluster,
            'MedoidIndex': med,
            'MedoidDate': med_date,
            'MedoidLabel': med_lbl,
            'MedoidMatchesMajority': match
        })
    return pd.DataFrame(rows)