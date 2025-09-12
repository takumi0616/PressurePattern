#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
results_(v5/v6) 配下の各 seed 実験ディレクトリにある以下を再帰的に探索・集計します:

1) 学習時評価ログ: evaluation_v*.log
   - 手法（--- [METHOD] --- で示されるブロック）ごとに
     [Summary] Macro Recall (基本ラベル) と (基本+応用) を抽出して
     平均・最小・最大・中央値を算出・表示します
   - ラベル 6A / 6B / 6C の
     - Correct（平均/最小/最大/中央値/合計）
     - Recall（平均）
     も「各ラベルの再現率（代表ノード群ベース）」セクションから集計

2) 検証時評価ログ: verification_results/*_som/*_verification.log
   - 手法ごとに [Summary] Macro Recall (基本ラベル) / (基本+応用) を抽出し、
     平均・最小・最大・中央値を算出・表示します

3) 学習結果: learning_result/*_som/*_results.log
   - NodewiseMatchRate（最終値）を集計（Mean/Min/Median/Max と matched/total の合計および全体比）

さらに、各表の Min / Max について「どの seed の値か」を表示します。

使い方:
  python src/PressurePattern/search_results_v5.py
    - デフォルトでは、このスクリプトのあるディレクトリの直下 'results_v5' を探索します

  python src/PressurePattern/search_results_v5.py --root /path/to/results_v6
    - 明示的に探索ルートを指定可能です

  python src/PressurePattern/search_results_v5.py --precision 4 --sort rank
    - 小数点以下の表示桁数や並び順を調整可能です
    - --sort は rank/name/basic_combo のいずれか
      * rank: 各表をそれぞれの平均値（降順）で並べる（デフォルト）
      * name: 手法名で昇順
      * basic_combo: 「基本」表は基本の平均降順、「基本+応用」表は基本+応用の平均降順（rankと同じ挙動）
"""

import os
import re
import argparse
from typing import Dict, List, Tuple, Any, Optional
import math
import statistics as stats


HEADER_RE = re.compile(r'^--- \[(.+?)\] ---')
BASIC_SUMMARY_RE = re.compile(r'^\[Summary\]\s*Macro Recall \(基本ラベル\)\s*=\s*([0-9.]+)')
COMBO_SUMMARY_RE = re.compile(r'^\[Summary\]\s*Macro Recall \(基本\+応用\)\s*=\s*([0-9.]+)')
# NodewiseMatchRate (from *_results.log, e.g., "NodewiseMatchRate = 0.358696 (matched 33/92 nodes)")
NODEWISE_RE = re.compile(r'NodewiseMatchRate\s*=\s*([0-9.]+)\s*\(matched\s*(\d+)\s*/\s*(\d+)\s*nodes\)')

# 「各ラベルの再現率（代表ノード群ベース）」ブロック中の 6A/6B/6C 行を抽出
# 例: " - 6A : N=   7 Correct=   3 Recall=0.4286 代表=[...]"
LABEL_LINE_RE = re.compile(
    r'^-\s*(6[ABC])\s*:\s*N=\s*(\d+)\s+Correct=\s*(\d+)\s+Recall=([0-9.]+)'
)

LABELS_TARGET = ("6A", "6B", "6C")


def get_seed_from_path(path: str) -> Optional[int]:
    m = re.search(r'seed(\d+)', path)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def parse_log(log_path: str) -> Dict[str, Dict[str, Any]]:
    """
    1つの evaluation_v*.log をパースして、
    {
      method_name: {
        "basic": float or None,
        "combo": float or None,
        "labels": {
          "6A": {"correct": int, "recall": float},
          "6B": {...},
          "6C": {...}
        }
      }, ...
    } を返す
    """
    methods: Dict[str, Dict[str, Any]] = {}
    current_method: str = ""
    in_basic_label_section: bool = False

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                line = raw_line.strip()

                # メソッドヘッダ
                m_header = HEADER_RE.match(line)
                if m_header:
                    current_method = m_header.group(1).strip()
                    if current_method not in methods:
                        methods[current_method] = {
                            "basic": None,
                            "combo": None,
                            "labels": {}
                        }
                    in_basic_label_section = False  # ヘッダを跨いだら一旦解除
                    continue

                if not current_method:
                    # メソッドブロックの外はスキップ
                    continue

                # セクション開始/終了の検知
                if "【各ラベルの再現率（代表ノード群ベース）】" in line:
                    in_basic_label_section = True
                    continue
                if line.startswith("【複合ラベル考慮の再現率（基本+応用）】"):
                    in_basic_label_section = False
                    # ここからは複合側のラベル表になるので 6A/6B/6C の抽出は行わない
                    #（要件は代表ノード群ベースの値を使うため）
                    continue

                # 要約（Summary）
                m_basic = BASIC_SUMMARY_RE.match(line)
                if m_basic:
                    try:
                        methods[current_method]["basic"] = float(m_basic.group(1))
                    except ValueError:
                        pass
                    continue

                m_combo = COMBO_SUMMARY_RE.match(line)
                if m_combo:
                    try:
                        methods[current_method]["combo"] = float(m_combo.group(1))
                    except ValueError:
                        pass
                    continue

                # ラベル 6A/6B/6C の抽出（代表ノード群ベース）
                if in_basic_label_section:
                    m_label = LABEL_LINE_RE.match(line)
                    if m_label:
                        lab = m_label.group(1)
                        if lab in LABELS_TARGET:
                            try:
                                # n = int(m_label.group(2))  # N は今回は未使用
                                correct = int(m_label.group(3))
                                recall = float(m_label.group(4))
                                methods[current_method]["labels"][lab] = {
                                    "correct": correct,
                                    "recall": recall,
                                }
                            except ValueError:
                                pass
    except FileNotFoundError:
        pass

    return methods


def parse_verification_log_for_summaries(log_path: str) -> Tuple[Optional[float], Optional[float]]:
    """
    verification_results/*_som/*_verification.log から
    [Summary] Macro Recall (基本ラベル) / (基本+応用) を抽出する
    """
    basic_val: Optional[float] = None
    combo_val: Optional[float] = None
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                line = raw_line.strip()
                m_basic = BASIC_SUMMARY_RE.match(line)
                if m_basic:
                    try:
                        basic_val = float(m_basic.group(1))
                    except Exception:
                        pass
                    continue
                m_combo = COMBO_SUMMARY_RE.match(line)
                if m_combo:
                    try:
                        combo_val = float(m_combo.group(1))
                    except Exception:
                        pass
                    continue
    except FileNotFoundError:
        pass
    return basic_val, combo_val


def collect_logs(root: str) -> Tuple[List[str], List[str], Dict[str, Dict[str, Any]]]:
    """
    root 以下からログを再帰的に収集し、手法別に集約する。
    返り値:
      (eval_log_paths, ver_log_paths, aggregate)
      aggregate: {
        method: {
          "basic": [float, ...],
          "basic_pairs": [(float, seed or None), ...],
          "combo": [float, ...],
          "combo_pairs": [(float, seed or None), ...],

          "ver_basic": [float, ...],
          "ver_basic_pairs": [(float, seed or None), ...],
          "ver_combo": [float, ...],
          "ver_combo_pairs": [(float, seed or None), ...],

          "nodewise": [float, ...],
          "nodewise_pairs": [(float, seed or None), ...],
          "nodewise_matched": [int, ...],
          "nodewise_total": [int, ...],

          "labels": {
            "6A": {"correct_sum": int, "corrects": [int, ...], "recalls": [float, ...], "count": int},
            "6B": {...},
            "6C": {...}
          }
        }
      }
    """
    eval_log_paths: List[str] = []
    ver_log_paths: List[str] = []

    # 収集: evaluation_v*.log
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if re.match(r"evaluation_v\d+\.log$", fn):
                eval_log_paths.append(os.path.join(dirpath, fn))

    aggregate: Dict[str, Dict[str, Any]] = {}

    # 初期化ヘルパ
    def ensure_method(method: str):
        if method not in aggregate:
            aggregate[method] = {
                "basic": [],
                "basic_pairs": [],
                "combo": [],
                "combo_pairs": [],
                "ver_basic": [],
                "ver_basic_pairs": [],
                "ver_combo": [],
                "ver_combo_pairs": [],
                # Node-wise metrics from *_results.log
                "nodewise": [],
                "nodewise_pairs": [],
                "nodewise_matched": [],
                "nodewise_total": [],
                "labels": {
                    "6A": {"correct_sum": 0, "corrects": [], "recalls": [], "count": 0},
                    "6B": {"correct_sum": 0, "corrects": [], "recalls": [], "count": 0},
                    "6C": {"correct_sum": 0, "corrects": [], "recalls": [], "count": 0},
                },
            }

    # 解析: evaluation_v*.log
    for p in sorted(eval_log_paths):
        parsed = parse_log(p)
        seed = get_seed_from_path(p)
        for method, vals in parsed.items():
            ensure_method(method)
            # Summary
            if vals.get("basic") is not None:
                aggregate[method]["basic"].append(vals["basic"])  # type: ignore[index]
                aggregate[method]["basic_pairs"].append((vals["basic"], seed))  # type: ignore[index]
            if vals.get("combo") is not None:
                aggregate[method]["combo"].append(vals["combo"])  # type: ignore[index]
                aggregate[method]["combo_pairs"].append((vals["combo"], seed))  # type: ignore[index]
            # Labels 6A/6B/6C（代表ノード群ベース）
            labels: Dict[str, Dict[str, float]] = vals.get("labels", {})
            for lab in LABELS_TARGET:
                info = labels.get(lab)
                if info:
                    try:
                        c = int(info["correct"])  # type: ignore[arg-type]
                        r = float(info["recall"])  # type: ignore[arg-type]
                        aggregate[method]["labels"][lab]["correct_sum"] += c  # type: ignore[index]
                        aggregate[method]["labels"][lab]["corrects"].append(c)  # type: ignore[index]
                        aggregate[method]["labels"][lab]["recalls"].append(r)  # type: ignore[index]
                        aggregate[method]["labels"][lab]["count"] += 1  # type: ignore[index]
                    except Exception:
                        # パース失敗時はスキップ
                        pass

    # 追加収集: learning_result/*_som/*_results.log から NodewiseMatchRate（最終値）を集計
    results_logs: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith("_results.log"):
                results_logs.append(os.path.join(dirpath, fn))

    for rp in sorted(results_logs):
        # 推定手法名: ディレクトリ名 '<name>_som' の前半を大文字化
        method_dir = os.path.basename(os.path.dirname(rp))  # e.g., 'euclidean_som'
        base_name = method_dir.rsplit("_som", 1)[0].upper()
        ensure_method(base_name)
        last_tuple = None  # (rate, matched, total)
        try:
            with open(rp, "r", encoding="utf-8", errors="ignore") as f:
                for raw in f:
                    m = NODEWISE_RE.search(raw)
                    if m:
                        try:
                            rate = float(m.group(1))
                            matched = int(m.group(2))
                            total = int(m.group(3))
                            last_tuple = (rate, matched, total)
                        except Exception:
                            pass
        except FileNotFoundError:
            last_tuple = None

        if last_tuple:
            rate, matched, total = last_tuple
            aggregate[base_name]["nodewise"].append(rate)  # type: ignore[index]
            aggregate[base_name]["nodewise_matched"].append(matched)  # type: ignore[index]
            aggregate[base_name]["nodewise_total"].append(total)  # type: ignore[index]
            aggregate[base_name]["nodewise_pairs"].append((rate, get_seed_from_path(rp)))  # type: ignore[index]

    # 収集/解析: verification_results/*_som/*_verification.log
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith("_verification.log"):
                vp = os.path.join(dirpath, fn)
                ver_log_paths.append(vp)
                method_dir = os.path.basename(os.path.dirname(vp))  # '<name>_som'
                base_name = method_dir.rsplit("_som", 1)[0].upper()
                ensure_method(base_name)
                vb, vc = parse_verification_log_for_summaries(vp)
                seed = get_seed_from_path(vp)
                if vb is not None:
                    aggregate[base_name]["ver_basic"].append(vb)  # type: ignore[index]
                    aggregate[base_name]["ver_basic_pairs"].append((vb, seed))  # type: ignore[index]
                if vc is not None:
                    aggregate[base_name]["ver_combo"].append(vc)  # type: ignore[index]
                    aggregate[base_name]["ver_combo_pairs"].append((vc, seed))  # type: ignore[index]

    return eval_log_paths, ver_log_paths, aggregate


def mean_or_nan(values: List[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def min_or_nan(values: List[float]) -> float:
    if not values:
        return float("nan")
    return min(values)


def max_or_nan(values: List[float]) -> float:
    if not values:
        return float("nan")
    return max(values)


def median_or_nan(values: List[float]) -> float:
    if not values:
        return float("nan")
    return stats.median(values)


def fmt_float(v: float, prec: int) -> str:
    return f"{v:.{prec}f}" if not math.isnan(v) else "NaN"


def fmt_seed(seed: Optional[int]) -> str:
    return f"{seed:d}" if seed is not None else "-"


def find_extreme_seeds(pairs: List[Tuple[float, Optional[int]]]) -> Tuple[Optional[int], Optional[int], float, float]:
    """
    pairs: [(value, seed), ...]
    return: (min_seed, max_seed, min_val, max_val)
    """
    if not pairs:
        return None, None, float("nan"), float("nan")
    values = [v for v, _ in pairs]
    min_v = min(values)
    max_v = max(values)
    min_seed = next((s for v, s in pairs if v == min_v), None)
    max_seed = next((s for v, s in pairs if v == max_v), None)
    return min_seed, max_seed, min_v, max_v


def main():
    parser = argparse.ArgumentParser(description="results_(v5/v6) のログから手法別の各種統計（学習/検証）を算出")
    default_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_v5")
    parser.add_argument(
        "--root",
        type=str,
        default=default_root,
        help=f"探索対象の results ディレクトリ (default: {default_root})",
    )
    parser.add_argument(
        "--sort",
        type=str,
        default="rank",
        choices=["rank", "name", "basic_combo"],
        help="表示順のソートキー: rank(各表で平均降順) / name(名前昇順) / basic_combo(基本/基本+応用を各平均降順)",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="小数点以下の表示桁数 (default: 4)",
    )
    args = parser.parse_args()

    root = args.root
    if not os.path.isdir(root):
        print(f"[ERROR] 指定のディレクトリが存在しません: {root}")
        return

    eval_paths, ver_paths, aggregate = collect_logs(root)

    print("==== results 集計 ====")
    print(f"探索対象: {root}")
    print(f"検出ログ数: evaluation={len(eval_paths)} verification={len(ver_paths)}")
    print("")

    # 並び順関数
    def key_by_name(item):
        return item[0]

    def key_by_mean_key(key: str):
        def _k(item):
            name, metrics = item
            return -mean_or_nan(metrics.get(key, []))  # type: ignore[arg-type,index]
        return _k

    def print_table(title: str, value_key: str, pair_key: str, sort_mode: str, prec: int):
        header = (
            f'{title:24s} '
            f'{"N":>5s} {"Mean":>10s} {"Min":>10s} {"Median":>10s} {"Max":>10s} '
            f'{"MinSeed":>8s} {"MaxSeed":>8s}'
        )
        print(header)
        print("-" * len(header))
        items = list(aggregate.items())
        if sort_mode == "name":
            items.sort(key=key_by_name)
        else:
            items.sort(key=key_by_mean_key(value_key))

        for method, metrics in items:
            vals: List[float] = metrics.get(value_key, [])  # type: ignore[assignment]
            pairs: List[Tuple[float, Optional[int]]] = metrics.get(pair_key, [])  # type: ignore[assignment]
            n = len(vals)
            mean_v = mean_or_nan(vals)
            min_v = min_or_nan(vals)
            med_v = median_or_nan(vals)
            max_v = max_or_nan(vals)
            min_seed, max_seed, _mv, _xv = find_extreme_seeds(pairs)
            print(
                f"{method:24s} {n:5d} "
                f"{fmt_float(mean_v, prec):>10s} {fmt_float(min_v, prec):>10s} "
                f"{fmt_float(med_v, prec):>10s} {fmt_float(max_v, prec):>10s} "
                f"{fmt_seed(min_seed):>8s} {fmt_seed(max_seed):>8s}"
            )
        print("")

    prec = args.precision

    # 学習(評価ログ) 基本ラベル
    print("==== 手法別 Macro Recall 統計（学習: 基本ラベル, evaluation_v*.log） ====")
    print_table("[基本] Method", "basic", "basic_pairs", args.sort, prec)

    # 学習(評価ログ) 基本+応用
    print("==== 手法別 Macro Recall 統計（学習: 基本+応用, evaluation_v*.log） ====")
    print_table("[基本+応用] Method", "combo", "combo_pairs", args.sort, prec)

    # 検証(verification) 基本ラベル
    print("==== 手法別 Macro Recall 統計（検証: 基本ラベル, *_verification.log） ====")
    print_table("[Ver基本] Method", "ver_basic", "ver_basic_pairs", args.sort, prec)

    # 検証(verification) 基本+応用
    print("==== 手法別 Macro Recall 統計（検証: 基本+応用, *_verification.log） ====")
    print_table("[Ver基本+応用] Method", "ver_combo", "ver_combo_pairs", args.sort, prec)

    # NodewiseMatchRate 統計（Mean/Min/Median/Max と matched/total の合計および全体比 + Min/Max seed）
    header_nodewise = (
        f'{"[Nodewise] Method":24s} '
        f'{"N":>5s} {"Mean":>10s} {"Min":>10s} {"Median":>10s} {"Max":>10s} '
        f'{"Σmatch":>10s} {"Σnodes":>10s} {"Overall":>10s} '
        f'{"MinSeed":>8s} {"MaxSeed":>8s}'
    )
    print("==== 手法別 NodewiseMatchRate 統計（learning_result/*_results.log の[Final Metrics]より） ====")
    print(header_nodewise)
    print("-" * len(header_nodewise))
    items3 = list(aggregate.items())

    if args.sort == "name":
        items3.sort(key=key_by_name)
    else:
        items3.sort(key=key_by_mean_key("nodewise"))

    for method, metrics in items3:
        rates: List[float] = metrics.get("nodewise", [])  # type: ignore[assignment]
        n = len(rates)
        mean_v = mean_or_nan(rates)
        min_v = min_or_nan(rates)
        med_v = median_or_nan(rates)
        max_v = max_or_nan(rates)
        sum_match = sum(metrics.get("nodewise_matched", []))  # type: ignore[arg-type]
        sum_total = sum(metrics.get("nodewise_total", []))  # type: ignore[arg-type]
        overall = (sum_match / sum_total) if sum_total > 0 else float("nan")
        min_seed, max_seed, _mv, _xv = find_extreme_seeds(metrics.get("nodewise_pairs", []))  # type: ignore[arg-type]
        print(
            f"{method:24s} {n:5d} "
            f"{fmt_float(mean_v, prec):>10s} {fmt_float(min_v, prec):>10s} "
            f"{fmt_float(med_v, prec):>10s} {fmt_float(max_v, prec):>10s} "
            f"{sum_match:10d} {sum_total:10d} {fmt_float(overall, prec):>10s} "
            f"{fmt_seed(min_seed):>8s} {fmt_seed(max_seed):>8s}"
        )
    print("")

    # 6A/6B/6C の統計（Correct 平均/最小/最大/中央値/合計、Recall 平均）
    def print_label_stats(label_key: str):
        title = f"==== ラベル {label_key} の統計（代表ノード群ベース: Correctの平均/最小/最大/中央値/合計、Recallの平均） ===="
        print(title)
        header = (
            f'{"Method":24s} '
            f'{"N":>5s} '
            f'{"Mean_C":>10s} {"Min_C":>10s} {"Med_C":>10s} {"Max_C":>10s} {"Sum_C":>10s} '
            f'{"Mean_R":>10s}'
        )
        print(header)
        print("-" * len(header))
        rows = []
        for method, metrics in aggregate.items():
            info = metrics["labels"][label_key]  # type: ignore[index]
            corrects: List[int] = info["corrects"]  # type: ignore[index]
            recalls: List[float] = info["recalls"]  # type: ignore[index]
            n = len(corrects)
            mean_c = stats.mean(corrects) if n > 0 else float("nan")
            min_c = min(corrects) if n > 0 else float("nan")
            med_c = stats.median(corrects) if n > 0 else float("nan")
            max_c = max(corrects) if n > 0 else float("nan")
            sum_c = info["correct_sum"]  # type: ignore[index]
            mean_r = mean_or_nan(recalls)
            rows.append((method, n, mean_c, min_c, med_c, max_c, sum_c, mean_r))

        # デフォルトはランキング（Mean_R 降順）: 同率は手法名で
        if args.sort == "name":
            rows.sort(key=lambda x: x[0])
        else:
            rows.sort(key=lambda x: (- (x[7] if not math.isnan(x[7]) else -1.0), x[0]))

        for method, n, mean_c, min_c, med_c, max_c, sum_c, mean_r in rows:
            print(
                f"{method:24s} {n:5d} "
                f"{fmt_float(float(mean_c), prec):>10s} {fmt_float(float(min_c), prec):>10s} "
                f"{fmt_float(float(med_c), prec):>10s} {fmt_float(float(max_c), prec):>10s} "
                f"{sum_c:10d} {fmt_float(mean_r, prec):>10s}"
            )
        print("")

    for lab in LABELS_TARGET:
        print_label_stats(lab)

    print("注記:")
    print(" - 学習(評価ログ)/検証(verification)の Macro Recall は [Summary] の値から算出（Mean/Min/Median/Max）。")
    print(" - 各表では MinSeed / MaxSeed に、最小/最大値が出た seed を表示（同値が複数ある場合は最初に検出したもの）。")
    print(" - 6A/6B/6C の Correct/Recall は「各ラベルの再現率（代表ノード群ベース）」の値を使用（学習評価ログ）。")
    print(" - デフォルトの並び順（--sort rank/basic_combo）は平均値の降順です。--sort name で手法名順。")


if __name__ == "__main__":
    main()
