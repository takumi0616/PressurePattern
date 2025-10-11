#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
改良版: 複数の results_* ディレクトリを横断集計し、見やすい表と総合評価(推奨手法)を表示

従来:
  - 1つのルート配下 (results_v5/v6 など) のみを探索

本改良:
  - 複数ルート (--roots) を同時に探索・統合できる
  - 各ルートごとの要約に加えて、「全ルート統合（Overall）」の集計を表示
  - 見やすい横断表（Method × Root のピボット風）を出力（Verification Basic の平均）
  - ばらつき（標準偏差）や総合スコアを用いた推奨手法表示（--recommend）
  - オプションで CSV 出力（--csv-out）

対象ログ:
1) 学習時評価ログ: evaluation_v*.log
   - 手法（--- [METHOD] ---）ごとに [Summary] Macro Recall (基本ラベル) / (基本+応用) を抽出
   - ラベル 6A / 6B / 6C の Correct（平均/最小/最大/中央値/合計）と Recall（平均）
     を「各ラベルの再現率（代表ノード群ベース）」から集計

2) 検証時評価ログ: verification_results/*_som/*_verification.log
   - 手法ごとに [Summary] Macro Recall (基本ラベル) / (基本+応用) を抽出

3) 学習結果: learning_result/*_som/*_results.log
   - NodewiseMatchRate（最終値）を集計（Mean/Min/Median/Max と matched/total の合計および全体比）

表示する表では Min / Max の seed も併記（どの seed で出た値か）。

使い方（例）:
  # デフォルトでは src/PressurePattern/ 直下にある以下3つが存在すれば自動探索します:
  #   results_v6_iter100, results_v6_iter1000, results_v6_iter10000
  python src/PressurePattern/search_results_v7.py

  # 任意のディレクトリを複数指定して横断集計
  python src/PressurePattern/search_results_v7.py --roots \
    src/PressurePattern/results_v6_iter100 \
    src/PressurePattern/results_v6_iter1000 \
    src/PressurePattern/results_v6_iter10000

  # 従来どおり単一ルートで集計（後方互換）
  python src/PressurePattern/search_results_v7.py --root src/PressurePattern/results_v6_iter100

  # 推奨手法の算出を表示（検証Basic重視、Combo/安定性/Nodewiseも加味）
  # すべて表示したい場合は --topk 0 を指定
  python src/PressurePattern/search_results_v7.py --recommend --topk 0

  # 横断ピボット表（Verification Basic の平均）の CSV を出力
  python src/PressurePattern/search_results_v7.py --csv-out summary_ver_basic.csv

  # NetCDF の基本ラベル分布をレポート（1991-01-01〜2000-12-31）
  python src/PressurePattern/search_results_v7.py --nc-report --nc-file src/PressurePattern/prmsl_era5_all_data_seasonal_large.nc --nc-start 1991-01-01 --nc-end 2000-12-31

オプション:
  --roots       複数の results ディレクトリを指定（最優先）
  --root        単一の results ディレクトリを指定（後方互換）
  --sort        並び順（rank/name/basic_combo）: デフォルト rank
  --precision   小数点以下の表示桁数: デフォルト 2
  --recommend   総合スコアに基づく推奨手法 Top-K を表示
  --topk        推奨手法の件数（--recommend 使用時のみ）: 0 以下で全件表示（デフォルト 0）
  --csv-out     横断ピボット表（Verification Basic 平均）の CSV 出力先
  --nc-report   NetCDF ファイルから指定期間の基本ラベル分布をレポート
  --nc-file     NetCDF ファイルパス（既定: src/PressurePattern/prmsl_era5_all_data_seasonal_large.nc）
  --nc-start    期間開始日（YYYY-MM-DD、既定: 1991-01-01）
  --nc-end      期間終了日（YYYY-MM-DD、既定: 2000-12-31）
"""

import os
import re
import argparse
from typing import Dict, List, Tuple, Any, Optional
import math
import statistics as stats
from decimal import Decimal, ROUND_HALF_UP
import csv
from collections import Counter
from datetime import datetime, timezone, timedelta
import sys
import atexit
import numpy as np
import pandas as pd

# xarray は存在すれば利用、無ければ後述の netCDF4 フォールバックを用いる
try:
    import xarray as xr  # type: ignore
except Exception:
    xr = None  # xarray が無い環境でも動作可能にする


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
# verification 用（例: "- 6A : N_base=   0 Correct_base=   0 Recall_base=0.0000 | N_comp=  28 Correct_comp=   0 Recall_comp=0.0000"）
VER_LABEL_LINE_RE = re.compile(
    r'^-\s*(6[ABC])\s*:\s*N_base=\s*(\d+)\s+Correct_base=\s*(\d+)\s+Recall_base=([0-9.]+)\s*\|\s*N_comp=\s*(\d+)\s+Correct_comp=\s*(\d+)\s+Recall_comp=([0-9.]+)'
)
# verification 用（base のみ。例: "- 6A : N_base=   1 Correct_base=   0 Recall_base=0.0000"）
VER_LABEL_BASE_ONLY_RE = re.compile(
    r'^-\s*(6[ABC])\s*:\s*N_base=\s*(\d+)\s+Correct_base=\s*(\d+)\s+Recall_base=([0-9.]+)'
)

LABELS_TARGET = ("6A", "6B", "6C")
# 全15の基本ラベル（列順）
LABELS_ALL = ("1", "2A", "2B", "2C", "2D", "3A", "3B", "3C", "3D", "4A", "4B", "5", "6A", "6B", "6C")

# 学習/評価ログ（results.log/evaluation.log）用: 全ラベル行
ALL_LABEL_LINE_RE = re.compile(
    r'^-\s*(1|2A|2B|2C|2D|3A|3B|3C|3D|4A|4B|5|6A|6B|6C)\s*:\s*N=\s*(\d+)\s+Correct=\s*(\d+)\s+Recall=([0-9.]+)'
)

# 検証ログ用（base | comp の両方が並記されている行）
VER_ALL_LABEL_LINE_RE = re.compile(
    r'^-\s*(1|2A|2B|2C|2D|3A|3B|3C|3D|4A|4B|5|6A|6B|6C)\s*:\s*N_base=\s*(\d+)\s+Correct_base=\s*(\d+)\s+Recall_base=([0-9.]+)\s*\|\s*N_comp=.*$'
)
# 検証ログ用（base のみの行）
VER_ALL_LABEL_BASE_ONLY_RE = re.compile(
    r'^-\s*(1|2A|2B|2C|2D|3A|3B|3C|3D|4A|4B|5|6A|6B|6C)\s*:\s*N_base=\s*(\d+)\s+Correct_base=\s*(\d+)\s+Recall_base=([0-9.]+)'
)


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
                # Typhoon detection metric (Recall-based)
                "typhoon": [],
                "typhoon_pairs": [],
                "typhoon_ver": [],
                "typhoon_ver_pairs": [],
                # 検証時の 6A/6B/6C 統計（従来）
                "ver_labels": {
                    "6A": {"correct_sum": 0, "corrects": [], "recalls": [], "count": 0},
                    "6B": {"correct_sum": 0, "corrects": [], "recalls": [], "count": 0},
                    "6C": {"correct_sum": 0, "corrects": [], "recalls": [], "count": 0},
                },
                # 学習時の 6A/6B/6C 統計（従来）
                "labels": {
                    "6A": {"correct_sum": 0, "corrects": [], "recalls": [], "count": 0},
                    "6B": {"correct_sum": 0, "corrects": [], "recalls": [], "count": 0},
                    "6C": {"correct_sum": 0, "corrects": [], "recalls": [], "count": 0},
                },
                # 追加: 全15基本ラベルの再現率（学習・検証）をラベル別に蓄積
                "train_label_recalls": {lab: [] for lab in LABELS_ALL},
                "ver_label_recalls": {lab: [] for lab in LABELS_ALL},
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
            # 台風補足（Recall）: 6A/6B の Recall を利用可能ラベルで平均して seed 指標とする
            try:
                ty_vals: List[float] = []
                info_6a = labels.get("6A")
                info_6b = labels.get("6B")
                if info_6a is not None and isinstance(info_6a.get("recall"), (int, float)):
                    ty_vals.append(float(info_6a["recall"]))
                if info_6b is not None and isinstance(info_6b.get("recall"), (int, float)):
                    ty_vals.append(float(info_6b["recall"]))
                if ty_vals:
                    ty_metric = sum(ty_vals) / len(ty_vals)
                    aggregate[method]["typhoon"].append(ty_metric)  # type: ignore[index]
                    aggregate[method]["typhoon_pairs"].append((ty_metric, seed))  # type: ignore[index]
            except Exception:
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
        # 追加収集: *_results.log 終盤の「各ラベルの再現率（代表ノード群ベース）」から全15基本ラベルの Recall を学習側として集計
        try:
            in_basic = False
            with open(rp, "r", encoding="utf-8", errors="ignore") as f:
                for raw in f:
                    s = raw.strip()
                    if "【各ラベルの再現率（代表ノード群ベース）】" in s:
                        in_basic = True
                        continue
                    if s.startswith("[Summary]"):
                        in_basic = False
                    if in_basic:
                        m_all = ALL_LABEL_LINE_RE.match(s)
                        if m_all:
                            lab = m_all.group(1)
                            try:
                                rec = float(m_all.group(4))
                            except Exception:
                                continue
                            if lab in LABELS_ALL:
                                aggregate[base_name]["train_label_recalls"][lab].append(rec)  # type: ignore[index]
        except Exception:
            pass

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
                # Typhoon (verification) 6A/6B recall per seed -> average, and collect verification label stats (6A/6B/6C)
                try:
                    ty_vals_ver: List[float] = []
                    with open(vp, "r", encoding="utf-8", errors="ignore") as vf:
                        for raw_v in vf:
                            s = raw_v.strip()
                            # まず verification 専用フォーマットを試す（全ラベル→6A/6B/6Cの順）
                            m_all = VER_ALL_LABEL_LINE_RE.match(s)
                            if m_all:
                                lab = m_all.group(1)
                                try:
                                    corr_base = int(m_all.group(3))
                                    rec_base = float(m_all.group(4))
                                except Exception:
                                    continue
                                if lab in ("6A", "6B"):
                                    ty_vals_ver.append(rec_base)
                                if lab in LABELS_TARGET:
                                    try:
                                        aggregate[base_name]["ver_labels"][lab]["correct_sum"] += corr_base  # type: ignore[index]
                                        aggregate[base_name]["ver_labels"][lab]["corrects"].append(corr_base)  # type: ignore[index]
                                        aggregate[base_name]["ver_labels"][lab]["recalls"].append(rec_base)  # type: ignore[index]
                                        aggregate[base_name]["ver_labels"][lab]["count"] += 1  # type: ignore[index]
                                    except Exception:
                                        pass
                                if lab in LABELS_ALL:
                                    try:
                                        aggregate[base_name]["ver_label_recalls"][lab].append(rec_base)  # type: ignore[index]
                                    except Exception:
                                        pass
                                continue
                            m_all_base = VER_ALL_LABEL_BASE_ONLY_RE.match(s)
                            if m_all_base:
                                lab = m_all_base.group(1)
                                try:
                                    corr_base = int(m_all_base.group(3))
                                    rec_base = float(m_all_base.group(4))
                                except Exception:
                                    continue
                                if lab in ("6A", "6B"):
                                    ty_vals_ver.append(rec_base)
                                if lab in LABELS_TARGET:
                                    try:
                                        aggregate[base_name]["ver_labels"][lab]["correct_sum"] += corr_base  # type: ignore[index]
                                        aggregate[base_name]["ver_labels"][lab]["corrects"].append(corr_base)  # type: ignore[index]
                                        aggregate[base_name]["ver_labels"][lab]["recalls"].append(rec_base)  # type: ignore[index]
                                        aggregate[base_name]["ver_labels"][lab]["count"] += 1  # type: ignore[index]
                                    except Exception:
                                        pass
                                if lab in LABELS_ALL:
                                    try:
                                        aggregate[base_name]["ver_label_recalls"][lab].append(rec_base)  # type: ignore[index]
                                    except Exception:
                                        pass
                                continue
                            m_ver = VER_LABEL_LINE_RE.match(s)
                            if m_ver:
                                lab = m_ver.group(1)
                                try:
                                    corr_base = int(m_ver.group(3))
                                    rec_base = float(m_ver.group(4))
                                except Exception:
                                    continue
                                if lab in ("6A", "6B"):
                                    ty_vals_ver.append(rec_base)
                                if lab in LABELS_TARGET:
                                    try:
                                        aggregate[base_name]["ver_labels"][lab]["correct_sum"] += corr_base  # type: ignore[index]
                                        aggregate[base_name]["ver_labels"][lab]["corrects"].append(corr_base)  # type: ignore[index]
                                        aggregate[base_name]["ver_labels"][lab]["recalls"].append(rec_base)  # type: ignore[index]
                                        aggregate[base_name]["ver_labels"][lab]["count"] += 1  # type: ignore[index]
                                    except Exception:
                                        pass
                                # 追加: 全15基本ラベルの base Recall を収集
                                if lab in LABELS_ALL:
                                    try:
                                        aggregate[base_name]["ver_label_recalls"][lab].append(rec_base)  # type: ignore[index]
                                    except Exception:
                                        pass
                                continue
                            # 追加: 複合系(Comp)の無い base のみの行にも対応
                            m_ver_base = VER_LABEL_BASE_ONLY_RE.match(s)
                            if m_ver_base:
                                lab = m_ver_base.group(1)
                                try:
                                    corr_base = int(m_ver_base.group(3))
                                    rec_base = float(m_ver_base.group(4))
                                except Exception:
                                    continue
                                if lab in ("6A", "6B"):
                                    ty_vals_ver.append(rec_base)
                                if lab in LABELS_TARGET:
                                    try:
                                        aggregate[base_name]["ver_labels"][lab]["correct_sum"] += corr_base
                                        aggregate[base_name]["ver_labels"][lab]["corrects"].append(corr_base)
                                        aggregate[base_name]["ver_labels"][lab]["recalls"].append(rec_base)
                                        aggregate[base_name]["ver_labels"][lab]["count"] += 1
                                    except Exception:
                                        pass
                                # 追加: 全15基本ラベルの base Recall を収集
                                if lab in LABELS_ALL:
                                    try:
                                        aggregate[base_name]["ver_label_recalls"][lab].append(rec_base)  # type: ignore[index]
                                    except Exception:
                                        pass
                                continue
                            # フォールバック: 学習形式に近い行にも対応
                            m_lab = LABEL_LINE_RE.match(s)
                            if m_lab:
                                lab = m_lab.group(1)
                                try:
                                    corr_v = int(m_lab.group(3))
                                    rec_v = float(m_lab.group(4))
                                except Exception:
                                    continue
                                if lab in ("6A", "6B"):
                                    ty_vals_ver.append(rec_v)
                                if lab in LABELS_TARGET:
                                    try:
                                        aggregate[base_name]["ver_labels"][lab]["correct_sum"] += corr_v  # type: ignore[index]
                                        aggregate[base_name]["ver_labels"][lab]["corrects"].append(corr_v)  # type: ignore[index]
                                        aggregate[base_name]["ver_labels"][lab]["recalls"].append(rec_v)  # type: ignore[index]
                                        aggregate[base_name]["ver_labels"][lab]["count"] += 1  # type: ignore[index]
                                    except Exception:
                                        pass
                                # 追加: 旧式（学習形式）行でも base として扱い収集
                                if lab in LABELS_ALL:
                                    try:
                                        aggregate[base_name]["ver_label_recalls"][lab].append(rec_v)  # type: ignore[index]
                                    except Exception:
                                        pass
                    if ty_vals_ver:
                        ty_metric_ver = sum(ty_vals_ver) / len(ty_vals_ver)
                        aggregate[base_name]["typhoon_ver"].append(ty_metric_ver)  # type: ignore[index]
                        aggregate[base_name]["typhoon_ver_pairs"].append((ty_metric_ver, seed))  # type: ignore[index]
                except Exception:
                    pass

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


def std_or_nan(values: List[float]) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return 0.0
    try:
        return stats.stdev(values)
    except Exception:
        return float("nan")


def has_any_values(aggregate: Dict[str, Dict[str, Any]], key: str) -> bool:
    """aggregate の各 method で指定 key の配列に1つでも値があれば True。"""
    for _m, metrics in aggregate.items():
        vals = metrics.get(key, [])
        if isinstance(vals, list) and len(vals) > 0:
            return True
    return False


def fmt_float(v: float, prec: int) -> str:
    if math.isnan(v):
        return "NaN"
    try:
        d = Decimal(str(v))
        exp = Decimal('1').scaleb(-prec)  # e.g., prec=2 -> 0.01
        rounded = d.quantize(exp, rounding=ROUND_HALF_UP)
        return str(rounded)  # keeps trailing zeros (e.g., '1.20')
    except Exception:
        # Fallback to standard formatting
        return f"{v:.{prec}f}"


def fmt_seed(seed: Optional[int]) -> str:
    return f"{seed:d}" if seed is not None else "-"


class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass
    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

def setup_global_tee_logging(log_path: str) -> None:
    """
    全ての標準出力/標準エラー出力を指定ファイルに必ず保存する Tee を有効化する。
    実行環境で `> xxx.log` などのリダイレクトがあっても、同時に log_path にも書き出される。
    """
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        f = open(log_path, "a", encoding="utf-8")  # 追記モード。毎回上書きにしたい場合は "w" に変更
    except Exception as e:
        # ファイルが開けない場合でも処理は継続（標準出力のみ）
        print(f"[WARN] ログファイル {log_path} を開けません: {e}")
        return
    # 現在の stdout/stderr を保持したまま Tee 化
    sys.stdout = _Tee(sys.stdout, f)  # type: ignore[assignment]
    sys.stderr = _Tee(sys.stderr, f)  # type: ignore[assignment]
    atexit.register(f.close)
    print(f"[INFO] Tee logging enabled -> {log_path}")

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


def seed_means_from_pairs(pairs: List[Tuple[float, Optional[int]]]) -> Dict[int, float]:
    """同一 seed が複数回出現する場合は平均して seed->平均値 の辞書を返す。"""
    from collections import defaultdict
    d: Dict[int, List[float]] = defaultdict(list)
    for v, s in pairs:
        if s is None:
            continue
        try:
            d[int(s)].append(float(v))
        except Exception:
            continue
    return {s: (sum(vals) / len(vals)) for s, vals in d.items() if len(vals) > 0}


def select_best_seed_balanced(maps: List[Dict[int, float]]) -> Optional[int]:
    """
    5 指標（ver_basic, ver_combo, basic, combo, nodewise）の seed 別値を受け取り、
    - 出現指標数が最大の seed を優先（全5指標に揃っていれば最優先）
    - 次点として平均値（利用可能な指標の単純平均）が最大の seed を選ぶ
    """
    from collections import Counter
    if not maps:
        return None
    counts = Counter()
    for m in maps:
        counts.update(m.keys())
    if not counts:
        return None

    max_cov = max(counts.values())
    while max_cov >= 1:
        candidates = [s for s, c in counts.items() if c == max_cov]
        if candidates:
            best_seed: Optional[int] = None
            best_avg = -float("inf")
            for s in candidates:
                vals = [m[s] for m in maps if s in m]
                if not vals:
                    continue
                avg = sum(vals) / len(vals)
                if avg > best_avg or (avg == best_avg and (best_seed is None or s < best_seed)):
                    best_seed = s
                    best_avg = avg
            if best_seed is not None:
                return best_seed
        max_cov -= 1
    return None


def pair_means_by_root_seed(pairs: List[Tuple[float, Optional[int]]], root_label: str) -> Dict[Tuple[str, int], float]:
    """
    同一 (root, seed) が複数回出現する場合は平均して {(root, seed): 平均値} を返す。
    """
    base = seed_means_from_pairs(pairs)
    return {(root_label, s): v for s, v in base.items()}


def select_best_root_seed_balanced(maps: List[Dict[Tuple[str, int], float]]) -> Optional[Tuple[str, int]]:
    """
    5 指標の (root, seed) 別値を受け取り、
    - 出現指標数が最大の (root, seed) を優先
    - 次に平均値（利用可能な指標の単純平均）が最大の (root, seed)
    """
    from collections import Counter
    if not maps:
        return None
    counts = Counter()
    for m in maps:
        counts.update(m.keys())
    if not counts:
        return None
    max_cov = max(counts.values())
    while max_cov >= 1:
        candidates = [k for k, c in counts.items() if c == max_cov]
        if candidates:
            best_key: Optional[Tuple[str, int]] = None
            best_avg = -float("inf")
            for key in candidates:
                vals = [m[key] for m in maps if key in m]
                if not vals:
                    continue
                avg = sum(vals) / len(vals)
                if avg > best_avg or (avg == best_avg and (best_key is None or key < best_key)):
                    best_key = key
                    best_avg = avg
            if best_key is not None:
                return best_key
        max_cov -= 1
    return None

def ensure_method_in_agg(agg: Dict[str, Dict[str, Any]], method: str):
    if method not in agg:
        agg[method] = {
            "basic": [],
            "basic_pairs": [],
            "combo": [],
            "combo_pairs": [],
            "ver_basic": [],
            "ver_basic_pairs": [],
            "ver_combo": [],
            "ver_combo_pairs": [],
            "nodewise": [],
            "nodewise_pairs": [],
            "nodewise_matched": [],
            "nodewise_total": [],
            "typhoon": [],
            "typhoon_pairs": [],
            "typhoon_ver": [],
            "typhoon_ver_pairs": [],
            "ver_labels": {
                "6A": {"correct_sum": 0, "corrects": [], "recalls": [], "count": 0},
                "6B": {"correct_sum": 0, "corrects": [], "recalls": [], "count": 0},
                "6C": {"correct_sum": 0, "corrects": [], "recalls": [], "count": 0},
            },
            "labels": {
                "6A": {"correct_sum": 0, "corrects": [], "recalls": [], "count": 0},
                "6B": {"correct_sum": 0, "corrects": [], "recalls": [], "count": 0},
                "6C": {"correct_sum": 0, "corrects": [], "recalls": [], "count": 0},
            },
            "train_label_recalls": {lab: [] for lab in LABELS_ALL},
            "ver_label_recalls": {lab: [] for lab in LABELS_ALL},
        }


def merge_aggregates(aggregates: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """
    複数の aggregate を統合
    """
    merged: Dict[str, Dict[str, Any]] = {}
    for agg in aggregates:
        for method, m in agg.items():
            ensure_method_in_agg(merged, method)
            # リスト系を伸長
            for key in [
                "basic", "basic_pairs",
                "combo", "combo_pairs",
                "ver_basic", "ver_basic_pairs",
                "ver_combo", "ver_combo_pairs",
                "nodewise", "nodewise_pairs",
                "nodewise_matched", "nodewise_total",
                "typhoon", "typhoon_pairs",
                "typhoon_ver", "typhoon_ver_pairs",
            ]:
                merged[method][key].extend(m.get(key, []))  # type: ignore[index]
            # ラベル系をマージ
            for lab in LABELS_TARGET:
                dst = merged[method]["labels"][lab]  # type: ignore[index]
                src = m.get("labels", {}).get(lab, {})
                dst["correct_sum"] += src.get("correct_sum", 0)  # type: ignore[index]
                dst["count"] += src.get("count", 0)  # type: ignore[index]
                dst["corrects"].extend(src.get("corrects", []))  # type: ignore[index]
                dst["recalls"].extend(src.get("recalls", []))  # type: ignore[index]
            # Merge verification label stats as well
            for lab in LABELS_TARGET:
                dstv = merged[method].setdefault("ver_labels", {
                    "6A": {"correct_sum": 0, "corrects": [], "recalls": [], "count": 0},
                    "6B": {"correct_sum": 0, "corrects": [], "recalls": [], "count": 0},
                    "6C": {"correct_sum": 0, "corrects": [], "recalls": [], "count": 0},
                })[lab]  # type: ignore[index]
                srcv = m.get("ver_labels", {}).get(lab, {})
                dstv["correct_sum"] += srcv.get("correct_sum", 0)  # type: ignore[index]
                dstv["count"] += srcv.get("count", 0)  # type: ignore[index]
                dstv["corrects"].extend(srcv.get("corrects", []))  # type: ignore[index]
                dstv["recalls"].extend(srcv.get("recalls", []))  # type: ignore[index]
            # 追加: 全15基本ラベル（学習/検証）の再現率をマージ
            dst_train = merged[method].setdefault("train_label_recalls", {lab: [] for lab in LABELS_ALL})
            src_train = m.get("train_label_recalls", {})
            for lab in LABELS_ALL:
                dst_train.setdefault(lab, [])
                dst_train[lab].extend(src_train.get(lab, []))
            dst_ver = merged[method].setdefault("ver_label_recalls", {lab: [] for lab in LABELS_ALL})
            src_ver = m.get("ver_label_recalls", {})
            for lab in LABELS_ALL:
                dst_ver.setdefault(lab, [])
                dst_ver[lab].extend(src_ver.get(lab, []))
    return merged


def print_table(aggregate: Dict[str, Dict[str, Any]], title: str, value_key: str, pair_key: str, sort_mode: str, prec: int):
    header = (
        f'{title:24s} '
        f'{"N":>5s} {"Mean":>10s} {"Std":>10s} {"Min":>10s} {"Median":>10s} {"Max":>10s} '
        f'{"MinSeed":>8s} {"MaxSeed":>8s}'
    )
    print(header)
    print("-" * len(header))
    # 並び順関数
    def key_by_name(item):
        return item[0]

    def key_by_mean_key(item):
        name, metrics = item
        return -mean_or_nan(metrics.get(value_key, []))  # type: ignore[arg-type,index]

    items = list(aggregate.items())
    if sort_mode == "name":
        items.sort(key=key_by_name)
    else:
        items.sort(key=key_by_mean_key)

    for method, metrics in items:
        vals: List[float] = metrics.get(value_key, [])  # type: ignore[assignment]
        pairs: List[Tuple[float, Optional[int]]] = metrics.get(pair_key, [])  # type: ignore[assignment]
        n = len(vals)
        mean_v = mean_or_nan(vals)
        min_v = min_or_nan(vals)
        med_v = median_or_nan(vals)
        max_v = max_or_nan(vals)
        min_seed, max_seed, _mv, _xv = find_extreme_seeds(pairs)
        std_v = std_or_nan(vals)
        print(
            f"{method:24s} {n:5d} "
            f"{fmt_float(mean_v, prec):>10s} {fmt_float(std_v, prec):>10s} {fmt_float(min_v, prec):>10s} "
            f"{fmt_float(med_v, prec):>10s} {fmt_float(max_v, prec):>10s} "
            f"{fmt_seed(min_seed):>8s} {fmt_seed(max_seed):>8s}"
        )
    print("")


def print_nodewise_table(aggregate: Dict[str, Dict[str, Any]], sort_mode: str, prec: int):
    header_nodewise = (
        f'{"[Nodewise] Method":24s} '
        f'{"N":>5s} {"Mean":>10s} {"Std":>10s} {"Min":>10s} {"Median":>10s} {"Max":>10s} '
        f'{"Σmatch":>10s} {"Σnodes":>10s} {"Overall":>10s} '
        f'{"MinSeed":>8s} {"MaxSeed":>8s}'
    )
    print("==== 手法別 NodewiseMatchRate 統計（learning_result/*_results.log の[Final Metrics]より） ====")
    print(header_nodewise)
    print("-" * len(header_nodewise))

    def key_by_name(item):
        return item[0]

    def key_by_mean_nodewise(item):
        name, metrics = item
        return -mean_or_nan(metrics.get("nodewise", []))  # type: ignore[arg-type,index]

    items3 = list(aggregate.items())
    if sort_mode == "name":
        items3.sort(key=key_by_name)
    else:
        items3.sort(key=key_by_mean_nodewise)

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
        std_v = std_or_nan(rates)
        print(
            f"{method:24s} {n:5d} "
            f"{fmt_float(mean_v, prec):>10s} {fmt_float(std_v, prec):>10s} {fmt_float(min_v, prec):>10s} "
            f"{fmt_float(med_v, prec):>10s} {fmt_float(max_v, prec):>10s} "
            f"{sum_match:10d} {sum_total:10d} {fmt_float(overall, prec):>10s} "
            f"{fmt_seed(min_seed):>8s} {fmt_seed(max_seed):>8s}"
        )
    print("")


def print_label_stats_tables(aggregate: Dict[str, Dict[str, Any]], sort_mode: str, prec: int):
    for label_key in LABELS_TARGET:
        title = f"==== ラベル {label_key} の統計（代表ノード群ベース: Correctの平均/最小/最大/中央値/合計、Recallの平均） ===="
        print(title)
        header = (
            f'{"Method":24s} '
            f'{"N":>5s} '
            f'{"Mean_C":>10s} {"Std_C":>10s} {"Min_C":>10s} {"Med_C":>10s} {"Max_C":>10s} {"Sum_C":>10s} '
            f'{"Mean_R":>10s} {"Std_R":>10s}'
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
            std_c = (stats.stdev(corrects) if n > 1 else (0.0 if n == 1 else float("nan")))
            min_c = min(corrects) if n > 0 else float("nan")
            med_c = stats.median(corrects) if n > 0 else float("nan")
            max_c = max(corrects) if n > 0 else float("nan")
            sum_c = info["correct_sum"]  # type: ignore[index]
            mean_r = mean_or_nan(recalls)
            std_r = std_or_nan(recalls)
            rows.append((method, n, mean_c, std_c, min_c, med_c, max_c, sum_c, mean_r, std_r))

        if sort_mode == "name":
            rows.sort(key=lambda x: x[0])
        else:
            # x[8] は Mean_R
            rows.sort(key=lambda x: (- (x[8] if not math.isnan(x[8]) else -1.0), x[0]))

        for method, n, mean_c, std_c, min_c, med_c, max_c, sum_c, mean_r, std_r in rows:
            print(
                f"{method:24s} {n:5d} "
                f"{fmt_float(float(mean_c), prec):>10s} {fmt_float(float(std_c), prec):>10s} {fmt_float(float(min_c), prec):>10s} "
                f"{fmt_float(float(med_c), prec):>10s} {fmt_float(float(max_c), prec):>10s} "
                f"{sum_c:10d} {fmt_float(mean_r, prec):>10s} {fmt_float(std_r, prec):>10s}"
            )
        print("")
    
    
def print_ver_label_stats_tables(aggregate: Dict[str, Dict[str, Any]], sort_mode: str, prec: int):
    for label_key in LABELS_TARGET:
        title = f"==== 検証 ラベル {label_key} の統計（代表ノード群ベース: Correctの平均/最小/最大/中央値/合計、Recallの平均） ===="
        print(title)
        header = (
            f'{"Method":24s} '
            f'{"N":>5s} '
            f'{"Mean_C":>10s} {"Std_C":>10s} {"Min_C":>10s} {"Med_C":>10s} {"Max_C":>10s} {"Sum_C":>10s} '
            f'{"Mean_R":>10s} {"Std_R":>10s}'
        )
        print(header)
        print("-" * len(header))
        rows = []
        for method, metrics in aggregate.items():
            info = metrics.get("ver_labels", {}).get(label_key, {"corrects": [], "recalls": [], "count": 0, "correct_sum": 0})
            corrects: List[int] = info.get("corrects", [])
            recalls: List[float] = info.get("recalls", [])
            n = len(corrects)
            mean_c = stats.mean(corrects) if n > 0 else float("nan")
            std_c = (stats.stdev(corrects) if n > 1 else (0.0 if n == 1 else float("nan")))
            min_c = min(corrects) if n > 0 else float("nan")
            med_c = stats.median(corrects) if n > 0 else float("nan")
            max_c = max(corrects) if n > 0 else float("nan")
            sum_c = info.get("correct_sum", 0)
            mean_r = mean_or_nan(recalls)
            std_r = std_or_nan(recalls)
            rows.append((method, n, mean_c, std_c, min_c, med_c, max_c, sum_c, mean_r, std_r))
        if sort_mode == "name":
            rows.sort(key=lambda x: x[0])
        else:
            # x[8] は Mean_R
            rows.sort(key=lambda x: (- (x[8] if not math.isnan(x[8]) else -1.0), x[0]))
        for method, n, mean_c, std_c, min_c, med_c, max_c, sum_c, mean_r, std_r in rows:
            print(
                f"{method:24s} {n:5d} "
                f"{fmt_float(float(mean_c), prec):>10s} {fmt_float(float(std_c), prec):>10s} {fmt_float(float(min_c), prec):>10s} "
                f"{fmt_float(float(med_c), prec):>10s} {fmt_float(float(max_c), prec):>10s} "
                f"{sum_c:10d} {fmt_float(mean_r, prec):>10s} {fmt_float(std_r, prec):>10s}"
            )
        print("")
    
    
def print_cross_root_ver_basic(per_root_aggregates: Dict[str, Dict[str, Dict[str, Any]]], overall_agg: Dict[str, Dict[str, Any]], prec: int):
    """
    横断ピボット風の表: Method × Root で Verification Basic の平均値を表示し、右端に Overall(mean) と Std を付与
    """
    if not per_root_aggregates:
        return
    roots = list(per_root_aggregates.keys())
    roots.sort()

    # ヘッダ
    header = f'{"[Pivot] VerBasic Mean":24s} '
    for r in roots:
        header += f'{os.path.basename(r):>12s} '
    header += f'{"Overall":>12s} {"Std":>12s} {"N_all":>6s}'
    print("==== 横断ピボット表（Verification Basic の平均） ====")
    print(header)
    print("-" * len(header))

    methods = set()
    for agg in per_root_aggregates.values():
        methods |= set(agg.keys())
    methods = sorted(methods)

    # 並びは overall の mean 降順
    def overall_mean(method: str) -> float:
        return mean_or_nan(overall_agg.get(method, {}).get("ver_basic", []))  # type: ignore[arg-type]

    methods.sort(key=lambda m: - (overall_mean(m) if not math.isnan(overall_mean(m)) else -1.0))

    for method in methods:
        row = f"{method:24s} "
        vb_all = overall_agg.get(method, {}).get("ver_basic", [])  # type: ignore[arg-type]
        overall_mean_v = mean_or_nan(vb_all)
        overall_std_v = std_or_nan(vb_all)
        n_all = len(vb_all)
        for r in roots:
            agg = per_root_aggregates[r]
            vals = agg.get(method, {}).get("ver_basic", [])  # type: ignore[arg-type]
            row += f"{fmt_float(mean_or_nan(vals), prec):>12s} "
        row += f"{fmt_float(overall_mean_v, prec):>12s} {fmt_float(overall_std_v, prec):>12s} {n_all:6d}"
        print(row)
    print("")


def print_per_label_pivot(aggregate: Dict[str, Dict[str, Any]], label_source: str, prec: int):
    """
    行=手法, 列=全15基本ラベル で、各ラベルの平均Recall（seed横断平均）を表示するピボット表。
    label_source: "train" -> *_results.log（学習）由来, "ver" -> *_verification.log（検証）由来
    """
    if not aggregate:
        return
    key = "train_label_recalls" if label_source == "train" else "ver_label_recalls"
    title = "横断ピボット表（学習: 基本ラベルごとの平均再現率）" if label_source == "train" else "横断ピボット表（検証: 基本ラベルごとの平均再現率）"
    print(f"==== {title} ====")
    header = f'{"[Pivot] Method":24s} ' + " ".join(f"{lab:>6s}" for lab in LABELS_ALL) + f' {"Overall":>8s} {"N_lab":>6s}'
    print(header)
    print("-" * len(header))

    methods = sorted(aggregate.keys())

    def overall_mean_method(meth: str) -> float:
        recs_map: Dict[str, List[float]] = aggregate.get(meth, {}).get(key, {})  # type: ignore[assignment]
        vals: List[float] = []
        for lab in LABELS_ALL:
            lv = mean_or_nan(recs_map.get(lab, []))
            if not math.isnan(lv):
                vals.append(lv)
        return mean_or_nan(vals) if vals else float("nan")

    methods.sort(key=lambda m: - (overall_mean_method(m) if not math.isnan(overall_mean_method(m)) else -1.0))

    for meth in methods:
        recs_map: Dict[str, List[float]] = aggregate.get(meth, {}).get(key, {})  # type: ignore[assignment]
        label_means: List[float] = []
        cells: List[str] = []
        n_lab = 0
        for lab in LABELS_ALL:
            mv = mean_or_nan(recs_map.get(lab, []))
            cells.append(fmt_float(mv, prec))
            if not math.isnan(mv):
                label_means.append(mv)
                n_lab += 1
        overall_m = mean_or_nan(label_means) if label_means else float("nan")
        print(f"{meth:24s} " + " ".join(f"{c:>6s}" for c in cells) + f" {fmt_float(overall_m, prec):>8s} {n_lab:6d}")
    print("")
    
    
def print_per_label_pivot_std(aggregate: Dict[str, Dict[str, Any]], label_source: str, prec: int):
    """
    行=手法, 列=全15基本ラベル で、各ラベルの標準偏差（seed横断の std）を表示するピボット表。
    label_source: "train" -> *_results.log（学習）由来, "ver" -> *_verification.log（検証）由来
    """
    if not aggregate:
        return
    key = "train_label_recalls" if label_source == "train" else "ver_label_recalls"
    title = "横断ピボット表（学習: 基本ラベルごとの標準偏差）" if label_source == "train" else "横断ピボット表（検証: 基本ラベルごとの標準偏差）"
    print(f"==== {title} ====")
    header = f'{"[Pivot-Std] Method":24s} ' + " ".join(f"{lab:>6s}" for lab in LABELS_ALL) + f' {"Overall":>8s} {"N_lab":>6s}'
    print(header)
    print("-" * len(header))

    methods = sorted(aggregate.keys())

    def overall_std_method(meth: str) -> float:
        recs_map: Dict[str, List[float]] = aggregate.get(meth, {}).get(key, {})  # type: ignore[assignment]
        vals: List[float] = []
        for lab in LABELS_ALL:
            lv = std_or_nan(recs_map.get(lab, []))
            if not math.isnan(lv):
                vals.append(lv)
        return mean_or_nan(vals) if vals else float("nan")

    # 並びは Overall（std の平均）降順
    methods.sort(key=lambda m: - (overall_std_method(m) if not math.isnan(overall_std_method(m)) else -1.0))

    for meth in methods:
        recs_map: Dict[str, List[float]] = aggregate.get(meth, {}).get(key, {})  # type: ignore[assignment]
        label_stds: List[float] = []
        cells: List[str] = []
        n_lab = 0
        for lab in LABELS_ALL:
            sv = std_or_nan(recs_map.get(lab, []))
            cells.append(fmt_float(sv, prec))
            if not math.isnan(sv):
                label_stds.append(sv)
                n_lab += 1
        overall_s = mean_or_nan(label_stds) if label_stds else float("nan")
        print(f"{meth:24s} " + " ".join(f"{c:>6s}" for c in cells) + f" {fmt_float(overall_s, prec):>8s} {n_lab:6d}")
    print("")
    
    
def print_all_tables_for_aggregate(aggregate: Dict[str, Dict[str, Any]], context_name: str, sort_mode: str, prec: int):
    print(f"==== 手法別 Macro Recall 統計（学習: 基本ラベル, evaluation_v*.log）[{context_name}] ====")
    print_table(aggregate, "[基本] Method", "basic", "basic_pairs", sort_mode, prec)

    if has_any_values(aggregate, "combo"):
        print(f"==== 手法別 Macro Recall 統計（学習: 基本+応用, evaluation_v*.log）[{context_name}] ====")
        print_table(aggregate, "[基本+応用] Method", "combo", "combo_pairs", sort_mode, prec)

    print(f"==== 手法別 Macro Recall 統計（検証: 基本ラベル, *_verification.log）[{context_name}] ====")
    print_table(aggregate, "[Ver基本] Method", "ver_basic", "ver_basic_pairs", sort_mode, prec)

    if has_any_values(aggregate, "ver_combo"):
        print(f"==== 手法別 Macro Recall 統計（検証: 基本+応用, *_verification.log）[{context_name}] ====")
        print_table(aggregate, "[Ver基本+応用] Method", "ver_combo", "ver_combo_pairs", sort_mode, prec)

    print_nodewise_table(aggregate, sort_mode, prec)
    print_label_stats_tables(aggregate, sort_mode, prec)
    print_ver_label_stats_tables(aggregate, sort_mode, prec)
    # 新規ピボット（全15基本ラベル）
    print_per_label_pivot(aggregate, "train", prec)
    print_per_label_pivot(aggregate, "ver", prec)
    # 新規ピボット（標準偏差）
    print_per_label_pivot_std(aggregate, "train", prec)
    print_per_label_pivot_std(aggregate, "ver", prec)


def print_overall_tables(overall_agg: Dict[str, Dict[str, Any]], sort_mode: str, prec: int):
    print("==== 手法別 Macro Recall 統計（学習: 基本ラベル, evaluation_v*.log）[Overall] ====")
    print_table(overall_agg, "[基本] Method", "basic", "basic_pairs", sort_mode, prec)

    if has_any_values(overall_agg, "combo"):
        print("==== 手法別 Macro Recall 統計（学習: 基本+応用, evaluation_v*.log）[Overall] ====")
        print_table(overall_agg, "[基本+応用] Method", "combo", "combo_pairs", sort_mode, prec)

    print("==== 手法別 Macro Recall 統計（検証: 基本ラベル, *_verification.log）[Overall] ====")
    print_table(overall_agg, "[Ver基本] Method", "ver_basic", "ver_basic_pairs", sort_mode, prec)

    if has_any_values(overall_agg, "ver_combo"):
        print("==== 手法別 Macro Recall 統計（検証: 基本+応用, *_verification.log）[Overall] ====")
        print_table(overall_agg, "[Ver基本+応用] Method", "ver_combo", "ver_combo_pairs", sort_mode, prec)

    print_nodewise_table(overall_agg, sort_mode, prec)
    print_label_stats_tables(overall_agg, sort_mode, prec)
    print_ver_label_stats_tables(overall_agg, sort_mode, prec)
    # 新規ピボット（全15基本ラベル）
    print_per_label_pivot(overall_agg, "train", prec)
    print_per_label_pivot(overall_agg, "ver", prec)
    # 新規ピボット（標準偏差）
    print_per_label_pivot_std(overall_agg, "train", prec)
    print_per_label_pivot_std(overall_agg, "ver", prec)


def recommend_methods(overall_agg: Dict[str, Dict[str, Any]], topk: int, prec: int, context_name: str = "Overall", roots_data: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None):
    """
    推奨手法を総合スコアで提示。
    スコア定義（Mean と Std を別々に集計; 単純和）:
      Score(SumMean) = BasicMean + ComboMean + TyphoonMean
        - BasicMean  = mean({VerBasicMean,  TrainBasicMean} 利用可能のみ)
        - ComboMean  = mean({VerComboMean,  TrainComboMean} 利用可能のみ)
        - TyphoonMean= TrainTyphoonRecallMean + VerTyphoonRecallMean
      Score(SumStd)  = BasicStd  + ComboStd  + TyphoonStd
        - BasicStd   = mean({VerBasicStd,   TrainBasicStd} 利用可能のみ)
        - ComboStd   = mean({VerComboStd,   TrainComboStd} 利用可能のみ)
        - TyphoonStd = TrainTyphoonRecallStd + VerTyphoonRecallStd
    注: Nodewise は参考指標でありスコアには含めません。
    """
    rows = []
    for method, metrics in overall_agg.items():
        vb_list: List[float] = metrics.get("ver_basic", [])  # type: ignore[assignment]
        vc_list: List[float] = metrics.get("ver_combo", [])  # type: ignore[assignment]
        tb_list: List[float] = metrics.get("basic", [])  # 学習(基本)
        tc_list: List[float] = metrics.get("combo", [])  # 学習(基本+応用)
        vb_mean = mean_or_nan(vb_list)
        vb_std = std_or_nan(vb_list)
        vc_mean = mean_or_nan(vc_list)
        vc_std = std_or_nan(vc_list)
        tb_mean = mean_or_nan(tb_list)
        tb_std = std_or_nan(tb_list)
        tc_mean = mean_or_nan(tc_list)
        tc_std = std_or_nan(tc_list)
        # Typhoon (train/evaluation) recall per seed across 6A/6B
        ty_list: List[float] = metrics.get("typhoon", [])  # type: ignore[assignment]
        ty_mean = mean_or_nan(ty_list)
        ty_std = std_or_nan(ty_list)
        # Typhoon (verification)
        ty_ver_list: List[float] = metrics.get("typhoon_ver", [])  # type: ignore[assignment]
        ty_ver_mean = mean_or_nan(ty_ver_list)
        ty_ver_std = std_or_nan(ty_ver_list)
        # overall nodewise ratio（集計比は参考算出だがスコアには含めない）
        sum_match = sum(metrics.get("nodewise_matched", []))  # type: ignore[arg-type]
        sum_total = sum(metrics.get("nodewise_total", []))  # type: ignore[arg-type]
        nodewise_overall = (sum_match / sum_total) if sum_total > 0 else float("nan")
        # nodewise mean/std
        nw_mean = mean_or_nan(metrics.get("nodewise", []))  # type: ignore[arg-type]
        nw_std = std_or_nan(metrics.get("nodewise", []))  # type: ignore[arg-type]
        # NaN 保護
        vb_m = 0.0 if math.isnan(vb_mean) else vb_mean
        vb_s = 0.0 if math.isnan(vb_std) else vb_std
        vc_m = 0.0 if math.isnan(vc_mean) else vc_mean
        vc_s = 0.0 if math.isnan(vc_std) else vc_std
        tb_m = 0.0 if math.isnan(tb_mean) else tb_mean
        tb_s = 0.0 if math.isnan(tb_std) else tb_std
        tc_m = 0.0 if math.isnan(tc_mean) else tc_mean
        tc_s = 0.0 if math.isnan(tc_std) else tc_std
        nw_m = 0.0 if math.isnan(nw_mean) else nw_mean
        nw_s = 0.0 if math.isnan(nw_std) else nw_std
        ty_m = 0.0 if math.isnan(ty_mean) else ty_mean
        ty_s = 0.0 if math.isnan(ty_std) else ty_std
        ty_ver_m = 0.0 if math.isnan(ty_ver_mean) else ty_ver_mean
        ty_ver_s = 0.0 if math.isnan(ty_ver_std) else ty_ver_std

        # グループ別スコア（利用可能な項目の平均）
        basic_terms: List[float] = []
        if vb_list:
            basic_terms.append(vb_m)
        if tb_list:
            basic_terms.append(tb_m)
        basic_grp = (sum(basic_terms) / len(basic_terms)) if basic_terms else 0.0

        combo_terms: List[float] = []
        if vc_list:
            combo_terms.append(vc_m)
        if tc_list:
            combo_terms.append(tc_m)
        combo_grp = (sum(combo_terms) / len(combo_terms)) if combo_terms else 0.0

        # Typhoon = TrainTyphoonRecallMean + VerTyphoonRecallMean
        typhoon_grp_mean = ty_m + ty_ver_m
        typhoon_grp_std  = ty_s + ty_ver_s

        # Std グループ（存在するものの平均、Typhoon は和）
        basic_std_terms: List[float] = []
        if vb_list:
            basic_std_terms.append(vb_s)
        if tb_list:
            basic_std_terms.append(tb_s)
        basic_grp_std = (sum(basic_std_terms) / len(basic_std_terms)) if basic_std_terms else 0.0

        combo_std_terms: List[float] = []
        if vc_list:
            combo_std_terms.append(vc_s)
        if tc_list:
            combo_std_terms.append(tc_s)
        combo_grp_std = (sum(combo_std_terms) / len(combo_std_terms)) if combo_std_terms else 0.0

        # 総合スコア（Mean/Std）をそれぞれ単純加算
        score_mean = basic_grp + combo_grp + typhoon_grp_mean
        score_std  = basic_grp_std + combo_grp_std + typhoon_grp_std

        # BestIter/BestSeed の決定（Overall のときは (iter, seed) を横断で評価）
        best_iter: str = context_name
        best_seed: Optional[int] = None
        if context_name == "Overall" and roots_data:
            maps_rs: List[Dict[Tuple[str, int], float]] = []
            for root_path, agg_root in roots_data.items():
                root_label = os.path.basename(root_path)
                vb_pairs = agg_root.get(method, {}).get("ver_basic_pairs", [])  # type: ignore[arg-type]
                vc_pairs = agg_root.get(method, {}).get("ver_combo_pairs", [])  # type: ignore[arg-type]
                tb_pairs = agg_root.get(method, {}).get("basic_pairs", [])      # type: ignore[arg-type]
                tc_pairs = agg_root.get(method, {}).get("combo_pairs", [])      # type: ignore[arg-type]
                nw_pairs = agg_root.get(method, {}).get("nodewise_pairs", [])   # type: ignore[arg-type]
                if vb_pairs: maps_rs.append(pair_means_by_root_seed(vb_pairs, root_label))
                if tb_pairs: maps_rs.append(pair_means_by_root_seed(tb_pairs, root_label))
                if vc_pairs: maps_rs.append(pair_means_by_root_seed(vc_pairs, root_label))
                if tc_pairs: maps_rs.append(pair_means_by_root_seed(tc_pairs, root_label))
                if nw_pairs: maps_rs.append(pair_means_by_root_seed(nw_pairs, root_label))
            # Typhoon per (root,seed)
            ty_pairs = agg_root.get(method, {}).get("typhoon_pairs", [])  # type: ignore[arg-type]
            if ty_pairs:
                maps_rs.append(pair_means_by_root_seed(ty_pairs, root_label))
            brs = select_best_root_seed_balanced(maps_rs)
            if brs is not None:
                best_iter, best_seed = brs[0], brs[1]
        if best_seed is None:
            # フォールバック: 集計単位内（iter別またはOverall統合）で seed のみでバランス評価（Basic/Combo/Nodewise/Typhoon）
            vb_map = seed_means_from_pairs(metrics.get("ver_basic_pairs", []))  # type: ignore[arg-type]
            tb_map = seed_means_from_pairs(metrics.get("basic_pairs", []))      # type: ignore[arg-type]
            vc_map = seed_means_from_pairs(metrics.get("ver_combo_pairs", []))  # type: ignore[arg-type]
            tc_map = seed_means_from_pairs(metrics.get("combo_pairs", []))      # type: ignore[arg-type]
            nw_map = seed_means_from_pairs(metrics.get("nodewise_pairs", []))   # type: ignore[arg-type]
            ty_map = seed_means_from_pairs(metrics.get("typhoon_pairs", []))    # type: ignore[arg-type]
            best_seed = select_best_seed_balanced([vb_map, tb_map, vc_map, tc_map, nw_map, ty_map])
            if best_seed is None:
                best_iter = context_name
        rows.append((method, score_mean, score_std, vb_mean, vb_std, vc_mean, vc_std, tb_mean, tb_std, tc_mean, tc_std, nw_mean, nw_std, ty_mean, ty_std, ty_ver_mean, ty_ver_std, best_iter, best_seed, len(vb_list)))

    # スコア降順で上位
    rows.sort(key=lambda x: (-x[1], x[0]))

    print(f"==== 総合推奨手法（暫定スコアに基づく）[{context_name}] ====")
    header = (
        f'{"Method":24s} {"Score(SumMean)":>16s} {"Score(SumStd)":>15s} '
        f'{"VerBasicMean":>13s} {"VerBasicStd":>12s} {"VerComboMean":>13s} {"VerComboStd":>12s} '
        f'{"TrainBasicMean":>15s} {"TrainBasicStd":>14s} {"TrainComboMean":>15s} {"TrainComboStd":>14s} '
        f'{"NodewiseOverallMean":>20s} {"NodewiseOverallStd":>19s} '
        f'{"TyphoonTrainMean":>17s} {"TyphoonTrainStd":>16s} {"TyphoonVerMean":>15s} {"TyphoonVerStd":>14s} '
        f'{"BestIter":>12s} {"BestSeed":>8s} {"N(VerB)":>8s}'
    )
    print(header)
    print("-" * len(header))
    k = len(rows) if (topk is None or topk <= 0 or topk > len(rows)) else topk
    for i, (method, score_mean, score_std, vb_mean, vb_std, vc_mean, vc_std, tb_mean, tb_std, tc_mean, tc_std, nw_mean, nw_std, ty_mean, ty_std, ty_ver_mean, ty_ver_std, best_iter, best_seed, n_vb) in enumerate(rows[:k]):
        print(
            f"{method:24s} {fmt_float(score_mean, prec):>16s} {fmt_float(score_std, prec):>15s} "
            f"{fmt_float(vb_mean, prec):>13s} {fmt_float(vb_std, prec):>12s} {fmt_float(vc_mean, prec):>13s} {fmt_float(vc_std, prec):>12s} "
            f"{fmt_float(tb_mean, prec):>15s} {fmt_float(tb_std, prec):>14s} {fmt_float(tc_mean, prec):>15s} {fmt_float(tc_std, prec):>14s} "
            f"{fmt_float(nw_mean, prec):>20s} {fmt_float(nw_std, prec):>19s} "
            f"{fmt_float(ty_mean, prec):>17s} {fmt_float(ty_std, prec):>16s} {fmt_float(ty_ver_mean, prec):>15s} {fmt_float(ty_ver_std, prec):>14s} "
            f"{best_iter:>12s} {fmt_seed(best_seed):>8s} {n_vb:8d}"
        )
    print("")
    print("注: スコアは Mean と Std を別集計した2列を表示します（いずれも単純和）。")
    print("  Score(SumMean) = BasicMean + ComboMean + (TrainTyphoonRecallMean + VerTyphoonRecallMean)")
    print("    - BasicMean = mean({VerBasicMean, TrainBasicMean} 利用可能のみ)")
    print("    - ComboMean = mean({VerComboMean, TrainComboMean} 利用可能のみ)")
    print("  Score(SumStd)  = BasicStd  + ComboStd  + (TrainTyphoonRecallStd + VerTyphoonRecallStd)")
    print("    - BasicStd  = mean({VerBasicStd,  TrainBasicStd} 利用可能のみ)")
    print("    - ComboStd  = mean({VerComboStd,  TrainComboStd} 利用可能のみ)")
    print("  Nodewise系は参考指標でスコアには不使用です。")
    print("")


def maybe_write_csv(csv_path: Optional[str], per_root_aggregates: Dict[str, Dict[str, Dict[str, Any]]], overall_agg: Dict[str, Dict[str, Any]], prec: int):
    """
    横断ピボット（Verification Basic の平均）の CSV を出力
    カラム: method, <root1>, <root2>, ..., overall_mean, overall_std, n_all
    """
    if not csv_path:
        return
    roots = list(per_root_aggregates.keys())
    roots.sort()
    methods = set()
    for agg in per_root_aggregates.values():
        methods |= set(agg.keys())
    methods = sorted(methods, key=lambda m: -mean_or_nan(overall_agg.get(m, {}).get("ver_basic", [])))  # type: ignore[arg-type]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["method"] + [os.path.basename(r) for r in roots] + ["overall_mean", "overall_std", "n_all"]
        writer.writerow(header)
        for method in methods:
            row = [method]
            vb_all = overall_agg.get(method, {}).get("ver_basic", [])  # type: ignore[arg-type]
            overall_mean_v = mean_or_nan(vb_all)
            overall_std_v = std_or_nan(vb_all)
            n_all = len(vb_all)
            for r in roots:
                agg = per_root_aggregates[r]
                vals = agg.get(method, {}).get("ver_basic", [])  # type: ignore[arg-type]
                row.append(fmt_float(mean_or_nan(vals), prec))
            row += [fmt_float(overall_mean_v, prec), fmt_float(overall_std_v, prec), n_all]
            writer.writerow(row)


def report_nc_labels(nc_path: str, start_date: str, end_date: str, basic_labels: Tuple[str, ...] = LABELS_ALL, prec: int = 3) -> None:
    """
    指定 NetCDF から期間 [start_date, end_date] を xarray で直接 .sel して抽出し、
    変数 'label' の基本ラベル出現分布を集計して表示する。
    main_v7.py の load_and_prepare_data_unified と同等の座標検出・期間フィルタ方式に合わせる。
    """
    print("==== NetCDF ラベル分布レポート ====")
    print(f"ファイル: {nc_path}")
    print(f"期間: {start_date} 〜 {end_date}")
    if not os.path.isfile(nc_path):
        print("[WARN] NetCDF ファイルが見つかりません。スキップします。")
        print("")
        return

    # ラベル正規化: main_v7.py の basic_label_or_none に準拠（簡約版を内蔵）
    def _normalize_to_base_candidate(label_str: Optional[str]) -> Optional[str]:
        import unicodedata, re as _re
        if label_str is None:
            return None
        s = str(label_str)
        s = unicodedata.normalize('NFKC', s)
        s = s.upper().strip()
        s = s.replace('＋', '+').replace('－', '-').replace('−', '-')
        # 英数字以外は除去
        s = _re.sub(r'[^0-9A-Z\+\-]', '', s)
        return s if s != '' else None

    def basic_label_or_none(label_str: Optional[str], base_labels: Tuple[str, ...]) -> Optional[str]:
        import re as _re
        cand = _normalize_to_base_candidate(label_str)
        if cand is None:
            return None
        # 完全一致を優先
        if cand in base_labels:
            return cand
        # 先頭一致 + 残りに英数字が無い（例: '2A+' → '2A'）
        for bl in base_labels:
            if cand == bl:
                return bl
            if cand.startswith(bl):
                rest = cand[len(bl):]
                if _re.search(r'[0-9A-Z]', rest) is None:
                    return bl
        return None

    try:
        ds = None
        # xarray で decode_times=True として開く（エンジンを順に試す）
        if xr is not None:
            open_errs = []
            for engine in (None, "h5netcdf", "scipy"):
                try:
                    ds = xr.open_dataset(nc_path, decode_times=True, engine=engine if engine else None)
                    break
                except Exception as e:
                    open_errs.append(str(e))
                    ds = None
            if ds is None:
                raise RuntimeError("xarray での読み込みに失敗: " + " | ".join(open_errs))

        # 時間座標名を判定
        if "valid_time" in ds:
            time_coord = "valid_time"
        elif "time" in ds:
            time_coord = "time"
        else:
            raise ValueError('No time coordinate named "valid_time" or "time".')

        # 期間でスライス（デコード済みのため文字列 slice でOK）
        sub = ds.sel({time_coord: slice(start_date, end_date)})

        # 該当数
        total = int(sub[time_coord].size)
        if total == 0:
            # xarray の decode_times=True でヒットしない場合に備え、numeric time でフォールバック
            try:
                open_errs_fb = []
                ds_fb = None
                for engine in (None, "h5netcdf", "scipy"):
                    try:
                        ds_fb = xr.open_dataset(nc_path, decode_times=False, engine=engine if engine else None)
                        break
                    except Exception as e:
                        open_errs_fb.append(str(e))
                        ds_fb = None
                if ds_fb is None:
                    print("[WARN] 指定期間に該当するデータがありません。フォールバック読み込みにも失敗しました。")
                    print("")
                    try:
                        ds.close()
                    except Exception:
                        pass
                    return

                # 時間座標を再判定
                if "valid_time" in ds_fb:
                    time_coord_fb = "valid_time"
                elif "time" in ds_fb:
                    time_coord_fb = "time"
                else:
                    print('[WARN] 指定期間に該当するデータがありません（時間座標が見つからず）。')
                    print("")
                    try:
                        ds_fb.close()
                    except Exception:
                        pass
                    try:
                        ds.close()
                    except Exception:
                        pass
                    return

                vt_vals = ds_fb[time_coord_fb].values
                if vt_vals.dtype.kind not in ("i", "u", "f"):
                    print("[WARN] 指定期間に該当するデータがありません（時間が数値でないためフォールバック不可）。")
                    print("")
                    try:
                        ds_fb.close()
                    except Exception:
                        pass
                    try:
                        ds.close()
                    except Exception:
                        pass
                    return

                # UTC で秒へ変換して範囲抽出
                s_ts = pd.Timestamp(start_date).tz_localize("UTC")
                e_ts = pd.Timestamp(end_date).tz_localize("UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                start_sec = int(s_ts.timestamp())
                end_sec = int(e_ts.timestamp())
                idx = np.where((vt_vals >= start_sec) & (vt_vals <= end_sec))[0]
                total_fb = int(idx.size)
                if total_fb == 0:
                    print("[WARN] 指定期間に該当するデータがありません。")
                    print("")
                    try:
                        ds_fb.close()
                    except Exception:
                        pass
                    try:
                        ds.close()
                    except Exception:
                        pass
                    return

                if "label" not in ds_fb.variables:
                    print("[WARN] 'label' 変数が見つからないため、分布集計をスキップします。")
                    print(f"対象データ数: {total_fb}")
                    print("")
                    try:
                        ds_fb.close()
                    except Exception:
                        pass
                    try:
                        ds.close()
                    except Exception:
                        pass
                    return

                labels_da = ds_fb["label"].isel({time_coord_fb: idx})
                raw_labels = labels_da.values.reshape(-1)

                # デコード・正規化して集計
                label_list: List[Optional[str]] = []
                for v in raw_labels:
                    try:
                        if isinstance(v, (bytes, bytearray)):
                            s = v.decode("utf-8", errors="ignore").strip()
                        else:
                            s = str(v).strip()
                    except Exception:
                        s = ""
                    label_list.append(s if s != "" else None)

                # 集計（基本ラベルに正規化してカウント）＋ 複合ラベル解析（フォールバック経路）
                counts: Dict[str, int] = {lab: 0 for lab in basic_labels}
                other = 0
                for lab in label_list:
                    bl = basic_label_or_none(lab, basic_labels)
                    if bl is not None:
                        counts[bl] += 1
                    else:
                        other += 1

                comp_counts = Counter()
                for lab in label_list:
                    s = _normalize_to_base_candidate(lab)
                    if not s:
                        continue
                    ops = [ch for ch in s if ch in ['+', '-']]
                    if len(ops) != 1:
                        continue
                    op = ops[0]
                    parts = s.split(op)
                    if len(parts) != 2:
                        continue
                    left, right = parts[0], parts[1]
                    if (left in basic_labels) and (right in basic_labels):
                        key = f"{left}{op}{right}"
                        comp_counts[key] += 1

                print(f"対象データ数: {total_fb}")
                header = f'{"Label":>6s} {"Count":>8s} {"Percent":>10s}'
                print(header)
                print("-" * len(header))
                base_sum_fb = 0
                for lab in basic_labels:
                    c = counts[lab]
                    base_sum_fb += c
                    pct = (c / total_fb * 100.0) if total_fb > 0 else float("nan")
                    print(f"{lab:>6s} {c:8d} {fmt_float(pct, prec):>10s}")
                base_sum_pct_fb = (base_sum_fb / total_fb * 100.0) if total_fb > 0 else float("nan")
                print("-" * len(header))
                print(f"{'TOTAL':>6s} {base_sum_fb:8d} {fmt_float(base_sum_pct_fb, prec):>10s}")
                if other > 0:
                    pct = (other / total_fb * 100.0)
                    print(f"{'OTHER':>6s} {other:8d} {fmt_float(pct, prec):>10s}")
                print("")

                comp_total_fb = sum(comp_counts.values())
                comp_kinds_fb = len(comp_counts)
                kinds_plus_fb = sum(1 for k in comp_counts if '+' in k)
                kinds_minus_fb = sum(1 for k in comp_counts if '-' in k)
                comp_pct_fb = (comp_total_fb / total_fb * 100.0) if total_fb > 0 else float("nan")

                print("【複合ラベル（Base±Base）統計】")
                print(f"- 出現総数: {comp_total_fb} / {total_fb} ({fmt_float(comp_pct_fb, prec)}%)")
                print(f"- 異なる複合ラベルの種類数: {comp_kinds_fb}  (+' 種類: {kinds_plus_fb}, -' 種類: {kinds_minus_fb})")
                if comp_kinds_fb > 0:
                    print("- 出現上位（最大 30 件）:")
                    for key, cnt in comp_counts.most_common(30):
                        pctk = (cnt / total_fb * 100.0) if total_fb > 0 else float('nan')
                        print(f"  {key:>8s} : {cnt:6d} ({fmt_float(pctk, prec)}%)")
                print("")
                try:
                    ds_fb.close()
                except Exception:
                    pass
                try:
                    ds.close()
                except Exception:
                    pass
                return
            except Exception as _e_fb:
                print(f"[WARN] フォールバック集計中に例外: {_e_fb}")
                print("")
                try:
                    ds.close()
                except Exception:
                    pass
                return

        # ラベル配列
        if "label" not in sub.variables:
            print("[WARN] 'label' 変数が見つからないため、分布集計をスキップします。")
            print(f"対象データ数: {total}")
            print("")
            try:
                ds.close()
            except Exception:
                pass
            return

        raw_labels = sub["label"].values  # 期待形状: (time,)
        raw_labels = raw_labels.reshape(-1)

        # デコード・正規化
        label_list: List[Optional[str]] = []
        for v in raw_labels:
            try:
                if isinstance(v, (bytes, bytearray)):
                    s = v.decode("utf-8", errors="ignore").strip()
                else:
                    s = str(v).strip()
            except Exception:
                s = ""
            label_list.append(s if s != "" else None)

        # 集計（基本ラベルに正規化してカウント）＋ 複合ラベル（Base±Base）解析
        counts: Dict[str, int] = {lab: 0 for lab in basic_labels}
        other = 0
        for lab in label_list:
            bl = basic_label_or_none(lab, basic_labels)
            if bl is not None:
                counts[bl] += 1
            else:
                other += 1

        # 複合ラベル（基本ラベル2つが + または - で接続）の出現集計
        comp_counts = Counter()
        for lab in label_list:
            s = _normalize_to_base_candidate(lab)
            if not s:
                continue
            ops = [ch for ch in s if ch in ['+', '-']]
            if len(ops) != 1:
                continue
            op = ops[0]
            parts = s.split(op)
            if len(parts) != 2:
                continue
            left, right = parts[0], parts[1]
            if (left in basic_labels) and (right in basic_labels):
                key = f"{left}{op}{right}"
                comp_counts[key] += 1

        # 出力（基本15ラベル分布＋合計）
        print(f"対象データ数: {total}")
        header = f'{"Label":>6s} {"Count":>8s} {"Percent":>10s}'
        print(header)
        print("-" * len(header))
        base_sum = 0
        for lab in basic_labels:
            c = counts[lab]
            base_sum += c
            pct = (c / total * 100.0) if total > 0 else float("nan")
            print(f"{lab:>6s} {c:8d} {fmt_float(pct, prec):>10s}")
        # 15ラベル合計
        base_sum_pct = (base_sum / total * 100.0) if total > 0 else float("nan")
        print("-" * len(header))
        print(f"{'TOTAL':>6s} {base_sum:8d} {fmt_float(base_sum_pct, prec):>10s}")
        if other > 0:
            other_pct = (other / total * 100.0)
            print(f"{'OTHER':>6s} {other:8d} {fmt_float(other_pct, prec):>10s}")
        print("")

        # 複合ラベルの詳細
        comp_total = sum(comp_counts.values())
        comp_kinds = len(comp_counts)
        kinds_plus = sum(1 for k in comp_counts if '+' in k)
        kinds_minus = sum(1 for k in comp_counts if '-' in k)
        comp_pct = (comp_total / total * 100.0) if total > 0 else float("nan")

        print("【複合ラベル（Base±Base）統計】")
        print(f"- 出現総数: {comp_total} / {total} ({fmt_float(comp_pct, prec)}%)")
        print(f"- 異なる複合ラベルの種類数: {comp_kinds}  (+' 種類: {kinds_plus}, -' 種類: {kinds_minus})")
        if comp_kinds > 0:
            print("- 出現上位（最大 30 件）:")
            for key, cnt in comp_counts.most_common(30):
                pctk = (cnt / total * 100.0) if total > 0 else float('nan')
                print(f"  {key:>8s} : {cnt:6d} ({fmt_float(pctk, prec)}%)")
        print("")
    except Exception as e:
        print(f"[ERROR] NetCDF レポート中に例外: {e}")
        print("")
    finally:
        try:
            if ds is not None:
                ds.close()
        except Exception:
            pass

def main():
    parser = argparse.ArgumentParser(description="複数 results_* ディレクトリのログから手法別の各種統計（学習/検証）を算出（横断・総合評価対応）")
    # 既定では src/PressurePattern/ 配下の results_v6_iter100/1000/10000 を自動探索（存在するものだけ）
    default_dir = os.path.dirname(os.path.abspath(__file__))
    default_root_single = os.path.join(default_dir, "results_v7_iter1000_128")

    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help=f"単一の results ディレクトリ（後方互換用途、--roots が指定されていれば無視）"
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
        default=2,
        help="小数点以下の表示桁数 (default: 2)",
    )
    parser.add_argument(
        "--recommend",
        action="store_true",
        help="総合スコアに基づく推奨手法 Top-K を表示"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=0,
        help="--recommend 使用時の推奨件数。0 以下で全件表示 (default: 0=all)"
    )
    args = parser.parse_args()
    # 常に固定ファイルへ tee 出力を有効化（どこで実行しても同じパスに保存）
    setup_global_tee_logging("/Users/takumi0616/Develop/docker_miniconda/src/PressurePattern/search_results_v7.log")


    # root 決定（--root 優先、無ければ既定ディレクトリ）
    if args.root:
        root = args.root
    else:
        root = default_root_single

    # 実在チェック
    if not os.path.isdir(root):
        print(f"[ERROR] 指定のディレクトリが存在しません: {root}")
        return

    # 収集・集計（単一ルート）
    eval_paths, ver_paths, aggregate = collect_logs(root)

    print("==== results 集計（単一ルート） ====")
    print(f"探索対象（root）: {root} (evaluation={len(eval_paths)} verification={len(ver_paths)})")
    print("")

    prec = args.precision
    sort_mode = args.sort

    ctx = os.path.basename(root)
    print_all_tables_for_aggregate(aggregate, ctx, sort_mode, prec)
    if args.recommend:
        recommend_methods(aggregate, args.topk, prec, context_name=ctx)

    # NetCDF ラベル分布レポートを自動で最後に追記
    print("==== ERA5 NetCDF ラベル分布レポート（自動） ====")
    nc_auto_path = os.path.join(default_dir, "prmsl_era5_all_data_seasonal_large.nc")
    report_nc_labels(nc_auto_path, "1991-01-01", "2000-12-31")

    print("注記:")
    print(" - Macro Recall は [Summary] の値から算出（Mean/Std/Min/Median/Max）。")
    print(" - MinSeed / MaxSeed に最小/最大値が出た seed を表示（同値が複数ある場合は最初のもの）。")
    print(" - 6A/6B/6C の Correct/Recall は「各ラベルの再現率（代表ノード群ベース）」の値を使用。")
    print(" - 本スクリプトは単一の results ディレクトリのみ評価し、複数 iter の横断集計は削除済み。")
    print(" - 実行完了時に ERA5 NetCDF（1991-01-01〜2000-12-31）のラベル分布レポートを末尾に自動追記します。")
    print(" - すべての出力は /Users/takumi0616/Develop/docker_miniconda/src/PressurePattern/search_results_v7.log にも Tee 保存されます。")


if __name__ == "__main__":
    main()
