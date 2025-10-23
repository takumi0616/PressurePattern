# -*- coding: utf-8 -*-
"""
複数バックボーン（simple / resnet_small / deformable / convnext_tiny）を順次実行するランナー

- 環境変数 MODEL_BACKBONE を切り替えつつ main_v1.py をサブプロセスとして実行
- 各バックボーンの出力（result/<backbone>/...）から mAP / loss / macro F1 を要約して表示
- コマンドは本スクリプトをユーザが手動実行する前提（ここでは実行しない）

使い方（例・実行はユーザが行ってください）:
  python src/PressurePattern/Classification/run_backbones.py \
      --backbones resnet_small,deformable,convnext_tiny

引数:
  --backbones: 実行するバックボーンのカンマ区切り（デフォルト: 全て）
  --extra-args: main_v1.py へそのまま渡す追加引数（現状 main_v1.py は引数なしを想定、将来拡張用）
  --dry-run: 実行せず計画のみ表示
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, Any, List

# プロジェクト内の main_v1.py 位置を解決
HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_PATH = os.path.join(HERE, "main_v1.py")

# main_v1_config.py と整合する定数
DEFAULT_OUTPUT_DIR = "result"
DEFAULT_MODEL_NAME = "cnn_v1_torch"
VALID_BACKBONES = ("simple", "resnet_small", "deformable", "convnext_tiny")


def summarize(backbone: str) -> Dict[str, Any]:
    """
    result/<backbone>/ 以下の成果物を読み、要約指標を返す
    """
    out_dir = os.path.join(HERE, DEFAULT_OUTPUT_DIR, backbone)
    model_tag = f"{DEFAULT_MODEL_NAME}_{backbone}"
    norm_path = os.path.join(out_dir, f"{model_tag}_norm.json")
    rep_path = os.path.join(out_dir, f"{model_tag}_val_report.json")

    summary = {"backbone": backbone, "out_dir": out_dir}
    try:
        if os.path.exists(norm_path):
            with open(norm_path, "r", encoding="utf-8") as f:
                norm = json.load(f)
            summary["best_val_map"] = norm.get("best_val_map")
            summary["best_val_loss"] = norm.get("best_val_loss")
            summary["best_epoch"] = norm.get("best_epoch")
        else:
            summary["best_val_map"] = None
            summary["best_val_loss"] = None
            summary["best_epoch"] = None
    except Exception as e:
        summary["best_val_map"] = None
        summary["best_val_loss"] = None
        summary["best_epoch"] = None
        summary["norm_error"] = str(e)

    try:
        if os.path.exists(rep_path):
            with open(rep_path, "r", encoding="utf-8") as f:
                rep = json.load(f)
            macro = rep.get("macro avg", {})
            weighted = rep.get("weighted avg", {})
            summary["macro_f1"] = macro.get("f1-score")
            summary["weighted_f1"] = weighted.get("f1-score")
        else:
            summary["macro_f1"] = None
            summary["weighted_f1"] = None
    except Exception as e:
        summary["macro_f1"] = None
        summary["weighted_f1"] = None
        summary["report_error"] = str(e)

    return summary


def print_summary_table(summaries: List[Dict[str, Any]]) -> None:
    # 簡易テーブルを整形して標準出力へ
    header = ["backbone", "best_epoch", "best_val_loss", "best_val_map", "macro_f1", "weighted_f1", "out_dir"]
    print("\n=== Summary ===")
    print("\t".join(header))
    for s in summaries:
        row = [
            str(s.get("backbone")),
            str(s.get("best_epoch")),
            f"{s.get('best_val_loss'):.6f}" if isinstance(s.get("best_val_loss"), (int, float)) else "NA",
            f"{s.get('best_val_map'):.6f}" if isinstance(s.get("best_val_map"), (int, float)) else "NA",
            f"{s.get('macro_f1'):.6f}" if isinstance(s.get("macro_f1"), (int, float)) else "NA",
            f"{s.get('weighted_f1'):.6f}" if isinstance(s.get("weighted_f1"), (int, float)) else "NA",
            str(s.get("out_dir")),
        ]
        print("\t".join(row))
    print("==============\n")


def run_one(backbone: str, extra_args: List[str], dry_run: bool = False) -> int:
    """
    指定バックボーンで main_v1.py を 1 回実行
    - 戻り値: サブプロセスの returncode（dry-run 時は 0）
    """
    if backbone not in VALID_BACKBONES:
        print(f"[runner] 無効なバックボーン指定: {backbone!r} -> スキップ")
        return 0

    env = os.environ.copy()
    env["MODEL_BACKBONE"] = backbone

    cmd = [sys.executable, MAIN_PATH]
    if extra_args:
        cmd.extend(extra_args)

    print(f"[runner] EXECUTE backbone={backbone} cmd={' '.join(cmd)}")
    if dry_run:
        return 0

    # サブプロセスで main_v1.py を実行（このスクリプト自体はコマンドを実行しません。ユーザが明示的に起動してください）
    proc = subprocess.run(cmd, env=env, cwd=HERE)
    return int(proc.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiple CNN backbones sequentially.")
    parser.add_argument(
        "--backbones",
        type=str,
        default=",".join(VALID_BACKBONES),
        help=f"バックボーン名のカンマ区切り（候補: {','.join(VALID_BACKBONES)}）",
    )
    parser.add_argument(
        "--extra-args",
        type=str,
        default="",
        help="main_v1.py に渡す追加引数（スペースは使えません。空白を含む場合は適宜シェル側で対応）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="実行せず、計画とコマンドのみ表示",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backbones = [b.strip().lower() for b in args.backbones.split(",") if b.strip()]
    extra = [a for a in args.extra_args.split(",") if a] if args.extra_args else []
    dry = bool(args.dry_run)

    print("[runner] 対象バックボーン:", backbones)
    print("[runner] 追加引数:", extra)
    print("[runner] dry-run:", dry)
    print(f"[runner] main_v1.py: {MAIN_PATH}")

    rcodes: List[int] = []
    for b in backbones:
        rc = run_one(b, extra_args=extra, dry_run=dry)
        rcodes.append(rc)

    # 実行後の要約（dry-run でも既存の成果物があれば読み込む）
    summaries = [summarize(b) for b in backbones]
    print_summary_table(summaries)

    # 何か失敗があれば終了コード非0にしたい場合
    if any(rc != 0 for rc in rcodes):
        sys.exit(1)


if __name__ == "__main__":
    main()
