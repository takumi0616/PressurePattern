# -*- coding: utf-8 -*-
"""
main_v1 (PyTorch, Simple): ERA5 由来の気圧配置データを用いた「シンプルな」CNN 学習スクリプト

方針（Keras の CIFAR10/weather ノートブックのシンプルさを踏襲）
- モデル: Conv -> ReLU -> MaxPool -> Dropout を3回 + Flatten -> Dense(512) -> Dropout -> 出力
  ※ 実装は src/PressurePattern/Classification/CNN.py の SimpleCNN と一致
- 最適化: Adam（固定学習率）、BCEWithLogitsLoss（マルチラベル）
- 余計な最適化: EMA, AMP, LRスケジューラ, torch.compile, 勾配クリップ, チェックポイント逐次保存 などは未使用
- 評価: 各エポックで val_loss と mAP（macro average precision）のみ

入出力:
- NetCDF の読み込み/前処理/正規化は従来どおり
- ./result 配下へ最終/ベスト重み, 学習履歴, 正規化統計, classification_report を保存
"""

from __future__ import annotations

import os
import re
import json
import random
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr

# LZ4圧縮されたNetCDFの読み込みに必要（h5netcdf経由）
try:
    import hdf5plugin  # noqa: F401
except Exception:
    hdf5plugin = None

warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import average_precision_score, classification_report

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import time
import logging

logger = logging.getLogger("main_v1_simple")
_START_TIME = time.perf_counter()
def _elapsed() -> float:
    return time.perf_counter() - _START_TIME
def logi(msg: str) -> None:
    logger.info(f"[+{_elapsed():.2f}s] {msg}")

# 相対/直下両対応のインポート
try:
    from .CNN import build_cnn, compute_pos_weights
    from .main_v1_config import (
        BASE_DIR,
        DATA_PATH,
        SELECTED_VARIABLES,
        USE_SEASONAL_AS_CHANNELS,
        SLP_TO_HPA,
        RANDOM_SEED,
        TRAIN_YEARS,
        VAL_YEARS,
        EPOCHS,
        BATCH_SIZE,
        LEARNING_RATE,
        WEIGHT_DECAY,
        PREDICTION_THRESHOLD,
        USE_POSITIVE_CLASS_WEIGHTS,
        OUTPUT_DIR,
        MODEL_NAME,
        FINAL_MODEL_PATH,
        BEST_WEIGHTS_PATH,
        HISTORY_JSON_PATH,
        NORM_STATS_JSON_PATH,
        VAL_REPORT_JSON_PATH,
        BASE_LABELS,
        VAR_CANDIDATES,
    )
except Exception:
    import os as _os, sys as _sys
    _HERE = _os.path.dirname(_os.path.abspath(__file__))
    if _HERE not in _sys.path:
        _sys.path.insert(0, _HERE)
    from CNN import build_cnn, compute_pos_weights
    from main_v1_config import (
        BASE_DIR,
        DATA_PATH,
        SELECTED_VARIABLES,
        USE_SEASONAL_AS_CHANNELS,
        SLP_TO_HPA,
        RANDOM_SEED,
        TRAIN_YEARS,
        VAL_YEARS,
        EPOCHS,
        BATCH_SIZE,
        LEARNING_RATE,
        WEIGHT_DECAY,
        PREDICTION_THRESHOLD,
        USE_POSITIVE_CLASS_WEIGHTS,
        OUTPUT_DIR,
        MODEL_NAME,
        FINAL_MODEL_PATH,
        BEST_WEIGHTS_PATH,
        HISTORY_JSON_PATH,
        NORM_STATS_JSON_PATH,
        VAL_REPORT_JSON_PATH,
        BASE_LABELS,
        VAR_CANDIDATES,
    )

_LABEL_TOKEN_PATTERN = re.compile(r"[+\-]")  # '+' または '-' で分割


def set_global_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _ensure_output_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(FINAL_MODEL_PATH), exist_ok=True)


def _to_datetime64(ds: xr.Dataset) -> pd.DatetimeIndex:
    t = pd.to_datetime(ds["valid_time"].values, unit="s", utc=True).tz_convert("Asia/Tokyo")
    return pd.DatetimeIndex(t)


def _subset_years_indices(times: pd.DatetimeIndex, years: List[int]) -> np.ndarray:
    mask = np.isin(times.year, years)
    return np.where(mask)[0]


def _msl_preprocess(da: xr.DataArray) -> xr.DataArray:
    out = da / 100.0  # Pa → hPa
    out.attrs["units"] = "hPa"
    return out


def _resolve_selected_vars(ds: xr.Dataset, selected: List[str], candidates: List[str]) -> List[str]:
    if any(s.upper() == "ALL" for s in selected):
        usable = [v for v in candidates if v in ds.variables]
        if not usable:
            raise RuntimeError("候補変数が NetCDF に見つかりません。")
        return usable
    else:
        missing = [v for v in selected if v not in ds.variables]
        if missing:
            raise KeyError(f"指定変数が NetCDF に存在しません: {missing}")
        return selected


def _extract_channels(
    ds: xr.Dataset,
    selected_vars: List[str],
    use_season: bool,
) -> Tuple[np.ndarray, List[str]]:
    arrays = []
    channel_names = []

    for var in selected_vars:
        da = ds[var]
        if set(["valid_time", "latitude", "longitude"]).issubset(set(da.dims)):
            da = da.transpose("valid_time", "latitude", "longitude")
        else:
            raise ValueError(f"変数 {var} の次元が想定外です: {da.dims}")

        if var == "msl" and SLP_TO_HPA:
            da = _msl_preprocess(da)

        arrays.append(da.values.astype(np.float32))
        channel_names.append(var)

    if use_season:
        for sname in ["f1_season", "f2_season"]:
            if sname not in ds:
                continue
            v = ds[sname].values.astype(np.float32)  # (T,)
            lat_len = ds.sizes["latitude"]
            lon_len = ds.sizes["longitude"]
            v3 = np.repeat(v[:, None, None], lat_len, axis=1)
            v3 = np.repeat(v3, lon_len, axis=2)
            arrays.append(v3)
            channel_names.append(sname)

    x = np.stack(arrays, axis=-1)  # (T, H, W, C)
    return x, channel_names


def _label_to_multihot(label_str: str, base_labels: List[str]) -> np.ndarray:
    """
    '3B+4B', '2A-1', '1' などをマルチホット(15次元)に変換。
    '+' も '-' も和集合として 1 を立てる。空/不明 は全 0。
    """
    vec = np.zeros(len(base_labels), dtype=np.float32)
    if not isinstance(label_str, str):
        return vec
    s = label_str.strip()
    if s in ("", "N/A", "NA", "None"):
        return vec

    tokens = [t.strip() for t in _LABEL_TOKEN_PATTERN.split(s) if t.strip()]
    for t in tokens:
        if t in base_labels:
            idx = base_labels.index(t)
            vec[idx] = 1.0
    return vec


def _build_dataset(
    ds: xr.Dataset,
    base_labels: List[str],
    selected_vars: List[str],
    use_season: bool,
    train_years: List[int],
    val_years: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int], List[str]]:
    times = _to_datetime64(ds)

    x_all, channel_names = _extract_channels(ds, selected_vars, use_season)  # (T,H,W,C)
    T, H, W, C = x_all.shape

    labels_raw = ds["label"].values
    label_arr = [_label_to_multihot(str(labels_raw[i]), base_labels) for i in range(T)]
    y_all = np.stack(label_arr, axis=0).astype(np.float32)  # (T, C_labels)

    # ラベルなしは除外
    valid_mask = (y_all.sum(axis=1) > 0.0)
    x_all = x_all[valid_mask]
    y_all = y_all[valid_mask]
    times = times[valid_mask]

    train_idx = _subset_years_indices(times, train_years)
    val_idx = _subset_years_indices(times, val_years)

    x_train, y_train = x_all[train_idx], y_all[train_idx]
    x_val, y_val = x_all[val_idx], y_all[val_idx]

    keep_stats = {
        "total_after_drop": int(x_all.shape[0]),
        "train_samples": int(x_train.shape[0]),
        "val_samples": int(x_val.shape[0]),
        "height": int(H),
        "width": int(W),
        "in_channels": int(C),
        "out_classes": int(y_all.shape[1]),
        "channels": channel_names,
    }
    return x_train, y_train, x_val, y_val, keep_stats, channel_names


def _compute_norm_stats(x_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    N, H, W, C = x_train.shape
    flat = x_train.reshape(N * H * W, C)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_norm(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean[None, None, None, :]) / std[None, None, None, :]


def _np_to_torchNCHW(x: np.ndarray) -> torch.Tensor:
    # (N,H,W,C) -> (N,C,H,W)
    return torch.from_numpy(np.transpose(x, (0, 3, 1, 2)))


def _save_json(obj: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    # ロギング初期化
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    global _START_TIME
    _START_TIME = time.perf_counter()
    logi("===== main_v1_simple (PyTorch): 気圧配置CNN 学習開始 =====")
    logi(f"データ: {DATA_PATH}")
    set_global_seed(RANDOM_SEED)
    _ensure_output_dirs()

    # データパス解決
    if os.path.isabs(DATA_PATH):
        data_path = DATA_PATH
    else:
        candidate = os.path.join(BASE_DIR, DATA_PATH)
        data_path = candidate if os.path.exists(candidate) else DATA_PATH

    # NetCDF 読み込み（h5netcdf優先）
    t0 = time.perf_counter()
    try:
        ds = xr.open_dataset(data_path, engine="h5netcdf")
    except Exception:
        ds = xr.open_dataset(data_path)
    logi(f"NetCDF 読み込み完了: path={data_path} ({time.perf_counter()-t0:.2f}s)")
    for v in ["valid_time", "latitude", "longitude", "label"]:
        if v not in ds:
            raise KeyError(f"NetCDF に '{v}' が見つかりません。")

    # 使える変数は全て使用
    selected_vars = _resolve_selected_vars(ds, SELECTED_VARIABLES, VAR_CANDIDATES)

    # 入力/ラベルの構築
    t_ds = time.perf_counter()
    x_train, y_train, x_val, y_val, keep, channels = _build_dataset(
        ds,
        base_labels=BASE_LABELS,
        selected_vars=selected_vars,
        use_season=USE_SEASONAL_AS_CHANNELS,
        train_years=TRAIN_YEARS,
        val_years=VAL_YEARS,
    )
    logi(f"データセット: train={x_train.shape[0]} val={x_val.shape[0]} in_ch={x_train.shape[-1]} HxW={x_train.shape[1]}x{x_train.shape[2]} ({time.perf_counter()-t_ds:.2f}s)")
    print(f"データ統計: {keep}")

    # 正規化（train 統計）
    t_norm = time.perf_counter()
    mean, std = _compute_norm_stats(x_train)
    x_train = _apply_norm(x_train, mean, std).astype(np.float32)
    x_val = _apply_norm(x_val, mean, std).astype(np.float32)
    logi(f"正規化: C={x_train.shape[-1]} ({time.perf_counter()-t_norm:.2f}s)")

    # Tensor 変換 (NCHW)
    x_train_t = _np_to_torchNCHW(x_train).float()
    y_train_t = torch.from_numpy(y_train).float()
    x_val_t = _np_to_torchNCHW(x_val).float()
    y_val_t = torch.from_numpy(y_val).float()
    logi(f"Tensor化: train={tuple(x_train_t.shape)} val={tuple(x_val_t.shape)}")

    # DataLoader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = min(4, os.cpu_count() or 1)
    pin_mem = device.type == "cuda"

    train_ds = TensorDataset(x_train_t, y_train_t)
    val_ds = TensorDataset(x_val_t, y_val_t)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    logi(f"DataLoader: batch_size={BATCH_SIZE} steps(train/val)=({len(train_loader)}/{len(val_loader)})")

    # モデル/最適化/損失
    in_ch = x_train_t.shape[1]
    num_classes = y_train_t.shape[1]
    model = build_cnn(in_channels=in_ch, num_classes=num_classes).to(device)
    try:
        n_params = sum(p.numel() for p in model.parameters())
        logi(f"モデル: params={n_params:,} in_ch={in_ch} out_classes={num_classes}")
    except Exception:
        pass

    pos_weight_tensor = None
    if USE_POSITIVE_CLASS_WEIGHTS:
        pos_w = compute_pos_weights(y_train).astype(np.float32)  # (C,)
        print("陽性クラス重み（先頭表示）:", [round(float(w), 2) for w in pos_w[:8]], "...")
        pos_weight_tensor = torch.from_numpy(pos_w).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 学習ループ（シンプル）
    best_val_loss = float("inf")
    best_epoch = -1
    best_map = -1.0
    history = {"train_loss": [], "val_loss": [], "val_map": []}

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
        train_loss = running / len(train_loader.dataset)

        # Validate
        model.eval()
        val_running = 0.0
        all_probs, all_trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                probs = torch.sigmoid(logits)
                val_running += loss.item() * xb.size(0)
                all_probs.append(probs.detach().cpu().numpy())
                all_trues.append(yb.detach().cpu().numpy())
        val_loss = val_running / len(val_loader.dataset)
        y_prob = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, num_classes), dtype=np.float32)
        y_true = np.concatenate(all_trues, axis=0) if all_trues else np.zeros((0, num_classes), dtype=np.float32)

        ap_per_class = []
        if y_true.shape[0] > 0:
            for c in range(y_true.shape[1]):
                try:
                    ap = average_precision_score(y_true[:, c], y_prob[:, c])
                except Exception:
                    ap = 0.0
                if not np.isfinite(ap):
                    ap = 0.0
                ap_per_class.append(ap)
        map_macro = float(np.mean(ap_per_class)) if ap_per_class else 0.0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_map"].append(map_macro)

        print(f"[Epoch {epoch:03d}/{EPOCHS}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_mAP={map_macro:.4f}")

        # ベスト更新（val_loss 最小）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_map = max(best_map, map_macro)
            torch.save(model.state_dict(), BEST_WEIGHTS_PATH)

    # 最終保存（ベストを FINAL として保存）
    try:
        state = torch.load(BEST_WEIGHTS_PATH, map_location=device)
        model.load_state_dict(state)
        torch.save(model.state_dict(), FINAL_MODEL_PATH)
        logi(f"最終モデルはベストエポック（val_loss最小）: epoch={best_epoch} best_val_loss={best_val_loss:.4f} best_mAP={best_map:.4f}")
    except Exception as e:
        logi(f"ベスト重み読込失敗（現状モデルを保存）: {e}")
        torch.save(model.state_dict(), FINAL_MODEL_PATH)

    # 履歴保存
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = float(best_val_loss)
    history["best_map"] = float(best_map)
    _save_json(history, HISTORY_JSON_PATH)

    # 損失曲線の保存
    loss_curve_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_loss_curve.png")
    try:
        plt.figure(figsize=(6, 4))
        epochs_range = np.arange(1, len(history["train_loss"]) + 1)
        plt.plot(epochs_range, history["train_loss"], label="train")
        plt.plot(epochs_range, history["val_loss"], label="val")
        if best_epoch >= 1:
            plt.axvline(best_epoch, color="red", linestyle="--", linewidth=1.5, label=f"best@{best_epoch}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss vs. Epochs (best val_loss epoch={best_epoch})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(loss_curve_path, dpi=150)
        plt.close()
        print(f"損失曲線の保存: {loss_curve_path}")
    except Exception as e:
        print("損失曲線の保存に失敗:", e)

    # 正規化統計の保存
    norm_info = {
        "channel_names": channels,
        "mean": [float(m) for m in mean],
        "std": [float(s) for s in std],
        "selected_variables": selected_vars,
        "use_seasonal_as_channels": USE_SEASONAL_AS_CHANNELS,
        "train_years": TRAIN_YEARS,
        "val_years": VAL_YEARS,
        "base_labels": BASE_LABELS,
        "best_val_map": best_map,
        "best_val_loss": float(best_val_loss),
        "best_epoch": best_epoch,
        "selection_metric": "val_loss"
    }
    _save_json(norm_info, NORM_STATS_JSON_PATH)
    print(f"モデル保存: {FINAL_MODEL_PATH}")
    print(f"ベスト重み保存: {BEST_WEIGHTS_PATH}")
    print(f"学習履歴保存: {HISTORY_JSON_PATH}")
    print(f"正規化統計保存: {NORM_STATS_JSON_PATH}")

    # ベスト重みでの検証レポート（threshold で 0/1 へ）
    model.eval()
    all_probs_best, all_trues_best = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            probs = torch.sigmoid(logits)
            all_probs_best.append(probs.detach().cpu().numpy())
            all_trues_best.append(yb.detach().cpu().numpy())
    y_prob_best = np.concatenate(all_probs_best, axis=0) if all_probs_best else np.zeros((0, num_classes), dtype=np.float32)
    y_true_best = np.concatenate(all_trues_best, axis=0) if all_trues_best else np.zeros((0, num_classes), dtype=np.float32)
    y_pred_best = (y_prob_best >= PREDICTION_THRESHOLD).astype(int)
    try:
        if y_true_best.shape[0] > 0:
            report = classification_report(
                y_true_best.astype(int), y_pred_best.astype(int),
                target_names=BASE_LABELS, zero_division=0, output_dict=True
            )
            _save_json(report, VAL_REPORT_JSON_PATH)
            logi(f"検証レポート保存: {VAL_REPORT_JSON_PATH}")
    except Exception as e:
        print("classification_report でエラー:", e)


    print("===== main_v1_simple (PyTorch): 完了 =====")


if __name__ == "__main__":
    main()
