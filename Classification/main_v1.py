# -*- coding: utf-8 -*-
"""
main_v1 (PyTorch): ERA5 由来の気圧配置データを用いたマルチラベル CNN 学習スクリプト

要件反映
- フレームワーク: TensorFlow ではなく PyTorch（swinunet_env に準拠）
- GPU: NVIDIA（CUDA 自動検出、混合精度 amp 使用）
- 変数: 使える候補変数は全て使用（gh500, t850, u/v(500/850), r(700/850), vo850, msl）
- 出力: ./result 配下に保存（main_v1_config.py の OUTPUT_DIR で制御）
- ラベル: 吉野の基本 15 クラスをマルチホット化（複合/移行は和集合で 1）
- 分割: 年単位（1991–1997: train, 1998–2000: val）
- 前処理: msl は Pa→hPa のみ（各時刻の領域平均は差し引かない）
- 正規化: 学習データのチャネル毎 mean/std
- 不均衡対策: BCEWithLogitsLoss の pos_weight を自動計算（陽性クラス逆頻度、クリップ）
- 評価: 各エポックで validation の mAP（macro 平均の average precision）を算出しベストモデルを保存
- 保存: 最終/ベスト重み, 学習履歴, 正規化統計, classification_report を ./result に出力
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
    import hdf5plugin  # noqa: F401  # フィルタ登録のためのimportだけでOK
except Exception:
    hdf5plugin = None

# FutureWarning を非表示（AMP の非推奨警告などを抑制）
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch import amp

from sklearn.metrics import average_precision_score, classification_report

# 追加インポート（学習レシピ/可視化）
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
try:
    from torch.optim.swa_utils import AveragedModel, update_bn
except Exception:
    AveragedModel = None
    update_bn = None

# 詳細ログと計時のためのユーティリティ
import time
import logging
from contextlib import contextmanager

logger = logging.getLogger("main_v1")
_START_TIME = time.perf_counter()
def _elapsed() -> float:
    return time.perf_counter() - _START_TIME
def logi(msg: str) -> None:
    logger.info(f"[+{_elapsed():.2f}s] {msg}")

# 相対インポート（モジュール実行）と、スクリプト実行の両対応
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
        # TENSORBOARD_LOGS_DIR,  # 未使用
        BASE_LABELS,
        VAR_CANDIDATES,
        USE_EMA,
        EVAL_WITH_EMA,
        EMA_UPDATE_BN,
    )
except Exception:
    import os as _os, sys as _sys
    _HERE = _os.path.dirname(_os.path.abspath(__file__))
    if _HERE not in _sys.path:
        _sys.path.insert(0, _HERE)
    # 直下ファイルとしての実行に対応
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
        # TENSORBOARD_LOGS_DIR,  # 未使用
        BASE_LABELS,
        VAR_CANDIDATES,
        USE_EMA,
        EVAL_WITH_EMA,
        EMA_UPDATE_BN,
    )

_LABEL_TOKEN_PATTERN = re.compile(r"[+\-]")  # '+' または '-' で分割


def set_global_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _ensure_output_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(FINAL_MODEL_PATH), exist_ok=True)


def _to_datetime64(ds: xr.Dataset) -> pd.DatetimeIndex:
    # valid_time は "seconds since 1970-01-01" の int64 (UTC)
    t = pd.to_datetime(ds["valid_time"].values, unit="s", utc=True).tz_convert("Asia/Tokyo")
    return pd.DatetimeIndex(t)


def _subset_years_indices(times: pd.DatetimeIndex, years: List[int]) -> np.ndarray:
    mask = np.isin(times.year, years)
    return np.where(mask)[0]


def _msl_preprocess(da: xr.DataArray) -> xr.DataArray:
    """
    前処理:
    - Pa -> hPa 変換のみ（各時刻の領域平均は差し引かない）
    """
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
    """
    ds から選択変数を取り出してチャネル方向に結合し、numpy (T, H, W, C) を返す。
    """
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
    '3B+4B', '2A-1', '1', などをマルチホット(15次元)に変換。
    ルール: '+' も '-' も和集合として 1 を立てる。空/不明('N/A')は全 0。
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
    """
    学習データからチャネル毎の mean/std（空間含む）を計算。
    x_train: (N, H, W, C)
    """
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
    logi("===== main_v1 (PyTorch): 気圧配置CNN 学習開始 =====")
    logi(f"データ: {DATA_PATH}")
    set_global_seed(RANDOM_SEED)
    _ensure_output_dirs()

    # データパス解決（main_v1.py からの相対パス指定を優先）
    if os.path.isabs(DATA_PATH):
        data_path = DATA_PATH
    else:
        # BASE_DIR（このファイルの場所）基準で存在チェックし、無ければ CWD 基準を試す
        candidate = os.path.join(BASE_DIR, DATA_PATH)
        data_path = candidate if os.path.exists(candidate) else DATA_PATH

    # LZ4（h5netcdf + hdf5plugin）対応でまず h5netcdf エンジンを試し、失敗時はデフォルトにフォールバック
    t0 = time.perf_counter()
    try:
        ds = xr.open_dataset(data_path, engine="h5netcdf")
    except Exception as e1:
        try:
            ds = xr.open_dataset(data_path)
        except Exception as e2:
            raise RuntimeError(f"NetCDFの読み込みに失敗しました。h5netcdf/LZ4対応が必要です: {e1} / fallback: {e2}")
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
    logi(f"データセット構築完了: train={x_train.shape[0]} val={x_val.shape[0]} in_ch={x_train.shape[-1]} HxW={x_train.shape[1]}x{x_train.shape[2]} ({time.perf_counter()-t_ds:.2f}s)")
    print(f"データ統計: {keep}")

    # 正規化（train 統計）
    t_norm = time.perf_counter()
    mean, std = _compute_norm_stats(x_train)
    x_train = _apply_norm(x_train, mean, std).astype(np.float32)
    x_val = _apply_norm(x_val, mean, std).astype(np.float32)
    logi(f"正規化適用完了: C={x_train.shape[-1]} ({time.perf_counter()-t_norm:.2f}s)")

    # NCHW へ変換
    x_train_t = _np_to_torchNCHW(x_train).float()
    x_train_t = x_train_t.contiguous(memory_format=torch.channels_last)
    y_train_t = torch.from_numpy(y_train).float()
    x_val_t = _np_to_torchNCHW(x_val).float()
    x_val_t = x_val_t.contiguous(memory_format=torch.channels_last)
    y_val_t = torch.from_numpy(y_val).float()
    logi(f"Tensor化完了: train={tuple(x_train_t.shape)} val={tuple(x_val_t.shape)}")

    # DataLoader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = min(4, os.cpu_count() or 1)
    pin_mem = device.type == "cuda"

    train_ds = TensorDataset(x_train_t, y_train_t)
    val_ds = TensorDataset(x_val_t, y_val_t)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    logi(f"DataLoader 準備完了: batch_size={BATCH_SIZE} train_steps/epoch={len(train_loader)} val_steps={len(val_loader)} workers={num_workers}")

    # モデル/最適化/損失
    in_ch = x_train_t.shape[1]
    num_classes = y_train_t.shape[1]
    model = build_cnn(in_channels=in_ch, num_classes=num_classes).to(device)
    # channels_last によるメモリレイアウト最適化
    try:
        model = model.to(memory_format=torch.channels_last)
    except Exception:
        pass
    # PyTorch 2.x であれば torch.compile による最適化を試行
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
        except Exception:
            pass
    # モデル情報ログ
    try:
        n_params = sum(p.numel() for p in model.parameters())
        logi(f"モデル構築完了: params={n_params:,} in_ch={in_ch} out_classes={num_classes}")
        if device.type == "cuda":
            torch.cuda.synchronize()
            mem_alloc = torch.cuda.memory_allocated() / (1024**2)
            mem_resv = torch.cuda.memory_reserved() / (1024**2)
            logi(f"CUDA メモリ: allocated={mem_alloc:.1f}MB reserved={mem_resv:.1f}MB")
    except Exception:
        pass

    pos_weight_tensor = None
    if USE_POSITIVE_CLASS_WEIGHTS:
        pos_w = compute_pos_weights(y_train).astype(np.float32)  # (C,)
        print("陽性クラス重み（先頭表示）:", [round(float(w), 2) for w in pos_w[:8]], "...")
        pos_weight_tensor = torch.from_numpy(pos_w).to(device)
        logi("陽性クラス重みの計算完了")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # AdamW: 正則化は重みのみに適用（BN/bias除外）
    decay_params, no_decay_params = [], []
    for n, p in model.named_parameters():
        if (p.ndim == 1) or n.endswith(".bias") or ("bn" in n) or ("norm" in n):
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=LEARNING_RATE,
    )

    scaler = amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # 学習率スケジューラ（Warmup + Cosine）
    warmup_epochs = max(1, int(0.1 * EPOCHS))
    min_lr_ratio = 0.1
    def lr_lambda(epoch: int):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))  # 線形ウォームアップ
        # Cosine decay: 1.0 → min_lr_ratio
        progress = float(epoch - warmup_epochs) / float(max(1, EPOCHS - warmup_epochs))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    # 初期LRログ
    init_lrs = [pg["lr"] for pg in optimizer.param_groups]
    logi(f"Optimizer/Scheduler 準備完了: AdamW lr={init_lrs} weight_decay={WEIGHT_DECAY} warmup_epochs={warmup_epochs}")
    logi(f"EMA settings: USE_EMA={USE_EMA} EVAL_WITH_EMA={EVAL_WITH_EMA} EMA_UPDATE_BN={EMA_UPDATE_BN}")

    # EMA（利用可能なら）
    ema_model = AveragedModel(model) if (USE_EMA and (AveragedModel is not None)) else None

    # ベスト保存（停止は行わず、指定エポックまで必ず学習）
    best_map = -1.0
    best_val_loss = float("inf")
    best_epoch = -1

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_map": [],
    }
    # 直前エポックのチェックポイント（上書き保存・クリーンアップ用）
    prev_epoch_ckpt_path = None

    VERBOSE_EVERY_N_EPOCHS = 5  # ログ出力間隔（エポック）

    for epoch in range(1, EPOCHS + 1):
        t_epoch0 = time.perf_counter()
        # -------- Train --------
        model.train()
        train_loss = 0.0
        batch_times = []
        total_batches = len(train_loader)
        logi(f"Epoch {epoch}/{EPOCHS} start: steps={total_batches}")
        t_train0 = time.perf_counter()
        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            bt0 = time.perf_counter()
            with amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            # 勾配クリッピング（安定化）
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            # EMA更新
            if ema_model is not None:
                ema_model.update_parameters(model)

            # バッチ処理時間記録・進捗ログ
            bt = time.perf_counter() - bt0
            batch_times.append(bt)
            if (batch_idx + 1) % max(1, total_batches // 5) == 0:
                logi(f"train {batch_idx+1}/{total_batches} ({(batch_idx+1)/total_batches*100:.1f}%) last_loss={loss.item():.4f} bt={bt:.2f}s")

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)
        t_train = time.perf_counter() - t_train0
        if batch_times:
            logi(f"Train 終了: avg_batch_time={np.mean(batch_times):.2f}s median={np.median(batch_times):.2f}s total={t_train:.2f}s")
        else:
            logi(f"Train 終了: total={t_train:.2f}s")

        # -------- Validate --------
        # EMAのBN統計を再推定（必要なら）
        if (EMA_UPDATE_BN and (ema_model is not None) and ("update_bn" in globals()) and (update_bn is not None)):
            try:
                logi("EMAモデルのBN統計を再推定(update_bn)")
                update_bn(train_loader, ema_model, device=device)
            except Exception as e:
                logi(f"update_bn に失敗（スキップ）: {e}")

        eval_model = ema_model if (USE_EMA and EVAL_WITH_EMA and (ema_model is not None)) else model
        eval_model.eval()
        val_loss = 0.0
        all_probs = []
        all_trues = []
        t_val0 = time.perf_counter()
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits = eval_model(xb)
                    loss = criterion(logits, yb)
                    probs = torch.sigmoid(logits)

                val_loss += loss.item() * xb.size(0)
                all_probs.append(probs.detach().cpu().numpy())
                all_trues.append(yb.detach().cpu().numpy())

        val_loss /= len(val_loader.dataset)
        y_prob = np.concatenate(all_probs, axis=0)
        y_true = np.concatenate(all_trues, axis=0)

        # mAP（macro average precision）
        ap_per_class = []
        for c in range(y_true.shape[1]):
            try:
                ap = average_precision_score(y_true[:, c], y_prob[:, c])
            except Exception:
                ap = 0.0
            if not np.isfinite(ap):
                ap = 0.0
            ap_per_class.append(ap)
        map_macro = float(np.mean(ap_per_class))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_map"].append(map_macro)

        # 改善時 or 指定間隔でのみログ出力
        improved_map = map_macro > best_map
        improved_loss = val_loss < best_val_loss
        t_val = time.perf_counter() - t_val0
        t_epoch = time.perf_counter() - t_epoch0
        if (epoch == 1) or (epoch % VERBOSE_EVERY_N_EPOCHS == 0) or (epoch == EPOCHS) or improved_loss or improved_map:
            logi(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_mAP={map_macro:.4f} time: train={t_train:.2f}s val={t_val:.2f}s total={t_epoch:.2f}s")

        # ベスト更新（選択基準は val_loss 最小）
        if improved_map:
            best_map = map_macro
        if improved_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            _to_save = model
            if USE_EMA and EVAL_WITH_EMA and (ema_model is not None):
                _to_save = ema_model
            torch.save(_to_save.state_dict(), BEST_WEIGHTS_PATH)

        # エポックチェックポイント保存（復旧用）
        try:
            ckpt_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_epoch_{epoch:03d}.ckpt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "ema_state": (ema_model.state_dict() if ema_model is not None else None),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_map": best_map,
                "best_epoch": best_epoch,
                "history": history,
                "config": {
                    "EPOCHS": EPOCHS,
                    "BATCH_SIZE": BATCH_SIZE,
                    "LEARNING_RATE": LEARNING_RATE,
                    "WEIGHT_DECAY": WEIGHT_DECAY,
                },
            }, ckpt_path)
            logi(f"エポックチェックポイント保存: {ckpt_path}")
            # 前回のチェックポイントを削除（最新のみ保持）
            if prev_epoch_ckpt_path and prev_epoch_ckpt_path != ckpt_path:
                try:
                    if os.path.exists(prev_epoch_ckpt_path):
                        os.remove(prev_epoch_ckpt_path)
                        logi(f"前回チェックポイント削除: {prev_epoch_ckpt_path}")
                except Exception as e:
                    logi(f"前回チェックポイント削除失敗: {e}")
            prev_epoch_ckpt_path = ckpt_path
        except Exception as e:
            logi(f"チェックポイント保存に失敗: {e}")

        # スケジューラ更新
        scheduler.step()


    # 最終保存（ベストエポックの重みを最終として採用）
    eval_model = ema_model if (USE_EMA and EVAL_WITH_EMA and (ema_model is not None)) else model
    try:
        state = torch.load(BEST_WEIGHTS_PATH, map_location=device)
        eval_model.load_state_dict(state)
        torch.save(eval_model.state_dict(), FINAL_MODEL_PATH)
        logi(f"最終モデルはベストエポック（val_loss最小）を採用: epoch={best_epoch} best_val_loss={best_val_loss:.4f} best_mAP={best_map:.4f}")
    except Exception as e:
        logi(f"ベスト重みの読込に失敗（最終を現状モデルで保存）: {e}")
        torch.save(model.state_dict(), FINAL_MODEL_PATH)

    history["best_epoch"] = best_epoch
    history["best_val_loss"] = float(best_val_loss)
    history["best_map"] = float(best_map)
    _save_json(history, HISTORY_JSON_PATH)

    # 損失曲線の保存（./result 配下へ）
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
    eval_model.eval()
    all_probs_best, all_trues_best = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = eval_model(xb)
                probs = torch.sigmoid(logits)
            all_probs_best.append(probs.detach().cpu().numpy())
            all_trues_best.append(yb.detach().cpu().numpy())
    y_prob_best = np.concatenate(all_probs_best, axis=0)
    y_true_best = np.concatenate(all_trues_best, axis=0)
    y_pred_best = (y_prob_best >= PREDICTION_THRESHOLD).astype(int)
    try:
        report = classification_report(
            y_true_best.astype(int), y_pred_best.astype(int),
            target_names=BASE_LABELS, zero_division=0, output_dict=True
        )
        _save_json(report, VAL_REPORT_JSON_PATH)
        logi(f"ベスト重みでの検証レポート保存: {VAL_REPORT_JSON_PATH}")
    except Exception as e:
        print("classification_report でエラー:", e)

    print("===== main_v1 (PyTorch): 完了 =====")


if __name__ == "__main__":
    main()
