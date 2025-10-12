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
- 前処理: README準拠（msl: Pa→hPa + 各時刻の領域平均を差し引く）
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

# 相対インポート（モジュール実行）と、スクリプト実行の両対応
try:
    from .CNN import build_cnn, compute_pos_weights
    from .main_v1_config import (
        BASE_DIR,
        DATA_PATH,
        SELECTED_VARIABLES,
        USE_SEASONAL_AS_CHANNELS,
        SLP_TO_HPA_AND_REMOVE_AREA_MEAN,
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
        SLP_TO_HPA_AND_REMOVE_AREA_MEAN,
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
    READMEの前処理:
    - Pa -> hPa 変換
    - 各時刻で空間平均（lat, lon）を差し引き
    """
    out = da / 100.0  # Pa → hPa
    mean2d = out.mean(dim=("latitude", "longitude"), skipna=True)
    # xarray は次元名に基づいて自動ブロードキャストされるため、そのまま減算する
    out = out - mean2d
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

        if var == "msl" and SLP_TO_HPA_AND_REMOVE_AREA_MEAN:
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
    print("===== main_v1 (PyTorch): 気圧配置CNN 学習開始 =====")
    print(f"データ: {DATA_PATH}")
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
    try:
        ds = xr.open_dataset(data_path, engine="h5netcdf")
    except Exception as e1:
        try:
            ds = xr.open_dataset(data_path)
        except Exception as e2:
            raise RuntimeError(f"NetCDFの読み込みに失敗しました。h5netcdf/LZ4対応が必要です: {e1} / fallback: {e2}")
    for v in ["valid_time", "latitude", "longitude", "label"]:
        if v not in ds:
            raise KeyError(f"NetCDF に '{v}' が見つかりません。")

    # 使える変数は全て使用
    selected_vars = _resolve_selected_vars(ds, SELECTED_VARIABLES, VAR_CANDIDATES)

    # 入力/ラベルの構築
    x_train, y_train, x_val, y_val, keep, channels = _build_dataset(
        ds,
        base_labels=BASE_LABELS,
        selected_vars=selected_vars,
        use_season=USE_SEASONAL_AS_CHANNELS,
        train_years=TRAIN_YEARS,
        val_years=VAL_YEARS,
    )
    print(f"データ統計: {keep}")

    # 正規化（train 統計）
    mean, std = _compute_norm_stats(x_train)
    x_train = _apply_norm(x_train, mean, std).astype(np.float32)
    x_val = _apply_norm(x_val, mean, std).astype(np.float32)

    # NCHW へ変換
    x_train_t = _np_to_torchNCHW(x_train).float()
    y_train_t = torch.from_numpy(y_train).float()
    x_val_t = _np_to_torchNCHW(x_val).float()
    y_val_t = torch.from_numpy(y_val).float()

    # DataLoader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = min(4, os.cpu_count() or 1)
    pin_mem = device.type == "cuda"

    train_ds = TensorDataset(x_train_t, y_train_t)
    val_ds = TensorDataset(x_val_t, y_val_t)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

    # モデル/最適化/損失
    in_ch = x_train_t.shape[1]
    num_classes = y_train_t.shape[1]
    model = build_cnn(in_channels=in_ch, num_classes=num_classes).to(device)

    pos_weight_tensor = None
    if USE_POSITIVE_CLASS_WEIGHTS:
        pos_w = compute_pos_weights(y_train).astype(np.float32)  # (C,)
        print("陽性クラス重み（先頭表示）:", [round(float(w), 2) for w in pos_w[:8]], "...")
        pos_weight_tensor = torch.from_numpy(pos_w).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # 早期終了/ベスト保存
    best_map = -1.0
    best_epoch = -1
    patience = 10
    wait = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_map": [],
    }

    VERBOSE_EVERY_N_EPOCHS = 5  # ログ出力間隔（エポック）

    for epoch in range(1, EPOCHS + 1):
        # -------- Train --------
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        # -------- Validate --------
        model.eval()
        val_loss = 0.0
        all_probs = []
        all_trues = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(xb)
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
        improved = map_macro > best_map
        if (epoch == 1) or (epoch % VERBOSE_EVERY_N_EPOCHS == 0) or (epoch == EPOCHS) or improved:
            print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_mAP={map_macro:.4f}")

        # ベスト更新
        if improved:
            best_map = map_macro
            best_epoch = epoch
            wait = 0
            torch.save(model.state_dict(), BEST_WEIGHTS_PATH)
        else:
            wait += 1

        # 早期終了
        if wait >= patience:
            print(f"Early stopping at epoch {epoch} (best mAP={best_map:.4f} @ epoch {best_epoch})")
            break

    # 最終保存
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    _save_json(history, HISTORY_JSON_PATH)

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
        "best_epoch": best_epoch,
    }
    _save_json(norm_info, NORM_STATS_JSON_PATH)
    print(f"モデル保存: {FINAL_MODEL_PATH}")
    print(f"ベスト重み保存: {BEST_WEIGHTS_PATH}")
    print(f"学習履歴保存: {HISTORY_JSON_PATH}")
    print(f"正規化統計保存: {NORM_STATS_JSON_PATH}")

    # 検証レポート（threshold で 0/1 へ）
    y_pred = (y_prob >= PREDICTION_THRESHOLD).astype(int)
    try:
        report = classification_report(
            y_true.astype(int), y_pred.astype(int),
            target_names=BASE_LABELS, zero_division=0, output_dict=True
        )
        _save_json(report, VAL_REPORT_JSON_PATH)
        print(f"検証レポート保存: {VAL_REPORT_JSON_PATH}")
    except Exception as e:
        print("classification_report でエラー:", e)

    print("===== main_v1 (PyTorch): 完了 =====")


if __name__ == "__main__":
    main()
