# -*- coding: utf-8 -*-
"""
PyTorch 版 CNN モデル定義

- CIFAR10/Weather の実装方針を参考に、Conv + BN + ReLU + MaxPool + Dropout のブロックを3層
- 入力は (N, C, H, W)、出力はロジット（BCEWithLogitsLoss を用いるため sigmoid はかけない）
- H, W 依存を小さくするため AdaptiveAvgPool2d(16,16) を挿入してパラメータを安定化
- マルチラベル分類（15 ラベル想定）

公開関数
- build_cnn(in_channels, num_classes): nn.Module を返す
- compute_pos_weights(y): 学習ラベル (N, C) から陽性クラス重みベクトル (C,) を算出
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, p_drop: float = 0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=p_drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32, p_drop=0.25),
            ConvBlock(32, 64, p_drop=0.25),
            ConvBlock(64, 128, p_drop=0.25),
            # 入力サイズ(161,161)想定でも、AdaptiveAvgPool2d で固定次元へ
            nn.AdaptiveAvgPool2d((16, 16)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),  # ロジットを出力（BCEWithLogitsLoss を使用）
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_cnn(in_channels: int, num_classes: int) -> nn.Module:
    """
    ベースライン CNN を生成
    """
    return SimpleCNN(in_channels=in_channels, num_classes=num_classes)


def compute_pos_weights(y: Sequence[Sequence[float] | np.ndarray]) -> np.ndarray:
    """
    学習ラベル行列 y (N, C) からクラス毎の陽性重みを計算。
      pos_weight_c = (N - N_pos_c) / max(N_pos_c, 1)
    0 陽性クラスは 1.0（重みなし）とし、極端な重みはクリップ。

    Parameters
    ----------
    y : (N, C) の 0/1 配列（list でも numpy でも可）

    Returns
    -------
    np.ndarray, shape=(C,)
    """
    y = np.asarray(y, dtype=np.float32)  # (N, C)
    n = float(y.shape[0])
    n_pos = np.sum(y, axis=0)  # (C,)
    n_pos_safe = np.where(n_pos > 0.0, n_pos, 1.0)
    pos_w = (n - n_pos_safe) / n_pos_safe
    pos_w = np.clip(pos_w, 1.0, 50.0)
    return pos_w.astype(np.float32)
