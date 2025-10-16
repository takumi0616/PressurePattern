# -*- coding: utf-8 -*-
"""
PyTorch 版 CNN モデル定義

- シンプルな CNN: Conv + ReLU + MaxPool + Dropout を3層
- 入力は (N, C, H, W)、出力はロジット（BCEWithLogitsLoss を用いるため sigmoid はかけない）
- 入力サイズ依存を小さくするため AdaptiveAvgPool2d(4,4) を使用
- マルチラベル分類（15 ラベル想定）

公開関数
- build_cnn(in_channels, num_classes): nn.Module を返す（SimpleCNN を生成）
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
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),

            # 入力サイズに依存せず、全結合の入力次元を固定化
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),  # ロジット（BCEWithLogitsLossを使用するためsigmoidはかけない）
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class SE(nn.Module):
    """
    Channel-wise 注意（Squeeze-and-Excitation）
    """
    def __init__(self, ch: int, r: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        hidden = max(1, ch // r)
        self.fc = nn.Sequential(
            nn.Linear(ch, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, ch, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class BasicBlock(nn.Module):
    """
    シンプルな残差ブロック（Conv-BN-ReLU x2 + SE + Dropout）
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, p_drop: float = 0.2, use_se: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SE(out_ch) if use_se else nn.Identity()
        self.down = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if (stride != 1 or in_ch != out_ch) else nn.Identity()
        )
        self.drop = nn.Dropout(p=p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.down(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.relu(out + identity)
        return self.drop(out)


class ResNetSmall(nn.Module):
    """
    小型残差CNN: [64,128,256] 幅、GAPヘッド、SE付き
    """
    def __init__(self, in_channels: int, num_classes: int, widths=(64, 128, 256)):
        super().__init__()
        w1, w2, w3 = widths
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, w1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(w1),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            BasicBlock(w1, w1, stride=1, p_drop=0.15, use_se=True),
            BasicBlock(w1, w1, stride=1, p_drop=0.15, use_se=True),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(w1, w2, stride=2, p_drop=0.2, use_se=True),
            BasicBlock(w2, w2, stride=1, p_drop=0.2, use_se=True),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(w2, w3, stride=2, p_drop=0.25, use_se=True),
            BasicBlock(w3, w3, stride=1, p_drop=0.25, use_se=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # GAP
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(w3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes),  # ロジット
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = self.head(x)
        return x


def build_cnn(in_channels: int, num_classes: int) -> nn.Module:
    """
    シンプルなCNNを生成（Conv-ReLU-MaxPool-Dropout x3 + Flatten-512-Dropout + 出力）
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
