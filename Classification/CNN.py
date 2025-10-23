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


# 追加: Deformable Conv（torchvision.ops が使える場合のみ）と ConvNeXtTiny 実装
try:
    from torchvision.ops import DeformConv2d  # type: ignore
    _HAVE_DEFORM = True
except Exception:
    DeformConv2d = None  # type: ignore
    _HAVE_DEFORM = False


class DeformableConvBlock(nn.Module):
    """
    DeformableConv2d を用いたブロック（フォールバック: dilation Conv）
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, p_drop: float = 0.2):
        super().__init__()
        k = 3
        self.use_deform = _HAVE_DEFORM
        pad = 1
        if self.use_deform:
            # オフセット生成 conv（2*k*k）
            self.offset = nn.Conv2d(in_ch, 2 * k * k, kernel_size=k, stride=stride, padding=pad, bias=True)
            self.conv = DeformConv2d(in_ch, out_ch, kernel_size=k, stride=stride, padding=pad, bias=False)
        else:
            # フォールバック: dilation conv で受容野を拡大
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=stride, padding=2, dilation=2, bias=False)
            self.offset = None
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_deform and self.offset is not None:
            offset = self.offset(x)
            out = self.conv(x, offset)
        else:
            out = self.conv(x)
        out = self.relu(self.bn(out))
        return self.drop(out)


class DeformResNetSmall(nn.Module):
    """
    Deformable 畳み込みを用いた小型ResNet風モデル（フォールバック付き）
    stem + [D-Block]*2 + [D-Block]*2 + [D-Block]*2 + GAP + ヘッド
    """
    def __init__(self, in_channels: int, num_classes: int, widths=(64, 128, 256)):
        super().__init__()
        w1, w2, w3 = widths
        self.stem = DeformableConvBlock(in_channels, w1, stride=1, p_drop=0.1)
        self.layer1 = nn.Sequential(
            DeformableConvBlock(w1, w1, stride=1, p_drop=0.1),
            DeformableConvBlock(w1, w1, stride=1, p_drop=0.1),
        )
        self.layer2 = nn.Sequential(
            DeformableConvBlock(w1, w2, stride=2, p_drop=0.15),
            DeformableConvBlock(w2, w2, stride=1, p_drop=0.15),
        )
        self.layer3 = nn.Sequential(
            DeformableConvBlock(w2, w3, stride=2, p_drop=0.2),
            DeformableConvBlock(w3, w3, stride=1, p_drop=0.2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(w3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = self.head(x)
        return x


class LayerNorm2d(nn.Module):
    """
    ConvNeXt 風: チャネル次元に対する LayerNorm（channels-last で適用）
    """
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (N,C,H,W) -> (N,H,W,C) -> LN -> (N,C,H,W)
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return x


class ConvNeXtBlock(nn.Module):
    """
    簡易 ConvNeXt ブロック
    - Depthwise Conv (k=7)
    - LayerNorm2d
    - Pointwise Conv (expansion) + GELU + Pointwise Conv (projection)
    - Stochastic Depth 省略、残差は恒等
    """
    def __init__(self, dim: int, drop_path: float = 0.0, mlp_ratio: float = 4.0):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.ln = LayerNorm2d(dim)
        hidden = int(dim * mlp_ratio)
        self.pw1 = nn.Conv2d(dim, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.dw(x)
        x = self.ln(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        return x + identity


class ConvNeXtTiny(nn.Module):
    """
    軽量 ConvNeXt 風ネットワーク
    """
    def __init__(self, in_channels: int, num_classes: int, dims=(64, 128, 256), depths=(2, 2, 6)):
        super().__init__()
        d1, d2, d3 = depths
        c1, c2, c3 = dims
        # Stem（Patchify 相当）
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=4, stride=4, padding=0),
            LayerNorm2d(c1),
        )
        # Stage1
        self.stage1 = nn.Sequential(*[ConvNeXtBlock(c1) for _ in range(d1)])
        self.down1 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=2, stride=2),
            LayerNorm2d(c2),
        )
        # Stage2
        self.stage2 = nn.Sequential(*[ConvNeXtBlock(c2) for _ in range(d2)])
        self.down2 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=2, stride=2),
            LayerNorm2d(c3),
        )
        # Stage3
        self.stage3 = nn.Sequential(*[ConvNeXtBlock(c3) for _ in range(d3)])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c3, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.head(x)
        return x


def build_cnn(in_channels: int, num_classes: int, backbone: str = "simple") -> nn.Module:
    """
    バックボーンを選択してモデルを生成
      backbone:
        - "simple":      SimpleCNN（既定）
        - "resnet_small":ResNetSmall（SE付き小型ResNet）
        - "deformable":  DeformResNetSmall（Deformable Conv、フォールバックあり）
        - "convnext_tiny":ConvNeXtTiny（軽量ConvNeXt風）
    """
    name = (backbone or "simple").lower()
    if name == "resnet_small":
        return ResNetSmall(in_channels=in_channels, num_classes=num_classes)
    if name == "deformable":
        return DeformResNetSmall(in_channels=in_channels, num_classes=num_classes)
    if name == "convnext_tiny":
        return ConvNeXtTiny(in_channels=in_channels, num_classes=num_classes)
    # default
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
