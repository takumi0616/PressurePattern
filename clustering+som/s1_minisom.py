# s1_minisom.py
# -*- coding: utf-8 -*-
import math
from typing import Tuple, Optional, List

import numpy as np
import torch
from torch import Tensor

try:
    from tqdm import trange
except Exception:
    def trange(n, **kwargs):
        return range(n)


def grid_auto_size(n_targets: int) -> Tuple[int, int]:
    """
    目標ノード数（例: クラスタ数）をすべてカバーする最小の正方格子サイズを返す。
    例: n_targets=19 -> (5, 5)
    """
    side = int(math.ceil(math.sqrt(max(1, int(n_targets)))))
    return side, side


def _to_device(x: np.ndarray, device: torch.device, dtype=torch.float32) -> Tensor:
    return torch.as_tensor(x, device=device, dtype=dtype)


def _as_numpy(x: Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


class MiniSom:
    """
    GPU対応 S1距離版 ミニバッチ・バッチSOM
    - activation_distance='s1' のみ対応
    - 学習は「ミニバッチ版 バッチSOM」。
      各ミニバッチでBMUをS1で決め、ガウス近傍重みで分子/分母を累積し、エポック末に weights = numerator / denominator。
    - 勾配ベースS1距離は2次元場として計算（(H, W)）し、GPUで加速。

    使い方:
      som = MiniSom(x, y, D, s1_field_shape=(H, W), device='cuda', random_seed=SEED)
      som.random_weights_init(data)  # data: (N, D), ここではメドイドの空間偏差[hPa]を想定
      som.train_batch(data, num_iteration=..., batch_size=..., nodes_chunk=...)  # GPUに合わせて
      winners = som.predict(data)  # (N, 2) のBMU座標
    """
    def __init__(self,
                 x: int,
                 y: int,
                 input_len: int,
                 sigma: float = 1.0,
                 learning_rate: float = 0.5,
                 neighborhood_function: str = 'gaussian',
                 topology: str = 'rectangular',
                 activation_distance: str = 's1',
                 random_seed: Optional[int] = None,
                 sigma_decay: str = 'asymptotic_decay',
                 s1_field_shape: Optional[Tuple[int, int]] = None,
                 device: Optional[str] = None,
                 dtype: torch.dtype = torch.float32,
                 nodes_chunk: int = 16):
        if activation_distance.lower() != 's1':
            raise ValueError('MiniSom: activation_distance="s1" のみサポートします。')
        if s1_field_shape is None:
            raise ValueError('MiniSom: s1_field_shape=(H, W) を指定してください。')
        if s1_field_shape[0] * s1_field_shape[1] != input_len:
            raise ValueError(f's1_field_shape {s1_field_shape} と input_len {input_len} が一致しません。')

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.dtype = dtype

        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

        self.x = int(x)
        self.y = int(y)
        self.m = self.x * self.y
        self.input_len = int(input_len)
        self.sigma0 = float(sigma)
        self.learning_rate = float(learning_rate)
        self.neighborhood_function = neighborhood_function
        self.topology = topology
        self.sigma_decay = sigma_decay
        self.s1_field_shape = s1_field_shape
        self.nodes_chunk = int(nodes_chunk)

        gx, gy = torch.meshgrid(torch.arange(self.x), torch.arange(self.y), indexing='ij')
        self.grid_coords = torch.stack([gx.flatten(), gy.flatten()], dim=1).to(self.device, torch.float32)
        H, W = s1_field_shape
        self.weights = (torch.rand((self.m, H, W), device=self.device, dtype=self.dtype) * 2 - 1)

        if self.neighborhood_function != 'gaussian':
            print('Warning: neighborhood_functionはgaussianのみ実装済みのためgaussianで処理します。')
            self.neighborhood_function = 'gaussian'

    def get_weights(self) -> np.ndarray:
        H, W = self.s1_field_shape
        w_flat = self.weights.reshape(self.m, H * W)
        w_grid = w_flat.reshape(self.x, self.y, H * W)
        return _as_numpy(w_grid)

    def random_weights_init(self, data: np.ndarray):
        H, W = self.s1_field_shape
        n = data.shape[0]
        if n < self.m:
            idx = np.random.choice(n, self.m, replace=True)
        else:
            idx = np.random.choice(n, self.m, replace=False)
        w0 = data[idx]  # (m, D)
        w0 = w0.reshape(self.m, H, W)
        self.weights = _to_device(w0, self.device, self.dtype).clone()

    def _sigma_at(self, t: int, max_iter: int) -> float:
        if self.sigma_decay == 'asymptotic_decay':
            return self.sigma0 / (1 + t / (max_iter / 2.0))
        elif self.sigma_decay == 'linear_decay':
            return max(1e-3, self.sigma0 * (1 - t / max_iter))
        return self.sigma0 / (1 + t / (max_iter / 2.0))

    @torch.no_grad()
    def _s1_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        Xb: (B, H, W)
        戻り値: (B, m) S1距離（小さいほど近い）
        """
        B, H, W = Xb.shape
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk

        dXdx = Xb[:, :, 1:] - Xb[:, :, :-1]
        dXdy = Xb[:, 1:, :] - Xb[:, :-1, :]
        mask_dx = torch.isfinite(dXdx).to(Xb.dtype)
        mask_dy = torch.isfinite(dXdy).to(Xb.dtype)
        dXdx = torch.where(torch.isfinite(dXdx), dXdx, torch.zeros_like(dXdx))
        dXdy = torch.where(torch.isfinite(dXdy), dXdy, torch.zeros_like(dXdy))

        out = torch.empty((B, self.m), device=Xb.device, dtype=self.dtype)
        dWdx_full = self.weights[:, :, 1:] - self.weights[:, :, :-1]
        dWdy_full = self.weights[:, 1:, :] - self.weights[:, :-1, :]

        for start in range(0, self.m, nodes_chunk):
            end = min(start + nodes_chunk, self.m)
            dWdx = dWdx_full[start:end]
            dWdy = dWdy_full[start:end]

            num_dx = (torch.abs(dWdx.unsqueeze(0) - dXdx.unsqueeze(1)) * mask_dx.unsqueeze(1)).sum(dim=(2, 3))
            num_dy = (torch.abs(dWdy.unsqueeze(0) - dXdy.unsqueeze(1)) * mask_dy.unsqueeze(1)).sum(dim=(2, 3))
            num = num_dx + num_dy

            den_dx = (torch.maximum(torch.abs(dWdx).unsqueeze(0), torch.abs(dXdx).unsqueeze(1)) * mask_dx.unsqueeze(1)).sum(dim=(2, 3))
            den_dy = (torch.maximum(torch.abs(dWdy).unsqueeze(0), torch.abs(dXdy).unsqueeze(1)) * mask_dy.unsqueeze(1)).sum(dim=(2, 3))
            denom = den_dx + den_dy

            s1 = 100.0 * num / (denom + 1e-12)
            out[:, start:end] = s1

        return out

    @torch.no_grad()
    def bmu_indices(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        dists = self._s1_distance_batch(Xb, nodes_chunk=nodes_chunk)
        bmu = torch.argmin(dists, dim=1)
        return bmu

    @torch.no_grad()
    def _neighborhood(self, bmu_flat: Tensor, sigma: float) -> Tensor:
        bmu_xy = self.grid_coords[bmu_flat]
        d2 = ((bmu_xy.unsqueeze(1) - self.grid_coords.unsqueeze(0)) ** 2).sum(dim=-1)
        h = torch.exp(-d2 / (2 * (sigma ** 2) + 1e-9))
        return h

    @torch.no_grad()
    def train_batch(self,
                    data: np.ndarray,
                    num_iteration: int,
                    batch_size: int = 32,
                    verbose: bool = True,
                    log_interval: int = 50,
                    update_per_iteration: bool = False,
                    shuffle: bool = True):
        N, D = data.shape
        H, W = self.s1_field_shape
        if D != H * W:
            raise ValueError(f'data dim {D} != H*W {H*W}.')

        Xall = _to_device(data, self.device, self.dtype).reshape(N, H, W)
        qhist: List[float] = []
        rng_idx = torch.arange(N, device=self.device)

        iterator = trange(num_iteration) if verbose else range(num_iteration)
        for it in iterator:
            sigma = self._sigma_at(it, num_iteration)

            numerator = torch.zeros_like(self.weights)
            denominator = torch.zeros((self.m,), device=self.device, dtype=self.dtype)

            if shuffle:
                perm = torch.randperm(N, device=self.device)
                idx_all = rng_idx[perm]
            else:
                idx_all = rng_idx

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                batch_idx = idx_all[start:end]
                Xb = Xall[batch_idx]

                bmu = self.bmu_indices(Xb, nodes_chunk=self.nodes_chunk)
                h = self._neighborhood(bmu, sigma)

                numerator += (h.unsqueeze(-1).unsqueeze(-1) * Xb.unsqueeze(1)).sum(dim=0)
                denominator += h.sum(dim=0)

                if update_per_iteration:
                    mask = (denominator > 0)
                    denom_safe = denominator.clone()
                    denom_safe[~mask] = 1.0
                    new_w = numerator / denom_safe.view(-1, 1, 1)
                    self.weights[mask] = new_w[mask]
                    numerator.zero_()
                    denominator.zero_()

            mask = (denominator > 0)
            if mask.any():
                denom_safe = denominator.clone()
                denom_safe[~mask] = 1.0
                new_w = numerator / denom_safe.view(-1, 1, 1)
                self.weights[mask] = new_w[mask]

            if (it % log_interval == 0) or (it == num_iteration - 1):
                qe = self.quantization_error(data, sample_limit=min(2048, N))
                qhist.append(qe)
                if verbose and hasattr(iterator, 'set_postfix'):
                    iterator.set_postfix(q_error=f"{qe:.5f}", sigma=f"{sigma:.3f}")

        return qhist

    @torch.no_grad()
    def quantization_error(self, data: np.ndarray, sample_limit: Optional[int] = None, batch_size: int = 64) -> float:
        N, D = data.shape
        H, W = self.s1_field_shape
        if sample_limit is not None and sample_limit < N:
            idx = np.random.choice(N, sample_limit, replace=False)
            X = data[idx]
        else:
            X = data
        X = _to_device(X, self.device, self.dtype).reshape(-1, H, W)

        total = 0.0
        cnt = 0
        for start in range(0, X.shape[0], batch_size):
            end = min(start + batch_size, X.shape[0])
            Xb = X[start:end]
            d = self._s1_distance_batch(Xb, nodes_chunk=self.nodes_chunk)
            mins = torch.min(d, dim=1).values
            total += float(mins.sum().item())
            cnt += Xb.shape[0]
        return total / max(1, cnt)

    @torch.no_grad()
    def predict(self, data: np.ndarray, batch_size: int = 64) -> np.ndarray:
        N, D = data.shape
        H, W = self.s1_field_shape
        X = _to_device(data, self.device, self.dtype).reshape(N, H, W)

        bmu_all = []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            Xb = X[start:end]
            bmu = self.bmu_indices(Xb, nodes_chunk=self.nodes_chunk)
            bmu_all.append(bmu)
        bmu_flat = torch.cat(bmu_all, dim=0)
        y = (bmu_flat % self.y).to(torch.long)
        x = (bmu_flat // self.y).to(torch.long)
        out = torch.stack([x, y], dim=1)
        return _as_numpy(out)