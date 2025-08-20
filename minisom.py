import math
import numpy as np
from typing import Tuple, Optional, List

import torch
from torch import Tensor

try:
    from tqdm import trange
except Exception:
    def trange(n, **kwargs):
        return range(n)

def _to_device(x: np.ndarray, device: torch.device, dtype=torch.float32) -> Tensor:
    return torch.as_tensor(x, device=device, dtype=dtype)

def _as_numpy(x: Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

class MiniSom:
    """
    PyTorch GPU版 SOM（batchSOM）
      - activation_distance: 's1'（Teweles-Wobus）, 'euclidean', 'ssim'
      - 学習は「ミニバッチ版バッチSOM」：BMU→近傍重み→分子/分母累積→一括更新
      - 全ての重い計算はGPU実行
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
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.dtype = dtype

        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

        self.x = x
        self.y = y
        self.m = x * y
        self.input_len = input_len
        self.sigma0 = float(sigma)
        self.learning_rate = float(learning_rate)
        self.neighborhood_function = 'gaussian'
        self.topology = topology
        self.sigma_decay = sigma_decay
        self.nodes_chunk = int(nodes_chunk)

        # 距離タイプ
        activation_distance = activation_distance.lower()
        if activation_distance not in ('s1', 'euclidean', 'ssim'):
            raise ValueError('activation_distance must be one of "s1","euclidean","ssim"')
        self.activation_distance = activation_distance

        # 画像形状
        if s1_field_shape is None:
            raise ValueError('s1_field_shape=(H,W) is required for all distances in this implementation.')
        if s1_field_shape[0] * s1_field_shape[1] != input_len:
            raise ValueError(f's1_field_shape={s1_field_shape} does not match input_len={input_len}.')
        self.field_shape = s1_field_shape
        H,W = s1_field_shape

        # グリッド座標
        gx, gy = torch.meshgrid(torch.arange(x), torch.arange(y), indexing='ij')
        self.grid_coords = torch.stack([gx.flatten(), gy.flatten()], dim=1).to(self.device, torch.float32)

        # 重み（(m,H,W)）
        self.weights = (torch.rand((self.m, H, W), device=self.device, dtype=self.dtype)*2 - 1)

        if self.neighborhood_function != 'gaussian':
            self.neighborhood_function = 'gaussian'

        # SSIM定数
        self.c1 = 1e-8
        self.c2 = 1e-8

    # ---------- ユーティリティ ----------
    def get_weights(self) -> np.ndarray:
        H,W = self.field_shape
        w_flat = self.weights.reshape(self.m, H*W)
        w_grid = w_flat.reshape(self.x, self.y, H*W)
        return _as_numpy(w_grid)

    def random_weights_init(self, data: np.ndarray):
        H,W = self.field_shape
        n = data.shape[0]
        if n < self.m:
            idx = np.random.choice(n, self.m, replace=True)
        else:
            idx = np.random.choice(n, self.m, replace=False)
        w0 = data[idx].reshape(self.m, H, W)
        self.weights = _to_device(w0, self.device, self.dtype).clone()

    # ---------- スケジューラ ----------
    def _sigma_at(self, t: int, max_iter: int) -> float:
        if self.sigma_decay == 'asymptotic_decay':
            return self.sigma0 / (1 + t / (max_iter / 2.0))
        elif self.sigma_decay == 'linear_decay':
            return max(1e-3, self.sigma0 * (1 - t / max_iter))
        else:
            return self.sigma0 / (1 + t / (max_iter / 2.0))

    # ---------- 近傍関数 ----------
    @torch.no_grad()
    def _neighborhood(self, bmu_flat: Tensor, sigma: float) -> Tensor:
        bmu_xy = self.grid_coords[bmu_flat]  # (B,2)
        d2 = ((bmu_xy.unsqueeze(1) - self.grid_coords.unsqueeze(0))**2).sum(dim=-1)  # (B,m)
        h = torch.exp(-d2 / (2 * (sigma**2) + 1e-9))
        return h

    # ---------- 距離計算 ----------
    @torch.no_grad()
    def _euclidean_distance_batch(self, Xb: Tensor) -> Tensor:
        """
        Xb: (B,H,W) -> 距離 (B,m)
        d^2 = sum((X-W)^2)
        """
        B,H,W = Xb.shape
        Xf = Xb.reshape(B, -1)                               # (B,D)
        Wf = self.weights.reshape(self.m, -1)                # (m,D)
        x2 = (Xf*Xf).sum(dim=1, keepdim=True)               # (B,1)
        w2 = (Wf*Wf).sum(dim=1, keepdim=True).T             # (1,m)
        cross = Xf @ Wf.T                                    # (B,m)
        d2 = x2 + w2 - 2*cross
        d2 = torch.clamp(d2, min=0.0)
        return torch.sqrt(d2 + 1e-12)

    @torch.no_grad()
    def _ssim_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int]=None) -> Tensor:
        """
        Xb: (B,H,W)
        戻り: (B,m) の "距離" = 1 - SSIM
        簡略形SSIM（全体1窓）。ノード方向チャンクでVRAM節約。
        """
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk
        B,H,W = Xb.shape
        out = torch.empty((B, self.m), device=Xb.device, dtype=self.dtype)

        mu_x = Xb.mean(dim=(1,2))                              # (B,)
        Xc = Xb - mu_x.view(B,1,1)
        var_x = (Xc*Xc).mean(dim=(1,2))                        # (B,)

        for start in range(0, self.m, nodes_chunk):
            end = min(start+nodes_chunk, self.m)
            Wc = self.weights[start:end]                       # (Mc,H,W)
            mu_w = Wc.mean(dim=(1,2))                          # (Mc,)
            Wc2 = Wc - mu_w.view(-1,1,1)
            var_w = (Wc2*Wc2).mean(dim=(1,2))                  # (Mc,)

            cov = (Xc.unsqueeze(1) * Wc2.unsqueeze(0)).mean(dim=(2,3))  # (B,Mc)

            l_num = (2*mu_x.view(B,1)*mu_w.view(1,-1) + self.c1)
            l_den = (mu_x.view(B,1)**2 + mu_w.view(1,-1)**2 + self.c1)
            c_num = (2*cov + self.c2)
            c_den = (var_x.view(B,1) + var_w.view(1,-1) + self.c2)
            ssim = (l_num*c_num) / (l_den*c_den + 1e-12)
            dist = 1.0 - ssim
            out[:, start:end] = dist

        return out

    @torch.no_grad()
    def _s1_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int]=None) -> Tensor:
        """
        Xb: (B,H,W) 戻り (B,m) S1距離
        """
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk
        B,H,W = Xb.shape
        dXdx = Xb[:,:,1:] - Xb[:,:,:-1]  # (B,H,W-1)
        dXdy = Xb[:,1:,:] - Xb[:,:-1,:]  # (B,H-1,W)

        out = torch.empty((B,self.m), device=Xb.device, dtype=self.dtype)
        dWdx_full = self.weights[:,:,1:] - self.weights[:,:,:-1]
        dWdy_full = self.weights[:,1:,:] - self.weights[:,:-1,:]

        for start in range(0, self.m, nodes_chunk):
            end = min(start+nodes_chunk, self.m)
            dWdx = dWdx_full[start:end]   # (Mc,H,W-1)
            dWdy = dWdy_full[start:end]   # (Mc,H-1,W)
            num_dx = (torch.abs(dWdx.unsqueeze(0) - dXdx.unsqueeze(1))).sum(dim=(2,3))
            num_dy = (torch.abs(dWdy.unsqueeze(0) - dXdy.unsqueeze(1))).sum(dim=(2,3))
            num = num_dx + num_dy

            den_dx = (torch.maximum(torch.abs(dWdx).unsqueeze(0), torch.abs(dXdx).unsqueeze(1))).sum(dim=(2,3))
            den_dy = (torch.maximum(torch.abs(dWdy).unsqueeze(0), torch.abs(dXdy).unsqueeze(1))).sum(dim=(2,3))
            denom = den_dx + den_dy
            s1 = 100.0 * num / (denom + 1e-12)
            out[:,start:end] = s1

        return out

    @torch.no_grad()
    def _distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int]=None) -> Tensor:
        if self.activation_distance == 's1':
            return self._s1_distance_batch(Xb, nodes_chunk=nodes_chunk)
        elif self.activation_distance == 'euclidean':
            return self._euclidean_distance_batch(Xb)
        elif self.activation_distance == 'ssim':
            return self._ssim_distance_batch(Xb, nodes_chunk=nodes_chunk)
        else:
            raise RuntimeError('Unknown activation_distance')

    @torch.no_grad()
    def bmu_indices(self, Xb: Tensor, nodes_chunk: Optional[int]=None) -> Tensor:
        dists = self._distance_batch(Xb, nodes_chunk=nodes_chunk)
        bmu = torch.argmin(dists, dim=1)
        return bmu

    # ---------- 学習 ----------
    @torch.no_grad()
    def train_batch(self,
                    data: np.ndarray,
                    num_iteration: int,
                    batch_size: int = 32,
                    verbose: bool = True,
                    log_interval: int = 50,
                    update_per_iteration: bool = False,
                    shuffle: bool = True):
        N,D = data.shape
        H,W = self.field_shape
        if D != H*W:
            raise ValueError(f'data dimension {D} != H*W {H*W}')
        Xall = _to_device(data, self.device, self.dtype).reshape(N,H,W)

        qhist: List[float] = []
        rng_idx = torch.arange(N, device=self.device)

        iterator = trange(num_iteration) if verbose else range(num_iteration)
        for it in iterator:
            sigma = self._sigma_at(it, num_iteration)

            numerator = torch.zeros_like(self.weights)  # (m,H,W)
            denominator = torch.zeros((self.m,), device=self.device, dtype=self.dtype)

            if shuffle:
                perm = torch.randperm(N, device=self.device)
                idx_all = rng_idx[perm]
            else:
                idx_all = rng_idx

            for start in range(0, N, batch_size):
                end = min(start+batch_size, N)
                batch_idx = idx_all[start:end]
                Xb = Xall[batch_idx]

                bmu = self.bmu_indices(Xb, nodes_chunk=self.nodes_chunk)    # (B,)
                h = self._neighborhood(bmu, sigma)                          # (B,m)
                numerator += (h.unsqueeze(-1).unsqueeze(-1) * Xb.unsqueeze(1)).sum(dim=0)
                denominator += h.sum(dim=0)

                if update_per_iteration:
                    mask = (denominator>0)
                    denom_safe = denominator.clone()
                    denom_safe[~mask] = 1.0
                    new_w = numerator / denom_safe.view(-1,1,1)
                    self.weights[mask] = new_w[mask]
                    numerator.zero_(); denominator.zero_()

            mask = (denominator>0)
            if mask.any():
                denom_safe = denominator.clone()
                denom_safe[~mask] = 1.0
                new_w = numerator / denom_safe.view(-1,1,1)
                self.weights[mask] = new_w[mask]

            if (it % log_interval == 0) or (it == num_iteration-1):
                qe = self.quantization_error(data, sample_limit=2048)
                qhist.append(qe)
                if verbose and hasattr(iterator, 'set_postfix'):
                    iterator.set_postfix(q_error=f"{qe:.6f}", sigma=f"{sigma:.3f}")

        return qhist

    # ---------- 評価 ----------
    @torch.no_grad()
    def quantization_error(self, data: np.ndarray, sample_limit: Optional[int]=None, batch_size: int=64) -> float:
        N,D = data.shape
        H,W = self.field_shape
        if sample_limit is not None and sample_limit < N:
            idx = np.random.choice(N, sample_limit, replace=False)
            X = data[idx]
        else:
            X = data
        X = _to_device(X, self.device, self.dtype).reshape(-1,H,W)
        total = 0.0; cnt=0
        for start in range(0, X.shape[0], batch_size):
            end = min(start+batch_size, X.shape[0])
            Xb = X[start:end]
            d = self._distance_batch(Xb, nodes_chunk=self.nodes_chunk)
            mins = torch.min(d, dim=1).values
            total += float(mins.sum().item())
            cnt += Xb.shape[0]
        return total/max(1,cnt)

    @torch.no_grad()
    def predict(self, data: np.ndarray, batch_size: int=64) -> np.ndarray:
        N,D = data.shape
        H,W = self.field_shape
        X = _to_device(data, self.device, self.dtype).reshape(N,H,W)
        bmu_all=[]
        for start in range(0, N, batch_size):
            end = min(start+batch_size, N)
            Xb = X[start:end]
            bmu = self.bmu_indices(Xb, nodes_chunk=self.nodes_chunk)
            bmu_all.append(bmu)
        bmu_flat = torch.cat(bmu_all, dim=0)
        y = (bmu_flat % self.y).to(torch.long)
        x = (bmu_flat // self.y).to(torch.long)
        out = torch.stack([x,y], dim=1)
        return _as_numpy(out)