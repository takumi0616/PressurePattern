import math
import numpy as np
import os
from typing import Tuple, Optional, List

import torch
from torch import Tensor
import torch.nn.functional as F

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
      - activation_distance（許可する手法のみ）:
          'euclidean'（ユークリッド）
          'ssim5'    （5x5 窓・C=0 の SSIM: 距離は 1-SSIM）
          's1'       （Teweles–Wobus S1）
          'kappa'    （κ 曲率距離：0.5 * Σ|κ(X)-κ(W)| / Σmax(|κ(X)|,|κ(W)|)）
          's1k'      （S1 と κ の RMS 合成；S1 と κ を行方向 min–max 正規化後に RMS）
          'emd'      （EMD：正負分離・部分マッチング。GPU対応Sinkhorn近似（ε→0, 反復増で厳密解に収束））
      - 学習は「ミニバッチ版バッチSOM」：BMU→近傍重み→分子/分母累積→一括更新
      - 全ての重い計算はGPU実行
      - σ（近傍幅）は学習全体で一方向に減衰させる（セグメント学習でも継続）
      - 任意頻度で“距離一貫性”のためのメドイド置換（ノード重みを最近傍サンプルへ置換）を実行可
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
                 nodes_chunk: int = 16,
                 ssim_window: int = 5,
                 area_weight: Optional[np.ndarray] = None):
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

        # 学習全体の反復管理（σ継続減衰用）
        self.global_iter: int = 0
        self.total_iters: Optional[int] = None

        # メドイド置換頻度（None: 不使用, k: k反復ごと）
        self.medoid_replace_every: Optional[int] = None

        # 評価用の固定サンプルインデックス（QEを安定化）
        self.eval_indices: Optional[Tensor] = None

        # 距離タイプ（許可手法に限定）
        activation_distance = activation_distance.lower()
        if activation_distance not in ('s1', 'euclidean', 'ssim5', 'kappa', 's1k', 'emd'):
            raise ValueError('activation_distance must be one of "s1","euclidean","ssim5","kappa","s1k","emd"')
        self.activation_distance = activation_distance

        # 画像形状
        if s1_field_shape is None:
            raise ValueError('s1_field_shape=(H,W) is required for all distances in this implementation.')
        if s1_field_shape[0] * s1_field_shape[1] != input_len:
            raise ValueError(f's1_field_shape={s1_field_shape} does not match input_len={input_len}.')
        self.field_shape = s1_field_shape
        H, W = s1_field_shape
        # Optional area weight (e.g., cos(lat)); used optionally for curvature-family, but safe to keep available
        self.area_w: Optional[Tensor] = None
        if area_weight is not None:
            aw = torch.as_tensor(area_weight, device=self.device, dtype=self.dtype)
            if aw.shape != (H, W):
                raise ValueError(f'area_weight shape {aw.shape} does not match field_shape {(H, W)}')
            self.area_w = aw

        # グリッド座標
        gx, gy = torch.meshgrid(torch.arange(x), torch.arange(y), indexing='ij')
        self.grid_coords = torch.stack([gx.flatten(), gy.flatten()], dim=1).to(self.device, torch.float32)

        # 重み（(m,H,W)）
        self.weights = (torch.rand((self.m, H, W), device=self.device, dtype=self.dtype) * 2 - 1)

        if self.neighborhood_function != 'gaussian':
            self.neighborhood_function = 'gaussian'

        # SSIM用移動窓カーネル（平均フィルタ）
        self._kernel5: Optional[Tensor] = None
        self._win5_size: int = int(ssim_window)
        if self._win5_size < 1:
            raise ValueError(f'ssim_window must be positive integer, got {self._win5_size}')
        self._win5_pad: int = self._win5_size // 2

    # ---------- 外部制御 ----------
    def set_total_iterations(self, total_iters: int):
        """学習全体の反復回数を設定（σ減衰の基準）。複数回train_batchを呼ぶ前に設定してください。"""
        self.total_iters = int(total_iters)

    def set_medoid_replace_every(self, k: Optional[int]):
        """k反復ごとにメドイド置換（各ノード重みを距離的に最も近いサンプルへ置換）を行う。Noneまたは0で無効。"""
        if k is None or k <= 0:
            self.medoid_replace_every = None
        else:
            self.medoid_replace_every = int(k)

    def set_eval_indices(self, idx: Optional[np.ndarray]):
        """
        評価（quantization_error/predict等の固定評価で使用）用のインデックスを設定。
        Noneで解除。idxはデータ配列に対する行インデックス。
        """
        if idx is None:
            self.eval_indices = None
        else:
            self.eval_indices = torch.as_tensor(idx, device=self.device, dtype=torch.long)

    # ---------- ユーティリティ ----------
    def get_weights(self) -> np.ndarray:
        H, W = self.field_shape
        w_flat = self.weights.reshape(self.m, H * W)
        w_grid = w_flat.reshape(self.x, self.y, H * W)
        return _as_numpy(w_grid)

    def random_weights_init(self, data: np.ndarray):
        H, W = self.field_shape
        n = data.shape[0]
        if n < self.m:
            idx = np.random.choice(n, self.m, replace=True)
        else:
            idx = np.random.choice(n, self.m, replace=False)
        w0 = data[idx].reshape(self.m, H, W)
        self.weights = _to_device(w0, self.device, self.dtype).clone()

    # ---------- スケジューラ ----------
    def _sigma_at_val(self, t: int, max_iter: int) -> float:
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
        d2 = ((bmu_xy.unsqueeze(1) - self.grid_coords.unsqueeze(0)) ** 2).sum(dim=-1)  # (B,m)
        h = torch.exp(-d2 / (2 * (sigma ** 2) + 1e-9))
        return h

    # ---------- 内部ヘルパ（SSIM） ----------
    @torch.no_grad()
    def _ensure_kernel5(self):
        if self._kernel5 is None:
            k = torch.ones((1, 1, self._win5_size, self._win5_size), device=self.device, dtype=self.dtype) / float(self._win5_size * self._win5_size)
            self._kernel5 = k

    def _ssim_pad_tuple(self) -> Tuple[int, int, int, int]:
        # Asymmetric SAME padding for arbitrary window size (odd/even)
        k = int(self._win5_size)
        pl = k // 2
        pr = k - 1 - pl
        pt = k // 2
        pb = k - 1 - pt
        return (pl, pr, pt, pb)

    # ---------- 距離計算（バッチ→全ノード） ----------
    @torch.no_grad()
    def _euclidean_distance_batch(self, Xb: Tensor) -> Tensor:
        """
        Xb: (B,H,W) -> 距離 (B,m)
        d^2 = sum((X-W)^2), 戻りは sqrt(d^2)（単調変換）
        """
        B, H, W = Xb.shape
        Xf = Xb.reshape(B, -1)                # (B,D)
        Wf = self.weights.reshape(self.m, -1) # (m,D)
        x2 = (Xf * Xf).sum(dim=1, keepdim=True)         # (B,1)
        w2 = (Wf * Wf).sum(dim=1, keepdim=True).T       # (1,m)
        cross = Xf @ Wf.T                                # (B,m)
        d2 = x2 + w2 - 2 * cross
        d2 = torch.clamp(d2, min=0.0)
        return torch.sqrt(d2 + 1e-12)

    @torch.no_grad()
    def _ssim5_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        SSIM: 5x5移動窓・C=0（分母のみ数値安定化）
        Xb: (B,H,W)
        戻り: (B,m) の "距離" = 1 - mean(SSIM_map)
        """
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk
        self._ensure_kernel5()
        eps = 1e-12
        B, H, W = Xb.shape
        out = torch.empty((B, self.m), device=Xb.device, dtype=self.dtype)

        # X 側のローカル統計
        X = Xb.unsqueeze(1)  # (B,1,H,W)
        X_pad = F.pad(X, self._ssim_pad_tuple(), mode='reflect')
        mu_x = F.conv2d(X_pad, self._kernel5, padding=0)                      # (B,1,H,W)
        mu_x2 = F.conv2d(X_pad * X_pad, self._kernel5, padding=0)             # (B,1,H,W)
        var_x = torch.clamp(mu_x2 - mu_x * mu_x, min=0.0)                     # (B,1,H,W)

        for start in range(0, self.m, nodes_chunk):
            end = min(start + nodes_chunk, self.m)
            Wc = self.weights[start:end].unsqueeze(1)                          # (Mc,1,H,W)
            W_pad = F.pad(Wc, self._ssim_pad_tuple(), mode='reflect')
            mu_w = F.conv2d(W_pad, self._kernel5, padding=0)                  # (Mc,1,H,W)
            mu_w2 = F.conv2d(W_pad * W_pad, self._kernel5, padding=0)         # (Mc,1,H,W)
            var_w = torch.clamp(mu_w2 - mu_w * mu_w, min=0.0)                 # (Mc,1,H,W)

            # 共分散: mean(x*w) - mu_x*mu_w
            prod = (X.unsqueeze(1) * Wc.unsqueeze(0)).reshape(B * (end - start), 1, H, W)  # (B*Mc,1,H,W)
            prod_pad = F.pad(prod, self._ssim_pad_tuple(), mode='reflect')
            mu_xw = F.conv2d(prod_pad, self._kernel5, padding=0).reshape(B, end - start, 1, H, W)  # (B,Mc,1,H,W)

            mu_x_b = mu_x.unsqueeze(1)                         # (B,1,1,H,W)
            mu_w_mc = mu_w.unsqueeze(0)                        # (1,Mc,1,H,W)
            var_x_b = var_x.unsqueeze(1)                       # (B,1,1,H,W)
            var_w_mc = var_w.unsqueeze(0)                      # (1,Mc,1,H,W)
            cov = mu_xw - (mu_x_b * mu_w_mc)                   # (B,Mc,1,H,W)

            # SSIMマップ（C1=C2=0だが分母にのみepsガード）
            l_num = 2 * (mu_x_b * mu_w_mc)
            l_den = (mu_x_b * mu_x_b + mu_w_mc * mu_w_mc)
            c_num = 2 * cov
            c_den = (var_x_b + var_w_mc)
            ssim_map = (l_num * c_num) / (l_den * c_den + eps)               # (B,Mc,1,H,W)

            # 空間平均
            ssim_avg = ssim_map.mean(dim=(2, 3, 4))                          # (B,Mc)
            out[:, start:end] = 1.0 - ssim_avg

        return out

    @torch.no_grad()
    def _kappa_field(self, X: Tensor) -> Tensor:
        """
        Curvature field κ = div(∇X/|∇X|) computed with centered differences on the inner common grid.
        Input: X (B,H,W), Output: (B,H-3,W-3)
        """
        eps = 1e-12
        B, H, W = X.shape
        dXdx = X[:, :, 1:] - X[:, :, :-1]        # (B,H,W-1)
        dXdy = X[:, 1:, :] - X[:, :-1, :]        # (B,H-1,W)
        gx = dXdx[:, :-1, :]                     # (B,H-1,W-1)
        gy = dXdy[:, :, :-1]                     # (B,H-1,W-1)
        mag = torch.sqrt(gx * gx + gy * gy + eps)
        nx = gx / (mag + eps)
        ny = gy / (mag + eps)
        dnx_dx = 0.5 * (nx[:, :, 2:] - nx[:, :, :-2])   # (B,H-1,W-3)
        dny_dy = 0.5 * (ny[:, 2:, :] - ny[:, :-2, :])   # (B,H-3,W-1)
        dnx_dx_c = dnx_dx[:, 1:-1, :]                   # (B,H-3,W-3)
        dny_dy_c = dny_dy[:, :, 1:-1]                   # (B,H-3,W-3)
        kappa = dnx_dx_c + dny_dy_c
        return kappa

    @torch.no_grad()
    def _kappa_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        Kappa curvature distance in [0,1]: D_k = 0.5 * Σ|κ(X)-κ(W)| / Σ max(|κ(X)|,|κ(W)|)
        Uses inner grid (H-3, W-3).
        """
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk
        eps = 1e-12
        kx = self._kappa_field(Xb)                               # (B,hk,wk)
        out = torch.empty((Xb.shape[0], self.m), device=Xb.device, dtype=self.dtype)
        for start in range(0, self.m, nodes_chunk):
            end = min(start + nodes_chunk, self.m)
            kw = self._kappa_field(self.weights[start:end])      # (Mc,hk,wk)
            diff = torch.abs(kx.unsqueeze(1) - kw.unsqueeze(0))  # (B,Mc,hk,wk)
            num = diff.flatten(2).sum(dim=2)                     # (B,Mc)
            den = torch.maximum(kx.abs().unsqueeze(1), kw.abs().unsqueeze(0)).flatten(2).sum(dim=2)
            out[:, start:end] = 0.5 * (num / (den + eps))
        return out


    @torch.no_grad()
    def _s1_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        Xb: (B,H,W) 戻り (B,m) S1距離
        """
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk
        B, H, W = Xb.shape
        dXdx = Xb[:, :, 1:] - Xb[:, :, :-1]  # (B,H,W-1)
        dXdy = Xb[:, 1:, :] - Xb[:, :-1, :]  # (B,H-1,W)

        out = torch.empty((B, self.m), device=Xb.device, dtype=self.dtype)
        dWdx_full = self.weights[:, :, 1:] - self.weights[:, :, :-1]
        dWdy_full = self.weights[:, 1:, :] - self.weights[:, :-1, :]

        for start in range(0, self.m, nodes_chunk):
            end = min(start + nodes_chunk, self.m)
            dWdx = dWdx_full[start:end]   # (Mc,H,W-1)
            dWdy = dWdy_full[start:end]   # (Mc,H-1,W)
            num_dx = (torch.abs(dWdx.unsqueeze(0) - dXdx.unsqueeze(1))).sum(dim=(2, 3))
            num_dy = (torch.abs(dWdy.unsqueeze(0) - dXdy.unsqueeze(1))).sum(dim=(2, 3))
            num = num_dx + num_dy

            den_dx = (torch.maximum(torch.abs(dWdx).unsqueeze(0), torch.abs(dXdx).unsqueeze(1))).sum(dim=(2, 3))
            den_dy = (torch.maximum(torch.abs(dWdy).unsqueeze(0), torch.abs(dXdy).unsqueeze(1))).sum(dim=(2, 3))
            denom = den_dx + den_dy
            s1 = 100.0 * num / (denom + 1e-12)
            out[:, start:end] = s1

        return out

    @torch.no_grad()
    def _s1k_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk
        d1 = self._s1_distance_batch(Xb, nodes_chunk=nodes_chunk)
        dk = self._kappa_distance_batch(Xb, nodes_chunk=nodes_chunk)
        eps = 1e-12
        d1_min = d1.min(dim=1, keepdim=True).values
        d1_max = d1.max(dim=1, keepdim=True).values
        dk_min = dk.min(dim=1, keepdim=True).values
        dk_max = dk.max(dim=1, keepdim=True).values
        d1n = (d1 - d1_min) / (d1_max - d1_min + eps)
        dkn = (dk - dk_min) / (dk_max - dk_min + eps)
        return torch.sqrt((d1n * d1n + dkn * dkn) / 2.0)



    @torch.no_grad()
    def _emd_to_ref(self, Xb: Tensor, ref: Tensor) -> Tensor:
        """
        Earth Mover's Distance d_W±(x,w) のGPU向け近似（エントロピー正則化 Sinkhorn）。
        速度最適化を追加:
          - ダウンサンプリング: adaptive_avg_pool2d により (H,W) → (H/α,W/α)（α=self.emd_downscale, 既定2）
          - AMP: CUDA環境での半精度/混合精度（self.emd_amp, 既定 True）
          - 早期収束: スケーリングベクトルの平均変化量が self.emd_tol（既定1e-3）未満でbreak
        正負の質量を分離し、各側で部分マッチング（スラック行/列）を balanced OT として解き、両側の平均を返す。
        """
        eps = 1e-12
        B, H, W = Xb.shape
        device, dtype = Xb.device, Xb.dtype
        sqrt2 = math.sqrt(2.0)

        # パラメータ（未設定なら既定値）
        emd_eps: float = float(getattr(self, "emd_epsilon", 0.03))     # 正則化強度
        emd_max_iter: int = int(getattr(self, "emd_max_iter", 200))    # 反復上限
        emd_chunk: int = int(getattr(self, "emd_chunk", 4096))         # cdist チャンク幅
        emd_downscale: int = int(getattr(self, "emd_downscale", 2))    # 距離専用ダウンサンプル係数(>=1)
        emd_tol: float = float(getattr(self, "emd_tol", 1e-3))         # 早期収束の閾値
        emd_amp: bool = bool(getattr(self, "emd_amp", False))           # AMP使用可否（CUDA時のみ有効）

        # ダウンサンプリング（計算規模を縮小）
        if emd_downscale > 1 and (H >= 4 or W >= 4):
            h2 = max(16, max(1, H // emd_downscale))
            w2 = max(16, max(1, W // emd_downscale))
            Xb = F.adaptive_avg_pool2d(Xb.unsqueeze(1), (h2, w2)).squeeze(1)            # (B,h2,w2)
            ref = F.adaptive_avg_pool2d(ref.unsqueeze(0).unsqueeze(0), (h2, w2)).squeeze(0).squeeze(0)  # (h2,w2)
            H, W = int(h2), int(w2)

        # グリッド座標（[0,1]^2）をキャッシュ
        if not hasattr(self, "_emd_coords_cache"):
            self._emd_coords_cache = {}
        key = (int(H), int(W), device)
        if key not in self._emd_coords_cache:
            yy = torch.linspace(0.0, 1.0, H, device=device, dtype=self.dtype)
            xx = torch.linspace(0.0, 1.0, W, device=device, dtype=self.dtype)
            Y, X = torch.meshgrid(yy, xx, indexing='ij')
            coords = torch.stack([Y.reshape(-1), X.reshape(-1)], dim=1)  # (N,2)
            self._emd_coords_cache[key] = coords
        coords = self._emd_coords_cache[key]  # (N,2)
        N = coords.shape[0]

        # AMP（混合精度）コンテキスト：CUDA時のみ有効
        amp_enabled = (device.type == 'cuda') and emd_amp
        with torch.amp.autocast('cuda', enabled=amp_enabled):
            # Precompute and cache full distance/kernel matrices to leverage VRAM
            if not hasattr(self, '_emd_D_cache'):
                self._emd_D_cache = {}
            if not hasattr(self, '_emd_K_cache'):
                self._emd_K_cache = {}
            if not hasattr(self, '_emd_KD_cache'):
                self._emd_KD_cache = {}

            # persistent cache dir (e.g., RESULT_DIR/data) - set externally as som.persist_cache_dir
            persist_dir = getattr(self, 'persist_cache_dir', None)
            if persist_dir is not None:
                os.makedirs(persist_dir, exist_ok=True)

            # Load or build D_full on CPU, key by (H,W)
            dkey = (int(H), int(W), 'cpu')
            D_full = None
            if dkey in self._emd_D_cache:
                D_full = self._emd_D_cache[dkey]
            else:
                D_path = None if persist_dir is None else os.path.join(persist_dir, f"D_{H}x{W}.pt")
                if D_path is not None and os.path.exists(D_path):
                    D_full = torch.load(D_path, map_location='cpu')
                else:
                    # compute deterministically in float32 outside autocast
                    with torch.amp.autocast('cuda', enabled=False):
                        D_dev = torch.cdist(coords.to(torch.float32), coords.to(torch.float32), p=2) / sqrt2
                    D_full = D_dev.detach().cpu()
                    if D_path is not None:
                        try:
                            torch.save(D_full, D_path)
                        except Exception:
                            pass
                self._emd_D_cache[dkey] = D_full
            # Load or build K_full and KD_full on CPU, key by (H,W,eps)
            eps_str = str(float(emd_eps)).replace('.', 'p')
            kkey = (int(H), int(W), 'cpu', eps_str)
            if kkey in self._emd_K_cache:
                K_cpu = self._emd_K_cache[kkey]
                KD_cpu = self._emd_KD_cache[kkey]
            else:
                K_path = None if persist_dir is None else os.path.join(persist_dir, f"K_{H}x{W}_eps{eps_str}.pt")
                KD_path = None if persist_dir is None else os.path.join(persist_dir, f"KD_{H}x{W}_eps{eps_str}.pt")
                if K_path is not None and KD_path is not None and os.path.exists(K_path) and os.path.exists(KD_path):
                    K_cpu = torch.load(K_path, map_location='cpu')
                    KD_cpu = torch.load(KD_path, map_location='cpu')
                else:
                    K_cpu = torch.exp(-(D_full / float(emd_eps)))
                    KD_cpu = K_cpu * D_full
                    if K_path is not None and KD_path is not None:
                        try:
                            torch.save(K_cpu, K_path)
                            torch.save(KD_cpu, KD_path)
                        except Exception:
                            pass
                self._emd_K_cache[kkey] = K_cpu
                self._emd_KD_cache[kkey] = KD_cpu

            # move to device/dtype for compute
            D_full = D_full.to(device=device, dtype=dtype, non_blocking=True)
            K_full = self._emd_K_cache[kkey].to(device=device, dtype=dtype, non_blocking=True)
            KD_full = self._emd_KD_cache[kkey].to(device=device, dtype=dtype, non_blocking=True)
            def _K_dot_v(v_real: Tensor, v_slack: Tensor, eps_reg: float) -> Tensor:
                # Use precomputed full kernel: K_full @ v + slack
                res = (K_full @ v_real.to(K_full.dtype)).to(dtype)
                res = res + v_slack.to(res.dtype)  # cost=0 → K_{i,slack}=1
                return res

            def _KT_dot_u(u_real: Tensor, u_slack: Tensor, eps_reg: float) -> Tensor:
                # Use precomputed full kernel: K_full^T @ u + slack
                res = (K_full.transpose(0, 1) @ u_real.to(K_full.dtype)).to(dtype)
                res = res + u_slack.to(res.dtype)  # cost=0 → K_{slack,j}=1
                return res

            def _cost_sum(u_real: Tensor, v_real: Tensor, eps_reg: float) -> Tensor:
                # Full-matrix form: u^T (K ⊙ D) v
                s = (u_real.to(KD_full.dtype) @ (KD_full @ v_real.to(KD_full.dtype))).to(dtype)
                return s

            def _emd_side_sinkhorn(m_src: Tensor, m_dst: Tensor) -> Tensor:
                # m_src, m_dst: (N,) on device. Returns scalar tensor distance in [0,1]
                M_src = float(m_src.sum().item()); M_dst = float(m_dst.sum().item())
                if M_src <= eps and M_dst <= eps:
                    return torch.zeros((), device=device, dtype=dtype)
                if (M_src <= eps and M_dst > eps) or (M_src > eps and M_dst <= eps):
                    return torch.ones((), device=device, dtype=dtype)

                if M_src >= M_dst:
                    # 目的側にスラック（列）: b = [b_real, b_slack], 合計 = M_src
                    a_real = m_src.clone()          # (N,)
                    b_real = m_dst.clone()          # (N,)
                    b_slack = torch.tensor(M_src - M_dst, device=device, dtype=dtype)
                    # 初期 u,v
                    u = torch.ones_like(a_real)
                    v_real = torch.ones_like(b_real)
                    v_slack = torch.ones((), device=device, dtype=dtype)
                    # 反復（早期収束）
                    for _ in range(emd_max_iter):
                        u_prev = u
                        v_prev = v_real
                        Kv = _K_dot_v(v_real, v_slack, emd_eps) + eps
                        u = a_real / Kv
                        KTu_real = _KT_dot_u(u, torch.zeros((), device=device, dtype=dtype), emd_eps) + eps
                        v_real = b_real / KTu_real
                        # slack 列: K^T u の slack 成分は Σ_i u_i
                        KTu_slack = u.sum() + eps
                        v_slack = b_slack / KTu_slack
                        # 早期収束チェック
                        du = (u - u_prev).abs().mean()
                        dv = (v_real - v_prev).abs().mean()
                        if float(torch.max(du, dv).item()) < emd_tol:
                            break
                    # コスト: 実セル間のみ。分母は M_match = M_dst
                    cost = _cost_sum(u, v_real, emd_eps)
                    denom = torch.tensor(M_dst, device=device, dtype=dtype)
                    res = (cost / (denom + eps)).clamp(0.0, 1.0)
                    return torch.nan_to_num(res, nan=1.0, posinf=1.0, neginf=0.0)
                else:
                    # 供給側にスラック（行）: a = [a_real, a_slack], 合計 = M_dst
                    a_real = m_src.clone()
                    a_slack = torch.tensor(M_dst - M_src, device=device, dtype=dtype)
                    b_real = m_dst.clone()
                    # 初期 u,v
                    u_real = torch.ones_like(a_real)
                    u_slack = torch.ones((), device=device, dtype=dtype)
                    v = torch.ones_like(b_real)
                    for _ in range(emd_max_iter):
                        u_prev = u_real
                        v_prev = v
                        # K v for real rows
                        Kv_real = torch.zeros((N,), device=device, dtype=coords.dtype)
                        for j0 in range(0, N, emd_chunk):
                            j1 = min(j0 + emd_chunk, N)
                            d = torch.cdist(coords, coords[j0:j1], p=2) / sqrt2  # (N, j1-j0)
                            Kv_real += (torch.exp(-d / emd_eps) @ v[j0:j1].to(d.dtype))
                        Kv_real = Kv_real.to(dtype)
                        Kv_slack = v.sum() + eps  # cost=0 → K_{slack,j}=1
                        u_real = a_real / (Kv_real + eps)
                        u_slack = a_slack / (Kv_slack)
                        # K^T u for real cols
                        KTu_real = torch.zeros((N,), device=device, dtype=coords.dtype)
                        for i0 in range(0, N, emd_chunk):
                            i1 = min(i0 + emd_chunk, N)
                            d = torch.cdist(coords[i0:i1], coords, p=2) / sqrt2  # (i1-i0, N)
                            K = torch.exp(-d / emd_eps)
                            KTu_real += (K.transpose(0, 1) @ u_real[i0:i1].to(K.dtype))
                        KTu_real = (KTu_real.to(dtype) + u_slack)  # slack 行からの寄与
                        v = b_real / (KTu_real + eps)
                        # 早期収束チェック
                        du = (u_real - u_prev).abs().mean()
                        dv = (v - v_prev).abs().mean()
                        if float(torch.max(du, dv).item()) < emd_tol:
                            break
                    # コスト: 実セル間のみ。分母は M_match = M_src
                    cost = _cost_sum(u_real, v, emd_eps)
                    denom = torch.tensor(M_src, device=device, dtype=dtype)
                    res = (cost / (denom + eps)).clamp(0.0, 1.0)
                    return torch.nan_to_num(res, nan=1.0, posinf=1.0, neginf=0.0)

            # バッチ処理（各サンプルで正負を分離して平均）- 全列同時SinkhornでGEMM活用
            Xp = torch.clamp(Xb, min=0.0).reshape(B, -1)       # (B,N)
            Xn_abs = torch.clamp(-Xb, min=0.0).reshape(B, -1)  # (B,N)
            Rp = torch.clamp(ref, min=0.0).reshape(-1)         # (N,)
            Rn_abs = torch.clamp(-ref, min=0.0).reshape(-1)    # (N,)

            def _K_dot_V(V_real: Tensor, v_slack: Tensor) -> Tensor:
                # V_real: (N,Bc), v_slack: (Bc,) -> (N,Bc)
                return (K_full @ V_real) + v_slack.unsqueeze(0)

            def _KT_dot_U(U_real: Tensor, u_slack: Tensor) -> Tensor:
                # U_real: (N,Bc), u_slack: (Bc,) -> (N,Bc)
                return (K_full.transpose(0, 1) @ U_real) + u_slack.unsqueeze(0)

            def _cost_sum_batch(U_real: Tensor, V_real: Tensor) -> Tensor:
                # returns (Bc,)
                return (U_real * (KD_full @ V_real)).sum(dim=0)

            def _emd_side_sinkhorn_batch(A_bN: Tensor, b_real_N: Tensor) -> Tensor:
                # A_bN: (B,N), b_real_N: (N,)
                Bc = A_bN.shape[0]
                out_side = torch.empty((Bc,), device=device, dtype=dtype)
                if Bc == 0:
                    return out_side
                # masses
                M_src = A_bN.sum(dim=1)           # (B,)
                M_dst = float(b_real_N.sum().item())

                # cases
                epsl = torch.tensor(eps, device=device, dtype=dtype)
                both_zero = (M_src <= epsl) & (M_dst <= eps)
                only_one_zero = ((M_src <= epsl) & (M_dst > eps)) | ((M_src > epsl) & (M_dst <= eps))
                maskA = (M_src > epsl) & (M_dst > eps) & (M_src >= M_dst)   # column slack (dest slack)
                maskB = (M_src > epsl) & (M_dst > eps) & (M_src <  M_dst)   # row slack (source slack)

                out_side[both_zero] = 0.0
                out_side[only_one_zero] = 1.0

                # common tensors
                aT = A_bN.transpose(0, 1)         # (N,B)

                # Case A: column slack, solve all columns in a single GEMM loop
                idxA = torch.nonzero(maskA, as_tuple=False).flatten()
                if idxA.numel() > 0:
                    aA = aT[:, idxA]                              # (N,BA)
                    b_slack = (M_src[idxA] - M_dst)               # (BA,)
                    # init
                    U = torch.ones_like(aA)                       # (N,BA)
                    V = torch.ones((N, idxA.numel()), device=device, dtype=dtype)
                    v_slack = torch.ones((idxA.numel(),), device=device, dtype=dtype)
                    bR = b_real_N.view(-1, 1).expand(-1, idxA.numel())  # (N,BA)
                    for _ in range(emd_max_iter):
                        U_prev = U
                        V_prev = V
                        Kv = _K_dot_V(V, v_slack) + eps
                        U = aA / Kv
                        KTu = _KT_dot_U(U, torch.zeros((idxA.numel(),), device=device, dtype=dtype)) + eps
                        V = bR / KTu
                        v_slack = b_slack / (U.sum(dim=0) + eps)
                        # early stop
                        if torch.max((U - U_prev).abs().mean(), (V - V_prev).abs().mean()) < emd_tol:
                            break
                    cost = _cost_sum_batch(U, V)                  # (BA,)
                    out_side[idxA] = (cost / (M_dst + eps)).clamp(0.0, 1.0)

                # Case B: row slack
                idxB = torch.nonzero(maskB, as_tuple=False).flatten()
                if idxB.numel() > 0:
                    aB = aT[:, idxB]                              # (N,BB)
                    a_slack = (M_dst - M_src[idxB])               # (BB,)
                    U = torch.ones_like(aB)                       # (N,BB)
                    u_slack = torch.ones((idxB.numel(),), device=device, dtype=dtype)
                    V = torch.ones((N, idxB.numel()), device=device, dtype=dtype)
                    bR = b_real_N.view(-1, 1).expand(-1, idxB.numel())  # (N,BB)
                    for _ in range(emd_max_iter):
                        U_prev = U
                        V_prev = V
                        Kv = (K_full @ V) + eps                   # (N,BB)
                        U = aB / Kv
                        u_slack = a_slack / (V.sum(dim=0) + eps)
                        KTu = (K_full.transpose(0, 1) @ U) + u_slack.unsqueeze(0) + eps
                        V = bR / KTu
                        if torch.max((U - U_prev).abs().mean(), (V - V_prev).abs().mean()) < emd_tol:
                            break
                    cost = _cost_sum_batch(U, V)                  # (BB,)
                    out_side[idxB] = (cost / (M_src[idxB] + eps)).clamp(0.0, 1.0)

                return out_side

            dp = _emd_side_sinkhorn_batch(Xp, Rp)    # (B,)
            dn = _emd_side_sinkhorn_batch(Xn_abs, Rn_abs)  # (B,)
            res = 0.5 * (dp + dn)
            return torch.nan_to_num(res, nan=1.0, posinf=1.0, neginf=0.0)

    @torch.no_grad()
    def _emd_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        EMD距離（バッチ vs 全ノード）: (B,m)
        GPU上のSinkhorn反復（エントロピー正則化OT）で計算するため高コスト（ε, 反復, chunk で調整）。
        """
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk
        B = Xb.shape[0]
        out = torch.empty((B, self.m), device=Xb.device, dtype=self.dtype)

        for start in range(0, self.m, nodes_chunk):
            end = min(start + nodes_chunk, self.m)
            for k in range(start, end):
                ref = self.weights[k]
                out[:, k] = self._emd_to_ref(Xb, ref)
        return torch.nan_to_num(out, nan=1.0, posinf=1.0, neginf=0.0)

    @torch.no_grad()
    def _distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        if self.activation_distance == 's1':
            return self._s1_distance_batch(Xb, nodes_chunk=nodes_chunk)
        elif self.activation_distance == 'euclidean':
            return self._euclidean_distance_batch(Xb)
        elif self.activation_distance == 'ssim5':
            return self._ssim5_distance_batch(Xb, nodes_chunk=nodes_chunk)
        elif self.activation_distance == 'kappa':
            return self._kappa_distance_batch(Xb, nodes_chunk=nodes_chunk)
        elif self.activation_distance == 's1k':
            return self._s1k_distance_batch(Xb, nodes_chunk=nodes_chunk)
        elif self.activation_distance == 'emd':
            return self._emd_distance_batch(Xb, nodes_chunk=nodes_chunk)
        else:
            raise RuntimeError('Unknown activation_distance')

    @torch.no_grad()
    def bmu_indices(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        dists = self._distance_batch(Xb, nodes_chunk=nodes_chunk)
        bmu = torch.argmin(dists, dim=1)
        return bmu

    # ---------- 距離計算（バッチ→単一参照：メドイド置換等で使用） ----------
    @torch.no_grad()
    def _euclidean_to_ref(self, Xb: Tensor, ref: Tensor) -> Tensor:
        # Xb: (B,H,W), ref: (H,W) -> (B,)
        diff = Xb - ref.view(1, *ref.shape)
        d2 = (diff * diff).sum(dim=(1, 2))
        return torch.sqrt(d2 + 1e-12)

    @torch.no_grad()
    def _ssim5_to_ref(self, Xb: Tensor, ref: Tensor) -> Tensor:
        # 1 - mean(SSIM_map(5x5, C=0)) 対参照
        self._ensure_kernel5()
        eps = 1e-12
        B, H, W = Xb.shape
        X = Xb.unsqueeze(1)
        R = ref.view(1, 1, H, W)

        X_pad = F.pad(X, (self._win5_pad, self._win5_pad, self._win5_pad, self._win5_pad), mode='reflect')
        R_pad = F.pad(R, (self._win5_pad, self._win5_pad, self._win5_pad, self._win5_pad), mode='reflect')

        mu_x = F.conv2d(X_pad, self._kernel5, padding=0)                 # (B,1,H,W)
        mu_r = F.conv2d(R_pad, self._kernel5, padding=0)                 # (1,1,H,W)

        mu_x2 = F.conv2d(X_pad * X_pad, self._kernel5, padding=0)
        mu_r2 = F.conv2d(R_pad * R_pad, self._kernel5, padding=0)
        var_x = torch.clamp(mu_x2 - mu_x * mu_x, min=0.0)
        var_r = torch.clamp(mu_r2 - mu_r * mu_r, min=0.0)

        mu_xr = F.conv2d(F.pad(X * R, (self._win5_pad, self._win5_pad, self._win5_pad, self._win5_pad), mode='reflect'),
                         self._kernel5, padding=0)
        cov = mu_xr - mu_x * mu_r

        l_num = 2 * (mu_x * mu_r)
        l_den = (mu_x * mu_x + mu_r * mu_r)
        c_num = 2 * cov
        c_den = (var_x + var_r)
        ssim_map = (l_num * c_num) / (l_den * c_den + eps)               # (B,1,H,W)
        ssim_avg = ssim_map.mean(dim=(1, 2, 3))                          # (B,)
        return 1.0 - ssim_avg

    @torch.no_grad()
    def _s1_to_ref(self, Xb: Tensor, ref: Tensor) -> Tensor:
        dXdx = Xb[:, :, 1:] - Xb[:, :, :-1]
        dXdy = Xb[:, 1:, :] - Xb[:, :-1, :]
        dRdx = ref[:, 1:] - ref[:, :-1]
        dRdy = ref[1:, :] - ref[:-1, :]
        num_dx = (torch.abs(dXdx - dRdx.view(1, *dRdx.shape))).sum(dim=(1, 2))
        num_dy = (torch.abs(dXdy - dRdy.view(1, *dRdy.shape))).sum(dim=(1, 2))
        den_dx = torch.maximum(torch.abs(dXdx), torch.abs(dRdx).view(1, *dRdx.shape)).sum(dim=(1, 2))
        den_dy = torch.maximum(torch.abs(dXdy), torch.abs(dRdy).view(1, *dRdy.shape)).sum(dim=(1, 2))
        s1 = 100.0 * (num_dx + num_dy) / (den_dx + den_dy + 1e-12)
        return s1


    @torch.no_grad()
    def _kappa_to_ref(self, Xb: Tensor, ref: Tensor) -> Tensor:
        """
        Kappa curvature distance to ref in [0,1]: D_k = 0.5 * Σ|κ(X)-κ(ref)| / Σ max(|κ(X)|,|κ(ref)|)
        """
        eps = 1e-12
        kx = self._kappa_field(Xb)                       # (B,hk,wk)
        kr = self._kappa_field(ref.unsqueeze(0)).squeeze(0)  # (hk,wk)
        num = torch.abs(kx - kr.unsqueeze(0)).flatten(1).sum(dim=1)
        den = torch.maximum(kx.abs(), kr.abs().unsqueeze(0)).flatten(1).sum(dim=1)
        return 0.5 * (num / (den + eps))

    @torch.no_grad()
    def _s1k_to_ref(self, Xb: Tensor, ref: Tensor) -> Tensor:
        eps = 1e-12
        d1 = self._s1_to_ref(Xb, ref)
        dk = self._kappa_to_ref(Xb, ref)
        d1_min, _ = d1.min(dim=0, keepdim=True)
        d1_max, _ = d1.max(dim=0, keepdim=True)
        dk_min, _ = dk.min(dim=0, keepdim=True)
        dk_max, _ = dk.max(dim=0, keepdim=True)
        d1n = (d1 - d1_min) / (d1_max - d1_min + eps)
        dkn = (dk - dk_min) / (dk_max - dk_min + eps)
        return torch.sqrt((d1n * d1n + dkn * dkn) / 2.0)



    @torch.no_grad()
    def _distance_to_ref(self, Xb: Tensor, ref: Tensor) -> Tensor:
        """
        現在のactivation_distanceに対応した「Xb vs 単一参照ref」の距離ベクトル(B,)
        """
        if self.activation_distance == 'euclidean':
            return self._euclidean_to_ref(Xb, ref)
        elif self.activation_distance == 'ssim5':
            return self._ssim5_to_ref(Xb, ref)
        elif self.activation_distance == 's1':
            return self._s1_to_ref(Xb, ref)
        elif self.activation_distance == 'kappa':
            return self._kappa_to_ref(Xb, ref)
        elif self.activation_distance == 's1k':
            return self._s1k_to_ref(Xb, ref)
        elif self.activation_distance == 'emd':
            return self._emd_to_ref(Xb, ref)
        else:
            raise RuntimeError('Unknown activation_distance')

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
        """
        σは self.total_iters を基準に self.global_iter + it で一方向に減衰。
        複数回に分けて呼んでも、set_total_iterations(total) 済みなら継続減衰します。
        """
        N, D = data.shape
        H, W = self.field_shape
        if D != H * W:
            raise ValueError(f'data dimension {D} != H*W {H*W}')
        Xall = _to_device(data, self.device, self.dtype).reshape(N, H, W)

        qhist: List[float] = []
        rng_idx = torch.arange(N, device=self.device)

        # total iters 未設定なら今回のnum_iterationを総回数とみなす
        if self.total_iters is None:
            self.total_iters = int(num_iteration)

        iterator = trange(num_iteration) if verbose else range(num_iteration)
        for it in iterator:
            # 学習全体での反復数に基づくσ
            sigma = self._sigma_at_val(self.global_iter + it, self.total_iters)

            numerator = torch.zeros_like(self.weights)  # (m,H,W)
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

                bmu = self.bmu_indices(Xb, nodes_chunk=self.nodes_chunk)    # (B,)
                h = self._neighborhood(bmu, sigma)                          # (B,m)
                numerator += (h.unsqueeze(-1).unsqueeze(-1) * Xb.unsqueeze(1)).sum(dim=0)
                denominator += h.sum(dim=0)

                if update_per_iteration:
                    mask = (denominator > 0)
                    denom_safe = denominator.clone()
                    denom_safe[~mask] = 1.0
                    new_w = numerator / denom_safe.view(-1, 1, 1)
                    self.weights[mask] = new_w[mask]
                    numerator.zero_(); denominator.zero_()

            # 1イテレーションの最後に一括更新
            mask = (denominator > 0)
            if mask.any():
                denom_safe = denominator.clone()
                denom_safe[~mask] = 1.0
                new_w = numerator / denom_safe.view(-1, 1, 1)
                self.weights[mask] = new_w[mask]

            # 任意頻度のメドイド置換（距離一貫性の改善）
            if (self.medoid_replace_every is not None) and (((self.global_iter + it + 1) % self.medoid_replace_every) == 0):
                # 全データでBMUを計算して各ノードの最近傍サンプルで置換
                bmu_all = self.bmu_indices(Xall, nodes_chunk=self.nodes_chunk)  # (N,)
                for node in range(self.m):
                    idxs = (bmu_all == node).nonzero(as_tuple=False).flatten()
                    if idxs.numel() == 0:
                        continue
                    Xn = Xall[idxs]                                   # (Bn,H,W)
                    ref = self.weights[node]                          # (H,W)
                    d = self._distance_to_ref(Xn, ref)                # (Bn,)
                    pos = int(torch.argmin(d).item())
                    self.weights[node] = Xn[pos]

            # ログ用QE
            if (it % log_interval == 0) or (it == num_iteration - 1):
                qe = self.quantization_error(data, sample_limit=2048)
                qhist.append(qe)
                if verbose and hasattr(iterator, 'set_postfix'):
                    iterator.set_postfix(q_error=f"{qe:.6f}", sigma=f"{sigma:.3f}")

        # グローバル反復を進める
        self.global_iter += num_iteration

        return qhist

    # ---------- 評価 ----------
    @torch.no_grad()
    def quantization_error(self, data: np.ndarray, sample_limit: Optional[int] = None, batch_size: int = 64) -> float:
        N, D = data.shape
        H, W = self.field_shape
        if self.eval_indices is not None:
            # 固定評価
            X = _to_device(data, self.device, self.dtype)[self.eval_indices].reshape(-1, H, W)
        else:
            if sample_limit is not None and sample_limit < N:
                idx = np.random.choice(N, sample_limit, replace=False)
                X = data[idx]
            else:
                X = data
            X = _to_device(X, self.device, self.dtype).reshape(-1, H, W)
        total = 0.0; cnt = 0
        for start in range(0, X.shape[0], batch_size):
            end = min(start + batch_size, X.shape[0])
            Xb = X[start:end]
            d = self._distance_batch(Xb, nodes_chunk=self.nodes_chunk)
            mins = torch.min(d, dim=1).values
            total += float(mins.sum().item())
            cnt += Xb.shape[0]
        return total / max(1, cnt)

    @torch.no_grad()
    def predict(self, data: np.ndarray, batch_size: int = 64) -> np.ndarray:
        N, D = data.shape
        H, W = self.field_shape
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
