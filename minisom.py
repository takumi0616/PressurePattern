import math
import numpy as np
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
      - activation_distance:
          'euclidean'（ユークリッド）
          'ssim5'    （論文仕様に近い 5x5 窓・C=0 のSSIM：移動窓平均）
          's1'       （Teweles–Wobus S1）
          's1ssim'   （S1とSSIM(5x5)の融合距離：サンプル毎min-max正規化後の等重み和）
          's1ssim5_hf'（HF-S1SSIM5: SSIM(5x5)でゲートするソフト階層化 D = u + (1-u)v）
          's1ssim5_and'（AND合成: 行方向min–max正規化後の D = max(U,V)）
          'pf_s1ssim'（比例融合: 正規化なしで D = dS1 * dSSIM）
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

        # 学習全体の反復管理（σ継続減衰用）
        self.global_iter: int = 0
        self.total_iters: Optional[int] = None

        # メドイド置換頻度（None: 不使用, k: k反復ごと）
        self.medoid_replace_every: Optional[int] = None

        # 評価用の固定サンプルインデックス（QEを安定化）
        self.eval_indices: Optional[Tensor] = None

        # 距離タイプ
        activation_distance = activation_distance.lower()
        if activation_distance not in ('s1', 'euclidean', 'ssim5', 's1ssim', 's1ssim5_hf', 's1ssim5_and', 'pf_s1ssim', 's1gssim', 'gssim'):
            raise ValueError('activation_distance must be one of "s1","euclidean","ssim5","s1ssim","s1ssim5_hf","s1ssim5_and","pf_s1ssim","s1gssim","gssim"')
        self.activation_distance = activation_distance

        # 画像形状
        if s1_field_shape is None:
            raise ValueError('s1_field_shape=(H,W) is required for all distances in this implementation.')
        if s1_field_shape[0] * s1_field_shape[1] != input_len:
            raise ValueError(f's1_field_shape={s1_field_shape} does not match input_len={input_len}.')
        self.field_shape = s1_field_shape
        H, W = s1_field_shape

        # グリッド座標
        gx, gy = torch.meshgrid(torch.arange(x), torch.arange(y), indexing='ij')
        self.grid_coords = torch.stack([gx.flatten(), gy.flatten()], dim=1).to(self.device, torch.float32)

        # 重み（(m,H,W)）
        self.weights = (torch.rand((self.m, H, W), device=self.device, dtype=self.dtype) * 2 - 1)

        if self.neighborhood_function != 'gaussian':
            self.neighborhood_function = 'gaussian'


        # 5x5移動窓SSIM用カーネル（平均フィルタ）
        self._kernel5: Optional[Tensor] = None
        self._win5_size: int = 5
        self._win5_pad: int = 2

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
    def _ensure_kernel5(self):
        if self._kernel5 is None:
            k = torch.ones((1, 1, self._win5_size, self._win5_size), device=self.device, dtype=self.dtype) / float(self._win5_size * self._win5_size)
            self._kernel5 = k

    @torch.no_grad()
    def _ssim5_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        論文仕様に近いSSIM: 5x5移動窓・C=0（分母のみ数値安定化）
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
        X_pad = F.pad(X, (self._win5_pad, self._win5_pad, self._win5_pad, self._win5_pad), mode='reflect')
        mu_x = F.conv2d(X_pad, self._kernel5, padding=0)                      # (B,1,H,W)
        mu_x2 = F.conv2d(X_pad * X_pad, self._kernel5, padding=0)             # (B,1,H,W)
        var_x = torch.clamp(mu_x2 - mu_x * mu_x, min=0.0)                     # (B,1,H,W)

        for start in range(0, self.m, nodes_chunk):
            end = min(start + nodes_chunk, self.m)
            Wc = self.weights[start:end].unsqueeze(1)                          # (Mc,1,H,W)
            W_pad = F.pad(Wc, (self._win5_pad, self._win5_pad, self._win5_pad, self._win5_pad), mode='reflect')
            mu_w = F.conv2d(W_pad, self._kernel5, padding=0)                  # (Mc,1,H,W)
            mu_w2 = F.conv2d(W_pad * W_pad, self._kernel5, padding=0)         # (Mc,1,H,W)
            var_w = torch.clamp(mu_w2 - mu_w * mu_w, min=0.0)                 # (Mc,1,H,W)

            # 共分散: mean(x*w) - mu_x*mu_w
            prod = (X.unsqueeze(1) * Wc.unsqueeze(0)).reshape(B * (end - start), 1, H, W)  # (B*Mc,1,H,W)
            prod_pad = F.pad(prod, (self._win5_pad, self._win5_pad, self._win5_pad, self._win5_pad), mode='reflect')
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
    def _s1gssim_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        Gradient-SSIM(+direction) と S1(行方向min-max正規化) のRMS合成距離。
        1) 勾配強度 |∇| のSSIM(5x5, C=0) => d_g ∈ [0,1]
        2) 勾配方向cosθの重み付き平均（重み=max(|∇X|,|∇W|)）=> d_dir ∈ [0,1]
        3) d_edge = 0.5*(d_g + d_dir)
        4) d_s1 = 行方向min-max正規化したS1 ∈ [0,1]
        出力: sqrt((d_edge^2 + d_s1^2)/2)
        """
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk
        eps = 1e-12
        B, H, W = Xb.shape

        # サンプル側の勾配と強度（共通領域 (H-1, W-1)）
        dXdx = Xb[:, :, 1:] - Xb[:, :, :-1]   # (B, H, W-1)
        dXdy = Xb[:, 1:, :] - Xb[:, :-1, :]   # (B, H-1, W)
        dXdx_c = dXdx[:, :-1, :]              # (B, H-1, W-1)
        dXdy_c = dXdy[:, :, :-1]              # (B, H-1, W-1)
        magX = torch.sqrt(dXdx_c * dXdx_c + dXdy_c * dXdy_c + eps)  # (B, H-1, W-1)

        # 勾配強度のローカル統計（5x5平均畳み込み）
        self._ensure_kernel5()
        Xg = magX.unsqueeze(1)  # (B,1,H-1,W-1)
        Xg_pad = F.pad(Xg, (2, 2, 2, 2), mode='reflect')
        mu_xg = F.conv2d(Xg_pad, self._kernel5, padding=0)                  # (B,1,H-1,W-1)
        mu_xg2 = F.conv2d(Xg_pad * Xg_pad, self._kernel5, padding=0)
        var_xg = torch.clamp(mu_xg2 - mu_xg * mu_xg, min=0.0)

        # ノード側の勾配（事前計算）
        dWdx_full = self.weights[:, :, 1:] - self.weights[:, :, :-1]  # (m, H, W-1)
        dWdy_full = self.weights[:, 1:, :] - self.weights[:, :-1, :]  # (m, H-1, W)

        # S1全体を先に計算して行方向min-max正規化用のmin/maxを得る
        dS1_all = self._s1_distance_batch(Xb, nodes_chunk=self.nodes_chunk)  # (B, m)
        min_s1 = dS1_all.min(dim=1, keepdim=True).values
        max_s1 = dS1_all.max(dim=1, keepdim=True).values

        out = torch.empty((B, self.m), device=Xb.device, dtype=self.dtype)
        for start in range(0, self.m, nodes_chunk):
            end = min(start + nodes_chunk, self.m)
            dWdx = dWdx_full[start:end]    # (Mc, H, W-1)
            dWdy = dWdy_full[start:end]    # (Mc, H-1, W)
            dWdx_c = dWdx[:, :-1, :]       # (Mc, H-1, W-1)
            dWdy_c = dWdy[:, :, :-1]       # (Mc, H-1, W-1)
            magW = torch.sqrt(dWdx_c * dWdx_c + dWdy_c * dWdy_c + eps)  # (Mc, H-1, W-1)

            # 1) 勾配強度のSSIM (C=0)
            Wg = magW.unsqueeze(1)                                         # (Mc,1,H-1,W-1)
            Wg_pad = F.pad(Wg, (2, 2, 2, 2), mode='reflect')
            mu_wg = F.conv2d(Wg_pad, self._kernel5, padding=0)             # (Mc,1,H-1,W-1)
            mu_wg2 = F.conv2d(Wg_pad * Wg_pad, self._kernel5, padding=0)
            var_wg = torch.clamp(mu_wg2 - mu_wg * mu_wg, min=0.0)

            # 修正: ブロードキャスト次元を合わせるため Xg にもノード次元を追加（(B,1,1,H-1,W-1) × (1,Mc,1,H-1,W-1)）
            prod = (Xg.unsqueeze(1) * Wg.unsqueeze(0)).reshape(B * (end - start), 1, magX.shape[1], magX.shape[2])
            prod_pad = F.pad(prod, (2, 2, 2, 2), mode='reflect')
            mu_xwg = F.conv2d(prod_pad, self._kernel5, padding=0).reshape(B, end - start, 1, magX.shape[1], magX.shape[2])

            mu_xg_b = mu_xg.unsqueeze(1)       # (B,1,1,H-1,W-1)
            mu_wg_mc = mu_wg.unsqueeze(0)      # (1,Mc,1,H-1,W-1)
            var_xg_b = var_xg.unsqueeze(1)
            var_wg_mc = var_wg.unsqueeze(0)
            l_num = 2.0 * (mu_xg_b * mu_wg_mc)
            l_den = (mu_xg_b * mu_xg_b + mu_wg_mc * mu_wg_mc)
            c_num = 2.0 * (mu_xwg - mu_xg_b * mu_wg_mc)
            c_den = (var_xg_b + var_wg_mc)
            ssim_map = (l_num * c_num) / (l_den * c_den + eps)             # (B,Mc,1,H-1,W-1)
            ssim_avg = ssim_map.mean(dim=(2, 3, 4))                        # (B,Mc)
            d_g = 0.5 * (1.0 - ssim_avg)                                   # (B,Mc) in [0,1]

            # 2) 勾配方向の一致（cosθの重み付き平均）
            magW2 = torch.sqrt(dWdx_c * dWdx_c + dWdy_c * dWdy_c + eps)    # (Mc, H-1, W-1)
            dot = dXdx_c.unsqueeze(1) * dWdx_c.unsqueeze(0) + dXdy_c.unsqueeze(1) * dWdy_c.unsqueeze(0)  # (B,Mc,H-1,W-1)
            denom = magX.unsqueeze(1) * magW2.unsqueeze(0) + eps
            cos = (dot / denom).clamp(-1.0, 1.0)
            wgt = torch.maximum(magX.unsqueeze(1), magW2.unsqueeze(0))
            s_dir = (cos * wgt).sum(dim=(2, 3)) / (wgt.sum(dim=(2, 3)) + eps)   # (B,Mc)
            d_dir = 0.5 * (1.0 - s_dir)                                         # (B,Mc) in [0,1]

            d_edge = 0.5 * (d_g + d_dir)                                        # (B,Mc)

            # 3) S1 を行方向min-max正規化し、RMS合成
            dS1n = (dS1_all[:, start:end] - min_s1) / (max_s1 - min_s1 + eps)   # (B,Mc)
            out[:, start:end] = torch.sqrt((d_edge * d_edge + dS1n * dS1n) / 2.0)

        return out

    @torch.no_grad()
    def _gssim_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        Gradient-Structure Similarity (G-SSIM-S1の勾配構造部分) に基づく距離。
        S_GS = Σ[w · S_mag · S_dir] / (Σ w + ε),  D = 1 - S_GS
          - S_mag = 2|∇X||∇W| / (|∇X|^2 + |∇W|^2 + ε)
          - S_dir = (1 + cosθ)/2,  cosθ = (∇X·∇W)/(||∇X||||∇W|| + ε)
          - w = max(|∇X|, |∇W|)
        """
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk
        eps = 1e-12
        B, H, W = Xb.shape

        # サンプル側勾配（共通領域）
        dXdx = Xb[:, :, 1:] - Xb[:, :, :-1]
        dXdy = Xb[:, 1:, :] - Xb[:, :-1, :]
        gx = dXdx[:, :-1, :]
        gy = dXdy[:, :, :-1]
        gmagX = torch.sqrt(gx * gx + gy * gy + eps)  # (B, H-1, W-1)

        out = torch.empty((B, self.m), device=Xb.device, dtype=self.dtype)

        dWdx_full = self.weights[:, :, 1:] - self.weights[:, :, :-1]  # (m, H, W-1)
        dWdy_full = self.weights[:, 1:, :] - self.weights[:, :-1, :]  # (m, H-1, W)

        for start in range(0, self.m, nodes_chunk):
            end = min(start + nodes_chunk, self.m)
            grx = dWdx_full[start:end, :-1, :]   # (Mc, H-1, W-1)
            gry = dWdy_full[start:end, :, :-1]   # (Mc, H-1, W-1)
            gmagW = torch.sqrt(grx * grx + gry * gry + eps)  # (Mc, H-1, W-1)

            gx_b = gx.unsqueeze(1)     # (B,1,H-1,W-1)
            gy_b = gy.unsqueeze(1)
            gX_b = gmagX.unsqueeze(1)  # (B,1,H-1,W-1)
            grx_m = grx.unsqueeze(0)   # (1,Mc,H-1,W-1)
            gry_m = gry.unsqueeze(0)
            gW_m = gmagW.unsqueeze(0)  # (1,Mc,H-1,W-1)

            dot = gx_b * grx_m + gy_b * gry_m
            cos = (dot / (gX_b * gW_m + eps)).clamp(-1.0, 1.0)
            Sdir = 0.5 * (1.0 + cos)
            Smag = (2.0 * gX_b * gW_m) / (gX_b * gX_b + gW_m * gW_m + eps)
            S = Smag * Sdir
            w = torch.maximum(gX_b, gW_m)
            sim = (S * w).sum(dim=(2, 3)) / (w.sum(dim=(2, 3)) + eps)  # (B,Mc)
            out[:, start:end] = 1.0 - sim

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
    def _distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        if self.activation_distance == 's1':
            return self._s1_distance_batch(Xb, nodes_chunk=nodes_chunk)
        elif self.activation_distance == 'euclidean':
            return self._euclidean_distance_batch(Xb)
        elif self.activation_distance == 'ssim5':
            return self._ssim5_distance_batch(Xb, nodes_chunk=nodes_chunk)
        elif self.activation_distance == 's1ssim':
            # 融合：各サンプル毎にノード方向min-max正規化して等重み和
            d1 = self._s1_distance_batch(Xb, nodes_chunk=nodes_chunk)    # (B,m)
            d2 = self._ssim5_distance_batch(Xb, nodes_chunk=nodes_chunk) # (B,m)
            min1 = d1.min(dim=1, keepdim=True).values
            max1 = d1.max(dim=1, keepdim=True).values
            min2 = d2.min(dim=1, keepdim=True).values
            max2 = d2.max(dim=1, keepdim=True).values
            dn1 = (d1 - min1) / (max1 - min1 + 1e-12)
            dn2 = (d2 - min2) / (max2 - min2 + 1e-12)
            return 0.5 * (dn1 + dn2)
        elif self.activation_distance == 's1ssim5_hf':
            # HF-S1SSIM5: SSIM(5x5)でゲートするソフト階層化 D = u + (1-u)*v
            dS1 = self._s1_distance_batch(Xb, nodes_chunk=nodes_chunk)      # (B,m)
            dSSIM = self._ssim5_distance_batch(Xb, nodes_chunk=nodes_chunk) # (B,m) = 1-SSIM
            eps = 1e-12
            min_s1 = dS1.min(dim=1, keepdim=True).values
            max_s1 = dS1.max(dim=1, keepdim=True).values
            min_ss = dSSIM.min(dim=1, keepdim=True).values
            max_ss = dSSIM.max(dim=1, keepdim=True).values
            v = (dS1 - min_s1) / (max_s1 - min_s1 + eps)   # normalized S1
            u = (dSSIM - min_ss) / (max_ss - min_ss + eps) # normalized dSSIM
            return u + (1.0 - u) * v
        elif self.activation_distance == 's1ssim5_and':
            # AND合成: 行方向min–max正規化 U,V を用いて D = max(U,V)
            dS1 = self._s1_distance_batch(Xb, nodes_chunk=nodes_chunk)
            dSSIM = self._ssim5_distance_batch(Xb, nodes_chunk=nodes_chunk)
            eps = 1e-12
            min_s1 = dS1.min(dim=1, keepdim=True).values
            max_s1 = dS1.max(dim=1, keepdim=True).values
            min_ss = dSSIM.min(dim=1, keepdim=True).values
            max_ss = dSSIM.max(dim=1, keepdim=True).values
            V = (dS1 - min_s1) / (max_s1 - min_s1 + eps)
            U = (dSSIM - min_ss) / (max_ss - min_ss + eps)
            return torch.maximum(U, V)
        elif self.activation_distance == 'pf_s1ssim':
            # 比例融合: 正規化なしで積 D = dS1 * dSSIM
            dS1 = self._s1_distance_batch(Xb, nodes_chunk=nodes_chunk)
            dSSIM = self._ssim5_distance_batch(Xb, nodes_chunk=nodes_chunk)
            return dS1 * dSSIM
        elif self.activation_distance == 's1gssim':
            return self._s1gssim_distance_batch(Xb, nodes_chunk=nodes_chunk)
        elif self.activation_distance == 'gssim':
            return self._gssim_distance_batch(Xb, nodes_chunk=nodes_chunk)
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
    def _s1gssim_to_ref(self, Xb: Tensor, ref: Tensor) -> Tensor:
        """
        s1gssim の対参照距離：
          d_edge = 0.5*(d_g + d_dir),  d_s1 = 行方向min-max正規化したS1
          D = sqrt((d_edge^2 + d_s1^2)/2)
        """
        eps = 1e-12
        B, H, W = Xb.shape

        # 勾配（共通領域）
        dXdx = Xb[:, :, 1:] - Xb[:, :, :-1]
        dXdy = Xb[:, 1:, :] - Xb[:, :-1, :]
        dRdx = ref[:, 1:] - ref[:, :-1]
        dRdy = ref[1:, :] - ref[:-1, :]

        dXdx_c, dXdy_c = dXdx[:, :-1, :], dXdy[:, :, :-1]
        dRdx_c, dRdy_c = dRdx[:-1, :], dRdy[:, :-1]
        magX = torch.sqrt(dXdx_c * dXdx_c + dXdy_c * dXdy_c + eps)  # (B, H-1, W-1)
        magR = torch.sqrt(dRdx_c * dRdx_c + dRdy_c * dRdy_c + eps)  # (H-1, W-1)

        # 勾配強度のSSIM(5x5, C=0)
        self._ensure_kernel5()
        Xg = magX.unsqueeze(1)                          # (B,1,.,.)
        Rg = magR.unsqueeze(0).unsqueeze(1)             # (1,1,.,.)
        Xg_pad = F.pad(Xg, (2, 2, 2, 2), mode='reflect')
        Rg_pad = F.pad(Rg, (2, 2, 2, 2), mode='reflect')
        mu_x = F.conv2d(Xg_pad, self._kernel5, padding=0)
        mu_r = F.conv2d(Rg_pad, self._kernel5, padding=0)
        mu_x2 = F.conv2d(Xg_pad * Xg_pad, self._kernel5, padding=0)
        mu_r2 = F.conv2d(Rg_pad * Rg_pad, self._kernel5, padding=0)
        var_x = torch.clamp(mu_x2 - mu_x * mu_x, min=0.0)
        var_r = torch.clamp(mu_r2 - mu_r * mu_r, min=0.0)
        mu_xr = F.conv2d(F.pad(Xg * Rg, (2, 2, 2, 2), mode='reflect'), self._kernel5, padding=0)
        cov = mu_xr - mu_x * mu_r
        ssim_map = (2 * mu_x * mu_r * 2 * cov) / ((mu_x * mu_x + mu_r * mu_r) * (var_x + var_r) + eps)
        ssim_avg = ssim_map.mean(dim=(1, 2, 3))
        d_g = 0.5 * (1.0 - ssim_avg)                    # (B,)

        # 勾配方向（加重平均）
        dot = dXdx_c * dRdx_c.unsqueeze(0) + dXdy_c * dRdy_c.unsqueeze(0)     # (B,.,.)
        denom = magX * magR.unsqueeze(0) + eps
        cos = (dot / denom).clamp(-1.0, 1.0)
        wgt = torch.maximum(magX, magR.unsqueeze(0))
        s_dir = (cos * wgt).flatten(1).sum(dim=1) / (wgt.flatten(1).sum(dim=1) + eps)     # (B,)
        d_dir = 0.5 * (1.0 - s_dir)

        d_edge = 0.5 * (d_g + d_dir)                    # (B,)

        # S1 の行方向min-max正規化（バッチ内）
        dS1 = self._s1_to_ref(Xb, ref)                  # (B,)
        min_s1, _ = dS1.min(dim=0, keepdim=True)
        max_s1, _ = dS1.max(dim=0, keepdim=True)
        dS1n = (dS1 - min_s1) / (max_s1 - min_s1 + eps)

        return torch.sqrt((d_edge * d_edge + dS1n * dS1n) / 2.0)

    @torch.no_grad()
    def _gssim_to_ref(self, Xb: Tensor, ref: Tensor) -> Tensor:
        """
        gssim の対参照距離： D = 1 - S_GS （勾配強度・方向の重み付き類似度）
        """
        eps = 1e-12
        B, H, W = Xb.shape
        dXdx = Xb[:, :, 1:] - Xb[:, :, :-1]
        dXdy = Xb[:, 1:, :] - Xb[:, :-1, :]
        gx = dXdx[:, :-1, :]
        gy = dXdy[:, :, :-1]
        gmagX = torch.sqrt(gx * gx + gy * gy + eps)     # (B, H-1, W-1)

        dRdx = ref[:, 1:] - ref[:, :-1]
        dRdy = ref[1:, :] - ref[:-1, :]
        grx = dRdx[:-1, :]
        gry = dRdy[:, :-1]
        gmagR = torch.sqrt(grx * grx + gry * gry + eps) # (H-1, W-1)

        dot = gx * grx.unsqueeze(0) + gy * gry.unsqueeze(0)                     # (B,.,.)
        cos = (dot / (gmagX * gmagR.unsqueeze(0) + eps)).clamp(-1.0, 1.0)
        Sdir = 0.5 * (1.0 + cos)
        Smag = (2.0 * gmagX * gmagR.unsqueeze(0)) / (gmagX * gmagX + gmagR.unsqueeze(0) * gmagR.unsqueeze(0) + eps)
        S = Smag * Sdir
        w = torch.maximum(gmagX, gmagR.unsqueeze(0))
        sim = (S * w).flatten(1).sum(dim=1) / (w.flatten(1).sum(dim=1) + eps)   # (B,)
        return 1.0 - sim

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
        elif self.activation_distance == 's1ssim':
            d1 = self._s1_to_ref(Xb, ref)
            d2 = self._ssim5_to_ref(Xb, ref)
            min1, _ = d1.min(dim=0, keepdim=True)
            max1, _ = d1.max(dim=0, keepdim=True)
            min2, _ = d2.min(dim=0, keepdim=True)
            max2, _ = d2.max(dim=0, keepdim=True)
            dn1 = (d1 - min1) / (max1 - min1 + 1e-12)
            dn2 = (d2 - min2) / (max2 - min2 + 1e-12)
            return 0.5 * (dn1 + dn2)
        elif self.activation_distance == 's1ssim5_hf':
            # HF-S1SSIM5 (to ref): 行方向（バッチ内）でmin-max正規化後、D = u + (1-u)*v
            dS1 = self._s1_to_ref(Xb, ref)       # (B,)
            dSSIM = self._ssim5_to_ref(Xb, ref)  # (B,)
            eps = 1e-12
            min_s1, _ = dS1.min(dim=0, keepdim=True)
            max_s1, _ = dS1.max(dim=0, keepdim=True)
            min_ss, _ = dSSIM.min(dim=0, keepdim=True)
            max_ss, _ = dSSIM.max(dim=0, keepdim=True)
            v = (dS1 - min_s1) / (max_s1 - min_s1 + eps)    # normalized S1
            u = (dSSIM - min_ss) / (max_ss - min_ss + eps)  # normalized dSSIM
            return u + (1.0 - u) * v
        elif self.activation_distance == 's1ssim5_and':
            # AND合成 (to ref): 行方向min–max正規化後、D = max(U,V)
            dS1 = self._s1_to_ref(Xb, ref)
            dSSIM = self._ssim5_to_ref(Xb, ref)
            eps = 1e-12
            min_s1, _ = dS1.min(dim=0, keepdim=True)
            max_s1, _ = dS1.max(dim=0, keepdim=True)
            min_ss, _ = dSSIM.min(dim=0, keepdim=True)
            max_ss, _ = dSSIM.max(dim=0, keepdim=True)
            V = (dS1 - min_s1) / (max_s1 - min_s1 + eps)
            U = (dSSIM - min_ss) / (max_ss - min_ss + eps)
            return torch.maximum(U, V)
        elif self.activation_distance == 'pf_s1ssim':
            # 比例融合 (to ref): 正規化なしで積
            dS1 = self._s1_to_ref(Xb, ref)
            dSSIM = self._ssim5_to_ref(Xb, ref)
            return dS1 * dSSIM
        elif self.activation_distance == 's1gssim':
            return self._s1gssim_to_ref(Xb, ref)
        elif self.activation_distance == 'gssim':
            return self._gssim_to_ref(Xb, ref)
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

            # ログ用QE（固定サブセットを外部から渡すのが推奨だが、API互換のためここはそのまま）
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
