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
    """
    概要:
      NumPy配列を指定デバイス・dtypeのtorch.Tensorへ変換する薄いユーティリティ。
    引数:
      - x (np.ndarray): 変換対象の配列。
      - device (torch.device): 配置先デバイス（CPU/GPU）。
      - dtype (torch.dtype): 変換後のデータ型（既定: torch.float32）。
    処理の詳細:
      - torch.as_tensorでコピー/ビューを作成し、device/dtypeを指定。
    戻り値:
      - Tensor: 指定条件のTensor。
    """
    return torch.as_tensor(x, device=device, dtype=dtype)


def _as_numpy(x: Tensor) -> np.ndarray:
    """
    概要:
      PyTorch Tensor を NumPy 配列へ安全に変換する。
    引数:
      - x (Tensor): 変換対象のテンソル。
    処理の詳細:
      - .detach() で計算グラフから切り離し、.cpu() でCPUへ移動してから .numpy() を呼ぶ。
    戻り値:
      - np.ndarray: 変換結果のNumPy配列。
    """
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
          'msssim'   （マルチスケールSSIM: D=1-Π_s SSIM_s^{w_s}）
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
        allowed_distances = ('s1', 'euclidean', 'ssim5', 'kappa', 's1k', 'msssim',
                            'msssim_kappa', 'ssim5_kappa', 'msssim_s1', 'ssim5_s1')
        if activation_distance not in allowed_distances:
            raise ValueError(f'activation_distance must be one of {allowed_distances}')
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
        """
        概要:
          学習全体で見込む総反復回数を設定し、σ（近傍幅）の減衰スケジュールの基準にする。
        引数:
          - total_iters (int): 学習全体の反復回数（train_batchを分割呼び出しする場合も合計値）。
        処理の詳細:
          - self.total_iters を更新。train_batch 内で self.global_iter と合わせて σ を計算。
        戻り値:
          - なし
        """
        self.total_iters = int(total_iters)

    def set_medoid_replace_every(self, k: Optional[int]):
        """
        概要:
          k 反復ごとに“メドイド置換”を行う頻度を設定する。None/0 の場合は無効。
        引数:
          - k (Optional[int]): 置換頻度（例: 100 なら100反復ごと）。None または 0 で無効。
        処理の詳細:
          - train_batch の各反復末に条件を満たせば、各ノードの重みを
            そのノードに割り当てられたサンプルの中で最も距離が近いサンプルへ置き換える。
        戻り値:
          - なし
        """
        if k is None or k <= 0:
            self.medoid_replace_every = None
        else:
            self.medoid_replace_every = int(k)

    def set_eval_indices(self, idx: Optional[np.ndarray]):
        """
        概要:
          評価時（quantization_error/predict等）に用いる固定サンプルの行インデックスを設定する。
        引数:
          - idx (Optional[np.ndarray]): データ配列に対する行インデックス配列。None で固定評価を解除。
        処理の詳細:
          - self.eval_indices をTensor化して保持。quantization_error でサンプリングの揺らぎを抑える用途に有効。
        戻り値:
          - なし
        """
        if idx is None:
            self.eval_indices = None
        else:
            self.eval_indices = torch.as_tensor(idx, device=self.device, dtype=torch.long)

    # ---------- ユーティリティ ----------
    def get_weights(self) -> np.ndarray:
        """
        概要:
          現在のSOMノード重みを (x, y, H*W) 形状のNumPy配列として取得する。
        引数:
          - なし（内部状態から取得）
        処理の詳細:
          - (m,H,W) の重みを (x,y,H*W) に整形し、CPUへ移してNumPy化。
        戻り値:
          - np.ndarray: 形状 (x, y, H*W) の重み配列。
        """
        H, W = self.field_shape
        w_flat = self.weights.reshape(self.m, H * W)
        w_grid = w_flat.reshape(self.x, self.y, H * W)
        return _as_numpy(w_grid)

    def random_weights_init(self, data: np.ndarray):
        """
        概要:
          入力データからランダムにサンプリングして、SOM重みを初期化する。
        引数:
          - data (np.ndarray): 形状 (N, H*W) の入力データ。
        処理の詳細:
          - N < m なら重複ありで、N >= m なら重複なしで m サンプルを抽出し、(m,H,W) に整形して設定。
        戻り値:
          - なし
        """
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
        """
        概要:
          学習反復 t における近傍幅 σ を減衰スケジュールに基づいて計算する。
        引数:
          - t (int): 現在の学習反復（グローバル）。
          - max_iter (int): 想定する総反復数（set_total_iterations で設定）。
        処理の詳細:
          - 'asymptotic_decay' または 'linear_decay' に従って σ を計算。
        戻り値:
          - float: 現在の σ 値。
        """
        if self.sigma_decay == 'asymptotic_decay':
            return self.sigma0 / (1 + t / (max_iter / 2.0))
        elif self.sigma_decay == 'linear_decay':
            return max(1e-3, self.sigma0 * (1 - t / max_iter))
        else:
            return self.sigma0 / (1 + t / (max_iter / 2.0))

    # ---------- 近傍関数 ----------
    @torch.no_grad()
    def _neighborhood(self, bmu_flat: Tensor, sigma: float) -> Tensor:
        """
        概要:
          BMUノードを中心としたガウシアン近傍重み h を計算する。
        引数:
          - bmu_flat (Tensor): 各サンプルのBMUをフラットインデックスで表した (B,) テンソル。
          - sigma (float): 近傍幅。
        処理の詳細:
          - SOMグリッド座標上のユークリッド距離の二乗 d^2 を計算し、h = exp(-d^2/(2σ^2)) を返す。
        戻り値:
          - Tensor: 形状 (B, m) の近傍重み行列。
        """
        bmu_xy = self.grid_coords[bmu_flat]  # (B,2)
        d2 = ((bmu_xy.unsqueeze(1) - self.grid_coords.unsqueeze(0)) ** 2).sum(dim=-1)  # (B,m)
        h = torch.exp(-d2 / (2 * (sigma ** 2) + 1e-9))
        return h

    # ---------- 内部ヘルパ（SSIM） ----------
    @torch.no_grad()
    def _ensure_kernel5(self):
        """
        概要:
          SSIM計算に用いる移動平均カーネル（self._kernel5）を遅延初期化する。
        引数:
          - なし（内部状態を参照）
        処理の詳細:
          - まだ作成されていなければ、(1,1,win,win) の平均フィルタを生成して保持。
        戻り値:
          - なし
        """
        if self._kernel5 is None:
            k = torch.ones((1, 1, self._win5_size, self._win5_size), device=self.device, dtype=self.dtype) / float(self._win5_size * self._win5_size)
            self._kernel5 = k

    def _ssim_pad_tuple(self) -> Tuple[int, int, int, int]:
        """
        概要:
          任意の窓サイズ（奇数/偶数）に対して出力サイズを維持する非対称SAMEパディング量を返す。
        引数:
          - なし（self._win5_size を参照）
        処理の詳細:
          - 左右/上下で (pl, pr, pt, pb) を計算（偶数長では左右非対称になる）。
        戻り値:
          - Tuple[int,int,int,int]: (left, right, top, bottom) のパディング。
        """
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
        概要:
          ユークリッド距離（L2）に基づき、入力バッチ各サンプルと全ノード重みの距離行列を計算する。
        引数:
          - Xb (Tensor): 入力バッチ (B,H,W)。Bはサンプル数、(H,W)は2D場。
        処理の詳細:
          - 入力と重みをそれぞれ (B,D)、(m,D) に平坦化し、||x||^2 + ||w||^2 - 2 x·w で二乗距離を一括算出。
          - 数値誤差での負値を clamp(0) し、最後に平方根を取って L2 距離へ変換（単調変換なのでBMUは不変）。
        戻り値:
          - Tensor: 形状 (B,m) の距離行列。各行がサンプル、各列がSOMノードに対応。
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
        概要:
          κ(X) = div(∇X/|∇X|) を内側グリッドで中心差分により計算し、κの2D場を返す。
        引数:
          - X (Tensor): 入力2D場のバッチ (B,H,W)。
        処理の詳細:
          - 勾配 (gx, gy) とその正規化 (nx, ny) を算出。
          - 中心差分で ∂nx/∂x + ∂ny/∂y を評価し、周縁2セル分を除いた (H-3,W-3) を返す。
        戻り値:
          - Tensor: 形状 (B,H-3,W-3) の曲率場。
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
        概要:
          Kappa 曲率距離 D_k を [0,1] 範囲で計算し、入力バッチ対全ノードの距離行列を返す。
        引数:
          - Xb (Tensor): 入力バッチ (B,H,W)。
          - nodes_chunk (Optional[int]): ノード側の分割サイズ。None なら self.nodes_chunk。
        処理の詳細:
          - κ(Z)=div(∇Z/|∇Z|) を中心差分で内側グリッド (H-3,W-3) 上に計算。
          - 距離は D_k = 0.5 * Σ|κ(X)-κ(W)| / Σmax(|κ(X)|,|κ(W)|) を用いる。
          - ノードをチャンクに分割して κ(W) を逐次計算し、(B,m) を埋める。
        戻り値:
          - Tensor: 形状 (B,m) の距離行列（各要素は [0,1]）。
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
        概要:
          Teweles–Wobus S1 距離をサンプル一括で全ノードに対して計算する。
        引数:
          - Xb (Tensor): 入力バッチ (B,H,W)。
          - nodes_chunk (Optional[int]): ノード側の分割処理チャンクサイズ。None なら self.nodes_chunk。
        処理の詳細:
          - 入力のx/y方向一次差分を計算し、各ノード重みの差分と比較。
          - |∇X-∇W| の総和を max(|∇X|,|∇W|) の総和で割り、100倍したS1を (B,m) で返す。
        戻り値:
          - Tensor: 形状 (B,m) のS1距離。
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
        """
        概要:
          S1 と κ（Kappa 曲率）距離を"理論値域"で正規化した後、RMSで合成した距離を返す。
        引数:
          - Xb (Tensor): 入力バッチ (B,H,W)。
          - nodes_chunk (Optional[int]): ノード分割処理のチャンク。
        処理の詳細:
          - _s1_distance_batch（値域おおむね[0,200]）と _kappa_distance_batch（[0,1]）を計算。
          - 行内min–maxではなく、理論値域で正規化（S1は200で割って[0,1]へ、κは[0,1]のまま）し、必要に応じてclamp。
          - 正規化後に RMS 合成: sqrt((S1n^2 + κ^2)/2) を返す。
        戻り値:
          - Tensor: 形状 (B,m) の合成距離。
        """
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk
        d1 = self._s1_distance_batch(Xb, nodes_chunk=nodes_chunk)   # ~[0,200]
        dk = self._kappa_distance_batch(Xb, nodes_chunk=nodes_chunk) # [0,1]
        # 理論値域で正規化（行内min–maxは使わない）
        d1n = torch.clamp(d1 / 200.0, min=0.0, max=1.0)
        dkn = torch.clamp(dk, min=0.0, max=1.0)
        return torch.sqrt((d1n * d1n + dkn * dkn) / 2.0)

    @torch.no_grad()
    def _msssim_kappa_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        概要:
          マルチスケールSSIMとκ（Kappa 曲率）距離を"理論値域"で正規化した後、RMSで合成した距離を返す。
        引数:
          - Xb (Tensor): 入力バッチ (B,H,W)。
          - nodes_chunk (Optional[int]): ノード分割処理のチャンク。
        処理の詳細:
          - _msssim_distance_batch（値域[0,1]）と _kappa_distance_batch（[0,1]）を計算。
          - 両方とも[0,1]範囲なので、そのままRMS合成: sqrt((MSSSIM^2 + κ^2)/2) を返す。
        戻り値:
          - Tensor: 形状 (B,m) の合成距離。
        """
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk
        dms = self._msssim_distance_batch(Xb, nodes_chunk=nodes_chunk)  # [0,1]
        dk = self._kappa_distance_batch(Xb, nodes_chunk=nodes_chunk)    # [0,1]
        return torch.sqrt((dms * dms + dk * dk) / 2.0)

    @torch.no_grad()
    def _ssim5_kappa_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        概要:
          SSIM5とκ（Kappa 曲率）距離を"理論値域"で正規化した後、RMSで合成した距離を返す。
        引数:
          - Xb (Tensor): 入力バッチ (B,H,W)。
          - nodes_chunk (Optional[int]): ノード分割処理のチャンク。
        処理の詳細:
          - _ssim5_distance_batch（値域[0,1]）と _kappa_distance_batch（[0,1]）を計算。
          - 両方とも[0,1]範囲なので、そのままRMS合成: sqrt((SSIM5^2 + κ^2)/2) を返す。
        戻り値:
          - Tensor: 形状 (B,m) の合成距離。
        """
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk
        ds5 = self._ssim5_distance_batch(Xb, nodes_chunk=nodes_chunk)  # [0,1]
        dk = self._kappa_distance_batch(Xb, nodes_chunk=nodes_chunk)   # [0,1]
        return torch.sqrt((ds5 * ds5 + dk * dk) / 2.0)

    @torch.no_grad()
    def _msssim_s1_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        概要:
          マルチスケールSSIMとS1距離を"理論値域"で正規化した後、RMSで合成した距離を返す。
        引数:
          - Xb (Tensor): 入力バッチ (B,H,W)。
          - nodes_chunk (Optional[int]): ノード分割処理のチャンク。
        処理の詳細:
          - _msssim_distance_batch（値域[0,1]）と _s1_distance_batch（値域おおむね[0,200]）を計算。
          - S1を200で割って[0,1]へ正規化し、RMS合成: sqrt((MSSSIM^2 + S1n^2)/2) を返す。
        戻り値:
          - Tensor: 形状 (B,m) の合成距離。
        """
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk
        dms = self._msssim_distance_batch(Xb, nodes_chunk=nodes_chunk)  # [0,1]
        d1 = self._s1_distance_batch(Xb, nodes_chunk=nodes_chunk)       # ~[0,200]
        d1n = torch.clamp(d1 / 200.0, min=0.0, max=1.0)
        return torch.sqrt((dms * dms + d1n * d1n) / 2.0)

    @torch.no_grad()
    def _ssim5_s1_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        概要:
          SSIM5とS1距離を"理論値域"で正規化した後、RMSで合成した距離を返す。
        引数:
          - Xb (Tensor): 入力バッチ (B,H,W)。
          - nodes_chunk (Optional[int]): ノード分割処理のチャンク。
        処理の詳細:
          - _ssim5_distance_batch（値域[0,1]）と _s1_distance_batch（値域おおむね[0,200]）を計算。
          - S1を200で割って[0,1]へ正規化し、RMS合成: sqrt((SSIM5^2 + S1n^2)/2) を返す。
        戻り値:
          - Tensor: 形状 (B,m) の合成距離。
        """
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk
        ds5 = self._ssim5_distance_batch(Xb, nodes_chunk=nodes_chunk)  # [0,1]
        d1 = self._s1_distance_batch(Xb, nodes_chunk=nodes_chunk)      # ~[0,200]
        d1n = torch.clamp(d1 / 200.0, min=0.0, max=1.0)
        return torch.sqrt((ds5 * ds5 + d1n * d1n) / 2.0)



    @torch.no_grad()
    def _gaussian_kernel1d(self, sigma: float, dtype: torch.dtype, device: torch.device) -> Tensor:
        """
        概要:
          1次元ガウスカーネルを生成するユーティリティ。σから半径≈3σを取り、奇数長のカーネルを返す。
        引数:
          - sigma (float): ガウス分布の標準偏差。0以下の場合は [1] を返す。
          - dtype (torch.dtype): 生成するカーネルのデータ型。
          - device (torch.device): 配置先デバイス（CPU/GPU）。
        処理の詳細:
          - 半径 r = ceil(3σ) を用い、[-r, r] の整数座標に対して exp(-x^2/(2σ^2)) を評価。
          - 総和で正規化して、重みが1になるように調整。
        戻り値:
          - Tensor: 形状 (2r+1,) の正規化済みカーネル。
        """
        # radius ~ 3*sigma, ensure odd length
        if sigma <= 0:
            k = torch.tensor([1.0], device=device, dtype=dtype)
            return k / k.sum()
        r = max(1, int(math.ceil(3.0 * float(sigma))))
        xs = torch.arange(-r, r + 1, device=device, dtype=dtype)
        k = torch.exp(-(xs * xs) / (2.0 * (sigma ** 2)))
        return k / (k.sum() + 1e-12)

    @torch.no_grad()
    def _gaussian_blur2d(self, Z: Tensor, sigma: float) -> Tensor:
        """
        概要:
          2次元配列に対して分離可能なガウスぼかし（横→縦）を適用する。
        引数:
          - Z (Tensor): 入力2Dテンソル (H,W)。
          - sigma (float): ぼかしの標準偏差。0や小さい値ではほぼ恒等になる。
        処理の詳細:
          - _gaussian_kernel1d で1Dカーネルを生成し、reflectパディングで conv2d を2回（横/縦）適用。
          - 分離可能フィルタにより計算量を低減。
        戻り値:
          - Tensor: 形状 (H,W) のぼかし後テンソル。
        """
        # Z: (H,W) -> (H,W), separable conv with reflect padding
        H, W = Z.shape
        device, dtype = Z.device, Z.dtype
        k1 = self._gaussian_kernel1d(sigma, dtype, device)
        k1c = k1.view(1, 1, 1, -1)
        k1r = k1.view(1, 1, -1, 1)
        # horizontal
        pad_h = (k1.shape[0] // 2)
        X = Z.view(1, 1, H, W)
        Xh = F.pad(X, (pad_h, pad_h, 0, 0), mode='reflect')
        Xh = F.conv2d(Xh, k1c)
        # vertical
        pad_v = (k1.shape[0] // 2)
        Xv = F.pad(Xh, (0, 0, pad_v, pad_v), mode='reflect')
        Xs = F.conv2d(Xv, k1r)
        return Xs.squeeze(0).squeeze(0)

    @torch.no_grad()
    def _distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        概要:
          現在設定されている activation_distance に応じて、(B,m) の距離行列を返すディスパッチャ。
        引数:
          - Xb (Tensor): 入力バッチ (B,H,W)。
          - nodes_chunk (Optional[int]): ノード分割処理のチャンク。
        処理の詳細:
          - 's1'/'euclidean'/'ssim5'/'kappa'/'s1k'/'msssim'/'msssim_kappa'/'ssim5_kappa'/'msssim_s1'/'ssim5_s1' の各実装を呼び分ける。
        戻り値:
          - Tensor: 形状 (B,m) の距離。
        """
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
        elif self.activation_distance == 'msssim':
            return self._msssim_distance_batch(Xb, nodes_chunk=nodes_chunk)
        elif self.activation_distance == 'msssim_kappa':
            return self._msssim_kappa_distance_batch(Xb, nodes_chunk=nodes_chunk)
        elif self.activation_distance == 'ssim5_kappa':
            return self._ssim5_kappa_distance_batch(Xb, nodes_chunk=nodes_chunk)
        elif self.activation_distance == 'msssim_s1':
            return self._msssim_s1_distance_batch(Xb, nodes_chunk=nodes_chunk)
        elif self.activation_distance == 'ssim5_s1':
            return self._ssim5_s1_distance_batch(Xb, nodes_chunk=nodes_chunk)
        else:
            raise RuntimeError('Unknown activation_distance')

    @torch.no_grad()
    def _msssim_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        概要:
          マルチスケールSSIMに基づく距離 D=1-Π_s SSIM_s^{w_s} を、バッチ対全ノードで計算する。
        引数:
          - Xb (Tensor): 入力バッチ (B,H,W)。
          - nodes_chunk (Optional[int]): ノード側の分割サイズ。None なら self.nodes_chunk。
        処理の詳細:
          - ノードをチャンクに分割し、各ノード重みを参照として _msssim_to_ref を呼び出す。
          - 各サンプルに対し (B,m) の距離行列を埋める。数値は [0,1] に収まる想定。
        戻り値:
          - Tensor: 形状 (B,m) の距離行列。
        """
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk
        B = Xb.shape[0]
        out = torch.empty((B, self.m), device=Xb.device, dtype=self.dtype)
        for start in range(0, self.m, nodes_chunk):
            end = min(start + nodes_chunk, self.m)
            for k in range(start, end):
                out[:, k] = self._msssim_to_ref(Xb, self.weights[k])
        return out





    @torch.no_grad()
    def bmu_indices(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        概要:
          入力バッチ各サンプルに対して、最短距離ノード（BMU）のフラットインデックスを返す。
        引数:
          - Xb (Tensor): 入力バッチ (B,H,W)。
          - nodes_chunk (Optional[int]): 距離計算でのノード分割チャンク。
        処理の詳細:
          - _distance_batch で (B,m) の距離を求め、各行の最小位置をargminで取得。
        戻り値:
          - Tensor: 形状 (B,) のBMUフラットインデックス。
        """
        dists = self._distance_batch(Xb, nodes_chunk=nodes_chunk)
        bmu = torch.argmin(dists, dim=1)
        return bmu

    # ---------- 距離計算（バッチ→単一参照：メドイド置換等で使用） ----------
    @torch.no_grad()
    def _euclidean_to_ref(self, Xb: Tensor, ref: Tensor) -> Tensor:
        """
        概要:
          参照1枚 ref に対する各サンプルのユークリッド距離（L2）を計算する。
        引数:
          - Xb (Tensor): 入力バッチ (B,H,W)。
          - ref (Tensor): 参照 (H,W)。
        処理の詳細:
          - 差分の二乗和をとり平方根を取って (B,) の距離ベクトルを返す。
        戻り値:
          - Tensor: 形状 (B,) の距離。
        """
        # Xb: (B,H,W), ref: (H,W) -> (B,)
        diff = Xb - ref.view(1, *ref.shape)
        d2 = (diff * diff).sum(dim=(1, 2))
        return torch.sqrt(d2 + 1e-12)

    @torch.no_grad()
    def _ssim5_to_ref(self, Xb: Tensor, ref: Tensor) -> Tensor:
        """
        概要:
          5x5 窓（C=0）のSSIMに基づく対参照距離 D = 1 - mean(SSIM_map) を計算する。
        引数:
          - Xb (Tensor): 入力バッチ (B,H,W)。
          - ref (Tensor): 参照 (H,W)。
        処理の詳細:
          - 反射パディングと移動平均により局所統計を算出し、SSIMマップを平均後 1-SSIM を距離とする。
        戻り値:
          - Tensor: 形状 (B,) の距離。
        """
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
        """
        概要:
          対参照の Teweles–Wobus S1 距離を計算する。
        引数:
          - Xb (Tensor): 入力バッチ (B,H,W)。
          - ref (Tensor): 参照 (H,W)。
        処理の詳細:
          - x/y方向の一次差分を用いて |∇X-∇ref| / max(|∇X|,|∇ref|) を計算し、100倍して返す。
        戻り値:
          - Tensor: 形状 (B,) のS1距離。
        """
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
        概要:
          対参照の Kappa 曲率距離 D_k を [0,1] で計算する。
        引数:
          - Xb (Tensor): 入力バッチ (B,H,W)。
          - ref (Tensor): 参照 (H,W)。
        処理の詳細:
          - κ(X)=div(∇X/|∇X|) を内側グリッドで算出し、0.5 * Σ|κ(X)-κ(ref)| / Σmax(|κ(X)|,|κ(ref)|) を返す。
        戻り値:
          - Tensor: 形状 (B,) の距離。
        """
        eps = 1e-12
        kx = self._kappa_field(Xb)                       # (B,hk,wk)
        kr = self._kappa_field(ref.unsqueeze(0)).squeeze(0)  # (hk,wk)
        num = torch.abs(kx - kr.unsqueeze(0)).flatten(1).sum(dim=1)
        den = torch.maximum(kx.abs(), kr.abs().unsqueeze(0)).flatten(1).sum(dim=1)
        return 0.5 * (num / (den + eps))

    @torch.no_grad()
    def _s1k_to_ref(self, Xb: Tensor, ref: Tensor) -> Tensor:
        """
        概要:
          対参照の S1 と κ 距離を“理論値域”で正規化し、RMSで合成した距離を返す。
        引数:
          - Xb (Tensor): 入力バッチ (B,H,W)。
          - ref (Tensor): 参照 (H,W)。
        処理の詳細:
          - _s1_to_ref（~[0,200]）および _kappa_to_ref（[0,1]）を計算。
          - 行内min–max正規化の代わりに、S1は200で割って[0,1]へ、κは[0,1]のままとしてclamp。
          - 正規化後に RMS 合成: sqrt((S1n^2 + κ^2)/2) を返す。
        戻り値:
          - Tensor: 形状 (B,) の合成距離。
        """
        d1 = self._s1_to_ref(Xb, ref)   # ~[0,200]
        dk = self._kappa_to_ref(Xb, ref) # [0,1]
        d1n = torch.clamp(d1 / 200.0, min=0.0, max=1.0)
        dkn = torch.clamp(dk, min=0.0, max=1.0)
        return torch.sqrt((d1n * d1n + dkn * dkn) / 2.0)


    @torch.no_grad()
    def _msssim_to_ref(self, Xb: Tensor, ref: Tensor) -> Tensor:
        """
        概要:
          単一参照 ref に対するマルチスケールSSIM距離を計算する（D = 1 - Π_s SSIM_s^{w_s}）。
        引数:
          - Xb (Tensor): 入力バッチ (B,H,W)。
          - ref (Tensor): 参照2D場 (H,W)。
        処理の詳細:
          - スケール s∈{1,2,4} で適応平均プーリングにより縮小し、各スケールで SSIM(5x5, C=0) を算出。
          - 各スケールSSIMを重み w=[0.5,0.3,0.2] でべき乗積（幾何平均相当）し、D=1-積 として距離化。
        戻り値:
          - Tensor: 形状 (B,) の距離ベクトル（概ね [0,1]）。
        """
        scales = [1, 2, 4]
        w = torch.tensor([0.5, 0.3, 0.2], device=Xb.device, dtype=self.dtype)
        w = w / w.sum()
        sims = []
        for s in scales:
            if s == 1:
                Xs = Xb
                Rs = ref
            else:
                h2 = max(2, Xb.shape[-2] // s)
                w2 = max(2, Xb.shape[-1] // s)
                Xs = F.adaptive_avg_pool2d(Xb.unsqueeze(1), (h2, w2)).squeeze(1)
                Rs = F.adaptive_avg_pool2d(ref.view(1, 1, *ref.shape), (h2, w2)).squeeze(0).squeeze(0)
            d = self._ssim5_to_ref(Xs, Rs)                 # (B,)
            sims.append((1.0 - d).clamp(0.0, 1.0))
        sim_prod = torch.ones_like(sims[0])
        for i in range(len(scales)):
            sim_prod = sim_prod * (sims[i] ** w[i])
        return 1.0 - sim_prod





    @torch.no_grad()
    def _msssim_kappa_to_ref(self, Xb: Tensor, ref: Tensor) -> Tensor:
        """対参照のMSSSIM+κ RMS合成距離"""
        dms = self._msssim_to_ref(Xb, ref)   # [0,1]
        dk = self._kappa_to_ref(Xb, ref)     # [0,1]
        return torch.sqrt((dms * dms + dk * dk) / 2.0)

    @torch.no_grad()
    def _ssim5_kappa_to_ref(self, Xb: Tensor, ref: Tensor) -> Tensor:
        """対参照のSSIM5+κ RMS合成距離"""
        ds5 = self._ssim5_to_ref(Xb, ref)    # [0,1]
        dk = self._kappa_to_ref(Xb, ref)     # [0,1]
        return torch.sqrt((ds5 * ds5 + dk * dk) / 2.0)

    @torch.no_grad()
    def _msssim_s1_to_ref(self, Xb: Tensor, ref: Tensor) -> Tensor:
        """対参照のMSSSIM+S1 RMS合成距離"""
        dms = self._msssim_to_ref(Xb, ref)   # [0,1]
        d1 = self._s1_to_ref(Xb, ref)        # ~[0,200]
        d1n = torch.clamp(d1 / 200.0, min=0.0, max=1.0)
        return torch.sqrt((dms * dms + d1n * d1n) / 2.0)

    @torch.no_grad()
    def _ssim5_s1_to_ref(self, Xb: Tensor, ref: Tensor) -> Tensor:
        """対参照のSSIM5+S1 RMS合成距離"""
        ds5 = self._ssim5_to_ref(Xb, ref)    # [0,1]
        d1 = self._s1_to_ref(Xb, ref)        # ~[0,200]
        d1n = torch.clamp(d1 / 200.0, min=0.0, max=1.0)
        return torch.sqrt((ds5 * ds5 + d1n * d1n) / 2.0)

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
        elif self.activation_distance == 'msssim':
            return self._msssim_to_ref(Xb, ref)
        elif self.activation_distance == 'msssim_kappa':
            return self._msssim_kappa_to_ref(Xb, ref)
        elif self.activation_distance == 'ssim5_kappa':
            return self._ssim5_kappa_to_ref(Xb, ref)
        elif self.activation_distance == 'msssim_s1':
            return self._msssim_s1_to_ref(Xb, ref)
        elif self.activation_distance == 'ssim5_s1':
            return self._ssim5_s1_to_ref(Xb, ref)
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
        概要:
          ミニバッチ版のバッチSOM学習を実行する。BMU算出→ガウシアン近傍重みによる分子/分母の累積→一括更新。
        引数:
          - data (np.ndarray): 学習データ (N, H*W)。
          - num_iteration (int): 今回実行する反復回数（self.global_iterに加算される）。
          - batch_size (int): ミニバッチサイズ。
          - verbose (bool): tqdm による進捗表示の有無。
          - log_interval (int): 量子化誤差（QE）を記録する間隔（反復数）。
          - update_per_iteration (bool): 反復内でも逐次更新を行うか（Trueなら分子/分母をその都度反映）。
          - shuffle (bool): 各反復のデータ順序をランダム化するか。
        処理の詳細:
          - total_iters が未設定なら今回の num_iteration を総回数とみなす。
          - 各反復で σ をスケジュールに従い更新しつつ、全データをミニバッチで走査。
          - BMU と近傍重みを用いて、各ノードの分子/分母を累積し、反復末に一括で weights を更新。
          - medoid_replace_every が設定されていれば、指定周期で各ノード重みを最近傍サンプルへ置換。
          - log_interval ごとに quantization_error を計算し履歴に格納。
        戻り値:
          - List[float]: 収集した量子化誤差（QE）の履歴。
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
        """
        概要:
          量子化誤差（QE）を計算する。各サンプルに対してBMUまでの距離の平均。
        引数:
          - data (np.ndarray): 入力データ (N, H*W)。
          - sample_limit (Optional[int]): サンプリングして評価する上限数（None なら全件）。
          - batch_size (int): バッチ分割のサイズ。
        処理の詳細:
          - eval_indices が設定されていれば固定インデックスに対して評価し、安定した比較を可能にする。
          - _distance_batch を用いて最小距離を集計し、平均を返す。
        戻り値:
          - float: QE の平均値。
        """
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
        """
        概要:
          入力データ各サンプルのBMU座標 (x,y) を推論する。
        引数:
          - data (np.ndarray): 入力データ (N, H*W)。
          - batch_size (int): 距離計算のバッチサイズ。
        処理の詳細:
          - _distance_batch → argmin によりフラットインデックスを取得し、(x,y) に変換して返す。
        戻り値:
          - np.ndarray: 形状 (N,2) の BMU 座標配列（列順は [x, y]）。
        """
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
