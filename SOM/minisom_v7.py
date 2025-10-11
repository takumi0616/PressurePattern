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
    """
    概要:
      NumPy配列を指定デバイス・dtypeのtorch.Tensorへ変換する薄いユーティリティ。
    引数:
      - x (np.ndarray): 変換対象の配列。
      - device (torch.device): 配置先デバイス（CPU/GPU）。
      - dtype (torch.dtype): 変換後のデータ型（既定: torch.float32）。
    処理:
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
    処理:
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
          'gssim'    （勾配構造類似：1 - S_GS）
          'kappa'    （κ 曲率距離：0.5 * Σ|κ(X)-κ(W)| / Σmax(|κ(X)|,|κ(W)|)）
          's1k'      （S1 と κ の RMS 合成；S1 と κ を行方向 min–max 正規化後に RMS）
          'gk'       （G-SSIM と κ の RMS 合成；κ を行方向 min–max 正規化後に RMS）
          's1gk'     （S1 + G-SSIM + κ の RMS 合成；S1 と κ を行方向 min–max 正規化）
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
        """
        概要:
          SOMのインスタンスを初期化する。距離タイプや近傍関数、学習スケジューラ、デバイスなどを設定。
        引数:
          - x, y (int): SOMグリッドの縦横サイズ。ノード数 m = x*y。
          - input_len (int): 入力ベクトル長 (= H*W)。s1_field_shapeと一致している必要がある。
          - sigma (float): 初期近傍幅 σ0。
          - learning_rate (float): 学習率（本実装では明示更新では未使用、互換のため保持）。
          - neighborhood_function (str): 近傍関数（'gaussian' のみサポート）。
          - topology (str): トポロジ（'rectangular'）。
          - activation_distance (str): BMU決定に使う距離タイプ（'euclidean'/'ssim5'/'s1'/'gssim'/'kappa'/'s1k'/'gk'/'s1gk'）。
          - random_seed (Optional[int]): 乱数シード。PyTorch/NumPyへ適用。
          - sigma_decay (str): σ減衰方式（'asymptotic_decay' or 'linear_decay'）。
          - s1_field_shape (Tuple[int,int]): 入力を2Dに復元する形状 (H,W)。
          - device (Optional[str]): 'cpu' | 'cuda' | 'cuda:N'。未指定なら自動判定。
          - dtype (torch.dtype): 計算dtype（既定: float32）。
          - nodes_chunk (int): SSIM/G-SSIM/κなどでノードを分割処理するチャンクサイズ。
          - ssim_window (int): SSIMの窓サイズ（奇数推奨）。
          - area_weight (Optional[np.ndarray]): 画素ごとの重みマップ（例: cos(lat)）。一部距離の重み付けに利用可。
        処理:
          - 乱数シード設定、各種属性/テンソルの初期化、SOM重みのランダム初期化。
          - SSIM用の移動平均カーネルの遅延初期化に備えたパラメータ設定。
        戻り値:
          - なし
        """
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
        if activation_distance not in ('s1', 'euclidean', 'ssim5', 'gssim', 'kappa', 's1k', 'gk', 's1gk'):
            raise ValueError('activation_distance must be one of "s1","euclidean","ssim5","gssim","kappa","s1k","gk","s1gk"')
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
        処理:
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
        処理:
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
        処理:
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
        処理:
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
        処理:
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
        処理:
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
        処理:
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
        処理:
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
        処理:
          - 左右/上下で (pl, pr, pt, pb) を計算（偶数長では左右非対称になる）。
        戻り値:
          - Tuple[int,int,int,int]: (left, right, top, bottom) のパディング。
        """
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

        値域:
          - 理論値: [0, +∞)
          - 実装上: sqrt(ε) ≈ 1e-6 以上（完全一致でも sqrt(d^2 + ε) によりわずかに正）
        根拠:
          - d^2 = Σ (x−w)^2 ≥ 0、距離 = sqrt(d^2)。非負性は二乗和と平方根から直ちに従う。
        """
        B, H, W = Xb.shape  # バッチ数Bと空間サイズH,Wを取得
        Xf = Xb.reshape(B, -1)                # (B,D) に平坦化（各サンプルの全画素を一次元ベクトル化）
        Wf = self.weights.reshape(self.m, -1) # (m,D) に平坦化（各ノード重みの全画素を一次元ベクトル化）
        x2 = (Xf * Xf).sum(dim=1, keepdim=True)         # (B,1) 各サンプルの二乗ノルム ||x||^2 を計算
        w2 = (Wf * Wf).sum(dim=1, keepdim=True).T       # (1,m) 各ノード重みの二乗ノルム ||w||^2 を転置してブロードキャスト準備
        cross = Xf @ Wf.T                                # (B,m) サンプルと各ノード重みの内積 x·w
        d2 = x2 + w2 - 2 * cross  # (B,m) ベクトル恒等式 ||x-w||^2 = ||x||^2 + ||w||^2 - 2 x·w により二乗距離を算出
        d2 = torch.clamp(d2, min=0.0)  # 数値誤差で負に振れた場合の下限0クリップ
        return torch.sqrt(d2 + 1e-12)  # L2距離へ平方根（εでゼロ割回避の数値安定化）

    @torch.no_grad()
    def _ssim5_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        SSIM: 5x5移動窓・C=0（分母のみ数値安定化）
        Xb: (B,H,W)
        戻り: (B,m) の "距離" = 1 - mean(SSIM_map)

        値域:
          - 理論値: [0, 2] 近傍（S∈[-1,1] → D=1−S∈[0,2]）
        根拠:
          - 輝度項 2μxμw/(μx^2+μw^2) ∈ [-1,1]、構造・コントラスト項 2cov/(σx^2+σw^2) ∈ [-1,1]。
            その積と画素平均も [-1,1]。よって距離 1−S は [0,2]。分母に ε を置くため端点近傍で微小な逸脱があり得る。
        """
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk  # ノードごとの分割処理サイズ（VRAM節約のためチャンク計算）
        self._ensure_kernel5()  # 5×5の平均フィルタカーネルを遅延初期化
        eps = 1e-12  # 数値安定化用の微小量
        B, H, W = Xb.shape  # 入力バッチと空間サイズ
        out = torch.empty((B, self.m), device=Xb.device, dtype=self.dtype)  # 出力距離配列 (B,m) を確保

        # X 側のローカル統計
        X = Xb.unsqueeze(1)  # (B,1,H,W) チャンネル次元を追加（畳み込みのため）
        X_pad = F.pad(X, self._ssim_pad_tuple(), mode='reflect')  # SAMEサイズを保つ反射パディング
        mu_x = F.conv2d(X_pad, self._kernel5, padding=0)                      # (B,1,H,W) 局所平均 μx
        mu_x2 = F.conv2d(X_pad * X_pad, self._kernel5, padding=0)             # (B,1,H,W) 局所平均 E[x^2]
        var_x = torch.clamp(mu_x2 - mu_x * mu_x, min=0.0)                     # (B,1,H,W) 分散 σx^2（負の丸め誤差を0で下限）

        for start in range(0, self.m, nodes_chunk):  # 全ノードをチャンクに分けて処理
            end = min(start + nodes_chunk, self.m)  # チャンク終端
            Wc = self.weights[start:end].unsqueeze(1)                          # (Mc,1,H,W) 重みパッチ
            W_pad = F.pad(Wc, self._ssim_pad_tuple(), mode='reflect')  # 反射パディング
            mu_w = F.conv2d(W_pad, self._kernel5, padding=0)                  # (Mc,1,H,W) 局所平均 μw
            mu_w2 = F.conv2d(W_pad * W_pad, self._kernel5, padding=0)         # (Mc,1,H,W) 局所平均 E[w^2]
            var_w = torch.clamp(mu_w2 - mu_w * mu_w, min=0.0)                 # (Mc,1,H,W) 分散 σw^2

            # 共分散: mean(x*w) - mu_x*mu_w
            prod = (X.unsqueeze(1) * Wc.unsqueeze(0)).reshape(B * (end - start), 1, H, W)  # (B*Mc,1,H,W) 画素積 x·w
            prod_pad = F.pad(prod, self._ssim_pad_tuple(), mode='reflect')  # 反射パディング
            mu_xw = F.conv2d(prod_pad, self._kernel5, padding=0).reshape(B, end - start, 1, H, W)  # (B,Mc,1,H,W) E[xw]

            mu_x_b = mu_x.unsqueeze(1)                         # (B,1,1,H,W) ブロードキャスト用に次元整形
            mu_w_mc = mu_w.unsqueeze(0)                        # (1,Mc,1,H,W)
            var_x_b = var_x.unsqueeze(1)                       # (B,1,1,H,W)
            var_w_mc = var_w.unsqueeze(0)                      # (1,Mc,1,H,W)
            cov = mu_xw - (mu_x_b * mu_w_mc)                   # (B,Mc,1,H,W) 共分散 cov(x,w)

            # SSIMマップ（C1=C2=0だが分母にのみepsガード）
            l_num = 2 * (mu_x_b * mu_w_mc)  # 輝度項の分子 2μxμw
            l_den = (mu_x_b * mu_x_b + mu_w_mc * mu_w_mc)  # 輝度項の分母 μx^2+μw^2
            c_num = 2 * cov  # 構造・コントラスト項の分子 2cov
            c_den = (var_x_b + var_w_mc)  # 構造・コントラスト項の分母 σx^2+σw^2
            ssim_map = (l_num * c_num) / (l_den * c_den + eps)               # (B,Mc,1,H,W) SSIM = (2μxμw)(2cov)/((μx^2+μw^2)(σx^2+σw^2))

            # 空間平均
            ssim_avg = ssim_map.mean(dim=(2, 3, 4))                          # (B,Mc) SSIMの画素平均
            out[:, start:end] = 1.0 - ssim_avg  # 距離 = 1 - SSIM

        return out  # (B,m) SSIM5距離を返す

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
        Uses inner grid (H-3, W-3)

        値域:
          - 理論値: [0, 1]
        根拠:
          - 各画素で |a−b| ≤ |a|+|b| ≤ 2·max(|a|,|b|)。総和に拡張しても比 ≤ 2、係数0.5で [0,1] に収まる。
        """
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk  # ノードをチャンク分割して曲率距離を計算
        eps = 1e-12  # 数値安定化
        kx = self._kappa_field(Xb)                               # (B,hk,wk) 入力の曲率場 κ(X)
        out = torch.empty((Xb.shape[0], self.m), device=Xb.device, dtype=self.dtype)  # 出力 (B,m)
        for start in range(0, self.m, nodes_chunk):  # チャンクごとの計算ループ
            end = min(start + nodes_chunk, self.m)  # チャンク終端
            kw = self._kappa_field(self.weights[start:end])      # (Mc,hk,wk) 重み側の曲率場 κ(W)
            diff = torch.abs(kx.unsqueeze(1) - kw.unsqueeze(0))  # (B,Mc,hk,wk) |κ(X)−κ(W)|
            num = diff.flatten(2).sum(dim=2)                     # (B,Mc) 分子：絶対差の総和
            den = torch.maximum(kx.abs().unsqueeze(1), kw.abs().unsqueeze(0)).flatten(2).sum(dim=2)  # (B,Mc) 分母：max(|κX|,|κW|)の総和
            out[:, start:end] = 0.5 * (num / (den + eps))  # (B,Mc) 0.5×正規化比（εでゼロ割回避）
        return out  # (B,m)

    @torch.no_grad()
    def _gssim_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        Gradient-Structure Similarity (G-SSIM) に基づく距離。D = 1 - S_GS
        S_GS = Σ[w·S_mag·S_dir] / (Σ w + ε),  w = max(|∇X|, |∇W|)

        値域:
          - 理論値: [0, 1]
        根拠:
          - Smag = 2GxGw/(Gx^2+Gw^2) ∈ [0,1]、Sdir = (1+cosθ)/2 ∈ [0,1]。
            w≥0 の重み付き平均 S は [0,1]、よって D=1−S は [0,1]。
        """
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk  # ノードをチャンク分割
        eps = 1e-12  # 数値安定化
        B, H, W = Xb.shape  # 入力サイズ

        # サンプル側勾配（共通領域）
        dXdx = Xb[:, :, 1:] - Xb[:, :, :-1]  # x方向一次差分
        dXdy = Xb[:, 1:, :] - Xb[:, :-1, :]  # y方向一次差分
        gx = dXdx[:, :-1, :]  # 共通内部格子へ位置合わせ（右端/下端を除外） 
        gy = dXdy[:, :, :-1]  # 共通内部格子へ位置合わせ
        gmagX = torch.sqrt(gx * gx + gy * gy + eps)  # (B, H-1, W-1) 勾配強度 G^x

        out = torch.empty((B, self.m), device=Xb.device, dtype=self.dtype)  # (B,m) 出力バッファ

        dWdx_full = self.weights[:, :, 1:] - self.weights[:, :, :-1]  # (m, H, W-1) 重みの x方向一次差分
        dWdy_full = self.weights[:, 1:, :] - self.weights[:, :-1, :]  # (m, H-1, W) 重みの y方向一次差分

        for start in range(0, self.m, nodes_chunk):  # チャンク処理
            end = min(start + nodes_chunk, self.m)
            grx = dWdx_full[start:end, :-1, :]   # (Mc, H-1, W-1) 共通内部格子に整列
            gry = dWdy_full[start:end, :, :-1]   # (Mc, H-1, W-1)
            gmagW = torch.sqrt(grx * grx + gry * gry + eps)  # (Mc, H-1, W-1) 勾配強度 G^w

            gx_b = gx.unsqueeze(1)     # (B,1,H-1,W-1) サンプル勾配x
            gy_b = gy.unsqueeze(1)     # (B,1,H-1,W-1) サンプル勾配y
            gX_b = gmagX.unsqueeze(1)  # (B,1,H-1,W-1) サンプル強度
            grx_m = grx.unsqueeze(0)   # (1,Mc,H-1,W-1) 重み勾配x
            gry_m = gry.unsqueeze(0)   # (1,Mc,H-1,W-1) 重み勾配y
            gW_m = gmagW.unsqueeze(0)  # (1,Mc,H-1,W-1) 重み強度

            dot = gx_b * grx_m + gy_b * gry_m  # 内積 g_x^x g_x^w + g_y^x g_y^w
            cos = (dot / (gX_b * gW_m + eps)).clamp(-1.0, 1.0)  # cosθ を計算し[-1,1]にクランプ
            Sdir = 0.5 * (1.0 + cos)  # 方向一致度 (1+cosθ)/2
            Smag = (2.0 * gX_b * gW_m) / (gX_b * gX_b + gW_m * gW_m + eps)  # 強度一致 2GxGw/(Gx^2+Gw^2)
            S = Smag * Sdir  # 強度×方向の合成類似度
            w = torch.maximum(gX_b, gW_m)  # 重み w = max(Gx,Gw)（強いエッジを重視）
            sim = (S * w).sum(dim=(2, 3)) / (w.sum(dim=(2, 3)) + eps)  # (B,Mc) 加重平均でS_GS
            out[:, start:end] = 1.0 - sim  # 距離 = 1 - S_GS

        return out  # (B,m)

    @torch.no_grad()
    def _s1_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        概要:
          Teweles–Wobus S1 距離をサンプル一括で全ノードに対して計算する。
        引数:
          - Xb (Tensor): 入力バッチ (B,H,W)。
          - nodes_chunk (Optional[int]): ノード側の分割処理チャンクサイズ。None なら self.nodes_chunk。
        処理:
          - 入力のx/y方向一次差分を計算し、各ノード重みの差分と比較。
          - |∇X-∇W| の総和を max(|∇X|,|∇W|) の総和で割り、100倍したS1を (B,m) で返す。
        戻り値:
          - Tensor: 形状 (B,m) のS1距離。

        値域:
          - 理論値: [0, 200]
        根拠:
          - 任意の a,b に対して |a−b| ≤ |a|+|b| ≤ 2·max(|a|,|b|)。総和でも同様に比 ≤ 2。
            100 倍する定義のため上限は 200。
        """
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk  # チャンクサイズ設定
        B, H, W = Xb.shape  # 入力寸法
        dXdx = Xb[:, :, 1:] - Xb[:, :, :-1]  # (B,H,W-1) 入力のx方向一次差分 Δx^x
        dXdy = Xb[:, 1:, :] - Xb[:, :-1, :]  # (B,H-1,W) 入力のy方向一次差分 Δy^x

        out = torch.empty((B, self.m), device=Xb.device, dtype=self.dtype)  # (B,m) 出力
        dWdx_full = self.weights[:, :, 1:] - self.weights[:, :, :-1]  # (m,H,W-1) 重みのx方向一次差分 Δx^w
        dWdy_full = self.weights[:, 1:, :] - self.weights[:, :-1, :]  # (m,H-1,W) 重みのy方向一次差分 Δy^w

        for start in range(0, self.m, nodes_chunk):  # ノードチャンクのループ
            end = min(start + nodes_chunk, self.m)
            dWdx = dWdx_full[start:end]   # (Mc,H,W-1) チャンク切り出し
            dWdy = dWdy_full[start:end]   # (Mc,H-1,W) チャンク切り出し
            num_dx = (torch.abs(dWdx.unsqueeze(0) - dXdx.unsqueeze(1))).sum(dim=(2, 3))  # (B,Mc) |Δx^x-Δx^w| の総和
            num_dy = (torch.abs(dWdy.unsqueeze(0) - dXdy.unsqueeze(1))).sum(dim=(2, 3))  # (B,Mc) |Δy^x-Δy^w| の総和
            num = num_dx + num_dy  # (B,Mc) 分子

            den_dx = (torch.maximum(torch.abs(dWdx).unsqueeze(0), torch.abs(dXdx).unsqueeze(1))).sum(dim=(2, 3))  # (B,Mc) max(|Δx^x|,|Δx^w|)
            den_dy = (torch.maximum(torch.abs(dWdy).unsqueeze(0), torch.abs(dXdy).unsqueeze(1))).sum(dim=(2, 3))  # (B,Mc) max(|Δy^x|,|Δy^w|)
            denom = den_dx + den_dy  # (B,Mc) 分母
            s1 = 100.0 * num / (denom + 1e-12)  # (B,Mc) S1 = 100 × 分子/分母（εでゼロ割回避）
            out[:, start:end] = s1  # 出力に格納

        return out  # (B,m)

    @torch.no_grad()
    def _s1k_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        概要:
          S1 と κ（Kappa 曲率）距離を行方向 min–max 正規化した後、RMSで合成した距離を返す。
        引数:
          - Xb (Tensor): 入力バッチ (B,H,W)。
          - nodes_chunk (Optional[int]): ノード分割処理のチャンク。
        処理:
          - _s1_distance_batch と _kappa_distance_batch を計算し、各行で min–max 正規化 → RMS 合成。
        戻り値:
          - Tensor: 形状 (B,m) の合成距離。

        値域:
          - 理論値: [0, 1]
        根拠:
          - 行方向 min–max 正規化により各成分が [0,1] に入り、RMS(二乗平均平方根) も [0,1] に収まる。
        """
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk  # チャンク設定
        d1 = self._s1_distance_batch(Xb, nodes_chunk=nodes_chunk)  # (B,m) S1距離を計算
        dk = self._kappa_distance_batch(Xb, nodes_chunk=nodes_chunk)  # (B,m) KAPPA距離を計算
        eps = 1e-12  # 数値安定化
        d1_min = d1.min(dim=1, keepdim=True).values  # (B,1) 各行の最小値
        d1_max = d1.max(dim=1, keepdim=True).values  # (B,1) 各行の最大値
        dk_min = dk.min(dim=1, keepdim=True).values  # (B,1) 各行の最小値
        dk_max = dk.max(dim=1, keepdim=True).values  # (B,1) 各行の最大値
        d1n = (d1 - d1_min) / (d1_max - d1_min + eps)  # (B,m) S1の行方向min–max正規化（入力 x を固定し、候補ノード方向でスケーリング）
        dkn = (dk - dk_min) / (dk_max - dk_min + eps)  # (B,m) KAPPAの行方向min–max正規化（入力 x を固定し、候補ノード方向でスケーリング）
        return torch.sqrt((d1n * d1n + dkn * dkn) / 2.0)  # (B,m) RMS統合（2指標）

    @torch.no_grad()
    def _gk_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        概要:
          G-SSIM 距離と κ 距離（行方向 min–max 正規化）をRMSで合成した距離を返す。
        引数:
          - Xb (Tensor): 入力バッチ (B,H,W)。
          - nodes_chunk (Optional[int]): ノード分割処理のチャンク。
        処理:
          - _gssim_distance_batch と _kappa_distance_batch を計算し、κは行方向min–max正規化後にRMS合成。
        戻り値:
          - Tensor: 形状 (B,m) の合成距離。
        """
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk  # チャンク設定
        dg = self._gssim_distance_batch(Xb, nodes_chunk=nodes_chunk)  # (B,m) G-SSIM距離（[0,1]：正規化不要）
        dk = self._kappa_distance_batch(Xb, nodes_chunk=nodes_chunk)  # (B,m) KAPPA距離
        eps = 1e-12  # 数値安定化
        dk_min = dk.min(dim=1, keepdim=True).values  # (B,1) 行ごとの最小
        dk_max = dk.max(dim=1, keepdim=True).values  # (B,1) 行ごとの最大
        dkn = (dk - dk_min) / (dk_max - dk_min + eps)  # (B,m) KAPPAの行方向min–max正規化（入力 x を固定し、候補ノード方向でスケーリング）
        return torch.sqrt((dg * dg + dkn * dkn) / 2.0)  # (B,m) RMS統合（2指標）

    @torch.no_grad()
    def _s1gk_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        S1GK distance: RMS of row-normalized S1, G-SSIM distance, and row-normalized Kappa curvature distance.
        Returns (B,m).

        値域:
          - 理論値: [0, 1]
        根拠:
          - 入力3成分（S1n, G-SSIM, KAn）がそれぞれ [0,1] にあり、3次元RMS の値域は [0,1]。
        """
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk  # チャンク設定
        eps = 1e-12  # 数値安定化
        d1 = self._s1_distance_batch(Xb, nodes_chunk=nodes_chunk)     # (B,m) S1距離
        dg = self._gssim_distance_batch(Xb, nodes_chunk=nodes_chunk)  # (B,m) G-SSIM距離（[0,1]：正規化不要）
        dk = self._kappa_distance_batch(Xb, nodes_chunk=nodes_chunk)  # (B,m) KAPPA距離
        d1_min = d1.min(dim=1, keepdim=True).values  # (B,1) S1の行最小
        d1_max = d1.max(dim=1, keepdim=True).values  # (B,1) S1の行最大
        dk_min = dk.min(dim=1, keepdim=True).values  # (B,1) KAPPAの行最小
        dk_max = dk.max(dim=1, keepdim=True).values  # (B,1) KAPPAの行最大
        d1n = (d1 - d1_min) / (d1_max - d1_min + eps)  # (B,m) S1の行方向min–max正規化（入力 x を固定し、候補ノード方向でスケーリング）
        dkn = (dk - dk_min) / (dk_max - dk_min + eps)  # (B,m) KAPPAの行方向min–max正規化（入力 x を固定し、候補ノード方向でスケーリング）
        return torch.sqrt((d1n * d1n + dg * dg + dkn * dkn) / 3.0)  # (B,m) RMS統合（3指標）

    @torch.no_grad()
    def _distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        概要:
          現在設定されている activation_distance に応じて、(B,m) の距離行列を返すディスパッチャ。
        引数:
          - Xb (Tensor): 入力バッチ (B,H,W)。
          - nodes_chunk (Optional[int]): ノード分割処理のチャンク。
        処理:
          - 's1'/'euclidean'/'ssim5'/'gssim'/'kappa'/'s1k'/'gk'/'s1gk' の各実装を呼び分ける。
        戻り値:
          - Tensor: 形状 (B,m) の距離。
        """
        if self.activation_distance == 's1':
            return self._s1_distance_batch(Xb, nodes_chunk=nodes_chunk)
        elif self.activation_distance == 'euclidean':
            return self._euclidean_distance_batch(Xb)
        elif self.activation_distance == 'ssim5':
            return self._ssim5_distance_batch(Xb, nodes_chunk=nodes_chunk)
        elif self.activation_distance == 'gssim':
            return self._gssim_distance_batch(Xb, nodes_chunk=nodes_chunk)
        elif self.activation_distance == 'kappa':
            return self._kappa_distance_batch(Xb, nodes_chunk=nodes_chunk)
        elif self.activation_distance == 's1k':
            return self._s1k_distance_batch(Xb, nodes_chunk=nodes_chunk)
        elif self.activation_distance == 'gk':
            return self._gk_distance_batch(Xb, nodes_chunk=nodes_chunk)
        elif self.activation_distance == 's1gk':
            return self._s1gk_distance_batch(Xb, nodes_chunk=nodes_chunk)
        else:
            raise RuntimeError('Unknown activation_distance')

    @torch.no_grad()
    def bmu_indices(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        概要:
          入力バッチ各サンプルに対して、最短距離ノード（BMU）のフラットインデックスを返す。
        引数:
          - Xb (Tensor): 入力バッチ (B,H,W)。
          - nodes_chunk (Optional[int]): 距離計算でのノード分割チャンク。
        処理:
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
        処理:
          - 差分の二乗和をとり平方根を取って (B,) の距離ベクトルを返す。
        戻り値:
          - Tensor: 形状 (B,) の距離。
        """
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
        処理:
          - 反射パディングと移動平均により局所統計を算出し、SSIMマップを平均後 1-SSIM を距離とする。
        戻り値:
          - Tensor: 形状 (B,) の距離。
        """
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
        処理:
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
        """
        概要:
          対参照の S1 と κ 距離を行方向min–max正規化し、RMSで合成した距離を返す。
        引数:
          - Xb (Tensor): 入力バッチ (B,H,W)。
          - ref (Tensor): 参照 (H,W)。
        処理:
          - _s1_to_ref, _kappa_to_ref を計算し、min–max正規化後に RMS 合成。
        戻り値:
          - Tensor: 形状 (B,) の合成距離。
        """
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
    def _gk_to_ref(self, Xb: Tensor, ref: Tensor) -> Tensor:
        """
        概要:
          対参照の G-SSIM 距離と κ 距離（min–max正規化）をRMSで合成した距離を返す。
        引数:
          - Xb (Tensor): 入力バッチ (B,H,W)。
          - ref (Tensor): 参照 (H,W)。
        処理:
          - _gssim_to_ref と _kappa_to_ref を計算し、κはmin–max正規化後にRMS合成。
        戻り値:
          - Tensor: 形状 (B,) の合成距離。
        """
        eps = 1e-12
        dg = self._gssim_to_ref(Xb, ref)
        dk = self._kappa_to_ref(Xb, ref)
        dk_min, _ = dk.min(dim=0, keepdim=True)
        dk_max, _ = dk.max(dim=0, keepdim=True)
        dkn = (dk - dk_min) / (dk_max - dk_min + eps)
        return torch.sqrt((dg * dg + dkn * dkn) / 2.0)

    @torch.no_grad()
    def _s1gk_to_ref(self, Xb: Tensor, ref: Tensor) -> Tensor:
        """
        S1GK (to ref): RMS of row-normalized S1, G-SSIM distance, and row-normalized Kappa curvature distance.
        """
        eps = 1e-12
        d1 = self._s1_to_ref(Xb, ref)        # (B,)
        dg = self._gssim_to_ref(Xb, ref)     # (B,)
        dk = self._kappa_to_ref(Xb, ref)     # (B,)
        d1_min, _ = d1.min(dim=0, keepdim=True)
        d1_max, _ = d1.max(dim=0, keepdim=True)
        dk_min, _ = dk.min(dim=0, keepdim=True)
        dk_max, _ = dk.max(dim=0, keepdim=True)
        d1n = (d1 - d1_min) / (d1_max - d1_min + eps)
        dkn = (dk - dk_min) / (dk_max - dk_min + eps)
        return torch.sqrt((d1n * d1n + dg * dg + dkn * dkn) / 3.0)

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
        elif self.activation_distance == 'gssim':
            return self._gssim_to_ref(Xb, ref)
        elif self.activation_distance == 'kappa':
            return self._kappa_to_ref(Xb, ref)
        elif self.activation_distance == 's1k':
            return self._s1k_to_ref(Xb, ref)
        elif self.activation_distance == 'gk':
            return self._gk_to_ref(Xb, ref)
        elif self.activation_distance == 's1gk':
            return self._s1gk_to_ref(Xb, ref)
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
          - update_per_iteration (bool): 反復内でも閾値ごとに逐次更新するか（Trueなら分子/分母をその都度反映）。
          - shuffle (bool): 各反復のデータ順序をランダム化するか。
        処理:
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
        処理:
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
        処理:
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
