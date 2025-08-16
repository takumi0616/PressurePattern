import math
import numpy as np
from typing import Tuple, Optional, List

import torch
from torch import Tensor

try:
    from tqdm import trange
except Exception:
    # tqdmが無い環境でも動くようにダミー
    def trange(n, **kwargs):
        return range(n)


def _to_device(x: np.ndarray, device: torch.device, dtype=torch.float32) -> Tensor:
    return torch.as_tensor(x, device=device, dtype=dtype)


def _as_numpy(x: Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


class MiniSom:
    """
    PyTorch GPU版 SOM + S1距離（Teweles & Wobus, 1954）対応
    - activation_distance='s1' のみサポート（本タスクの主題のため）
    - 学習は「ミニバッチ版バッチSOM」：
        1) ミニバッチごとにS1でBMUを決める
        2) BMUに基づくガウス近傍重みを計算し、分子/分母を累積
        3) 1エポックの最後に weights = numerator / denominator を更新
       もしくは、iterationごとに更新するモードの両方をサポート
    - 全ての重い計算（S1距離、BMU推定、学習更新）はGPUで実行

    使い方メモ：
      som = MiniSom(x, y, input_len, s1_field_shape=(nlat, nlon), device='cuda')
      som.random_weights_init(data)   # data: (N, D)
      som.train_batch(data, num_iteration=..., batch_size=..., ...)
      winners = som.predict(data)     # data全体のBMU
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
        """
        :param x, y: SOMグリッドサイズ
        :param input_len: 特徴次元（= nlat*nlon）
        :param sigma: 初期近傍半径
        :param learning_rate: （現状は重み更新の混合率には未使用、将来的な拡張用）
        :param activation_distance: 's1' 固定を想定
        :param s1_field_shape: (nlat, nlon) 必須
        :param device: 'cuda' または 'cpu'. Noneなら自動判定
        :param dtype: torch.float32 推奨（fp16は総和精度が落ちるので非推奨）
        :param nodes_chunk: S1距離計算時にノード方向を分割するチャンクサイズ（VRAMに合わせる）
        """
        if activation_distance.lower() != 's1':
            raise ValueError('This implementation supports only activation_distance="s1".')

        if s1_field_shape is None:
            raise ValueError('activation_distance="s1" requires s1_field_shape=(nlat, nlon).')

        if s1_field_shape[0] * s1_field_shape[1] != input_len:
            raise ValueError(f's1_field_shape={s1_field_shape} does not match input_len={input_len}.')

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.dtype = dtype

        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

        self.x = x
        self.y = y
        self.m = x * y  # ノード数
        self.input_len = input_len
        self.sigma0 = float(sigma)
        self.learning_rate = float(learning_rate)
        self.neighborhood_function = neighborhood_function
        self.topology = topology
        self.sigma_decay = sigma_decay
        self.s1_field_shape = s1_field_shape
        self.nodes_chunk = int(nodes_chunk)

        # SOMノードの座標（格子） (m, 2)
        gx, gy = torch.meshgrid(torch.arange(x), torch.arange(y), indexing='ij')
        self.grid_coords = torch.stack([gx.flatten(), gy.flatten()], dim=1).to(self.device, torch.float32)

        # 重み（画像として扱うため (m, H, W) に持つ）
        H, W = s1_field_shape
        # [-1, 1]の一様乱数
        weights = torch.rand((self.m, H, W), device=self.device, dtype=self.dtype) * 2 - 1
        # 単純に初期化（画像正規化はここではしない）
        self.weights = weights

        # 近傍関数設定（現状gaussianのみ）
        if self.neighborhood_function != 'gaussian':
            print('Warning: only gaussian neighborhood is implemented; falling back to gaussian.')
            self.neighborhood_function = 'gaussian'

    # ---------- ユーティリティ ----------
    def get_weights(self) -> np.ndarray:
        """
        戻り値: (x, y, input_len) のnumpy
        """
        w = self.weights  # (m, H, W)
        H, W = self.s1_field_shape
        w_flat = w.reshape(self.m, H * W)
        w_grid = w_flat.reshape(self.x, self.y, H * W)
        return _as_numpy(w_grid)

    def random_weights_init(self, data: np.ndarray):
        """
        データからランダムにノード数分をサンプリングして初期重みにする
        """
        H, W = self.s1_field_shape
        n = data.shape[0]
        if n < self.m:
            idx = np.random.choice(n, self.m, replace=True)
        else:
            idx = np.random.choice(n, self.m, replace=False)
        w0 = data[idx]  # (m, D)
        w0 = w0.reshape(self.m, H, W)
        self.weights = _to_device(w0, self.device, self.dtype).clone()

    # ---------- スケジューラ ----------
    def _sigma_at(self, t: int, max_iter: int) -> float:
        if self.sigma_decay == 'asymptotic_decay':
            return self.sigma0 / (1 + t / (max_iter / 2.0))
        elif self.sigma_decay == 'linear_decay':
            return max(1e-3, self.sigma0 * (1 - t / max_iter))
        else:
            # デフォルト
            return self.sigma0 / (1 + t / (max_iter / 2.0))

    # ---------- 距離・BMU ----------
    @torch.no_grad()
    def _s1_distance_batch(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        Xb: (B, H, W) on device
        戻り値: (B, m) S1距離（小さいほど近い）
        ノード方向をチャンク分割してVRAMを使いすぎないように計算
        """
        B, H, W = Xb.shape
        if nodes_chunk is None:
            nodes_chunk = self.nodes_chunk

        # 入力の勾配（欠損があればマスク）
        dXdx = Xb[:, :, 1:] - Xb[:, :, :-1]  # (B, H, W-1)
        dXdy = Xb[:, 1:, :] - Xb[:, :-1, :]  # (B, H-1, W)
        mask_dx = torch.isfinite(dXdx).to(Xb.dtype)  # 1/0
        mask_dy = torch.isfinite(dXdy).to(Xb.dtype)

        # NaNを0に（マスクで制御）
        dXdx = torch.where(torch.isfinite(dXdx), dXdx, torch.zeros_like(dXdx))
        dXdy = torch.where(torch.isfinite(dXdy), dXdy, torch.zeros_like(dXdy))

        out = torch.empty((B, self.m), device=Xb.device, dtype=self.dtype)

        # 事前に現在のweightsの勾配を計算（チャンク内で切り出すのでフルを持っておく）
        dWdx_full = self.weights[:, :, 1:] - self.weights[:, :, :-1]  # (m, H, W-1)
        dWdy_full = self.weights[:, 1:, :] - self.weights[:, :-1, :]  # (m, H-1, W)

        # チャンクループ
        for start in range(0, self.m, nodes_chunk):
            end = min(start + nodes_chunk, self.m)
            dWdx = dWdx_full[start:end]  # (Mc, H, W-1)
            dWdy = dWdy_full[start:end]  # (Mc, H-1, W)

            # 分子
            num_dx = (torch.abs(dWdx.unsqueeze(0) - dXdx.unsqueeze(1)) * mask_dx.unsqueeze(1)).sum(dim=(2, 3))
            num_dy = (torch.abs(dWdy.unsqueeze(0) - dXdy.unsqueeze(1)) * mask_dy.unsqueeze(1)).sum(dim=(2, 3))
            num = num_dx + num_dy  # (B, Mc)

            # 分母
            den_dx = (torch.maximum(torch.abs(dWdx).unsqueeze(0), torch.abs(dXdx).unsqueeze(1)) * mask_dx.unsqueeze(1)).sum(dim=(2, 3))
            den_dy = (torch.maximum(torch.abs(dWdy).unsqueeze(0), torch.abs(dXdy).unsqueeze(1)) * mask_dy.unsqueeze(1)).sum(dim=(2, 3))
            denom = den_dx + den_dy  # (B, Mc)

            s1 = 100.0 * num / (denom + 1e-12)
            out[:, start:end] = s1

        return out  # (B, m)

    @torch.no_grad()
    def bmu_indices(self, Xb: Tensor, nodes_chunk: Optional[int] = None) -> Tensor:
        """
        Xb: (B, H, W)
        戻り値: (B,) ノードのフラットインデックス（0..m-1）
        """
        dists = self._s1_distance_batch(Xb, nodes_chunk=nodes_chunk)  # (B, m)
        bmu = torch.argmin(dists, dim=1)
        return bmu

    @torch.no_grad()
    def _neighborhood(self, bmu_flat: Tensor, sigma: float) -> Tensor:
        """
        bmu_flat: (B,) BMUインデックス（0..m-1）
        戻り値: (B, m) 各サンプルのBMUに対する全ノードの近傍重み（ガウス）
        """
        # BMU座標 (B, 2)
        bmu_xy = self.grid_coords[bmu_flat]  # float32
        # 全ノード座標 (m, 2)
        # 距離二乗 (B, m)
        d2 = ((bmu_xy.unsqueeze(1) - self.grid_coords.unsqueeze(0)) ** 2).sum(dim=-1)
        h = torch.exp(-d2 / (2 * (sigma ** 2) + 1e-9))
        return h  # (B, m)

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
        ミニバッチ版 バッチSOM学習
        :param data: (N, D) numpy
        :param num_iteration: 反復回数（おおまかにエポックのように扱う）
        :param batch_size: ミニバッチサイズ（VRAMに合わせて調整）
        :param log_interval: 進捗ログ間隔（iteration単位）
        :param update_per_iteration: Trueなら各iterationで即時weights更新、Falseなら分子/分母を累積してepochごとに更新
        :param shuffle: ミニバッチ抽出時にシャッフル
        :return: 量子化誤差（S1）の履歴（各log_intervalで）
        """
        N, D = data.shape
        H, W = self.s1_field_shape
        if D != H * W:
            raise ValueError(f'data dimension {D} does not match s1_field_shape {self.s1_field_shape}')

        Xall = _to_device(data, self.device, self.dtype).reshape(N, H, W)

        qhist: List[float] = []
        rng_idx = torch.arange(N, device=self.device)

        iterator = trange(num_iteration) if verbose else range(num_iteration)
        for it in iterator:
            sigma = self._sigma_at(it, num_iteration)

            if update_per_iteration:
                # 分子/分母を都度更新（オンラインに近い挙動）
                numerator = torch.zeros_like(self.weights)  # (m, H, W)
                denominator = torch.zeros((self.m,), device=self.device, dtype=self.dtype)
            else:
                # 1 iteration 内で全バッチを累積して最後に一括更新
                numerator = torch.zeros_like(self.weights)
                denominator = torch.zeros((self.m,), device=self.device, dtype=self.dtype)

            if shuffle:
                perm = torch.randperm(N, device=self.device)
                idx_all = rng_idx[perm]
            else:
                idx_all = rng_idx

            # バッチループ
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                batch_idx = idx_all[start:end]
                Xb = Xall[batch_idx]  # (B, H, W)

                # BMU
                bmu = self.bmu_indices(Xb, nodes_chunk=self.nodes_chunk)  # (B,)
                # 近傍重み (B, m)
                h = self._neighborhood(bmu, sigma)  # (B, m)

                # 分子/分母に累積
                # numerator += sum_i h_i[:, None, None] * Xb_i  をベクトル化
                # -> (B, m, H, W) = (B, m, 1, 1) * (B, 1, H, W) をBでsum
                numerator += (h.unsqueeze(-1).unsqueeze(-1) * Xb.unsqueeze(1)).sum(dim=0)  # (m, H, W)
                denominator += h.sum(dim=0)  # (m,)

                if update_per_iteration:
                    # ここで即時更新（分母が0のノードはそのまま）
                    mask = (denominator > 0)
                    denom_safe = denominator.clone()
                    denom_safe[~mask] = 1.0
                    new_w = numerator / denom_safe.view(-1, 1, 1)
                    self.weights[mask] = new_w[mask]
                    # 分子/分母リセット
                    numerator.zero_()
                    denominator.zero_()

            # 1 iterationの最後に更新
            mask = (denominator > 0)
            if mask.any():
                denom_safe = denominator.clone()
                denom_safe[~mask] = 1.0
                new_w = numerator / denom_safe.view(-1, 1, 1)
                self.weights[mask] = new_w[mask]

            # ログ
            if (it % log_interval == 0) or (it == num_iteration - 1):
                qe = self.quantization_error(data, sample_limit=2048)
                qhist.append(qe)
                if verbose:
                    if hasattr(iterator, 'set_postfix'):
                        iterator.set_postfix(q_error=f"{qe:.5f}", sigma=f"{sigma:.3f}")

        return qhist

    # ---------- 評価 ----------
    @torch.no_grad()
    def quantization_error(self, data: np.ndarray, sample_limit: Optional[int] = None, batch_size: int = 64) -> float:
        """
        S1距離のBMU距離の平均（小さいほどよい）
        :param sample_limit: 評価用にランダムサンプリング（Noneなら全体）
        :param batch_size: 推論時のバッチサイズ
        """
        N, D = data.shape
        H, W = self.s1_field_shape
        if sample_limit is not None and sample_limit < N:
            idx = np.random.choice(N, sample_limit, replace=False)
            X = data[idx]
        else:
            X = data
        X = _to_device(X, self.device, self.dtype).reshape(-1, H, W)
        # 距離最小値の平均
        total = 0.0
        cnt = 0
        for start in range(0, X.shape[0], batch_size):
            end = min(start + batch_size, X.shape[0])
            Xb = X[start:end]
            d = self._s1_distance_batch(Xb, nodes_chunk=self.nodes_chunk)  # (B, m)
            mins = torch.min(d, dim=1).values  # (B,)
            total += float(mins.sum().item())
            cnt += Xb.shape[0]
        return total / max(1, cnt)

    @torch.no_grad()
    def predict(self, data: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """
        全データのBMU座標を返す
        戻り値: (N, 2) (x, y)
        """
        N, D = data.shape
        H, W = self.s1_field_shape
        X = _to_device(data, self.device, self.dtype).reshape(N, H, W)

        bmu_all = []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            Xb = X[start:end]
            bmu = self.bmu_indices(Xb, nodes_chunk=self.nodes_chunk)  # (B,)
            bmu_all.append(bmu)
        bmu_flat = torch.cat(bmu_all, dim=0)  # (N,)
        # フラット -> (x, y)
        y = (bmu_flat % self.y).to(torch.long)
        x = (bmu_flat // self.y).to(torch.long)
        out = torch.stack([x, y], dim=1)
        return _as_numpy(out)
